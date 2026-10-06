"""A persistent cache of judge verdicts, so re-inspecting costs no endpoint time.

Reading a chunk's verdict is the expensive part of looking at retrieval: the
20-question run paid ~820 graded chunks (~45-60 min, sequential against an
endpoint shared with everything else in this repo), and without a cache every
`--question-id` inspection of a question you looked at yesterday pays it again.
That price is what stops people from looking, which is how the first run's
18/20 reformulation fallback survived -- nobody re-opened the rows.

Cache file: one JSON object per line, keyed `(question_id, chunk_id)` and stamped
with `judge_prompt_sha()`. A record whose sha is not the current rubric's is
*ignored*, not deleted: the rubric may be reverted, and `--stats` reporting how
many verdicts a rubric change would invalidate is worth more than a tidy file.

Chunk *content* is not part of the key. `chunk_id` identifies the text for a
frozen corpus, so rebuilding the index without bumping `JUDGE_PROMPT_VERSION`
would serve stale verdicts for new prose. The tradeoff is deliberate (see
`judge_prompt.judge_prompt_sha`); the counterweight is that `context.md` records
`index_sha256` next to `judge_prompt_sha`, so a run that pairs a new index with
old verdicts is auditable after the fact.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from loguru import logger

from medical_rag.data.load_medqa import MedQAQuestion
from medical_rag.generation.llm import LLMClient, LLMError
from medical_rag.retrieval.retriever import RetrievedChunk

from medical_rag.eval.rubric import (
    JUDGE_SYSTEM_PROMPT,
    JudgeVerdict,
    build_judge_prompt,
    judge_prompt_sha,
)
from medical_rag.eval.runlog import EndpointCircuitBreaker
from medical_rag.paths import REPO_ROOT

DEFAULT_JUDGE_CACHE = (
    REPO_ROOT
    / "outputs"
    / "exploration"
    / "retrieval_tuning"
    / "judge_cache"
    / "judged_chunks.jsonl"
)

# 40 chunks/batch is what the 20-question run used and where the endpoint's
# structured output started truncating mid-list on longer prompts; half that is
# the default because a lost batch now costs a re-ask instead of a lost run.
DEFAULT_BATCH = 24


def verdict_key(question_id: str, chunk_id: str) -> tuple[str, str]:
    return (str(question_id), str(chunk_id))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_cache(
    path: str | Path = DEFAULT_JUDGE_CACHE, *, judge_sha: str | None = None
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, int]]:
    """Load the cache and report what was skipped, never silently.

    Returns `(verdicts, counts)` with `counts = {"loaded", "stale_prompt",
    "malformed"}`. Torn lines are skipped -- the appender is not transactional,
    and a crash mid-line during a 45-minute run must not make the file unreadable.
    """
    path = Path(path)
    judge_sha = judge_prompt_sha() if judge_sha is None else judge_sha
    verdicts: dict[tuple[str, str], dict[str, Any]] = {}
    counts = {"loaded": 0, "stale_prompt": 0, "malformed": 0, "missing_file": 0}
    if not path.exists():
        counts["missing_file"] = 1
        return verdicts, counts

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["malformed"] += 1
                continue
            if record.get("judge_prompt_sha") != judge_sha:
                counts["stale_prompt"] += 1
                continue
            verdicts[verdict_key(record["question_id"], record["chunk_id"])] = record
            counts["loaded"] += 1
    return verdicts, counts


def append_records(path: str | Path, records: Sequence[dict[str, Any]]) -> None:
    """Append verdicts, one line each, flushed -- a cache lost to a kill -9 is a
    cache that charged the endpoint for work it cannot show."""
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def import_run(
    run_dir: str | Path,
    cache_path: str | Path = DEFAULT_JUDGE_CACHE,
    *,
    judge_sha: str | None = None,
) -> dict[str, int]:
    """Seed the cache from a finished run's `results.jsonl`.

    The 20-question run already paid for ~820 verdicts; without this, every
    inspection of those questions re-bills them. `source_run` is stamped on each
    record so a verdict can be traced back to the run that produced it, and rows
    with `error` are skipped -- an error row's judgments dict is empty anyway, but
    the filter keeps a partially-written row from being trusted.
    """
    run_dir = Path(run_dir)
    results = run_dir / "results.jsonl"
    if not results.exists():
        raise SystemExit(f"{results} does not exist -- nothing to import")

    judge_sha = judge_prompt_sha() if judge_sha is None else judge_sha
    existing, _ = load_cache(cache_path, judge_sha=judge_sha)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    questions = 0
    for line in results.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "error" in row:
            continue
        questions += 1
        for chunk_id, judgment in (row.get("judgments") or {}).items():
            key = verdict_key(row["question_id"], chunk_id)
            if key in existing or key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "question_id": row["question_id"],
                    "chunk_id": chunk_id,
                    "relevance": judgment.get("relevance", "irrelevant"),
                    "supports_options": judgment.get("supports_options", []),
                    "reason": judgment.get("reason", ""),
                    "judge_prompt_sha": judge_sha,
                    "model": row.get("judge_model"),
                    "judged_at": _now(),
                    "source_run": run_dir.name,
                }
            )
    append_records(cache_path, records)
    logger.info(
        "imported {} from {}: {} question rows -> {} new verdicts ({} already cached)",
        results.name,
        run_dir,
        questions,
        len(records),
        len(existing),
    )
    return {"questions": questions, "imported": len(records), "already_cached": len(existing)}


@dataclass
class JudgeOutcome:
    """What a judging pass cost, next to what it produced.

    `missing_judgment` counts chunks the judge simply did not return a verdict
    for -- the prompt asks for one per chunk id, so a shortfall is the model
    dropping items from a long list, and a cell rendered from a partial verdict
    set has to say so rather than score the gap as `irrelevant`.
    """

    verdicts: dict[str, dict[str, Any]]
    cached: int = 0
    judged: int = 0
    missing_judgment: int = 0
    batches: int = 0
    new_chunk_ids: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


async def ensure_verdicts(
    question: MedQAQuestion,
    chunks: Sequence[RetrievedChunk],
    cache: dict[tuple[str, str], dict[str, Any]],
    *,
    client: LLMClient,
    cache_path: str | Path = DEFAULT_JUDGE_CACHE,
    breaker: EndpointCircuitBreaker | None = None,
    batch: int = DEFAULT_BATCH,
    judge_sha: str | None = None,
    source_run: str | None = None,
) -> JudgeOutcome:
    """Verdicts for every chunk, from cache where possible; judge only the rest.

    Raises `LLMError` if a batch fails outright (the caller writes an error row),
    but verdicts already obtained are appended to the cache first -- a run that
    dies on question 12 should not have to re-pay the first 11.
    """
    judge_sha = judge_prompt_sha() if judge_sha is None else judge_sha
    outcome = JudgeOutcome(verdicts={})
    todo: list[RetrievedChunk] = []
    for chunk in chunks:
        record = cache.get(verdict_key(question.id, chunk.chunk_id))
        if record is None:
            todo.append(chunk)
        else:
            outcome.verdicts[chunk.chunk_id] = record
            outcome.cached += 1

    for start in range(0, len(todo), max(1, batch)):
        group = todo[start : start + max(1, batch)]
        prompt = build_judge_prompt(question, group)
        try:
            verdict = await client.agenerate_structured(
                prompt, JudgeVerdict, system_prompt=JUDGE_SYSTEM_PROMPT
            )
        except LLMError as exc:
            if breaker is not None:
                breaker.record_failure(exc)
            raise
        if breaker is not None:
            breaker.record_success()
        outcome.batches += 1

        returned = {j.chunk_id: j for j in verdict.judgments}
        records: list[dict[str, Any]] = []
        for chunk in group:
            judgment = returned.get(chunk.chunk_id)
            if judgment is None:
                outcome.missing_judgment += 1
                continue
            record = {
                "question_id": question.id,
                "chunk_id": chunk.chunk_id,
                "relevance": judgment.relevance,
                "supports_options": sorted({o.upper() for o in judgment.supports_options}),
                "reason": judgment.reason,
                "judge_prompt_sha": judge_sha,
                "model": client.config.name,
                "judged_at": _now(),
                "source_run": source_run,
            }
            records.append(record)
            cache[verdict_key(question.id, chunk.chunk_id)] = record
            outcome.verdicts[chunk.chunk_id] = record
            outcome.new_chunk_ids.append(chunk.chunk_id)
        append_records(cache_path, records)
        outcome.judged += len(records)

    if outcome.missing_judgment:
        outcome.problems.append(
            f"judge returned no verdict for {outcome.missing_judgment}/{len(chunks)} chunk(s)"
        )
    return outcome



def stats(cache_path: str | Path = DEFAULT_JUDGE_CACHE) -> dict[str, Any]:
    """How much endpoint work the cache represents, and how much is stale."""
    path = Path(cache_path)
    tally: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "total": 0,
        "malformed": 0,
        "by_prompt_sha": {},
        "by_source_run": {},
        "by_relevance": {},
    }
    if not path.exists():
        return tally
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            tally["malformed"] += 1
            continue
        tally["total"] += 1
        for field_name, bucket in (
            ("judge_prompt_sha", "by_prompt_sha"),
            ("source_run", "by_source_run"),
            ("relevance", "by_relevance"),
        ):
            key = str(record.get(field_name))
            tally[bucket][key] = tally[bucket].get(key, 0) + 1
    tally["current_prompt_sha"] = judge_prompt_sha()
    tally["usable_now"] = tally["by_prompt_sha"].get(tally["current_prompt_sha"], 0)
    return tally


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default=str(DEFAULT_JUDGE_CACHE), help="cache jsonl path")
    parser.add_argument(
        "--import-run",
        action="append",
        default=[],
        metavar="RUN_DIR",
        help="seed the cache from a run's results.jsonl (repeatable)",
    )
    parser.add_argument("--stats", action="store_true", help="summarize the cache and exit")
    args = parser.parse_args(argv)

    if not args.import_run and not args.stats:
        parser.print_help()
        return 2

    for run_dir in args.import_run:
        counts = import_run(run_dir, args.cache)
        print(
            f"{run_dir}: {counts['questions']} question rows, "
            f"{counts['imported']} verdicts imported, "
            f"{counts['already_cached']} already cached"
        )

    if args.stats:
        summary = stats(args.cache)
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        print(
            f"\ncurrent judge prompt sha: {summary['current_prompt_sha']} "
            f"-> {summary['usable_now']}/{summary['total']} verdicts reusable"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

