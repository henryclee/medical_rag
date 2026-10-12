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
    DEFAULT_RUBRIC,
    JudgeVerdict,
    RubricSpec,
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
    rubric: RubricSpec = DEFAULT_RUBRIC,
    source_run: str | None = None,
) -> JudgeOutcome:
    """Verdicts for every chunk, from cache where possible; judge only the rest.

    `rubric` defaults to the frozen production rubric; passing a variant (e.g.
    `STABILITY_VARIANT_RUBRIC`) grades under that rubric's own sha, so its
    verdicts live alongside the default rubric's in the same cache without ever
    colliding or cross-serving: lookup, records and the system prompt all come
    from the one object. The loose `judge_sha` string override is gone on
    purpose -- a verdict must never be graded under a wording the string doesn't
    pin down (D4/C5).

    Raises `LLMError` if a batch fails outright (the caller writes an error row),
    but verdicts already obtained are appended to the cache first -- a run that
    dies on question 12 should not have to re-pay the first 11.
    """
    judge_sha = rubric.sha()
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
                prompt, JudgeVerdict, system_prompt=rubric.system_prompt
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



def audit_run(run_dir: str | Path, *, cache_path: str | Path | None = None) -> dict[str, Any]:
    """ADR-0008's clause, executed: every judged run must show zero drops.

    For each non-error row of `results.jsonl`, `set(row["judgments"])` must
    equal the union of `row["candidates"]` over the methods that row actually
    carries -- exactly the ADR's invariant, and stricter than the run-time
    `missing_judgment` counter, which only saw the chunks a *single pass* fed
    the judge. A shortfall that survived a whole run (a resume re-writing only
    part of a row, an import from an older harness, a verdict dropped by a
    torn per-question commit) is invisible everywhere else: the affected slots
    read as "no verdict", which reads as "not supporting" in every score.

    Returns a summary dict; `problems` names every `(question_id, chunk_id)`
    shortfall so a re-ask can target one batch instead of re-running.
    """
    path = Path(run_dir) / "results.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} does not exist -- nothing to audit")

    live_cache: dict[tuple[str, str], dict[str, Any]] = {}
    if cache_path is not None:
        live_cache, _ = load_cache(cache_path)

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "rows": 0,
        "error_rows": 0,
        "clean_rows": 0,
        "judgment_slots": 0,
        "missing_judgments": 0,
        "surplus_judgments": 0,
        "recoverable_from_cache": 0,
        "problems": [],
    }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "error" in row:
                summary["error_rows"] += 1
                continue
            summary["rows"] += 1
            judged = set(row.get("judgments") or {})
            candidates = row.get("candidates") or {}
            expected = {chunk_id for ids in candidates.values() for chunk_id in ids}
            summary["judgment_slots"] += len(expected)
            missing = sorted(expected - judged)
            surplus = sorted(judged - expected)
            if not missing and not surplus:
                summary["clean_rows"] += 1
                continue
            summary["missing_judgments"] += len(missing)
            summary["surplus_judgments"] += len(surplus)
            question_id = str(row.get("question_id"))
            for chunk_id in missing:
                owners = sorted(name for name, ids in candidates.items() if chunk_id in ids)
                problem = f"q{question_id}: no judgment for {chunk_id} (candidate of {owners})"
                if live_cache and verdict_key(question_id, chunk_id) in live_cache:
                    summary["recoverable_from_cache"] += 1
                    problem += " -- but a verdict exists in the cache (import_run can repair)"
                summary["problems"].append(problem)
            for chunk_id in surplus:
                summary["problems"].append(
                    f"q{question_id}: judgment for {chunk_id} has no candidate slot in this row"
                )
    return summary


def _audit_summary_text(summary: dict[str, Any]) -> str:
    lines = [
        f"audit of {summary['run_dir']}:",
        f"  rows: {summary['rows']} clean, {summary['error_rows']} error-row(s) skipped",
        f"  judgment slots: {summary['judgment_slots']}",
        f"  missing judgments: {summary['missing_judgments']}",
        f"  surplus judgments: {summary['surplus_judgments']}",
        f"  recoverable from live cache: {summary['recoverable_from_cache']}",
    ]
    if summary["problems"]:
        lines.append(f"  problems ({len(summary['problems'])}):")
        lines.extend(f"    - {problem}" for problem in summary["problems"][:40])
        if len(summary["problems"]) > 40:
            lines.append(f"    ... and {len(summary['problems']) - 40} more")
    elif summary["rows"]:
        lines.append("  ADR-0008 invariant holds: judgments == union(candidates) on every row")
    return "\n".join(lines)


def verify_cache(cache_path: str | Path = DEFAULT_JUDGE_CACHE) -> dict[str, Any]:
    """Read the whole cache without skipping anything (OQ6/D5 -- `judge.py --verify`).

    `load_cache` is deliberately forgiving: a torn line is skipped so a kill -9
    cannot make the file unreadable. That forgiveness is exactly what can hide a
    growing tear, so the verify pass reports what a load would silently drop:
    malformed lines, duplicate keys (which line wins), stale-sha counts, and
    records that fail `ChunkJudgment`'s shape. Only `malformed` / `duplicate_keys`
    / `invalid` are exit-non-zero material in `main`; everything else is a report.
    """
    path = Path(cache_path)
    tally: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "lines": 0,
        "valid": 0,
        "malformed": 0,
        "invalid": 0,
        "duplicate_keys": 0,
        "by_prompt_sha": {},
        "duplicate_key_detail": [],
        "invalid_detail": [],
    }
    if not path.exists():
        return tally

    seen: dict[tuple[str, str], int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            tally["lines"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                tally["malformed"] += 1
                if len(tally["invalid_detail"]) < 10:
                    tally["invalid_detail"].append(f"line {line_number}: torn/unparseable JSON")
                continue
            key = verdict_key(record.get("question_id", ""), record.get("chunk_id", ""))
            sha = str(record.get("judge_prompt_sha"))
            tally["by_prompt_sha"][sha] = tally["by_prompt_sha"].get(sha, 0) + 1
            if key in seen:
                tally["duplicate_keys"] += 1
                if len(tally["duplicate_key_detail"]) < 10:
                    tally["duplicate_key_detail"].append(
                        f"line {line_number}: duplicate ({key[0]}, {key[1]}) -- "
                        f"first seen on line {seen[key]}; last write wins"
                    )
            else:
                seen[key] = line_number
            shape_problems = []
            for field_name in ("relevance", "supports_options", "reason"):
                if field_name not in record:
                    shape_problems.append(f"missing {field_name}")
            if record.get("relevance") not in {"relevant", "partial", "irrelevant"}:
                shape_problems.append(f"relevance={record.get('relevance')!r} is not a verdict level")
            if not isinstance(record.get("supports_options"), list):
                shape_problems.append("supports_options is not a list")
            if shape_problems:
                tally["invalid"] += 1
                if len(tally["invalid_detail"]) < 10:
                    tally["invalid_detail"].append(
                        f"line {line_number} ({key[0]}, {key[1]}): " + "; ".join(shape_problems)
                    )
            else:
                tally["valid"] += 1
    return tally


def compact_cache(cache_path: str | Path = DEFAULT_JUDGE_CACHE) -> dict[str, Any]:
    """One-shot rewrite: drop malformed lines, dedupe keys keeping the LAST write.

    (OQ6/D5.) The append-only per-question commit stays -- it is the resume
    story -- so `compact_cache` runs deliberately after a build, not routinely:
    it is the one pass allowed to rewrite the file, and it keeps the original as
    `.bak` so the rewrite itself is revertable. Keep-last matches what
    `load_cache` already serves (later lines overwrite earlier in its dict), so
    compaction never changes what a load returns.
    """
    path = Path(cache_path)
    if not path.exists():
        raise SystemExit(f"{path} does not exist -- nothing to compact")

    kept: dict[tuple[str, str], dict[str, Any]] = {}
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            key = verdict_key(record.get("question_id", ""), record.get("chunk_id", ""))
            kept[key] = record  # last write wins, same as load_cache

    backup = path.with_suffix(path.suffix + ".bak")
    if backup.exists():
        backup.unlink()
    path.rename(backup)
    try:
        with path.open("w", encoding="utf-8") as handle:
            for record in kept.values():
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except BaseException:
        backup.rename(path)  # put the original back; a failed compaction changes nothing
        raise
    return {
        "path": str(path),
        "backup": str(backup),
        "records_written": len(kept),
        "malformed_dropped": malformed,
    }


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
    parser.add_argument("--verify", action="store_true", help="full-strictness cache read (OQ6)")
    parser.add_argument("--compact", action="store_true", help="dedupe+torn-drop rewrite (keeps .bak)")
    parser.add_argument(
        "--audit-run",
        action="append",
        default=[],
        metavar="RUN_DIR",
        help="check judgments == union(candidates) for a run dir (repeatable)",
    )
    args = parser.parse_args(argv)

    if not args.import_run and not args.stats and not args.verify and not args.compact and not args.audit_run:
        parser.print_help()
        return 2

    for run_dir in args.import_run:
        counts = import_run(run_dir, args.cache)
        print(
            f"{run_dir}: {counts['questions']} question rows, "
            f"{counts['imported']} verdicts imported, "
            f"{counts['already_cached']} already cached"
        )

    exit_code = 0

    for run_dir in args.audit_run:
        summary = audit_run(run_dir, cache_path=args.cache)
        print(_audit_summary_text(summary))
        if summary["missing_judgments"] or summary["surplus_judgments"]:
            exit_code = 1

    if args.verify:
        report = verify_cache(args.cache)
        print(json.dumps({k: v for k, v in report.items() if k != "invalid_detail"}, indent=2, ensure_ascii=False, sort_keys=True))
        for detail in report["invalid_detail"]:
            print(f"  {detail}")
        if report["duplicate_key_detail"]:
            print("duplicate keys (last write wins):")
            for detail in report["duplicate_key_detail"]:
                print(f"  {detail}")
        if report["malformed"] or report["duplicate_keys"] or report["invalid"]:
            exit_code = 1

    if args.compact:
        report = compact_cache(args.cache)
        print(
            f"compacted {report['path']}: {report['records_written']} records, "
            f"{report['malformed_dropped']} malformed dropped, original kept at {report['backup']}"
        )

    if args.stats:
        summary = stats(args.cache)
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
        print(
            f"\ncurrent judge prompt sha: {summary['current_prompt_sha']} "
            f"-> {summary['usable_now']}/{summary['total']} verdicts reusable"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

