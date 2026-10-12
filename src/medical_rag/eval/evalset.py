"""The oracle's review set: stratified hand-check and verdict snapshot (R2 A3/E2).

Two artifacts come out of this module, both committed:

* `experiments/retrieval/R2/hand_check.md` + `hand_check_records.csv` -- the
  stratified hand-check. A human grades ~30 (question, chunk) pairs blind
  (judge verdicts hidden), then `evalset score` reveals them and computes the
  per-class agreement table that gate G1 reads. Strata are the plan's five:
  the four known-bad control chunks, cached `relevant`, cached `partial`,
  random `irrelevant`, and letter-discrimination (chunks whose
  `supports_options` excludes the gold letter -- the judge's riskiest call).
* `experiments/retrieval/evalset/verdicts_<sha>.jsonl` +
  `snapshot_metadata.json` -- the oracle snapshot: every verdict the current
  rubric sha can produce, exported with provenance (git rev, rubric text,
  judge model and both sampling layers, dataset/corpus identity, sample file)
  so a paper's numbers stay reproducible after the cache moves on.

Determinism: stratum selection is `stable_rank_key(seed, pair_id)` from
`eval.sample` -- blake2b of `seed:id`, not Python's salted `hash()` -- so a
rebuild reproduces the committed review set record-for-record.

CLI: `scripts/evalset.py build|score|export|check` (this module's `main` too).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from medical_rag.config import load_config
from medical_rag.data.load_medqa import DATASET_ID, MedQAQuestion, load_medqa
from medical_rag.eval.judge import DEFAULT_JUDGE_CACHE, load_cache, stats
from medical_rag.eval.rubric import DEFAULT_RUBRIC, RubricSpec, judge_prompt_sha
from medical_rag.eval.sample import stable_rank_key
from medical_rag.retrieval.strategy import resolve_index_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
EVALSET_DIR = REPO_ROOT / "experiments" / "retrieval" / "evalset"
HAND_CHECK_DIR = REPO_ROOT / "experiments" / "retrieval" / "R2"
HAND_CHECK_SEED = 2
HAND_CHECK_VERSION = "hand-check-v1"

# The plan's known-bad control chunks (§A3 S1): boilerplate the R1 leniency hunt
# flagged. Every record here must be human-confirmed `irrelevant` for G1 to pass.
CONTROL_CHUNK_IDS: tuple[str, ...] = (
    "article-128082_187",
    "article-17453_34",
    "article-40888_0",
    "article-128082_185",
)

STRATUM_SIZES: dict[str, int] = {
    "S1_control": 8,       # 2 records per control chunk, spread over >= 5 questions
    "S2_relevant": 8,      # cached verdict `relevant`
    "S3_partial": 8,       # cached verdict `partial`
    "S4_irrelevant": 4,    # random `irrelevant`, S1 chunks excluded
    "S5_letter": 2,        # supports_options non-empty WITHOUT the gold letter
}

RECORD_FIELDS = (
    "pair_id", "question_id", "chunk_id", "stratum", "gold_letter",
    "question_text", "option_A", "option_B", "option_C", "option_D",
    "chunk_title", "chunk_text",
    "human_relevance", "human_supports", "human_note",
    "judge_relevance", "judge_supports", "judge_reason",
)


@dataclass(frozen=True)
class HandCheckPair:
    """One (question, chunk) record the human grades; judge columns stay empty
    until `--score` fills them in."""

    question_id: str
    chunk_id: str
    stratum: str
    question: MedQAQuestion
    chunk_title: str
    chunk_text: str
    judge_relevance: str = ""
    judge_supports: list[str] | None = None
    judge_reason: str = ""


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_records(cache_path: str | Path) -> list[dict[str, Any]]:
    """The current-sha verdict slice as flat records (load_cache is keyed by pair)."""
    verdicts, _counts = load_cache(cache_path)
    return list(verdicts.values())


def _pair_id(question_id: str, chunk_id: str) -> str:
    return f"q{question_id}::{chunk_id}"


def _ranked(records: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Records in seeded blake2b order -- the deterministic `random`."""
    return sorted(records, key=lambda r: stable_rank_key(seed, _pair_id(r["question_id"], r["chunk_id"])))


def _spread_over_questions(ranked: list[dict[str, Any]], n: int, min_questions: int) -> list[dict[str, Any]]:
    """Take `n` records, greedy by question so they spread across many questions.

    Walks the ranked list taking at most one record per question per round, so
    a single question's chunk pile cannot monopolise the stratum (S1 must
    spread over >= 5 questions to show the control chunk is bad *everywhere*,
    not just in one retrieval context).
    """
    picked: list[dict[str, Any]] = []
    used: set[str] = set()
    for _round in range(n):  # each round = one pass over unused questions
        for record in ranked:
            if record["question_id"] in used:
                continue
            picked.append(record)
            used.add(record["question_id"])
            if len(picked) == n:
                return picked
            break  # one per round; the next round skips this question
        else:
            break
    if len({r["question_id"] for r in picked}) < min_questions and len(picked) < n:
        raise SystemExit(
            f"could not spread {len(picked)} records over {min_questions} questions -- pool too small"
        )
    return picked


def build_hand_check(
    cache_path: str | Path = DEFAULT_JUDGE_CACHE,
    *,
    seed: int = HAND_CHECK_SEED,
    config_path: str = "config/default.yaml",
    questions: list[MedQAQuestion] | None = None,
    resolve_chunk: Any = None,
) -> list[HandCheckPair]:
    """The seeded stratified selection over the cache (plan §A3).

    `resolve_chunk(question_id, chunk_id) -> (title, prose)` fetches chunk
    prose; the default resolves through `eval.store.resolve_chunks` against the
    LanceDB table (tests inject a stub). Raises SystemExit when a stratum's
    pool cannot fill its quota -- a silently short stratum would make the
    agreement table read as more evidence than it is.
    """
    if resolve_chunk is None:
        from medical_rag.eval.store import resolve_chunks  # noqa: PLC0415 (offline import below)
        from medical_rag.retrieval.index import load_index
        from medical_rag.retrieval.strategy import resolve_index_dir

        config = load_config(config_path)
        table = load_index(resolve_index_dir(config))

        def resolve_chunk(chunk_id: str) -> tuple[str, str]:
            chunks, _source = resolve_chunks([chunk_id], table=table)
            chunk = chunks.get(chunk_id)
            if chunk is None:
                raise SystemExit(f"chunk {chunk_id} not found in the store")
            return chunk.title, chunk.content

    if questions is None:
        questions = load_medqa("train")
    by_id = {q.id: q for q in questions}

    records = _load_records(cache_path)
    if not records:
        raise SystemExit(
            f"no verdicts usable at judge_sha={judge_prompt_sha()!r} in {cache_path}"
        )
    controls = set(CONTROL_CHUNK_IDS)
    gold_by_question = {q.id: q.answer_idx.upper() for q in questions}

    s2_pool = [r for r in records if r["relevance"] == "relevant"]
    s3_pool = [r for r in records if r["relevance"] == "partial"]
    s4_pool = [
        r for r in records
        if r["relevance"] == "irrelevant" and r["chunk_id"] not in controls
    ]
    s5_pool = [
        r for r in records
        if r.get("supports_options") and gold_by_question.get(r["question_id"])
        and gold_by_question[r["question_id"]] not in r["supports_options"]
    ]

    # S1: two records per control chunk, spread over >= 5 distinct questions.
    s1: list[dict[str, Any]] = []
    for chunk_id in CONTROL_CHUNK_IDS:
        per_chunk = _ranked([r for r in records if r["chunk_id"] == chunk_id], seed)
        if len(per_chunk) < 2:
            raise SystemExit(f"control chunk {chunk_id} has {len(per_chunk)} verdicts, need 2")
        s1.extend(per_chunk[:2])
    if len({r["question_id"] for r in s1}) < 5:
        raise SystemExit("S1 controls did not spread over 5 questions -- re-check the cache")

    selected: dict[str, list[dict[str, Any]]] = {
        "S1_control": s1,
        "S2_relevant": _spread_over_questions(_ranked(s2_pool, seed), STRATUM_SIZES["S2_relevant"], 4),
        "S3_partial": _spread_over_questions(_ranked(s3_pool, seed), STRATUM_SIZES["S3_partial"], 4),
        "S4_irrelevant": _spread_over_questions(_ranked(s4_pool, seed), STRATUM_SIZES["S4_irrelevant"], 4),
        "S5_letter": _spread_over_questions(_ranked(s5_pool, seed), STRATUM_SIZES["S5_letter"], 2),
    }
    for stratum, picked in selected.items():
        if len(picked) < STRATUM_SIZES[stratum]:
            raise SystemExit(f"stratum {stratum} filled {len(picked)}/{STRATUM_SIZES[stratum]} -- cache too small")

    pairs: list[HandCheckPair] = []
    for stratum in ("S1_control", "S2_relevant", "S3_partial", "S4_irrelevant", "S5_letter"):
        for record in selected[stratum]:
            question = by_id.get(record["question_id"])
            if question is None:
                raise SystemExit(f"question {record['question_id']} not in the split pool")
            title, prose = resolve_chunk(record["chunk_id"])
            pairs.append(
                HandCheckPair(
                    question_id=record["question_id"],
                    chunk_id=record["chunk_id"],
                    stratum=stratum,
                    question=question,
                    chunk_title=title,
                    chunk_text=prose,
                    judge_relevance=record["relevance"],
                    judge_supports=list(record.get("supports_options") or []),
                    judge_reason=record.get("reason", ""),
                )
            )
    return pairs


def _record(pair: HandCheckPair) -> dict[str, Any]:
    """The CSV row: prose, gold letter at the bottom of the prompt-visible block,
    judge columns empty in pass 1, filled by `score_hand_check` in pass 2."""
    return {
        "pair_id": _pair_id(pair.question_id, pair.chunk_id),
        "question_id": pair.question_id,
        "chunk_id": pair.chunk_id,
        "stratum": pair.stratum,
        "gold_letter": pair.question.answer_idx.upper(),
        "question_text": pair.question.question,
        "option_A": pair.question.options.get("A", ""),
        "option_B": pair.question.options.get("B", ""),
        "option_C": pair.question.options.get("C", ""),
        "option_D": pair.question.options.get("D", ""),
        "chunk_title": pair.chunk_title,
        "chunk_text": pair.chunk_text,
        "human_relevance": "",
        "human_supports": "",
        "human_note": "",
        "judge_relevance": pair.judge_relevance,
        "judge_supports": ",".join(sorted(pair.judge_supports or [])),
        "judge_reason": pair.judge_reason,
    }


def render_hand_check(pairs: Sequence[HandCheckPair], out_dir: str | Path = HAND_CHECK_DIR) -> list[Path]:
    """Pass 1 artifact: `hand_check.md` + `hand_check_records.csv`, judge columns blank.

    The markdown hides the judge's verdicts (that is the blindness); it shows
    chunk prose, question, options, and the gold letter at the bottom, per the
    plan's two-pass procedure.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "hand_check_records.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(_record(pair))

    md_path = out_dir / "hand_check.md"
    counts = Counter(pair.stratum for pair in pairs)
    lines = [
        "# R2 stratified hand-check (A3) -- pass 1, blind",
        "",
        f"- seed: `{HAND_CHECK_SEED}` | selection: `{HAND_CHECK_VERSION}` "
        "| grading: fill `human_relevance` (relevant/partial/irrelevant) and "
        "`human_supports` (comma-separated option letters, or empty) in the CSV.",
        f"- {len(pairs)} records across {len(counts)} strata: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        "- judge verdicts are deliberately absent from this page -- `evalset score` "
        "reveals them against your filled columns.",
        "",
    ]
    for index, pair in enumerate(pairs, start=1):
        lines += [
            f"## {index}. `{pair.chunk_id}` (q{pair.question_id}, {pair.stratum})",
            "",
            f"**Chunk:** {pair.chunk_title}",
            "",
            "```",
            pair.chunk_text.strip(),
            "```",
            "",
            f"**Question {pair.question_id}:** {pair.question.question}",
            "",
            "```",
            *(
                f"({letter}) {pair.question.options.get(letter, '')}"
                for letter in ("A", "B", "C", "D")
            ),
            "```",
            "",
            f"Gold letter (kept out of your eye-line until you've graded): "
            f"||{pair.question.answer_idx.upper()}||",
            "",
            "---",
            "",
        ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return [md_path, csv_path]


# The CSV's legal vocabulary -- checked on --score so a typo does not silently
# become a disagreement.
RELEVANCE_LEVELS = ("relevant", "partial", "irrelevant")


def _parse_supports(raw: str) -> set[str]:
    return {token.strip().upper() for token in raw.split(",") if token.strip()}


def score_hand_check(csv_path: str | Path = HAND_CHECK_DIR / "hand_check_records.csv") -> str:
    """Pass 2: reveal the judge columns, emit the per-class agreement table.

    Returns the markdown table as text (FINDINGS embeds it verbatim). The plan's
    constraint: per-class numbers only -- the table has no pooled row anywhere.
    Gate G1 reads this: controls 100% human-`irrelevant`, per-class relevance
    agreement >= 80%, at most one judge-over-human (relevant where human says
    irrelevant) systematic error per class.
    """
    path = Path(csv_path)
    if not path.exists():
        raise SystemExit(f"{path} not found -- run `evalset build` and fill the CSV first")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    problems: list[str] = []
    for row in rows:
        if row["human_relevance"] not in RELEVANCE_LEVELS:
            problems.append(f"{row['pair_id']}: human_relevance={row['human_relevance']!r} "
                            f"is not one of {RELEVANCE_LEVELS}")
        for letter in _parse_supports(row["human_supports"]):
            if letter not in {"A", "B", "C", "D"}:
                problems.append(f"{row['pair_id']}: human_supports has option {letter!r}")
    if problems:
        raise SystemExit("CSV not gradeable:\n  " + "\n  ".join(problems))

    lines = [
        "## Hand-check agreement, per stratum (A3 pass 2)",
        "",
        "| stratum | n | relevance agree | relevant↔partial conf. | judge over human (rel. vs irr.) | human over judge | supports exact-set | gold-support correct |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    controls_irrelevant = True
    for stratum in sorted({row["stratum"] for row in rows}):
        stratum_rows = [row for row in rows if row["stratum"] == stratum]
        n = len(stratum_rows)
        agree = sum(
            1 for row in stratum_rows if row["human_relevance"] == row["judge_relevance"]
        )
        confusion = sum(
            1 for row in stratum_rows
            if {row["human_relevance"], row["judge_relevance"]} == {"relevant", "partial"}
        )
        # The gate's systematic error: judge lifts to relevant what a human calls
        # irrelevant -- the leniency direction. The reverse drift is recorded but
        # not gated on.
        judge_over = sum(
            1 for row in stratum_rows
            if row["judge_relevance"] == "relevant" and row["human_relevance"] == "irrelevant"
        )
        human_over = sum(
            1 for row in stratum_rows
            if row["human_relevance"] == "relevant" and row["judge_relevance"] == "irrelevant"
        )
        supports_exact = sum(
            1 for row in stratum_rows
            if _parse_supports(row["human_supports"]) == _parse_supports(row["judge_supports"])
        )
        gold_correct = sum(
            1 for row in stratum_rows
            if (row["gold_letter"] in _parse_supports(row["human_supports"]))
            == (row["gold_letter"] in _parse_supports(row["judge_supports"]))
        )
        if stratum == "S1_control":
            controls_irrelevant = all(
                row["human_relevance"] == "irrelevant" for row in stratum_rows
            )
        lines.append(
            f"| {stratum} | {n} | {agree}/{n} ({agree / n * 100:.0f}%) | {confusion} "
            f"| {judge_over} | {human_over} | {supports_exact}/{n} | {gold_correct}/{n} |"
        )

    lines += [
        "",
        f"Controls (S1) all human-confirmed irrelevant: "
        f"{'YES -- G1 control condition met' if controls_irrelevant else 'NO -- G1 FAILS on controls'}",
        "",
        "_Per-stratum only, by design -- no pooled agreement number exists for this artifact._",
        "",
    ]
    return "\n".join(lines)


SNAPSHOT_SAMPLE_FILE = REPO_ROOT / "eval" / "sample_v2.json"
OMLX_MODEL_SETTINGS = Path.home() / ".omlx" / "model_settings.json"
def _server_sampling(api_model_name: str) -> dict[str, Any]:
    """What `~/.omlx/model_settings.json` forces on the judge model (OQ4/D6).

    oMLX applies per-model sampling from its own settings file, outside this
    repo and outside git; `params_snapshot()` cannot see it. The snapshot pastes
    the judge model's entry verbatim, `force_sampling` included, so the
    "record-only, never change mid-track" rule has something to compare the
    next export against. Missing file -> an explicit marker, not a silent {}.
    """
    if not OMLX_MODEL_SETTINGS.exists():
        return {"_missing": str(OMLX_MODEL_SETTINGS)}
    data = json.loads(OMLX_MODEL_SETTINGS.read_text(encoding="utf-8"))
    models = data.get("models") or {}
    entry = models.get(api_model_name)
    if entry is None:
        return {"_note": f"judge api_model_name {api_model_name!r} not found; "
                         "models known: " + ", ".join(sorted(models))}
    return entry


def _index_sha(index_dir: Path) -> str | None:
    """A cheap-but-stable fingerprint of the LanceDB index directory.

    Runs recorded no index sha, so the snapshot computes one: every file's
    relative path + size + mtime folded through blake2b. ~700 MB of data is not
    re-hashed; a rebuild with different chunks changes some mtime/size and moves
    the fingerprint. `None` when the index dir is missing.
    """
    if not index_dir.exists():
        return None
    digest = hashlib.blake2b(digest_size=16)
    for path in sorted(p for p in index_dir.rglob("*") if p.is_file()):
        stat = path.stat()
        digest.update(f"{path.relative_to(index_dir)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def export_snapshot(
    out_dir: str | Path = EVALSET_DIR,
    *,
    cache_path: str | Path = DEFAULT_JUDGE_CACHE,
    config_path: str = "config/default.yaml",
    rubric: RubricSpec = DEFAULT_RUBRIC,
) -> list[Path]:
    """Stage E2: the oracle snapshot -- verdicts of the current rubric sha only.

    Writes `verdicts_<sha>.jsonl` (one verdict per line, **no chunk prose** --
    the corpus itself is the prose-of-record) and `snapshot_metadata.json` with
    the provenance FINDINGS cites: git rev, full rubric text + sha, judge model
    identity, client-side and server-side sampling, dataset + split, corpus
    provenance from the last run's `context.md`, the sample file + sha, and
    cache stats. Overwrites both files in place -- the snapshot is regenerated,
    never hand-edited.
    """
    sha = rubric.sha()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, _counts = load_cache(cache_path, judge_sha=sha)
    verdicts_path = out_dir / f"verdicts_{sha}.jsonl"
    ordered = sorted(records.items())
    with verdicts_path.open("w", encoding="utf-8") as handle:
        for (_question_id, _chunk_id), record in ordered:
            slim = {key: record[key] for key in (
                "question_id", "chunk_id", "relevance", "supports_options",
                "reason", "judged_at", "source_run", "model",
            ) if key in record}
            handle.write(json.dumps(slim, ensure_ascii=False) + "\n")

    config = load_config(config_path)
    judge_model = config.models["judge_model"]
    context_path, index_sha = _latest_run_context()
    cache_stats = stats(cache_path)
    sample_path = SNAPSHOT_SAMPLE_FILE
    sample = json.loads(sample_path.read_text(encoding="utf-8")) if sample_path.exists() else None

    metadata = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_rev": _git_rev(),
        "rubric": {
            "sha": sha,
            "version": rubric.version,
            "system_prompt": rubric.system_prompt,
            "json_instruction": rubric.json_instruction,
        },
        "judge_model": {
            "name": judge_model.name,
            "api_model_name": judge_model.api_model_name,
            "base_url": judge_model.base_url,
            "client_sampling": {
                "temperature": judge_model.temperature,
                "top_p": judge_model.top_p,
                "max_new_tokens": judge_model.max_new_tokens,
            },
            "server_sampling": _server_sampling(judge_model.api_model_name),
            "_note": "server-side values are pasted verbatim from ~/.omlx/model_settings.json "
            "(OQ4/D6): record-only, never changed mid-track; changing them is a new-oracle event",
        },
        "dataset": {"id": DATASET_ID, "split": config.dev_split},
        "corpus": {
            "name": config.retrieval.corpus,
            "embedding_model": config.retrieval.embedding_model,
            "reranker_model": config.retrieval.reranker_model,
            "index_dir": str(resolve_index_dir(config)),
            "index_sha256": _index_sha(resolve_index_dir(config)),
            "provenance_source": (str(context_path) if context_path else
                                  "no run context.md found -- provenance incomplete"),
        },
        "sample": (
            {"path": str(sample_path.relative_to(REPO_ROOT)),
             "sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest()[:12],
             "n": sample.get("size"), "seed": sample.get("seed"),
             "version": sample.get("version")}
            if sample else {"path": None, "_note": "sample_v2.json not found"}
        ),
        "cache": {
            "path": (str(Path(cache_path).relative_to(REPO_ROOT))
                     if Path(cache_path).is_relative_to(REPO_ROOT) else str(cache_path)),
            "total_lines": cache_stats["total"],
            "by_prompt_sha": dict(sorted(cache_stats["by_prompt_sha"].items())),
            "current_sha_usable": cache_stats["usable_now"],
            "verdicts_exported": len(records),
        },
    }
    metadata_path = out_dir / "snapshot_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return [verdicts_path, metadata_path]


def check_snapshot(out_dir: str | Path = EVALSET_DIR) -> bool:
    """The snapshot equals the current-sha slice of the live cache, right now.

    Recomputes the expected export and compares pair keys + verdict fields;
    returns False (and prints the diff class) if the cache has moved since the
    last export -- re-export, never hand-edit.
    """
    sha = DEFAULT_RUBRIC.sha()
    out_dir = Path(out_dir)
    verdicts_path = out_dir / f"verdicts_{sha}.jsonl"
    if not verdicts_path.exists():
        print(f"snapshot missing: {verdicts_path}")
        return False
    live, _ = load_cache(DEFAULT_JUDGE_CACHE, judge_sha=sha)
    exported: dict[tuple[str, str], dict[str, Any]] = {}
    for lineno, line in enumerate(verdicts_path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"snapshot unreadable: {verdicts_path.name} line {lineno} is not valid "
                  f"json ({exc}) -- re-run `evalset export`")
            return False
        exported[(record["question_id"], record["chunk_id"])] = record
    if set(exported) != set(live):
        only_live = len(set(live) - set(exported))
        only_exported = len(set(exported) - set(live))
        print(f"snapshot drift: {only_live} verdict(s) in cache but not exported, "
              f"{only_exported} exported but not in the current-sha slice")
        return False
    for key, live_record in live.items():
        slim = {k: live_record[k] for k in ("question_id", "chunk_id", "relevance",
                                            "supports_options", "reason", "judged_at",
                                            "source_run", "model") if k in live_record}
        if exported[key] != slim:
            print(f"snapshot drift: verdict {key} differs from the cache")
            return False
    print(f"snapshot OK: {len(exported)} verdict(s) match the current-sha cache slice")
    return True


def _latest_run_context() -> tuple[Path | None, str | None]:
    """(context.md, judge_prompt_sha) of the newest retrieval_tuning run.

    The provenance FINDINGS cites is what the run itself recorded -- its rubric
    sha and cache line -- not what the current config says; a config edit after
    the build must not silently re-brand the snapshot's corpus. Sorted by mtime,
    not name: `serve_smoke` sorts above `serve_final` but is older.
    """
    runs_root = REPO_ROOT / "outputs" / "exploration" / "retrieval_tuning"
    if not runs_root.exists():
        return None, None
    run_dirs = sorted(
        (d for d in runs_root.iterdir() if (d / "context.md").exists()),
        key=lambda d: (d / "context.md").stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        return None, None
    context = run_dirs[0] / "context.md"
    sha = None
    for line in context.read_text(encoding="utf-8").splitlines():
        if "judge_prompt_sha" in line and "`" in line:
            sha = line.split("`")[1]
            break
    return context, sha


def main(argv: Sequence[str] | None = None, resolve_chunk: Any = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", default=str(DEFAULT_JUDGE_CACHE), help="judge cache jsonl")
    parser.add_argument("--out-dir", default=str(EVALSET_DIR), help="snapshot / hand-check dir")
    parser.add_argument("--hand-check-dir", default=str(HAND_CHECK_DIR),
                        help="where the hand-check md+csv live")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--seed", type=int, default=HAND_CHECK_SEED)
    sub = parser.add_subparsers(dest="cmd")

    build = sub.add_parser("build", help="pass 1: the blind hand-check md + csv")
    score = sub.add_parser("score", help="pass 2: reveal judge verdicts, per-stratum table")
    score.add_argument("--csv", default=None, help="filled hand-check csv (default: <dir>/hand_check_records.csv)")
    export = sub.add_parser("export", help="Stage E2: verdicts_<sha>.jsonl + snapshot_metadata.json")
    check = sub.add_parser("check", help="assert the snapshot matches the live cache slice")

    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return 2

    if args.cmd == "build":
        pairs = build_hand_check(args.cache, seed=args.seed, config_path=args.config,
                                 resolve_chunk=resolve_chunk)
        paths = render_hand_check(pairs, args.hand_check_dir)
        for path in paths:
            print(f"wrote {path}")
        counts = Counter(pair.stratum for pair in pairs)
        print(", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + f" (seed={args.seed})")
        return 0

    if args.cmd == "score":
        csv_path = args.csv or (Path(args.hand_check_dir) / "hand_check_records.csv")
        print(score_hand_check(csv_path))
        return 0

    if args.cmd == "export":
        for path in export_snapshot(args.out_dir, cache_path=args.cache, config_path=args.config):
            print(f"wrote {path}")
        return 0

    if args.cmd == "check":
        return 0 if check_snapshot(args.out_dir) else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())