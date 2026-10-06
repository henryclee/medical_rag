"""Retrieval-tuning judge harness (see TUNING.md).

Phase 6 found that at `top_k=5`, the gold answer's *exact wording* reached
the retrieved excerpts for only 4/20 questions -- so three quarters of Phase
6's rows cannot distinguish "the model ignored the context" from "the
context never had the answer" (`experiments/phase6/FINDINGS.md`). This
script builds a semantic version of that measurement: `judge_model` (a
larger model, defined in `config/models.yaml`) grades whether each retrieved
chunk actually helps answer the question, blind to the gold letter -- it
reports `relevance` and which option letter(s) a chunk provides evidence
for (`supports_options`) -- and gold-option recall is computed
*programmatically* afterwards. That keeps the label reusable and never
leaks the answer into the grading prompt.

The grid is five strategies -- dense, dense+rerank (today's production config),
BM25 (see `retrieval/lexical.py`), hybrid RRF fusion of dense+BM25, and hybrid+rerank
(`retrieval/fusion.py`) -- crossed with two query variants: the question text (`__orig`)
and a reformulated information need (`__reform`), rewritten with
`build_reformulation_prompt()`. Ten cells, defined once in `retrieval/strategy.py`, which
`eval/inspector.py` renders cell by cell.

For each pinned question, every cell's candidate chunk_ids are unioned and judged
once per unseen chunk -- so ten methods cost no more judge time than one.
Verdicts live in a persistent cache (`eval/judge.py`) keyed
`(question_id, chunk_id, judge_prompt_sha)`, so a rerun pays only for chunks no
run has graded before, and the chunk *text* the run surfaced is written to a
`chunks.jsonl` sidecar so the run can be re-read without loading the index.

The first run of this script (2026-09-29) reported a `reform_dense` method whose
reformulation fell back to the raw question on 18 of 20 questions; `FINDINGS.md`
quotes numbers from it that the run never measured. The parsing bug behind that
is fixed in `eval/rewrite.py`, and `render_findings` now prints an integrity
section that states how many rows fell back before any Δ can be read.

This is the measurement the retrieval track (R-phases in `PLAN.md`) iterates
against, which is why it lives in the package rather than in an experiment
directory: `eval/` is production code now. It follows `scripts/exploration/`'s
run conventions (`RunWriter`, `error_row`, `EndpointCircuitBreaker`,
`--dry-run`/`--resume`/`--note`) via `eval/runlog.py` and builds its retriever
through `retrieval.strategy.build_retriever`, the same entry point the frozen
Phase 5-6 scripts use.

    set -a; source .env; set +a
    .venv/bin/python scripts/run_retrieval_grid.py --dry-run
    .venv/bin/python scripts/run_retrieval_grid.py --limit 2   # live smoke
    .venv/bin/python scripts/run_retrieval_grid.py             # full sample
    .venv/bin/python scripts/run_retrieval_grid.py --rerender \
        outputs/exploration/retrieval_tuning/20260929T120215Z   # rebuild report, 0 calls
"""

import argparse
import asyncio
import json
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from loguru import logger

from medical_rag.config import ExperimentConfig, load_config
from medical_rag.data.load_medqa import MedQAQuestion, load_medqa
from medical_rag.eval.judge import DEFAULT_JUDGE_CACHE, ensure_verdicts, load_cache
from medical_rag.eval.metrics import aggregate
from medical_rag.eval.rewrite import load_template, reformulate_query
# Only the sha: the harness stamps it on every verdict but never words a grading
# prompt itself -- `eval.judge.ensure_verdicts` builds that from `eval.rubric`, so
# there is one wording and the cache key cannot drift from it.
from medical_rag.eval.rubric import judge_prompt_sha
from medical_rag.eval.runlog import (
    DEFAULT_RUN_ROOT,
    DEFAULT_SAMPLE_SEED,
    EndpointCircuitBreaker,
    RunWriter,
    error_row,
    parse_notes,
    select_by_ids,
    warn_on_degenerate_sample,
)
from medical_rag.eval.store import append_chunks
from medical_rag.generation.llm import LLMClient, LLMError
from medical_rag.generation.preflight import Preflight, preflight
from medical_rag.retrieval.lexical import DEFAULT_BM25_PATH, load_or_build_bm25_index
from medical_rag.retrieval.strategy import (
    BASE_STRATEGIES,
    METHOD_GRID,
    QUERY_VARIANTS,
    aretrieve_grid,
    build_retriever,
    canonical_method,
    resolve_index_dir,
)

PHASE = "retrieval_tuning"
CONDITION_ID = "retrieval_tuning_judge"

# The same 20 dev-split questions Phase 5/6 answered, per PLAN.md's "Reuse
# Phase 5's pinned 20 questions" constraint -- this side-track measures
# retrieval on the sample the downstream comparisons already run on.
SAMPLE_QUESTION_IDS: list[str] = [
    "1312", "2391", "3998", "4002", "5209", "5949", "6205", "6264", "6727",
    "6753", "6771", "7159", "7502", "7553", "8966", "9293", "9473", "9572",
    "9597", "10064",
]

K_VALUES = (5, 10, 20)  # reported k values; `metrics.K_VALUES` holds the same tuple

# The grid, from `retrieval.strategy`: five strategies x two query variants. This
# module used to carry its own six-name list with `reform_dense` as a sixth
# strategy, which is what let a broken reformulation hide as a method -- the ids
# now come from the one module `eval.inspector` also uses.
METHOD_NAMES = METHOD_GRID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/default.yaml", help="path to default.yaml")
    parser.add_argument(
        "--split", default=None, help="MedQA split to resolve ids against (default: config.dev_split)"
    )
    parser.add_argument(
        "--question-ids", nargs="*", default=None,
        help=f"override the pinned sample (default: the {len(SAMPLE_QUESTION_IDS)} ids in "
        "SAMPLE_QUESTION_IDS)",
    )
    parser.add_argument(
        "--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED,
        help="recorded in context.md only; the sample itself is pinned by id, not redrawn",
    )
    parser.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    parser.add_argument(
        "--index-dir", default=None, help="LanceDB index directory (default: data/index/<corpus>)"
    )
    parser.add_argument(
        "--bm25-index", default=str(DEFAULT_BM25_PATH), help="pickled BM25 index path"
    )
    parser.add_argument(
        "--rebuild-bm25", action="store_true",
        help="rebuild the BM25 index from the LanceDB table even if a pickle exists",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="candidates each method retrieves before rerank/fusion (default: "
        "config.retrieval.top_k_retrieve)",
    )
    parser.add_argument(
        "--rerank-k", type=int, default=None,
        help="chunks kept after rerank (default: config.retrieval.top_k_rerank)",
    )
    parser.add_argument(
        "--bm25-top-k", type=int, default=None,
        help="BM25 depth, separate from --top-k (default: --top-k). BM25's low recall is "
        "the open question; this separates 'too shallow' from 'wrong vocabulary'",
    )
    parser.add_argument(
        "--rerank-with", choices=("question", "reform"), default="question",
        help="which string the cross-encoder scores. 'question' (default) makes "
        "orig-vs-reform a single-variable change: only the embedding query differs",
    )
    parser.add_argument(
        "--no-reform", action="store_true",
        help="drop the reform column -- 5 methods instead of 10, and one LLM call less per question",
    )
    parser.add_argument(
        "--reform-prompt", default=None, metavar="FILE",
        help="reformulation template with a {question} placeholder, replacing production's "
        "(see TUNING.md: exploratory prompts never go into src/)",
    )
    parser.add_argument(
        "--judge-cache", default=str(DEFAULT_JUDGE_CACHE),
        help="append-only verdict cache; read before any judge call and written after",
    )
    parser.add_argument(
        "--judge-batch", type=int, default=24,
        help="chunks per judge call (the 20-question run used 40 and the model started "
        "dropping items; a lost batch now costs a re-ask, not a lost run)",
    )
    parser.add_argument("--output-dir", default=DEFAULT_RUN_ROOT, help="run-artifact root")
    parser.add_argument(
        "--findings-path", default="experiments/retrieval_tuning/FINDINGS.md",
        help="where the rendered comparison table is written",
    )
    parser.add_argument(
        "--resume", default=None, metavar="RUN_DIR",
        help="re-open an existing run dir and skip questions already judged",
    )
    parser.add_argument(
        "--rerender", default=None, metavar="RUN_DIR",
        help="re-render --findings-path from an existing run's results.jsonl and exit -- "
        "no index, no BM25, no endpoint. Runs from before the grid carry legacy method "
        "ids (`reform_dense`); those are canonicalised onto the grid and flagged.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the plan (sample, methods, cost) and make no calls of any kind",
    )
    parser.add_argument(
        "--note", action="append", default=[], metavar="KEY=VALUE",
        help="record a fact about this run in context.md; repeatable",
    )
    return parser.parse_args()


# --- reformulation and retrieval ---------------------------------------------
#
# Both live in shared modules now: `reformulate.reformulate_query()` (tolerant
# parsing, and a fallback that is *recorded* instead of only logged) and
# `strategies.aretrieve_grid()` (5 strategies x 2 query variants). They used to
# be in this file, and `gather_candidates()` implemented a sixth "method" called
# `reform_dense` -- one retrieval on a rewritten query, judged against nothing
# comparable. The grid replaced it; keeping a second copy of either here is how
# FINDINGS.md and a report end up disagreeing.


# --- plan / cost printing ----------------------------------------------------


def print_plan(
    args: argparse.Namespace,
    config: ExperimentConfig,
    sample: list[MedQAQuestion],
    gold: Any,
    cache_counts: dict[str, int] | None = None,
) -> None:
    split = args.split or config.dev_split
    top_k = args.top_k or config.retrieval.top_k_retrieve
    rerank_k = args.rerank_k or config.retrieval.top_k_rerank
    bm25_top_k = args.bm25_top_k or top_k
    print(f"phase={PHASE} condition_id={CONDITION_ID} split={split}")
    print(f"pinned sample ({len(sample)} question(s)):")
    print("  " + ",".join(question.id for question in sample))
    print(f"  gold letters: {dict(sorted(gold.items()))}")
    print()
    print(f"methods compared (same verdict cache): {', '.join(METHOD_NAMES)}")
    print(
        f"  = {len(METHOD_GRID)} cells: {' x '.join(BASE_STRATEGIES)} "
        f"x {' x '.join(QUERY_VARIANTS)}"
    )
    print(f"k values reported: {K_VALUES}")
    print(
        f"depth: top_k={top_k} -> rerank {rerank_k}; bm25_top_k={bm25_top_k}; "
        f"reranker shown the {args.rerank_with}"
    )
    print()
    judge = config.models["judge_model"]
    reform = config.models["model_a"]
    print(f"judge: {judge.api_model_name} @ {judge.base_url}")
    print(f"reformulator: {reform.api_model_name} @ {reform.base_url}")
    if args.reform_prompt:
        print(f"reform prompt: OVERRIDDEN by {args.reform_prompt} (not production wording)")
    if not args.no_reform:
        print("reform fallback policy: tolerant parse, then structured retry, then recorded "
              "fallback -- a fell-back row is marked and its __reform column is a copy")
    if cache_counts:
        print(
            f"verdict cache: {cache_counts.get('loaded', 0)} usable ({args.judge_cache})"
            + (f", {cache_counts.get('stale_prompt', 0)} stale" if cache_counts.get("stale_prompt") else "")
        )
    print()
    # Judge cost is bounded by the chunks the grid surfaces, not by the number of
    # methods -- that is the whole point of judging the union once. Stating it as
    # "20 calls" understated it: 10 methods union to ~70-80 chunks per question.
    union_estimate = int(2.9 * top_k)
    batches = max(1, -(-union_estimate // args.judge_batch))
    print(
        f"planned calls, worst case (empty cache): {len(sample)} reformulation + "
        f"~{len(sample) * batches} judge batches ({union_estimate}ish chunks/question "
        f"at {args.judge_batch} per batch). Cached pairs cost nothing."
    )


# --- metrics -------------------------------------------------------------


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-method semantic_recall@k / relevant_fraction@k / first-relevant-rank.

    Thin wrapper over `metrics.aggregate`, which holds the formulas. The wrapper
    stays because `render_findings` and older notebooks call it here, and because
    pinning `METHOD_NAMES` at the call site is what guarantees the FINDINGS table
    has a row for every cell of the grid -- including one whose column came back
    empty, whose absence is itself a result.

    Rows are canonicalised first (see `canonicalize_rows`), so a run stored with
    the old flat method ids aggregates into the same grid as a new one instead of
    silently producing `n=0` for every method.
    """
    return aggregate(canonicalize_rows(rows), METHOD_NAMES, K_VALUES)


def load_run_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Read a run's `results.jsonl` back, skipping anything unparseable.

    Used by `--rerender`. Torn final lines are a normal artifact of killing a run
    mid-write, and a re-render that dies on one is a re-render you can't do.
    """
    path = Path(run_dir)
    if path.is_dir():
        path = path / "results.jsonl"
    if not path.exists():
        raise SystemExit(f"no results.jsonl to re-render at {path}")
    rows: list[dict[str, Any]] = []
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
    if skipped:
        logger.warning(f"{path}: skipped {skipped} unparseable line(s)")
    if not rows:
        raise SystemExit(f"{path} contained no readable rows")
    return rows


def canonicalize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-key stored candidate lists onto `METHOD_GRID` ids.

    Runs before the grid stored `dense`, `reform_dense`, ... as six peer methods.
    `strategies.LEGACY_METHOD_ALIASES` maps those onto grid cells and marks them
    aliased, because `reform_dense` was *one* retrieval with nothing comparable
    beside it -- renaming it silently would let the same number read as a measured
    reform result. Anything unrecognised is dropped and recorded rather than
    guessed at.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        candidates = row.get("candidates") or {}
        mapped: dict[str, Any] = {}
        moved: list[str] = []
        dropped: list[str] = []
        for name, chunk_ids in candidates.items():
            try:
                method, aliased = canonical_method(name)
            except ValueError:
                dropped.append(name)
                continue
            if method in mapped:
                dropped.append(name)
                continue
            mapped[method] = chunk_ids
            if aliased:
                moved.append(f"{name}->{method}")
        if not moved and not dropped:
            out.append(row)
            continue
        new_row = dict(row)
        new_row["candidates"] = mapped
        if moved:
            new_row["legacy_method_ids"] = sorted(set(moved))
        if dropped:
            new_row["dropped_method_ids"] = dropped
        out.append(new_row)
    return out


def render_findings(
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    run_dir: Path,
    legacy: Sequence[str] = (),
) -> str:
    """The tracked table, plus the integrity check that makes it readable.

    The Δ table is the reason the grid exists. `recall@k` per method was already
    computable from six flat columns, and it produced a conclusion about
    reformulation that its own `reform_dense` column could not support. Comparing
    each strategy against *itself* under the other query variant is the only
    reading that isolates the lever -- and the integrity block states how many of
    this run's rows even had a rewritten query to compare.
    """
    usable = [row for row in rows if "error" not in row]
    lines = [
        "# Retrieval tuning -- judge harness FINDINGS",
        "",
        f"Run: `{run_dir}` -- {metrics['n_questions']} question(s) judged.",
        "",
        "`semantic_recall@k`: fraction of questions where the gold option letter is in some "
        "chunk's `supports_options` within the top-k. `relevant_fraction@k`: mean fraction of "
        "the top-k chunks judged `relevant`/`partial`. `mean_first_relevant_rank`: average rank "
        "of the first `relevant` chunk (blank = never, for the questions it covers).",
        "",
        "Methods are `<strategy>__<query variant>` (`strategies.METHOD_GRID`): the five "
        "strategies run on the question text (`__orig`) and on the reformulated information "
        "need (`__reform`). `inspect_retrieval.py --from-run <this run>` renders the chunks "
        "behind every number below.",
        "",
        "| method | n | " + " | ".join(f"recall@{k}" for k in K_VALUES) + " | "
        + " | ".join(f"rel_frac@{k}" for k in K_VALUES) + " | mean_first_rank |",
        "| --- | --- | " + " | ".join("---" for _ in K_VALUES) + " | "
        + " | ".join("---" for _ in K_VALUES) + " | --- |",
    ]
    for method, stats in metrics["methods"].items():
        recall_cells = " | ".join(_fmt_pct(stats["per_k"][k]["semantic_recall"]) for k in K_VALUES)
        frac_cells = " | ".join(_fmt_pct(stats["per_k"][k]["relevant_fraction"]) for k in K_VALUES)
        rank_cell = "-" if stats["mean_first_relevant_rank"] is None else stats["mean_first_relevant_rank"]
        lines.append(f"| {method} | {stats['n']} | {recall_cells} | {frac_cells} | {rank_cell} |")

    lines += [
        "",
        "## Reformulation, per strategy (Δ = `__reform` minus `__orig`)",
        "",
        "| strategy | " + " | ".join(f"Δrecall@{k}" for k in K_VALUES)
        + " | Δmean_first_rank | identical columns |",
        "| --- | " + " | ".join("---" for _ in K_VALUES) + " | --- | --- |",
    ]
    for base in BASE_STRATEGIES:
        orig = metrics["methods"].get(f"{base}__orig")
        reform = metrics["methods"].get(f"{base}__reform")
        if not orig or not reform:
            lines.append(f"| {base} | " + " | ".join("-" for _ in K_VALUES) + " | - | -- |")
            continue
        deltas = []
        for k in K_VALUES:
            here, there = orig["per_k"][k]["semantic_recall"], reform["per_k"][k]["semantic_recall"]
            deltas.append("-" if here is None or there is None else f"{100 * (there - here):+.0f}pp")
        first = (
            "-"
            if orig["mean_first_relevant_rank"] is None
            or reform["mean_first_relevant_rank"] is None
            else f"{reform['mean_first_relevant_rank'] - orig['mean_first_relevant_rank']:+.2f}"
        )
        lines.append(f"| {base} | " + " | ".join(deltas) + f" | {first} | {_identical_share(usable, base)} |")

    if legacy:
        lines += [
            "",
            "## Stored method ids",
            "",
            "This run predates the grid; its flat ids were aliased onto grid cells by "
            "`strategies.LEGACY_METHOD_ALIASES`: "
            + ", ".join(f"`{item}`" for item in legacy)
            + ".",
            "",
            "Cells with `n = 0` were never retrieved by that run -- it ran the original query "
            "through every strategy plus exactly one dense retrieval on the rewritten query. "
            "Their Δ row is blank, not zero: a lever that was not run did not fail to work.",
        ]

    fallbacks = [row for row in usable if _reform_fallback(row)]
    lines += [
        "",
        "## Reformulation integrity",
        "",
        "A fell-back reformulation returns the question unchanged, so that row's `__reform` "
        "column is a copy of `__orig` and contributes a delta of exactly zero. Counting those "
        "as measurements is the bug this section exists to make impossible to miss.",
        "",
        f"- questions whose reformulation **fell back**: **{len(fallbacks)}/{len(usable)}**"
        + (f" ({', '.join('q' + str(row['question_id']) for row in fallbacks)})" if fallbacks else ""),
    ]
    stored_errors = {
        str(row["question_id"]): _reform_record(row).get("error") for row in fallbacks
    }
    for question_id, error in sorted(stored_errors.items()):
        if error:
            lines.append(f"  - q{question_id}: {str(error)[:170]}")
    if fallbacks and not any(stored_errors.values()):
        # Worth one line, not eighteen: the run predates `ReformulationError`,
        # so the cause is unrecoverable for every question equally.
        lines.append(
            "  - no per-question cause was stored for any of these -- the run predates "
            "`reformulate.ReformulationError`, so *why* they fell back is unrecoverable"
        )
    if usable and len(fallbacks) == len(usable):
        lines.append(
            "- **every** `__reform` column above is a copy of its `__orig` column: the Δ "
            "column measures BM25/fusion determinism, not reformulation. Do not quote it."
        )
    elif fallbacks:
        lines.append(
            "- read the Δ table with these rows excluded or re-run: each contributes a "
            "forced zero and pulls every Δ toward nothing."
        )
    newly = sum(len(row.get("new_judgments") or []) for row in usable)
    records_newly = any("new_judgments" in row for row in usable)
    cache_path = next(
        (row["judge_cache"] for row in usable if row.get("judge_cache")), None
    )
    if records_newly:
        cache_note = (
            f"- chunks newly judged by this run: {newly}"
            + (
                f" (the rest came from `{cache_path}` -- cached verdicts are reused across runs, "
                "so this is what the run *paid for*, not what it read)"
                if cache_path
                else " (this run recorded no cache path, so the split between paid and reused "
                "cannot be recovered)"
            )
        )
    else:
        cache_note = (
            "- chunks newly judged by this run: not recorded -- this run predates the verdict "
            "cache, so every judgment in it was paid for inside the run and none of it was "
            "reusable afterwards"
        )
    lines += ["", cache_note, "", "## Example judgments", ""]
    example_relevant = _find_example(rows, "relevant")
    example_irrelevant = _find_example(rows, "irrelevant")
    for label, example in (("relevant", example_relevant), ("irrelevant", example_irrelevant)):
        if example is None:
            continue
        question_id, chunk_id, judgment = example
        lines += [
            f"- **{label}** -- q{question_id}, chunk `{chunk_id}`: "
            f"relevance={judgment['relevance']} supports_options={judgment['supports_options']} "
            f"-- \"{judgment['reason']}\"",
        ]

    lines += [
        "",
        "## Recommendation",
        "",
        "Fill in by hand after reading the table above *and* running "
        "`inspect_retrieval.py --from-run <this run> --open`: which strategy the Phase 6 rerun "
        "should use, and whether reformulation moved recall on the strategies where the rewrite "
        "actually reached the embedder. A strategy whose `__reform` list is identical to "
        "`__orig` proves nothing in either direction -- check the integrity section above "
        "before believing any Δ in the table.",
        "",
    ]
    return "\n".join(lines)


def _reform_record(row: dict[str, Any]) -> dict[str, Any]:
    """The reformulation record, new shape or the legacy two-key shape.

    Tolerating both is deliberate: `--resume`d runs from before this change carry
    `reformulated_query`/`reformulation_fallback` only, and a findings renderer
    that crashed on them would have made those runs unreportable -- the opposite of
    the point, which is that their reformulation column must be visible as broken.
    """
    record = row.get("reformulation")
    if isinstance(record, dict):
        return record
    return {
        "query": row.get("reformulated_query"),
        "fallback": row.get("reformulation_fallback"),
        "error": None,
    }


def _reform_fallback(row: dict[str, Any]) -> bool:
    return bool(_reform_record(row).get("fallback"))


def _identical_share(rows: list[dict[str, Any]], base: str) -> str:
    """Questions where this strategy's `__orig` and `__reform` lists are the same order.

    For `dense` on the first run this was 20/20 -- the tell that the lever never
    fired. Identical columns in a run where the rewrite *did* reach the embedder is
    a legitimate finding (the corpus answers both queries the same way), which is
    why this reports a share per strategy instead of asserting one cause.
    """
    same = total = 0
    for row in rows:
        candidates = row.get("candidates") or {}
        orig, reform = candidates.get(f"{base}__orig"), candidates.get(f"{base}__reform")
        if orig is None or reform is None:
            continue
        total += 1
        same += int(list(orig) == list(reform))
    return "-" if not total else f"{same}/{total}"



def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def _find_example(
    rows: list[dict[str, Any]], relevance: str
) -> tuple[str, str, dict[str, Any]] | None:
    for row in rows:
        if "error" in row:
            continue
        for chunk_id, judgment in row["judgments"].items():
            if judgment.get("relevance") == relevance:
                return row["question_id"], chunk_id, judgment
    return None


# --- main ---------------------------------------------------------------


def rerender(args: argparse.Namespace) -> int:
    """Rebuild `--findings-path` from stored rows without touching the endpoint.

    `--resume` re-opens a run to finish judging it; this re-opens one to *read* it
    again. The two differ in cost: re-rendering is free, and after the grid and the
    integrity block landed, the honest FINDINGS for the pre-grid run was a rendering
    change, not a new experiment.
    """
    rows = canonicalize_rows(load_run_rows(Path(args.rerender)))
    legacy = sorted({moved for row in rows for moved in row.get("legacy_method_ids") or []})
    summary = compute_metrics(rows)
    findings = render_findings(summary, rows, Path(args.rerender), legacy=legacy)
    Path(args.findings_path).write_text(findings + "\n", encoding="utf-8")
    print(findings)
    print(f"\nre-rendered {len(rows)} row(s) from {args.rerender} -> {args.findings_path}")
    return 0


async def amain(args: argparse.Namespace) -> int:
    if args.rerender:
        return rerender(args)

    config = load_config(args.config)
    split = args.split or config.dev_split
    ids = list(args.question_ids or SAMPLE_QUESTION_IDS)
    pool = load_medqa(split)
    try:
        sample = select_by_ids(pool, ids)
    except KeyError as exc:
        raise SystemExit(str(exc.args[0]) if exc.args else exc) from exc
    if args.limit is not None:
        sample = sample[: args.limit]
    if not sample:
        raise SystemExit("no questions left to judge after --limit")
    gold = warn_on_degenerate_sample(sample)

    top_k = args.top_k or config.retrieval.top_k_retrieve
    rerank_k = args.rerank_k or config.retrieval.top_k_rerank
    bm25_top_k = args.bm25_top_k or top_k
    index_dir = resolve_index_dir(config, args.index_dir)
    template = load_template(args.reform_prompt)
    judge_sha = judge_prompt_sha()
    # Read the cache before printing the plan: "this run costs 20 judge calls" and
    # "this run costs 4, the other 16 are already graded" are different decisions,
    # and the plan is where that decision gets made.
    cache, cache_counts = load_cache(args.judge_cache, judge_sha=judge_sha)

    print_plan(args, config, sample, gold, cache_counts)
    if args.dry_run:
        print(
            "\ndry run -- no index load, no BM25 build, no preflight, no requests. Would write "
            f"to {Path(args.output_dir) / PHASE}/<UTC-stamp>/ (results.jsonl, chunks.jsonl, "
            f"context.md) and render {args.findings_path}"
        )
        return 0

    retriever, static = await asyncio.to_thread(build_retriever, config, str(index_dir))
    print(f"retrieval ready: {static['index_rows']} rows on {static['device']}")

    bm25_index = await asyncio.to_thread(
        load_or_build_bm25_index, index_dir, args.bm25_index, rebuild=args.rebuild_bm25
    )
    print(f"BM25 index ready: {len(bm25_index.chunk_ids)} chunk(s) ({args.bm25_index})\n")

    problems: list[str] = []
    async with AsyncExitStack() as stack:
        judge_model = config.models["judge_model"]
        reform_model = config.models["model_a"]
        judge = await stack.enter_async_context(LLMClient(judge_model))
        reformulator = await stack.enter_async_context(LLMClient(reform_model))

        for model, client in ((judge_model, judge), (reform_model, reformulator)):
            decision, lines = await preflight(model, client)
            for line in lines:
                print(f"  {line}")
            if decision == Preflight.BLOCK:
                raise SystemExit(f"{model.name} failed preflight -- aborting before any calls")
        print()

        judge_breaker = EndpointCircuitBreaker(judge_model.name)
        reform_breaker = EndpointCircuitBreaker(reform_model.name)

        with RunWriter(PHASE, root=args.output_dir, run_dir=args.resume) as writer:
            done = writer.completed_keys()
            if done:
                print(f"resume: {len(done)} question(s) already judged in {writer.dir}\n")
            writer.write_context(
                config=config,
                model_keys=["judge_model", "model_a"],
                questions=sample,
                split=split,
                sample_seed=args.sample_seed,
                extra={
                    "condition_id": CONDITION_ID,
                    "methods": ", ".join(METHOD_NAMES),
                    "grid": f"{len(BASE_STRATEGIES)} strategies x {len(QUERY_VARIANTS)} query variants",
                    "top_k": f"{top_k} -> rerank {rerank_k}",
                    "bm25_top_k": bm25_top_k,
                    "rerank_shown": args.rerank_with,
                    "bm25_index": str(args.bm25_index),
                    "reformulation": (
                        "skipped (--no-reform)"
                        if args.no_reform
                        else f"prompt: {args.reform_prompt or 'production build_reformulation_prompt()'}"
                    ),
                    "judge_prompt_sha": judge_sha,
                    "judge_cache": f"{args.judge_cache} ({cache_counts.get('loaded', 0)} cached)",
                    "judge_batch": args.judge_batch,
                    **parse_notes(args.note),
                },
            )
            print(f"run dir: {writer.dir}\n", flush=True)

            for index, question in enumerate(sample, start=1):
                key = (judge_model.name, question.id, "judge")
                if key in done:
                    logger.info("q{} already judged -- skipping", question.id)
                    continue
                if judge_breaker.tripped or reform_breaker.tripped:
                    problems.append(
                        "an endpoint circuit-broke -- abandoning the remaining "
                        f"{len(sample) - index + 1} question(s)"
                    )
                    break

                print(f"--- q{question.id} ({index}/{len(sample)}) ---")
                reform = None
                if not args.no_reform:
                    try:
                        reform = await reformulate_query(question, reformulator, template=template)
                        reform_breaker.record_success()
                    except LLMError as exc:
                        reform_breaker.record_failure(exc)
                        writer.write_row(
                            error_row(
                                model=reform_model, question=question, tag="judge",
                                condition_id=CONDITION_ID, split=split, exc=exc, wall_s=0.0,
                            )
                        )
                        problems.append(f"q{question.id}: reformulation call failed: {exc}")
                        continue

                try:
                    grid = await aretrieve_grid(
                        question.question,
                        retriever,
                        bm25_index,
                        reformulated_text=reform.query if reform else None,
                        top_k=top_k,
                        rerank_k=rerank_k,
                        bm25_top_k=bm25_top_k,
                        rerank_with=args.rerank_with,
                    )
                except Exception as exc:  # noqa: BLE001 - retrieval/bm25 failure, not an LLM error
                    problems.append(f"q{question.id}: candidate generation failed: {exc}")
                    continue

                chunk_lookup = grid.all_chunks()
                union_ids = sorted(chunk_lookup)
                union_chunks = [chunk_lookup[chunk_id] for chunk_id in union_ids]

                try:
                    outcome = await ensure_verdicts(
                        question,
                        union_chunks,
                        cache,
                        client=judge,
                        cache_path=args.judge_cache,
                        breaker=judge_breaker,
                        batch=args.judge_batch,
                        judge_sha=judge_sha,
                        source_run=writer.stamp,
                    )
                except LLMError as exc:
                    # `ensure_verdicts` already appended whatever earlier batches
                    # returned, so a failure here costs the failed batch, not the
                    # question's whole candidate set, on the next `--resume`.
                    writer.write_row(
                        error_row(
                            model=judge_model, question=question, tag="judge",
                            condition_id=CONDITION_ID, split=split, exc=exc, wall_s=0.0,
                        )
                    )
                    problems.append(f"q{question.id}: judge call failed: {exc}")
                    continue
                problems.extend(f"q{question.id}: {problem}" for problem in outcome.problems)

                judgments = {
                    chunk_id: {
                        "relevance": record["relevance"],
                        "supports_options": record["supports_options"],
                        "reason": record.get("reason", ""),
                    }
                    for chunk_id, record in outcome.verdicts.items()
                }
                row = {
                    "model": judge_model.name,
                    "question_id": question.id,
                    "tag": "judge",
                    "condition_id": CONDITION_ID,
                    "split": split,
                    "question": question.question,
                    "options": question.options,
                    "correct_answer": question.answer_idx.upper(),
                    "queries": grid.variant_queries,
                    "reformulation": (
                        reform.model_dump()
                        if reform
                        else {"query": None, "fallback": False, "method": "skipped (--no-reform)"}
                    ),
                    "candidates": {
                        name: [chunk.chunk_id for chunk in chunks]
                        for name, chunks in grid.lists.items()
                    },
                    "judgments": judgments,
                    "new_judgments": outcome.new_chunk_ids,
                    "judge_cache": str(args.judge_cache),
                    "judge_prompt_sha": judge_sha,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                writer.write_row(row)
                # The sidecar is the difference between a run you can re-read and a
                # run you can only re-aggregate: `results.jsonl` stores ids, and the
                # text they name otherwise costs a 467 MB pickle load to recover.
                append_chunks(writer.dir / "chunks.jsonl", union_chunks)
                identical = sum(
                    1
                    for base in BASE_STRATEGIES
                    if grid.lists.get(f"{base}__orig") == grid.lists.get(f"{base}__reform")
                )
                print(
                    f"  {len(union_ids)} unique chunk(s) across {len(grid.lists)} method(s): "
                    f"{outcome.cached} cached, {outcome.judged} newly judged in "
                    f"{outcome.batches} batch(es)"
                    + (f"; reformulation FELL BACK ({reform.error})" if reform and reform.fallback else "")
                    + (f"; {identical}/{len(BASE_STRATEGIES)} strategies retrieved identically")
                )

            rows = writer.load_rows()

    rows = canonicalize_rows(rows)
    legacy = sorted({moved for row in rows for moved in row.get("legacy_method_ids") or []})
    metrics = compute_metrics(rows)
    findings = render_findings(metrics, rows, writer.dir, legacy=legacy)
    Path(args.findings_path).write_text(findings + "\n", encoding="utf-8")
    print("\n" + findings)
    print(f"\nartifacts: {writer.dir}")
    print(f"findings written to: {args.findings_path}")

    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(amain(parse_args())))


if __name__ == "__main__":
    main()
