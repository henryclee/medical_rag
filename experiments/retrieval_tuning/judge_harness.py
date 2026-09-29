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

Six retrieval levers are compared against the same judge cache: dense-only,
dense+rerank (today's production config), BM25 (new lexical index, see
`bm25_index.py`), hybrid RRF fusion of dense+BM25 with and without rerank
(`hybrid.py`), and reformulation-as-retrieval (rewriting the query with the
already-implemented `build_reformulation_prompt()` before a dense
retrieval). For each of the pinned 20 questions, every method's candidate
chunk_ids are unioned and judged with **one** batched structured call to
`judge_model` -- so comparing six methods costs the same ~20 judge calls as
comparing one.

Side track, not a numbered phase (PLAN.md "Next actions"): production code
under `src/medical_rag/` is not touched. Follows `scripts/exploration/`'s
conventions (`RunWriter`, `error_row`, `EndpointCircuitBreaker`,
`--dry-run`/`--resume`/`--note`) via `_common.py`, and reuses `raw_rag.py`'s
`preflight()`/`build_retriever()`/`resolve_index_dir()` rather than
reimplementing them.

    set -a; source .env; set +a
    .venv/bin/python experiments/retrieval_tuning/judge_harness.py --dry-run
    .venv/bin/python experiments/retrieval_tuning/judge_harness.py --limit 2   # live smoke
    .venv/bin/python experiments/retrieval_tuning/judge_harness.py             # full 20
"""

import argparse
import asyncio
import json
import sys
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

from loguru import logger
from pydantic import BaseModel

from medical_rag.config import ExperimentConfig, load_config
from medical_rag.data.load_medqa import MedQAQuestion, load_medqa
from medical_rag.generation.llm import LLMClient, LLMError
from medical_rag.generation.prompt import build_reformulation_prompt, format_options
from medical_rag.retrieval.retriever import RetrievedChunk, Retriever

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "exploration"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DEFAULT_RUN_ROOT,
    DEFAULT_SAMPLE_SEED,
    EndpointCircuitBreaker,
    RunWriter,
    error_row,
    parse_notes,
    select_by_ids,
    warn_on_degenerate_sample,
)
from raw_rag import Preflight, build_retriever, preflight, resolve_index_dir  # noqa: E402
from bm25_index import DEFAULT_BM25_PATH, load_or_build_bm25_index  # noqa: E402
from hybrid import rrf_fuse  # noqa: E402

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

K_VALUES = (5, 10, 20)
RELEVANT_LEVELS = {"relevant", "partial"}

# One retrieval pass per method; order here is the order FINDINGS.md reports.
METHOD_NAMES = ("dense", "dense_rerank", "bm25", "hybrid", "hybrid_rerank", "reform_dense")


class ChunkJudgment(BaseModel):
    chunk_id: str
    relevance: Literal["relevant", "partial", "irrelevant"]
    supports_options: list[str]
    reason: str


class JudgeVerdict(BaseModel):
    judgments: list[ChunkJudgment]


_JUDGE_SYSTEM_PROMPT: str = (
    "You are grading whether retrieved medical reference excerpts help answer "
    "a USMLE-style multiple-choice question. You are not answering it."
)

_JUDGE_JSON_INSTRUCTION: str = (
    "Respond with JSON only, in exactly this shape:\n"
    '{"judgments": [{"chunk_id": "<the chunk_id shown above>", '
    '"relevance": "relevant" | "partial" | "irrelevant", '
    '"supports_options": ["<subset of the option letters above, or []>"], '
    '"reason": "<one short sentence>"}]}\n'
    "Include exactly one judgment per chunk_id listed above, using each "
    'chunk_id verbatim. "relevant" means the excerpt directly helps '
    'distinguish which option is correct; "partial" means it is topically '
    'related but does not resolve the question; "irrelevant" means it does '
    "not help at all. List an option letter in supports_options only if the "
    "excerpt gives evidence for that specific option -- never guess which "
    "option is correct if the excerpt does not say."
)


def build_judge_prompt(question: MedQAQuestion, chunks: Sequence[RetrievedChunk]) -> str:
    """One batched, gold-blind relevance prompt over every candidate chunk.

    The correct answer is never shown -- only the question and its four
    options -- so `supports_options` is the judge's own read of the evidence,
    not an echo of what it was told. Gold-option recall is derived later by
    checking programmatically whether the gold letter is in that set.
    """
    numbered = "\n\n".join(
        f"[{index}] chunk_id={chunk.chunk_id} title={chunk.title}\n{chunk.content}"
        for index, chunk in enumerate(chunks, start=1)
    )
    return "\n\n".join(
        [
            "You are grading retrieved reference excerpts for a medical exam "
            "question. You are NOT told which option is correct -- judge each "
            "excerpt on its own medical content.\n\n"
            f"Question: {question.question}",
            f"Options:\n{format_options(question.options)}",
            f"Candidate excerpts:\n{numbered}",
            _JUDGE_JSON_INSTRUCTION,
        ]
    )


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
        "--dry-run", action="store_true",
        help="print the plan (sample, methods, cost) and make no calls of any kind",
    )
    parser.add_argument(
        "--note", action="append", default=[], metavar="KEY=VALUE",
        help="record a fact about this run in context.md; repeatable",
    )
    return parser.parse_args()


# --- reformulation-as-retrieval ----------------------------------------------


async def reformulate_query(question: MedQAQuestion, client: LLMClient) -> tuple[str, bool]:
    """Rewrite the query with the existing reformulation prompt; fall back on failure.

    Reuses `build_reformulation_prompt()` as-is rather than waiting for
    Phase 7's `Reformulator` module, to get an early read on PLAN.md's open
    question 15 (does reformulation move retrieval recall at all). Every
    fallback is logged, per PLAN.md's convention for the real `Reformulator`.
    """
    prompt = build_reformulation_prompt(question.question)
    try:
        result = await client.agenerate(prompt)
        data = json.loads(result.content)
        need = data.get("information_need")
        if isinstance(need, str) and need.strip():
            return need.strip(), False
        logger.warning(
            "q{}: reformulation JSON had no usable information_need ({!r}) -- "
            "falling back to the original question",
            question.id,
            data,
        )
    except LLMError as exc:
        logger.warning(
            "q{}: reformulation call failed ({}) -- falling back to the original question",
            question.id,
            exc,
        )
        raise
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning(
            "q{}: reformulation response was not parsable JSON ({}) -- "
            "falling back to the original question",
            question.id,
            exc,
        )
    return question.question, True


# --- candidate generation per method -----------------------------------------


async def gather_candidates(
    question: MedQAQuestion,
    retriever: Retriever,
    bm25_index: Any,
    reformulator: LLMClient,
    top_k: int,
    rerank_k: int,
) -> tuple[dict[str, list[RetrievedChunk]], str, bool]:
    """One retrieval pass per method for `question`.

    Dense candidates are retrieved once and reused for `dense`,
    `dense_rerank`, and as one input to `hybrid` -- `Retriever.rerank()`
    truncates and overwrites `score`, so the pre-rerank list is what feeds
    fusion, matching `raw_rag.py`'s `retrieve_pass()`.
    """
    dense = await asyncio.to_thread(retriever.retrieve, question.question, top_k)
    dense_rerank = await asyncio.to_thread(retriever.rerank, question.question, dense, rerank_k)
    bm25 = await asyncio.to_thread(bm25_index.search, question.question, top_k)
    hybrid = rrf_fuse(dense, bm25, top_k=top_k)
    hybrid_rerank = await asyncio.to_thread(retriever.rerank, question.question, hybrid, rerank_k)

    reformulated_query, fallback = await reformulate_query(question, reformulator)
    reform_dense = await asyncio.to_thread(retriever.retrieve, reformulated_query, top_k)

    candidates = {
        "dense": dense,
        "dense_rerank": dense_rerank,
        "bm25": bm25,
        "hybrid": hybrid,
        "hybrid_rerank": hybrid_rerank,
        "reform_dense": reform_dense,
    }
    return candidates, reformulated_query, fallback


# --- plan / cost printing ----------------------------------------------------


def print_plan(
    args: argparse.Namespace, config: ExperimentConfig, sample: list[MedQAQuestion], gold: Any
) -> None:
    split = args.split or config.dev_split
    print(f"phase={PHASE} condition_id={CONDITION_ID} split={split}")
    print(f"pinned sample ({len(sample)} question(s)):")
    print("  " + ",".join(question.id for question in sample))
    print(f"  gold letters: {dict(sorted(gold.items()))}")
    print()
    print(f"methods compared (same judge cache): {', '.join(METHOD_NAMES)}")
    print(f"k values reported: {K_VALUES}")
    print()
    judge = config.models["judge_model"]
    reform = config.models["model_a"]
    print(f"judge: {judge.api_model_name} @ {judge.base_url}")
    print(f"reformulator: {reform.api_model_name} @ {reform.base_url}")
    print()
    print(
        f"planned calls: {len(sample)} judge call(s) + {len(sample)} reformulation "
        f"call(s) = {2 * len(sample)} total (one judge call per question covers every "
        "method's union of candidates)"
    )


# --- metrics -------------------------------------------------------------


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-method semantic_recall@k / relevant_fraction@k / first-relevant-rank.

    Reads only successful rows (`"error" not in row`); each row already
    carries every method's ordered candidate ids and the judge's verdicts for
    the union of chunk_ids any method surfaced for that question.
    """
    usable = [row for row in rows if "error" not in row]
    summary: dict[str, Any] = {"n_questions": len(usable), "methods": {}}
    if not usable:
        return summary

    for method in METHOD_NAMES:
        per_k = {k: {"recall_hits": 0, "relevant_fractions": []} for k in K_VALUES}
        first_ranks: list[int] = []
        n = 0
        for row in usable:
            candidates = row["candidates"].get(method)
            if candidates is None:
                continue
            n += 1
            judgments = row["judgments"]
            gold = row["correct_answer"]

            for k in K_VALUES:
                top = candidates[:k]
                hit = any(
                    gold in judgments.get(chunk_id, {}).get("supports_options", [])
                    for chunk_id in top
                )
                per_k[k]["recall_hits"] += int(hit)
                relevant = sum(
                    1
                    for chunk_id in top
                    if judgments.get(chunk_id, {}).get("relevance") in RELEVANT_LEVELS
                )
                per_k[k]["relevant_fractions"].append(relevant / len(top) if top else 0.0)

            rank = next(
                (
                    rank
                    for rank, chunk_id in enumerate(candidates, start=1)
                    if judgments.get(chunk_id, {}).get("relevance") == "relevant"
                ),
                None,
            )
            if rank is not None:
                first_ranks.append(rank)

        summary["methods"][method] = {
            "n": n,
            "per_k": {
                k: {
                    "semantic_recall": round(per_k[k]["recall_hits"] / n, 3) if n else None,
                    "relevant_fraction": round(
                        sum(per_k[k]["relevant_fractions"]) / n, 3
                    )
                    if n
                    else None,
                }
                for k in K_VALUES
            },
            "mean_first_relevant_rank": round(sum(first_ranks) / len(first_ranks), 2)
            if first_ranks
            else None,
        }
    return summary


def render_findings(metrics: dict[str, Any], rows: list[dict[str, Any]], run_dir: Path) -> str:
    """The tracked comparison table plus a few judgments worth reading by hand."""
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

    lines += ["", "## Example judgments", ""]
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
        "Fill in by hand after reading the table above: which method the Phase 6 rerun should "
        "use, and whether reformulation-as-retrieval (`reform_dense`) moved recall enough to "
        "keep the reformulation condition alive for Phase 7.",
        "",
    ]
    return "\n".join(lines)


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


async def amain(args: argparse.Namespace) -> int:
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
    index_dir = resolve_index_dir(config, args.index_dir)

    print_plan(args, config, sample, gold)
    if args.dry_run:
        print(
            "\ndry run -- no index load, no BM25 build, no preflight, no requests. Would write "
            f"to {Path(args.output_dir) / PHASE}/<UTC-stamp>/ (results.jsonl, context.md) and "
            f"render {args.findings_path}"
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
                    "top_k": f"{top_k} -> rerank {rerank_k}",
                    "bm25_index": str(args.bm25_index),
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
                try:
                    candidates, reformulated_query, fallback = await gather_candidates(
                        question, retriever, bm25_index, reformulator, top_k, rerank_k
                    )
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
                except Exception as exc:  # noqa: BLE001 - retrieval/bm25 failure, not an LLM error
                    problems.append(f"q{question.id}: candidate generation failed: {exc}")
                    continue

                chunk_lookup: dict[str, RetrievedChunk] = {}
                for chunks in candidates.values():
                    for chunk in chunks:
                        chunk_lookup.setdefault(chunk.chunk_id, chunk)
                union_ids = sorted(chunk_lookup)
                union_chunks = [chunk_lookup[chunk_id] for chunk_id in union_ids]

                prompt = build_judge_prompt(question, union_chunks)
                try:
                    verdict = await judge.agenerate_structured(
                        prompt, JudgeVerdict, system_prompt=_JUDGE_SYSTEM_PROMPT
                    )
                    judge_breaker.record_success()
                except LLMError as exc:
                    judge_breaker.record_failure(exc)
                    writer.write_row(
                        error_row(
                            model=judge_model, question=question, tag="judge",
                            condition_id=CONDITION_ID, split=split, exc=exc, wall_s=0.0,
                            prompt=prompt,
                        )
                    )
                    problems.append(f"q{question.id}: judge call failed: {exc}")
                    continue

                judgments = {item.chunk_id: item.model_dump() for item in verdict.judgments}
                missing = set(union_ids) - set(judgments)
                if missing:
                    logger.warning(
                        "q{}: judge omitted {} of {} candidate chunk(s): {}",
                        question.id, len(missing), len(union_ids), sorted(missing),
                    )

                row = {
                    "model": judge_model.name,
                    "question_id": question.id,
                    "tag": "judge",
                    "condition_id": CONDITION_ID,
                    "split": split,
                    "correct_answer": question.answer_idx.upper(),
                    "reformulated_query": reformulated_query,
                    "reformulation_fallback": fallback,
                    "candidates": {
                        name: [chunk.chunk_id for chunk in chunks]
                        for name, chunks in candidates.items()
                    },
                    "judgments": judgments,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                writer.write_row(row)
                print(
                    f"  judged {len(union_ids)} unique chunk(s) across {len(candidates)} "
                    f"method(s){' (reformulation fell back)' if fallback else ''}"
                )

            rows = writer.load_rows()

    metrics = compute_metrics(rows)
    findings = render_findings(metrics, rows, writer.dir)
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
