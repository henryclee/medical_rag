#!/usr/bin/env python
"""Score a retrieval strategy against the judge cache, with zero endpoint calls.

This is the command the retrieval track iterates against. The oracle is the
cached judge verdicts (`eval/judge.py`, ADR-0008), so scoring costs index work
and arithmetic, not completions -- which is the whole reason R4/R5 can afford
to try things.

Every strategy in one pass is nearly free: `retrieve_base()` computes the dense
list, the lexical list, the fused list and both reranked lists anyway, so the
default here compares all five rather than making you re-load a 467 MB BM25
pickle per hypothesis. Narrow it with repeated `--strategy`.

The number is only as good as the cache's coverage of the candidate slots, and a
new strategy is exactly the thing whose chunks nobody has judged yet -- so
coverage is printed as loudly as recall, and `--require-coverage 1.0` turns a
partially-unjudged run into a non-zero exit. A 40% recall over slots that are
60% unjudged is a measurement of the cache, not of retrieval.

    .venv/bin/python scripts/eval_retrieval.py --dry-run     # resolved config, nothing loaded
    .venv/bin/python scripts/eval_retrieval.py               # all five strategies, pinned sample
    .venv/bin/python scripts/eval_retrieval.py --strategy hybrid --k 5 10 20

`--rerender` on `run_retrieval_grid.py` is the other gate: it rebuilds
FINDINGS.md from a stored run offline. Use this one for hypotheses that have
never been retrieved; use that one to prove a refactor moved no number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from medical_rag.config import load_config
from medical_rag.eval.harness import SAMPLE_QUESTION_IDS
from medical_rag.eval.judge import DEFAULT_JUDGE_CACHE, load_cache
from medical_rag.eval.metrics import K_VALUES, aggregate
from medical_rag.eval.rubric import judge_prompt_sha
from medical_rag.eval.runlog import select_by_ids
from medical_rag.retrieval.lexical import DEFAULT_BM25_PATH, load_or_build_bm25_index
from medical_rag.retrieval.strategy import BASE_STRATEGIES, RetrievalStrategy, build_retriever

DEFAULT_K = tuple(K_VALUES)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/default.yaml", help="path to default.yaml")
    parser.add_argument(
        "--strategy",
        action="append",
        default=None,
        choices=list(BASE_STRATEGIES),
        help="strategy to score; repeatable. Default: every strategy this config computes.",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        dest="question_ids",
        default=None,
        help=f"question to score; repeatable. Default: the pinned sample "
        f"({len(SAMPLE_QUESTION_IDS)} ids).",
    )
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K))
    parser.add_argument("--judge-cache", default=str(DEFAULT_JUDGE_CACHE))
    parser.add_argument("--index-dir", default=None, help="default: data/index/<corpus>")
    parser.add_argument("--bm25-index", default=str(DEFAULT_BM25_PATH))
    parser.add_argument("--rebuild-bm25", action="store_true")
    parser.add_argument("--json", default=None, help="write per-question detail here")
    parser.add_argument(
        "--require-coverage",
        type=float,
        default=0.0,
        metavar="FRACTION",
        help="exit non-zero unless this share of scored candidate slots had a cached verdict",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved strategy, sample and cache state, then exit -- loads no "
        "index, no models, and makes no calls",
    )
    return parser.parse_args(argv)


def coverage_of(rows: list[dict[str, Any]], methods: list[str]) -> tuple[int, int]:
    """(judged slots, total slots) across every scored method and question."""
    judged = total = 0
    for row in rows:
        for method in methods:
            candidates = row["candidates"].get(method) or []
            total += len(candidates)
            judged += sum(1 for chunk_id in candidates if chunk_id in row["judgments"])
    return judged, total


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    strategy = RetrievalStrategy.from_config(config, args.strategy[0] if args.strategy else None)
    methods = args.strategy or list(BASE_STRATEGIES)
    ids = args.question_ids or list(SAMPLE_QUESTION_IDS)
    sha = judge_prompt_sha()

    verdicts, counts = load_cache(args.judge_cache, judge_sha=sha)

    print(f"strategy   : {strategy}")
    print(f"index      : {config.retrieval.corpus} / {config.retrieval.embedding_model} "
          f"/ {config.retrieval.reranker_model}")
    print(f"scored as  : {', '.join(methods)} at k={', '.join(str(k) for k in args.k)}")
    print(f"questions  : {len(ids)} pinned ids from the {config.dev_split} split")
    print(f"rubric sha : {sha[:12]}")
    print(f"cache      : {args.judge_cache}")
    print(
        f"           : {counts['loaded']} verdict(s) usable under this rubric "
        f"({counts['stale_prompt']} stale, {counts['malformed']} malformed"
        + (", FILE MISSING" if counts["missing_file"] else "")
        + ")"
    )

    if args.dry_run:
        print("\n--dry-run: stopped here. Nothing loaded, no calls made.")
        return 0

    # Imported here rather than at the top: `load_medqa` reads the HF dataset
    # cache, and `--dry-run` exists to be answerable from config alone.
    from medical_rag.data.load_medqa import load_medqa

    questions = select_by_ids(load_medqa(config.dev_split), ids)

    retriever, static = build_retriever(config, args.index_dir)
    bm25 = load_or_build_bm25_index(
        Path(static["index_dir"]), args.bm25_index, rebuild=args.rebuild_bm25
    )

    rows: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        lists = strategy.lists(question.question, retriever, bm25)
        candidates = {name: [c.chunk_id for c in lists[name]] for name in methods}
        judgments = {
            chunk_id: verdicts[(question.id, chunk_id)]
            for chunk_id in {c for ids_ in candidates.values() for c in ids_}
            if (question.id, chunk_id) in verdicts
        }
        rows.append(
            {
                "question_id": question.id,
                "correct_answer": question.answer_idx,
                "candidates": candidates,
                "judgments": judgments,
            }
        )
        detail.append({"question_id": question.id, "gold": question.answer_idx, **candidates})
        print(f"  [{index}/{len(questions)}] q{question.id} gold={question.answer_idx}")

    summary = aggregate(rows, methods, ks=args.k)
    judged, total = coverage_of(rows, methods)
    share = (judged / total) if total else 0.0

    print(f"\n{'method':<16}{'n':>4}" + "".join(f"{'recall@' + str(k):>11}" for k in args.k))
    print("-" * (20 + 11 * len(args.k)))
    for method in methods:
        cell = summary["methods"].get(method)
        if not cell:
            print(f"{method:<16}{0:>4}{'-':>11}")
            continue
        line = f"{method:<16}{cell['n']:>4}"
        for k in args.k:
            value = cell["per_k"][k]["semantic_recall"]
            line += f"{'-' if value is None else f'{value * 100:.0f}%':>11}"
        first = cell["mean_first_relevant_rank"]
        line += f"   first_relevant={first if first is not None else '-'}"
        print(line)

    print(
        f"\ncoverage: {judged}/{total} scored candidate slots have a cached verdict "
        f"({share * 100:.1f}%)"
    )
    if share < 1.0:
        print(
            "  WARNING: unjudged slots count as *not* supporting the gold option, so every\n"
            "  number above is a floor for the strategies whose chunks the cache has never\n"
            "  seen. Run the grid (`scripts/run_retrieval_grid.py`) to judge new candidates\n"
            "  before believing a comparison that favours an older, better-judged strategy."
        )

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "strategy": strategy.__dict__,
                    "judge_prompt_sha": sha,
                    "coverage": {"judged": judged, "total": total},
                    "summary": summary,
                    "questions": detail,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path}")

    if share < args.require_coverage:
        print(
            f"\nFAIL: coverage {share * 100:.1f}% is below the required "
            f"{args.require_coverage * 100:.1f}%",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
