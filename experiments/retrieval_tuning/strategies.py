"""The retrieval-strategy grid: five strategies x two query variants.

`judge_harness.py` and `inspect_retrieval.py` both import this module, so the
number in `FINDINGS.md` and the cell a human reads in the browser are computed
by the same function. Otherwise a viewer quietly drifts from the table it is
meant to explain.

Two independent variables, deliberately orthogonal:

* **strategy** -- `dense`, `dense_rerank`, `bm25`, `hybrid`, `hybrid_rerank`.
* **query variant** -- `orig` (the question text) and `reform` (the rewritten
  information need). Reformulation is *not* a sixth strategy: a strategy that
  only ever ran on one query cannot say whether a win came from the lever or
  from the rewrite, which is how `reform_dense` got read as evidence in the
  first run while being byte-identical to `dense` on 18 of 20 questions.

Method ids are therefore ``f"{base}__{variant}"`` -- `dense__reform`,
`hybrid_rerank__orig` -- 10 cells.

One knob inside the grid needs naming because it is a real second variable:
which string the *cross-encoder* is shown. Default is the **question**, not the
reformulated query, matching `raw_rag.py`'s `retrieve_pass()`, so
`dense__orig` vs `dense__reform` isolates the embedding step alone. Flip it with
`--rerank-with reform` and record the choice in `context.md`; do not compare a
run across the two settings.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from medical_rag.retrieval.retriever import RetrievedChunk

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hybrid import rrf_fuse  # noqa: E402

BASE_STRATEGIES: tuple[str, ...] = (
    "dense",
    "dense_rerank",
    "bm25",
    "hybrid",
    "hybrid_rerank",
)
QUERY_VARIANTS: tuple[str, ...] = ("orig", "reform")

# Order here is display order in FINDINGS.md and in the inspection grid: the
# five strategies for the original query, then the same five for the rewritten
# one, so a column-to-column read is a same-row read.
METHOD_GRID: tuple[str, ...] = tuple(
    f"{base}__{variant}" for variant in QUERY_VARIANTS for base in BASE_STRATEGIES
)

# Which list a reranked cell was drawn from, for the "#7 -> #1" movement marker
# the inspector renders. `hybrid_rerank` reranks the *fused* list, not dense.
PRERANK_SOURCE: dict[str, str] = {"dense_rerank": "dense", "hybrid_rerank": "hybrid"}


def method_id(base: str, variant: str) -> str:
    """Build a grid id, rejecting an off-grid pair rather than inventing one."""
    if base not in BASE_STRATEGIES:
        raise ValueError(f"unknown strategy {base!r} (known: {', '.join(BASE_STRATEGIES)})")
    if variant not in QUERY_VARIANTS:
        raise ValueError(f"unknown query variant {variant!r} (known: {', '.join(QUERY_VARIANTS)})")
    return f"{base}__{variant}"


def split_method(method: str) -> tuple[str, str]:
    """`"hybrid_rerank__orig"` -> `("hybrid_rerank", "orig")`."""
    base, sep, variant = method.partition("__")
    if not sep or base not in BASE_STRATEGIES or variant not in QUERY_VARIANTS:
        raise ValueError(
            f"{method!r} is not a grid method (expected <{'|'.join(BASE_STRATEGIES)}>__"
            f"<{'|'.join(QUERY_VARIANTS)}>)"
        )
    return base, variant


def base_of(method: str) -> str:
    return split_method(method)[0]


def variant_of(method: str) -> str:
    return split_method(method)[1]


def grid_for(variants: Sequence[str]) -> tuple[str, ...]:
    """The grid restricted to some query variants (`("orig",)` when reformulation is off)."""
    unknown = [variant for variant in variants if variant not in QUERY_VARIANTS]
    if unknown:
        raise ValueError(f"unknown query variant(s): {', '.join(unknown)}")
    return tuple(method for method in METHOD_GRID if variant_of(method) in set(variants))


# Method ids used by the 2026-09-29 run, before the grid existed. Kept so
# `--from-run` can render that run instead of erroring on it -- and deliberately
# kept *visible*: the inspector labels an aliased cell `legacy`, because a row
# where `reform_dense` was byte-identical to `dense` is evidence about a parsing
# bug, not about reformulation, and silently renaming it `dense__reform` would
# let it read as a measured reform result.
LEGACY_METHOD_ALIASES: dict[str, str] = {
    "dense": "dense__orig",
    "dense_rerank": "dense_rerank__orig",
    "bm25": "bm25__orig",
    "hybrid": "hybrid__orig",
    "hybrid_rerank": "hybrid_rerank__orig",
    "reform_dense": "dense__reform",
}


def canonical_method(name: str) -> tuple[str, bool]:
    """Map a stored method name to a grid id -> `(method, was_aliased)`."""
    if name in METHOD_GRID:
        return name, False
    if name in LEGACY_METHOD_ALIASES:
        return LEGACY_METHOD_ALIASES[name], True
    raise ValueError(f"{name!r} is neither a grid method nor a known legacy id")



@dataclass
class GridResult:
    """Every cell's ranked list, plus the query each variant actually searched on.

    `reused_variant` records the honest shortcut: when the reformulated query
    came back identical to the question (a fallback, or a rewrite that just
    echoed it), the reform lists *are* the orig lists and no second retrieval
    ran. Rendering that as two empty columns would read like a retrieval failure
    instead of the reformulation failure it is.
    """

    lists: dict[str, list[RetrievedChunk]] = field(default_factory=dict)
    variant_queries: dict[str, str] = field(default_factory=dict)
    reused_variant: str | None = None
    rerank_with: str = "question"

    def methods(self) -> tuple[str, ...]:
        return tuple(self.lists)

    def all_chunks(self) -> dict[str, RetrievedChunk]:
        """Every distinct chunk the grid surfaced, first occurrence winning."""
        lookup: dict[str, RetrievedChunk] = {}
        for chunks in self.lists.values():
            for chunk in chunks:
                lookup.setdefault(chunk.chunk_id, chunk)
        return lookup


def retrieve_base(
    query: str,
    retriever: Any,
    bm25_index: Any,
    *,
    top_k: int,
    rerank_k: int,
    bm25_top_k: int | None = None,
    rerank_query: str | None = None,
) -> dict[str, list[RetrievedChunk]]:
    """The five strategy lists for one query string.

    Dense and BM25 are retrieved once each and reused as the inputs to rerank
    and fusion -- `Retriever.rerank()` truncates and overwrites `score`, so the
    pre-rerank list must be kept to feed fusion, and `rrf_fuse()` returns copies
    rather than mutating, so the same objects are safe to share. `bm25_top_k`
    defaults to `top_k`; raising it is a lever, not a fix, and the first run's
    `bm25` recall@20 of 10% is the reason the knob exists at all.
    """
    scored_query = query if rerank_query is None else rerank_query
    dense = retriever.retrieve(query, top_k)
    lexical = bm25_index.search(query, bm25_top_k or top_k)
    hybrid = rrf_fuse(dense, lexical, top_k=top_k)
    return {
        "dense": dense,
        "dense_rerank": retriever.rerank(scored_query, dense, rerank_k),
        "bm25": lexical,
        "hybrid": hybrid,
        "hybrid_rerank": retriever.rerank(scored_query, hybrid, rerank_k),
    }



def retrieve_grid(
    question_text: str,
    retriever: Any,
    bm25_index: Any,
    *,
    reformulated_text: str | None = None,
    top_k: int,
    rerank_k: int,
    bm25_top_k: int | None = None,
    rerank_with: str = "question",
) -> GridResult:
    """Every cell of the grid. Blocks; call `aretrieve_grid()` from async code.

    `reformulated_text=None` (or identical to `question_text`) yields the
    orig-only grid -- 5 cells -- which is what `--no-reform` and a fell-back
    reformulation should produce.
    """
    result = GridResult(rerank_with=rerank_with)
    queries: dict[str, str] = {"orig": question_text}
    if reformulated_text and reformulated_text.strip() != question_text.strip():
        queries["reform"] = reformulated_text
    elif reformulated_text:
        result.reused_variant = "reform"

    for variant, query in queries.items():
        # The reform column keeps the *question* as the reranker's string by
        # default, so the only difference between the columns is what got
        # embedded. See the module docstring.
        rerank_query = query if (variant == "orig" or rerank_with == "reform") else question_text
        for base, chunks in retrieve_base(
            query,
            retriever,
            bm25_index,
            top_k=top_k,
            rerank_k=rerank_k,
            bm25_top_k=bm25_top_k,
            rerank_query=rerank_query,
        ).items():
            result.lists[method_id(base, variant)] = chunks

    if result.reused_variant == "reform":
        for base in BASE_STRATEGIES:
            result.lists[method_id(base, "reform")] = result.lists[method_id(base, "orig")]

    result.variant_queries = queries
    return result


async def aretrieve_grid(
    question_text: str,
    retriever: Any,
    bm25_index: Any,
    **kwargs: Any,
) -> GridResult:
    """`retrieve_grid()` off the event loop.

    One grid is four embedding/vector passes, two BM25 scans over 380k postings
    and two 20-pair rerank passes -- seconds of MPS work that would otherwise
    stall the loop talking to the judge.
    """
    return await asyncio.to_thread(retrieve_grid, question_text, retriever, bm25_index, **kwargs)


def prerank_map(method: str, lists: dict[str, list[RetrievedChunk]]) -> dict[str, int]:
    """`chunk_id -> rank in the pre-rerank source list`, for `*_rerank` cells.

    Empty when the cell has no recorded source, which is the caller's cue to
    render a plain rank instead of a movement arrow -- the alternative is a
    silent "#1 -> #1" that looks like a rerank agreeing with itself.
    """
    source_base = PRERANK_SOURCE.get(base_of(method))
    if source_base is None:
        return {}
    source = lists.get(method_id(source_base, variant_of(method)))
    if source is None:
        return {}
    return {chunk.chunk_id: rank for rank, chunk in enumerate(source, start=1)}

