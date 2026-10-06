"""Oracle corpus-ceiling probe: is the recall ceiling retrieval, or the corpus?

The judge harness can only grade chunks it already retrieved. That makes
`semantic_recall@k` a measurement over a candidate set it chose, so it cannot
separate two very different findings: "the supporting chunk exists in StatPearls
but retrieval ranked it #60" (a query-side lever can win this) versus "nothing
in StatPearls supports the gold option" (no reformulation, no k, no reranker can
win this, and the Phase 6 rerun has nothing to find). `FINDINGS.md` puts
`dense__orig` at 40/55/65 recall@5/10/20, so for roughly a third of the pinned 20
the top-20 held no gold-supporting chunk -- and the harness is silent on which of
the two cases those are.

Two halves, both free of completions:

* **containment scan** -- a literal substring search over all 380,454 chunk
  bodies, which the BM25 index already holds in memory (`lexical.BM25Index`
  keeps parallel `chunk_ids`/`titles`/`contents` lists), so this costs no LanceDB
  scan and no model load beyond the pickle.
* **oracle retrieval** -- the *gold option text* used as the query through
  `strategy.retrieve_base()`. If a query that literally contains the answer
  cannot surface a chunk that literally contains it, the bottleneck is the index,
  not the phrasing.

**Both use the gold answer, so neither is a retrieval lever.** They bound what
query-side work could win; they must never be cited as a strategy's score. Every
result carries `label` and the server renders it in a bordered panel with that
warning, because this repo has already published a wrong number off an unlabelled
column (`reform_dense`, whose +5pp recall@20 came from a rewrite that fell back on
18/20 questions).

And the bound is one-directional: a literal *miss* is not proof the corpus lacks
the information. Phase 8 found 32/40 rows chose wording appearing nowhere in the
excerpts -- these models answer from understanding, not string-match -- so a
paraphrased answer can be fully supported by chunks this scan cannot see. Treat
"0 matches" as "not stated in these words," not "not in StatPearls."
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from medical_rag.retrieval.retriever import RetrievedChunk
from medical_rag.retrieval.strategy import BASE_STRATEGIES, retrieve_base

ORACLE_LABEL = (
    "oracle: the query contains the gold answer -- a diagnostic ceiling, "
    "NOT a retrieval lever. Never cite this as a strategy's score."
)

# Below this, a "needle" is a common word, not a finding: `beta` matches thousands
# of chunks and the count says nothing about whether the answer is stated.
MIN_NEEDLE_CHARS = 3

# Above this many matches the text is boilerplate-ish and the hit list is a
# sample, not a census. Reported rather than hidden -- the count is the finding.
GENERIC_MATCH_COUNT = 5_000


@dataclass
class ContainmentHit:
    """One chunk that states the probe text verbatim."""

    chunk_id: str
    title: str
    offset: int
    excerpt: str


@dataclass
class CeilingResult:
    """What the probe found, with its own oracle label attached to it.

    `best_rank` maps every grid strategy to the best rank at which a
    containing chunk appeared in *that strategy's* list when searched with the
    probe text -- `None` meaning it never surfaced. That column is the actual
    ceiling statement: a strategy that misses the literal answer even when you
    query with the literal answer has an indexing problem, not a phrasing one.
    """

    query: str
    corpus_chunks: int
    matches: int
    hits: list[ContainmentHit] = field(default_factory=list)
    truncated: bool = False
    lists: dict[str, list[RetrievedChunk]] = field(default_factory=dict)
    best_rank: dict[str, int | None] = field(default_factory=dict)
    surfaced: dict[str, int] = field(default_factory=dict)
    warning: str = ""
    label: str = ORACLE_LABEL
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Row-shaped, so a probe can be appended to a session's artifacts."""
        return {
            "oracle": True,
            "label": self.label,
            "query": self.query,
            "corpus_chunks": self.corpus_chunks,
            "matches": self.matches,
            "truncated": self.truncated,
            "warning": self.warning,
            "elapsed_ms": self.elapsed_ms,
            "hits": [hit.__dict__ for hit in self.hits],
            "candidates": {m: [c.chunk_id for c in chunks] for m, chunks in self.lists.items()},
            "best_rank": self.best_rank,
            "surfaced": self.surfaced,
        }


@dataclass
class Scan:
    """What a containment scan found, keeping the census and the sample distinct.

    `ids` is every matching chunk and `hits` only the first `limit`, because they
    answer different questions: `ids` is what a strategy had to hit (scoring
    `best_rank` against the *displayed* sample would report "retrieval never
    surfaced it" for a chunk that surfaced at #4 and simply was not among the 20
    excerpts shown), while `hits` is prose for a human to read. On the real corpus
    a two-word phrase like "CT angiography" matches 278 chunks, so the sample is
    almost always a small fraction of the census and the difference is not edge-case.
    """

    matches: int
    ids: set[str]
    hits: list[ContainmentHit]
    truncated: bool


def containment_scan(
    index: Any, needle: str, *, limit: int = 20, window: int = 90
) -> Scan:
    """Count chunks whose body contains `needle` verbatim, case-insensitively.

    A regex over the in-memory lists rather than a LanceDB `LIKE`: the BM25 pickle
    already has every body, and a 380k-row `.to_pandas()` to answer one question
    is the cost this directory exists to avoid.
    """
    needle = (needle or "").strip()
    if len(needle) < MIN_NEEDLE_CHARS:
        raise ValueError(f"probe text must be at least {MIN_NEEDLE_CHARS} characters, got {needle!r}")
    pattern = re.compile(re.escape(needle), re.IGNORECASE)
    contents: Sequence[str] = index.contents
    titles: Sequence[str] = index.titles
    chunk_ids: Sequence[str] = index.chunk_ids

    ids: set[str] = set()
    hits: list[ContainmentHit] = []
    for i, content in enumerate(contents):
        found = pattern.search(content)
        if found is None:
            continue
        chunk_id = chunk_ids[i]
        ids.add(chunk_id)
        if len(hits) < limit:
            start = max(0, found.start() - window // 2)
            hits.append(
                ContainmentHit(
                    chunk_id=chunk_id,
                    title=titles[i] if i < len(titles) else "",
                    offset=found.start(),
                    excerpt=content[start : found.end() + window].strip(),
                )
            )
    return Scan(matches=len(ids), ids=ids, hits=hits, truncated=len(ids) > len(hits))


def best_ranks(
    lists: dict[str, Sequence[RetrievedChunk]], containing: set[str]
) -> tuple[dict[str, int | None], dict[str, int]]:
    """Per strategy: best rank of a containing chunk, and how many surfaced.

    Empty `containing` yields all-`None` rather than an empty dict, so the caller
    renders "did not surface" for every strategy instead of a blank row that
    looks like a missing measurement.
    """
    ranks: dict[str, int | None] = {}
    counts: dict[str, int] = {}
    for method, chunks in lists.items():
        seen = [rank for rank, chunk in enumerate(chunks, start=1) if chunk.chunk_id in containing]
        ranks[method] = seen[0] if seen else None
        counts[method] = len(seen)
    return ranks, counts


def probe(
    query_text: str,
    retriever: Any,
    bm25_index: Any,
    *,
    top_k: int,
    rerank_k: int,
    bm25_top_k: int | None = None,
    limit: int = 20,
) -> CeilingResult:
    """Run both halves of the probe on one string.

    The cross-encoder is shown the probe text (via `retrieve_base`'s default
    `rerank_query=None`), not the question -- the point is "what does this string
    retrieve", so swapping the reranker's string in would reintroduce a second
    variable into a single-variable measurement.
    """
    text = (query_text or "").strip()
    if not text:
        raise ValueError("probe text is empty")

    started = time.monotonic()
    scan = containment_scan(bm25_index, text, limit=limit)
    lists = retrieve_base(
        text, retriever, bm25_index, top_k=top_k, rerank_k=rerank_k, bm25_top_k=bm25_top_k
    )
    # Scored against every matching chunk, not the displayed sample -- see Scan.
    ranks, counts = best_ranks(lists, scan.ids)

    warning = ""
    if scan.matches > GENERIC_MATCH_COUNT:
        warning = (
            f"{scan.matches} matches means this text is boilerplate, not a finding -- the count "
            "is not evidence about where the answer is stated"
        )
    elif scan.matches == 0:
        warning = (
            "0 literal matches: StatPearls does not state this in these words. That is NOT "
            "proof it is absent -- a paraphrased answer can still be supported (see this "
            "module's docstring)"
        )

    return CeilingResult(
        query=text,
        corpus_chunks=len(bm25_index.contents),
        matches=scan.matches,
        hits=scan.hits,
        truncated=scan.truncated,
        lists=lists,
        best_rank={method: ranks.get(method) for method in _display_order(lists)},
        surfaced={method: counts.get(method, 0) for method in _display_order(lists)},
        warning=warning,
        elapsed_ms=int((time.monotonic() - started) * 1000),
    )


def _display_order(lists: dict[str, Any]) -> list[str]:
    """Grid order, falling back to insertion order for off-grid keys."""
    known = [base for base in BASE_STRATEGIES if base in lists]
    return known + [key for key in lists if key not in BASE_STRATEGIES]
