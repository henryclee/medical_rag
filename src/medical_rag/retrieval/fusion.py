"""Reciprocal Rank Fusion of ranked chunk lists, for the hybrid retrieval lever.

RRF needs only each list's rank order, not its (incomparable) score scale --
which matters here because dense scores are bounded cosine similarities and
BM25 scores are unbounded term-frequency scores. `Retriever.rerank()` in
`src/medical_rag/retrieval/retriever.py` shows the same lesson from the
other side: it discards the pre-rerank score rather than mixing it with the
cross-encoder logit.
"""

from medical_rag.retrieval.retriever import RetrievedChunk

RRF_K = 60


def rrf_fuse(
    *ranked_lists: list[RetrievedChunk], top_k: int, k: int = RRF_K
) -> list[RetrievedChunk]:
    """Fuse any number of ranked chunk lists by `sum(1 / (k + rank))`.

    A chunk absent from a list simply does not contribute that term, rather
    than being penalized with a fabricated worst-case rank -- so a chunk only
    BM25 found and dense retrieval never surfaced (or vice versa) is not
    treated as if every method considered and rejected it.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(chunk.chunk_id, chunk)

    ordered_ids = sorted(scores, key=lambda chunk_id: -scores[chunk_id])[:top_k]
    return [
        first_seen[chunk_id].model_copy(update={"score": scores[chunk_id]})
        for chunk_id in ordered_ids
    ]
