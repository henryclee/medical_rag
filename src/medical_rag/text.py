"""Text normalization shared by the repo's row-level containment tests.

Two places needed the same question answered -- "is this phrase inside that
chunk?" -- and had two different implementations of it. `eval.runlog`'s
`chunk_match_rank` (did the context we handed a model contain the thing it
needed) casefolded and dropped typographic quotes and dashes; the corpus-wide
oracle probe in `retrieval.ceiling` ran a case-insensitive regex over raw text.
Same question, two answers, and no way to tell from a number which one produced
it.

This module is now the single implementation of the normalized form. The corpus
probe deliberately still scans raw text, for two reasons written up in
`retrieval/ceiling.py`: it needs byte offsets into the original body to render
its excerpt window, and normalizing 380,454 bodies per probe multiplies a scan
that R3 is about to run hundreds of times. The consequence is stated there too:
a typographic variant ("well-circumscribed" vs an em-dashed corpus spelling)
reads as a miss there, so the probe's ceiling is a lower bound.
"""

from typing import Any, Sequence

# Typographic characters a StatPearls excerpt and a MedQA option can disagree on
# for reasons that mean nothing. Dropped rather than folded, so the helpers below
# stay substring tests and do not grow into a parser.
_STRIP_CHARS = dict.fromkeys(
    map(ord, "\u2018\u2019\u201c\u201d\u2013\u2014\u2212'\""), None
)


def normalize(text: str) -> str:
    """Casefold, drop quotes/dashes, collapse whitespace -- for containment tests."""
    return " ".join(text.translate(_STRIP_CHARS).casefold().split())


def contains(needle: str, haystack: str) -> bool:
    """Normalized substring test. False on an empty needle, which is not a match."""
    probe = normalize(needle)
    return bool(probe) and probe in normalize(haystack)


def chunk_match_rank(needle: str, chunks: Sequence[Any]) -> int | None:
    """1-based index of the first chunk containing `needle`, else None.

    The cheapest available answer to "did the context we handed the model
    actually contain the thing it needed?", which is where every RAG failure
    classification eventually reduces. It is a string test, so read it with both
    of its errors in mind: it *under*-counts (an excerpt explaining
    "noncaseating granulomas" without the gold phrase "sarcoidosis" reads as
    absent, though it is the useful excerpt) and it *over*-counts (an excerpt
    that names the gold option in order to rule it out reads as present).
    Under-counting is the commoner error at this corpus size, so treat a low hit
    rate as evidence about the metric before evidence about the retriever, and
    never call this accuracy.

    Accepts anything with a `.content` (a `RetrievedChunk` or a stand-in).
    """
    for index, chunk in enumerate(chunks, start=1):
        if contains(needle, chunk.content):
            return index
    return None
