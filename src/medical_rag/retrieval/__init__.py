"""Retrieval -- the half of this repo the project is now about.

`strategy.py` is the entry point: it defines the five strategies the study
measures and the single fan-out (`retrieve_base`) that computes all of them from
one query, so a strategy means the same thing in a findings table as it does in
a shipped call. `lexical.py` (BM25) and `fusion.py` (RRF) exist because the
first grid put the lexical leg at 0/0/10 -- the legs are cheap to re-measure and
expensive to get wrong, so they are named rather than inlined. `ceiling.py` is
the oracle probe: an upper bound, never a lever.

Chunk geometry is deliberately absent from this package. The corpus and its
chunking are frozen while the judge verdicts that score every number key on
`chunk_id` (`eval/judge.py`); re-chunking would orphan them, so chunking lives in
the backlog, not in a knob.
"""
