# ADR-0009: Corpus and chunking are frozen for the retrieval track

## Status
Accepted, with an expiry: it holds until a scheduled chunking phase wants to
become real, at which point that phase must first pay for this one's cache.

## Context
Verdicts are keyed `(question_id, chunk_id, judge_prompt_sha)`
(`src/medical_rag/eval/judge.py`). `chunk_id` is the only thing tying a verdict to
the prose it graded -- chunk *content* is deliberately not part of the key, because
storing and hashing 380,454 bodies per lookup costs more than the failure it
prevents. That design is what makes re-grading free, and it has a consequence that
was never written down until now: change how the corpus is chunked and every
`chunk_id` changes, so every cached verdict silently becomes a verdict about
something else.

"Silently" is the problem. `load_cache()` filters on `judge_prompt_sha` and has
nothing to compare a chunk's text against, so re-chunking without bumping the
rubric version serves *plausible* verdicts for new prose. The counterweight today is
only auditability after the fact: `context.md` records `index_sha256` next to
`judge_prompt_sha`, so a run that pairs a new index with old verdicts can be caught
-- by someone who thinks to look.

Chunking also looks tempting because it is not actually configured anywhere. The
corpus is 380,454 chunks from 9,652 articles, cut by a vendored port of MedRAG's
article-section algorithm (`src/medical_rag/data/chunking.py`) -- not by a token
window. `config/default.yaml` nonetheless carried `chunk_size: 512` and
`chunk_overlap: 64`, and nothing read them.

## Decision
For the R-phases, the corpus and its chunking are frozen: same NCBI StatPearls
archive, same section splitter, same 380,454 chunks, same LanceDB table.

- `retrieval.chunk_size` and `retrieval.chunk_overlap` are **deleted from
  `config/default.yaml` and `RetrievalConfig`**, not wired up. Making them live
  would be a second lie: the vendored chunker never consults them, so a knob that
  moves nothing is worse than no knob. Corpus shape is documented as provenance
  instead.
- Re-chunking is an **explicit cache-rebuild event**, owned by the backlog. A phase
  that changes chunk geometry must state the re-judge cost before starting --
  approximately the whole R2 judge spend, since nothing survives -- and bump
  `JUDGE_PROMPT_VERSION` as part of the same change so the old verdicts are
  discarded loudly rather than reused quietly.
- `data/chunking.py` stays in the tree with no callers, as the natural home of that
  backlog lever.

## Consequences
Easier: R3-R5 can run dozens of hypotheses against one stable oracle, and a
regression between two runs means the strategy changed rather than the ground
moving underneath it. Deleting the fake knobs makes `config/default.yaml` mean what
it says -- every field in it is now read by `RetrievalStrategy.from_config`.

Harder: some of the ceiling this track measures may be an artifact of section-sized
chunks, and R3 cannot separate "StatPearls never said it" from "the passage that
says it is split across a chunk boundary" -- the probe's bound is already
one-directional (0 matches means *not in these words*), and freezing the chunking
makes that ambiguity permanent for this track. Chunk-level recall is measured over
the chunks we happen to have.

And the freeze is a constraint on tooling, not a claim that the chunking is good.
`lexical`'s degeneracy (59 distinct chunks filling 100 top-5 slots) may turn out to
be partly a chunking symptom; if R4 fixes the tokenizer and the degeneracy
survives, that is the evidence that re-chunking deserves its cache bill.

## Evidence
`src/medical_rag/eval/judge.py` (the key, and why content is not in it);
`src/medical_rag/data/chunking.py` (the vendored MedRAG section splitter -- no
token window anywhere); `../../experiments/retrieval_tuning/FINDINGS.md` (the
lexical degeneracy that R4 attacks, measured on this frozen chunking);
`../../PLAN.md`'s R1 audit trail (the call to delete the two dead keys rather than
wire them; argued in §9.2 of `git show 8654e08:refactor_plan.md`).
