# ADR-0008: The judge cache is the ground truth

## Status
Accepted. R2 must validate the oracle before R4/R5's numbers are quoted; that is
a gate inside this decision, not a follow-up.

## Context
Every retrieval number in this repo is a *semantic* claim: does this chunk help
answer this question, and which option does it give evidence for. A string match
cannot make that claim -- Phase 6's literal recount found the gold option's wording
in the excerpts for 4/20 questions, while the judge grading the same corpus found
the gold option supported in 40% of `dense__orig`'s top-5 (`../../experiments/phase6/FINDINGS.md`
vs `../../experiments/retrieval_tuning/FINDINGS.md`). Somebody has to decide what
"supports" means, and that somebody is `judge_model`.

There is no cheaper substitute. No gold set of `(question, chunk)` relevance
labels exists for MedQA-over-StatPearls, producing one by hand is the kind of
multi-day annotation this project does not have, and the option letter cannot be
used as the label -- leaking it into grading is exactly what the rubric forbids
("NOT told which option is correct"), so gold-option recall is derived
*programmatically* from `supports_options` after the fact.

The cache is already the load-bearing artifact in practice: 926 verdicts are
reusable under the current rubric (`judge_prompt_sha 21e27467277a`), each one paid
for in completions on the shared `:8080`. `scripts/eval_retrieval.py` scores a
strategy by looking verdicts up. Which means the cache is the oracle whether or not
 anyone voted on it -- and today it fails the repo's own evidence rule, because it
lives in gitignored `outputs/`.

## Decision
`judge_model`'s cached verdicts are the measurement oracle for the retrieval
track. Specifically:

- **Verdicts are never re-asked to change an answer.** A number is recomputed from
  the cache; the judge is called only for `(question_id, chunk_id)` pairs it has
  never seen. This is the existing rule in `eval/judge.py`, promoted from
  convenience to policy.
- **The rubric is a versioned part of the measurement.** `judge_prompt_sha` stays a
  component of the cache key, and stale-sha verdicts are ignored rather than
  deleted -- a rubric may be reverted, and `--stats` reporting how many verdicts a
  rubric change would invalidate is worth more than a tidy file.
- **The oracle is tracked.** R2 commits a compact snapshot (ids, relevance,
  `supports_options`, reason -- no chunk prose) under
  `experiments/retrieval/evalset/`. The live cache stays regenerable-but-expensive;
  the snapshot is what a reviewer reads.
- **A new lever has to enter the grid to become measurable.** Harness scores are
  computed over the union of candidate lists it actually retrieved. Scoring a new
  strategy against the existing cache is legal and cheap, and every unjudged slot
  counts as *not* supporting -- so a fresh strategy's number is a floor, and
  `eval_retrieval.py` prints coverage next to recall and fails on
  `--require-coverage` for exactly this reason.
- **R2 validates the oracle before trusting it:** a stratified hand-check of ~30
  verdicts that deliberately includes the known-bad boilerplate chunks as controls
  and reports agreement per class rather than as one percentage; a silent-drop audit
  proving `judgments == candidates` for every batch at `--judge-batch 24`; one rubric
  variant stored under its own sha to measure verdict stability; and a test, not a
  reading, that gold answer text cannot reach the prompt.

## Consequences
Easier: a hypothesis costs index work and arithmetic, so the number of hypotheses
per afternoon stops being limited by endpoint throughput; the measurement is
reproducible by one command; and the rubric's identity is auditable rather than
remembered.

Harder, and this is the real cost: **the judge's endpoint-side settings now shape
ground truth.** `judge_model` runs on the same oMLX server as the old arms, and
its sampling parameters are applied outside git -- the exact problem ADR-0006 left
open for `model_a` (open question 16), now elevated: a knob nobody sends but the
server supplies decides what counts as evidence. R2 records them next to the
snapshot.

Judgement errors become systematic rather than random, because one model grades
every cell: if it is lenient about case-report boilerplate, *every* strategy's
lexical recall is inflated together and the comparison still looks clean. The
boilerplate controls in R2's hand-check exist to catch precisely that, and a
strategy's win is not believed until its supporting verdicts have been read.

Coverage is a permanent tax on novelty: any new candidate pool arrives unjudged,
so each R-phase that adds a lever must budget judge calls for it up front --
`--require-coverage` exists to make an unjudged comparison fail loudly instead of
quietly reading as a regression.

## Evidence
`../../experiments/retrieval_tuning/TUNING.md` (the rubric, and the judge
reliability this ADR promotes from "open question" to "gate"); `src/medical_rag/eval/judge.py`
(cache key, ignore-don't-delete, the `missing_judgment` counter that records a
judge dropping items from a long list); `scripts/eval_retrieval.py` (coverage
printed beside recall); `../../experiments/phase6/FINDINGS.md` (the 4/20 literal
recount that semantic judging replaced).
