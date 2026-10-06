# ADR-0007: Retrieval-first; the answer-accuracy study is cut

## Status
Accepted. Supersedes the *consequences* of
[ADR-0002](./0002-instruct-reasoning-model-pair.md) -- the pair itself stays, and
Phases 5-6's measurements stay citable as evidence -- without retracting anything
they measured.

## Context
The project was scoped as a two-arm factorial: does RAG move answer accuracy, and
does the effect depend on which LLM answers (Phases 7-13). Phases 5-6 ran and
produced a result that the plan under-weighted rather than obeyed.

Phase 6 measured whether the retrieved excerpts contained the gold option's
wording at all: **4/20 questions at k=5**
(`../../experiments/phase6/FINDINGS.md`). Three quarters of that phase's rows
therefore could not distinguish "the model ignored the context" from "the context
never contained it" -- by its own admission. Phases 7-13 would spend ~7 h of
shared-endpoint generation per condition per arm to measure how models behave on
a context that usually lacks the answer, and attribute the result to the model.

The retrieval-tuning grid then measured the shipped funnel semantically instead of
by string match (`../../experiments/retrieval_tuning/FINDINGS.md`, run
`20260929T120215Z`, n=20): `dense__orig` 40/55/65 semantic recall@5/10/20,
`dense_rerank__orig` 45/45/45, `hybrid__orig` 30/40/55,
`hybrid_rerank__orig` 25/25/25, `bm25__orig` 0/0/10. So the ceiling on that
first grid is 65% on a sample too small to defend, the *best* leg is
degenerate (59 distinct chunks fill the 100 top-5 slots; one chunk is rank-1 for
6/20 questions), and the rerank columns cannot rise above 45 because the funnel
hands the reranker 5 candidates and asks for 5. Meanwhile the one generation-side
lever the grid did test -- reformulation -- fell back to the raw question on 18/20
questions, which is why its `__reform` column is the `__orig` column copied.

Two asymmetries decided it. Cost: a retrieval hypothesis now costs index work and
arithmetic against a cached verdict set (`scripts/eval_retrieval.py`, zero
completions), while a generation hypothesis costs hours of endpoint time that is
shared with the judge. Attribution: with recall in the 40s, a generation result is
a measurement of retrieval wearing a model label.

## Decision
Retrieval quality is the project's object, and the generation-side study is cut
rather than deferred: Phases 7-13 (reformulator, verifier, runner, statistics,
taxonomy, confirmatory run), their stub modules under `src/medical_rag/`,
`scripts/run_pilot.py` / `run_experiment.py`, `config/conditions.yaml`, and
`tests/test_reformulator.py` / `tests/test_verifier.py` are deleted, not disabled.
Deletion over comment-out: the factorial's whole value was that the tree carried
one set of live assumptions.

The new plan is `R1`-`R6` in `../../PLAN.md`. The `R` prefix is deliberate:
`Phase 7` and `Phase 8` mean *reformulation* and *verification* in
`../../experiments/phase6/FINDINGS.md` and in the frozen tooling, and reusing those
numbers would silently reassign every historical citation.

`model_a` and `model_b` stay in `config/models.yaml`, and `raw_rag.py`,
`closed_book.py`, `probe_models.py`, `audit_run.py` and the answer/verification
prompt builders stay in the tree, labelled frozen Phases 5-6 evidence tooling. Cut
the study, not the evidence the study produced.

## Consequences
Easier: hypotheses become cheap enough to run routinely -- every strategy the grid
computes comes out of one retrieval pass, so comparing five costs no more endpoint
time than one, and none at all once the verdict cache covers the candidates. The
measurement stops depending on a model's decoding.

Harder: the oracle is now a model's judgement rather than an answer key, which is
a different kind of trust to earn -- see [ADR-0008](./0008-judge-cache-as-ground-truth.md).
The pinned 20 questions stay, but at n=20 a 5 pp difference is one question, so the
sample has to grow before a delta can be reported. And Phases 5-6's findings about
model behaviour are now context, not results: they were measured through a funnel
this project considers unsolved.

## Evidence
`../../experiments/phase6/FINDINGS.md` (4/20 wording recall at k=5; the
"cannot distinguish" admission); `../../experiments/retrieval_tuning/FINDINGS.md`
(run `20260929T120215Z`: the grid, the `*_rerank` cap, the 18/20 fallback, the
lexical degeneracy); `../../refactor_plan.md` §1-3 for the full argument with
`path:line` citations.
