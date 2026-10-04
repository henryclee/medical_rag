# Retrieval tuning -- judge harness FINDINGS

Run: `outputs/exploration/retrieval_tuning/20260929T120215Z` -- 20 question(s) judged.

`semantic_recall@k`: fraction of questions where the gold option letter is in some chunk's `supports_options` within the top-k. `relevant_fraction@k`: mean fraction of the top-k chunks judged `relevant`/`partial`. `mean_first_relevant_rank`: average rank of the first `relevant` chunk (blank = never, for the questions it covers).

Methods are `<strategy>__<query variant>` (`strategies.METHOD_GRID`): the five strategies run on the question text (`__orig`) and on the reformulated information need (`__reform`). `inspect_retrieval.py --from-run <this run>` renders the chunks behind every number below.

| method | n | recall@5 | recall@10 | recall@20 | rel_frac@5 | rel_frac@10 | rel_frac@20 | mean_first_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense__orig | 20 | 40% | 55% | 65% | 48% | 44% | 40% | 6.36 |
| dense_rerank__orig | 20 | 45% | 45% | 45% | 40% | 40% | 40% | 2.8 |
| bm25__orig | 20 | 0% | 0% | 10% | 1% | 2% | 5% | 6.0 |
| hybrid__orig | 20 | 30% | 40% | 55% | 28% | 25% | 24% | 10.31 |
| hybrid_rerank__orig | 20 | 25% | 25% | 25% | 19% | 19% | 19% | 1.0 |
| dense__reform | 20 | 40% | 55% | 70% | 52% | 47% | 43% | 6.36 |
| dense_rerank__reform | 0 | - | - | - | - | - | - | - |
| bm25__reform | 0 | - | - | - | - | - | - | - |
| hybrid__reform | 0 | - | - | - | - | - | - | - |
| hybrid_rerank__reform | 0 | - | - | - | - | - | - | - |

## Reformulation, per strategy (Δ = `__reform` minus `__orig`)

| strategy | Δrecall@5 | Δrecall@10 | Δrecall@20 | Δmean_first_rank | identical columns |
| --- | --- | --- | --- | --- | --- |
| dense | +0pp | +0pp | +5pp | +0.00 | 18/20 |
| dense_rerank | - | - | - | - | - |
| bm25 | - | - | - | - | - |
| hybrid | - | - | - | - | - |
| hybrid_rerank | - | - | - | - | - |

## Stored method ids

This run predates the grid; its flat ids were aliased onto grid cells by `strategies.LEGACY_METHOD_ALIASES`: `bm25->bm25__orig`, `dense->dense__orig`, `dense_rerank->dense_rerank__orig`, `hybrid->hybrid__orig`, `hybrid_rerank->hybrid_rerank__orig`, `reform_dense->dense__reform`.

Cells with `n = 0` were never retrieved by that run -- it ran the original query through every strategy plus exactly one dense retrieval on the rewritten query. Their Δ row is blank, not zero: a lever that was not run did not fail to work.

## Reformulation integrity

A fell-back reformulation returns the question unchanged, so that row's `__reform` column is a copy of `__orig` and contributes a delta of exactly zero. Counting those as measurements is the bug this section exists to make impossible to miss.

- questions whose reformulation **fell back**: **18/20** (q1312, q2391, q3998, q4002, q5209, q5949, q6205, q6264, q6727, q6753, q6771, q7502, q7553, q8966, q9293, q9473, q9572, q10064)
  - no per-question cause was stored for any of these -- the run predates `reformulate.ReformulationError`, so *why* they fell back is unrecoverable
- read the Δ table with these rows excluded or re-run: each contributes a forced zero and pulls every Δ toward nothing.

- chunks newly judged by this run: not recorded -- this run predates the verdict cache, so every judgment in it was paid for inside the run and none of it was reusable afterwards

## Example judgments

- **relevant** -- q1312, chunk `article-17026_4`: relevance=relevant supports_options=['B'] -- "The excerpt describes abdominal angina (chronic mesenteric ischemia) which presents with severe postprandial pain, weight loss, and lacks peritoneal signs, strongly supporting CT angiography to confirm the diagnosis."
- **irrelevant** -- q1312, chunk `article-127545_111`: relevance=irrelevant supports_options=[] -- "The excerpt discusses child abuse and neglect, which is completely unrelated to the medical case of an older adult with abdominal pain and weight loss."

## Recommendation

Fill in by hand after reading the table above *and* running `inspect_retrieval.py --from-run <this run> --open`: which strategy the Phase 6 rerun should use, and whether reformulation moved recall on the strategies where the rewrite actually reached the embedder. A strategy whose `__reform` list is identical to `__orig` proves nothing in either direction -- check the integrity section above before believing any Δ in the table.

