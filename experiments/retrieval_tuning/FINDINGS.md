# Retrieval tuning -- judge harness FINDINGS

Run: `outputs/exploration/retrieval_tuning/20260929T120215Z` -- 20 question(s) judged.

`semantic_recall@k`: fraction of questions where the gold option letter is in some chunk's `supports_options` within the top-k. `relevant_fraction@k`: mean fraction of the top-k chunks judged `relevant`/`partial`. `mean_first_relevant_rank`: average rank of the first `relevant` chunk (blank = never, for the questions it covers).

| method | n | recall@5 | recall@10 | recall@20 | rel_frac@5 | rel_frac@10 | rel_frac@20 | mean_first_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense | 20 | 40% | 55% | 65% | 48% | 44% | 40% | 6.36 |
| dense_rerank | 20 | 45% | 45% | 45% | 40% | 40% | 40% | 2.8 |
| bm25 | 20 | 0% | 0% | 10% | 1% | 2% | 5% | 6.0 |
| hybrid | 20 | 30% | 40% | 55% | 28% | 25% | 24% | 10.31 |
| hybrid_rerank | 20 | 25% | 25% | 25% | 19% | 19% | 19% | 1.0 |
| reform_dense | 20 | 40% | 55% | 70% | 52% | 47% | 43% | 6.36 |

## Example judgments

- **relevant** -- q1312, chunk `article-17026_4`: relevance=relevant supports_options=['B'] -- "The excerpt describes abdominal angina (chronic mesenteric ischemia) which presents with severe postprandial pain, weight loss, and lacks peritoneal signs, strongly supporting CT angiography to confirm the diagnosis."
- **irrelevant** -- q1312, chunk `article-127545_111`: relevance=irrelevant supports_options=[] -- "The excerpt discusses child abuse and neglect, which is completely unrelated to the medical case of an older adult with abdominal pain and weight loss."

## Recommendation

Fill in by hand after reading the table above: which method the Phase 6 rerun should use, and whether reformulation-as-retrieval (`reform_dense`) moved recall enough to keep the reformulation condition alive for Phase 7.

