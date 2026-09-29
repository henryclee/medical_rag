# Retrieval tuning

Side track, not a numbered phase (see `PLAN.md` "Next actions"). Phase 6 came
back underwhelming, and its own FINDINGS diagnosed why the result is hard to
trust: at `top_k=5`, the gold answer's *exact wording* showed up in the
retrieved excerpts for only 4/20 questions (8/40 rows). Three-quarters of
Phase 6's rows can't distinguish "the model ignored good context" from "the
context never had the answer." Before sinking more phases into prompts and
downstream modules, validate the retrieval step itself: build a harness that
uses a judge model to grade whether retrieved chunks are actually useful for
answering the question, then use that signal to compare retrieval levers.

Runs against Phase 5's pinned 20-question sample:
`1312 2391 3998 4002 5209 5949 6205 6264 6727 6753 6771 7159 7502 7553 8966
9293 9473 9572 9597 10064` (via `select_by_ids` in
`scripts/exploration/_common.py`).

## Judge design

One structured, batched call per question to `judge_model` (already defined
in `config/models.yaml`, served on the same `:8080` oMLX endpoint as
`model_a`/`model_b`) via
`LLMClient(config.models["judge_model"]).agenerate_structured()`. The judge
sees the question, its four options, and every candidate chunk (title +
content) — **never the correct answer**. For each chunk it reports:

```python
class ChunkJudgment(BaseModel):
    chunk_id: str
    relevance: Literal["relevant", "partial", "irrelevant"]
    supports_options: list[str]   # subset of {"A", "B", "C", "D"}
    reason: str

class JudgeVerdict(BaseModel):
    judgments: list[ChunkJudgment]
```

Gold-option recall is then computed *programmatically*, by checking whether
the gold letter is in a chunk's `supports_options` — the judge is never told
which option is correct, so the label is reusable across metrics and can't
be gamed by an answer leaking into the grading prompt.

Prompt and schema live in the harness script, not in
`src/medical_rag/generation/prompt.py` — this is exploratory grading of
retrieval candidates, not the production verification prompt (Phase 8's
`Verifier`), and it should not shape the frozen interfaces until a lever
proves itself.

## Harness architecture — judge each chunk once, reuse everywhere

For each of the 20 questions:

1. Generate every method's candidate chunk_id list (see Levers, below).
2. Take the **union** of chunk_ids surfaced by any method for that question.
3. Issue **one** batched judge call over that union.
4. Cache verdicts to `judged_chunks.jsonl`, keyed by `(question_id,
   chunk_id)`, following `_common.py`'s `RunWriter` / `completed_keys()`
   resume convention.

Every lever's metrics are then computed by looking up cached verdicts —
comparing N methods costs the same ~20 judge calls as comparing one, and a
new lever only costs judge calls for chunk_ids not already seen.

## Metrics

Computed per method at k ∈ {5, 10, 20}:

- **`semantic_recall@k`** — fraction of the 20 questions where the gold
  option letter appears in some chunk's `supports_options` within the top-k.
- **`relevant_fraction@k`** — mean fraction of the top-k chunks judged
  `relevant` or `partial`.
- **`mean_first_relevant_rank`** — average rank of the first `relevant`
  chunk (questions with none count as beyond k).

## Levers to compare

All against the same 20 questions and the same judge cache:

1. **Dense only, no rerank** — `Retriever.retrieve()`, current
   `top_k_retrieve=20`.
2. **Dense + cross-encoder rerank** — current production config
   (`Retriever.rerank()`, `top_k_rerank=5`).
3. **BM25** — new lightweight lexical index (below).
4. **Hybrid (RRF)** — Reciprocal Rank Fusion of the dense and BM25 rank
   lists (`rrf_score = Σ 1/(60 + rank)` summed over every list a chunk
   appears in), top 20 by fused score; measured with and without
   cross-encoder rerank applied to the fused set.
5. **Reformulation-as-retrieval** — rewrite the query with the already
   implemented `build_reformulation_prompt()`
   (`src/medical_rag/generation/prompt.py`) via
   `LLMClient(config.models["model_a"])`, then `Retriever.retrieve()` with
   the rewritten query. Compares recall against the original-query dense
   run and reuses existing production code — this answers PLAN.md's open
   question 15 before Phase 7 builds the full `Reformulator` module.

## New BM25 index

No lexical index exists in the repo yet. Added self-contained under this
directory — not `src/medical_rag/retrieval/` — since it's unproven; it only
gets promoted into production code if a lever built on it wins at this
tuning stage.

- `bm25_index.py` — pulls `(chunk_id, title, content)` for all 380,454 rows
  from the existing LanceDB table (`load_index()` + `.to_pandas()`),
  tokenizes with a simple lowercase regex splitter, builds
  `rank_bm25.BM25Okapi`, and pickles the chunk_id list + BM25 object to
  `data/index/statpearls_bm25.pkl` (gitignored, regenerable — same
  convention as the vector index). One-time build, cached; rebuilds only on
  an explicit `--rebuild-bm25` flag.
- `hybrid.py` — the RRF fusion function over two ranked chunk_id lists.
- New dependency: `rank_bm25` (pure Python), exact-pinned in
  `pyproject.toml` like the project's other dependencies.

## Harness script

`judge_harness.py`, following `scripts/exploration/`'s established CLI
conventions (`--dry-run`, `--resume`, `--note KEY=VALUE`), reusing
`RunWriter`, `error_row`/`result_row`, and `EndpointCircuitBreaker` from
`scripts/exploration/_common.py` for the judge calls.

Outputs:

- `outputs/exploration/retrieval_tuning/<UTC-stamp>/` (gitignored,
  regenerable): `judged_chunks.jsonl`, per-method candidate lists,
  `context.md`.
- `experiments/retrieval_tuning/FINDINGS.md` (tracked, written by the
  harness run — not by hand): the recall@k / relevant_fraction@k table per
  method, a few concrete chunk-judgment examples (a `relevant` and an
  `irrelevant` verdict each, to sanity-check the judge's calibration), and
  the resulting recommendation for what the Phase 6 rerun should use.

## Open questions this side-track should record, not silently resolve

- Is `judge_model` (`Qwen3.8-Flash-Next`) actually more capable than
  `model_a`/`model_b` for this grading task, or just differently sized?
  Worth a quick spot-check against a handful of hand-labeled chunks before
  trusting its verdicts at scale.
- The BM25 tokenizer is a simple splitter, not a real search engine's
  stemmed/stopword-aware tokenizer — good enough to compare against dense
  retrieval directionally, not a production-grade lexical index.
- All three models (`model_a`, `model_b`, `judge_model`) share one `:8080`
  oMLX process — judge calls run sequentially with the same
  `EndpointCircuitBreaker` discipline as Phase 6, not assumed free
  concurrency.

## After this

Rerun Phase 6 end-to-end against the tuned retriever (the original Phase 6
run stays on disk as the dense-only baseline), then Phase 7 compares
reformulation against that rerun, opening from this experiment's
`FINDINGS.md` rather than a fresh recall recount.
