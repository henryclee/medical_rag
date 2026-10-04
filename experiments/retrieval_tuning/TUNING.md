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
4. Cache verdicts to the shared
   `outputs/exploration/retrieval_tuning/judge_cache/judged_chunks.jsonl`,
   keyed by `(question_id, chunk_id, judge_prompt_sha)`. The sha is part of
   the key on purpose: a verdict is a fact about a chunk *under one rubric*,
   so editing `judge_prompt.py` invalidates every cached verdict instead of
   silently mixing two generations of grading into one table.

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
  regenerable): `results.jsonl` (one row per question: the 10 candidate
  lists, the judge's verdicts for their union, the reformulation record
  including its `raw_response` and any parse error, `new_judgments`,
  queries, provenance), `chunks.jsonl` (the chunk text those ids point at —
  see `chunk_store.py`, which is what lets a run be re-read without the
  380k-row index), and `context.md`. It also creates `chains/`, and leaves it
  empty: this harness never calls `RunWriter.write_chain`, because the exchange
  worth keeping is already on the row (`reformulation.raw_response`, the parse
  error) or in the shared judge cache (verdict + reason). Both run dirs on disk
  confirm it — `chains/` with zero files. Do not cite `chains/` as evidence for
  a retrieval-tuning run; cite the row. Per-question markdown dumps are
  `inspect_retrieval.py`'s job, not the harness's — the row already carries the
  question, options, and every query variant used.
- `outputs/exploration/retrieval_tuning/judge_cache/judged_chunks.jsonl`
  (gitignored, append-only, shared across runs; `judge_cache.py --stats`
  prints how much of it is still usable under the current rubric).
- `experiments/retrieval_tuning/FINDINGS.md` (tracked, written by the
  harness run — not by hand, and re-writable from stored rows with
  `--rerender <run>`, which is free): the recall@k / relevant_fraction@k
  table per grid cell, the per-strategy Δ table, the reformulation
  integrity block, a few concrete chunk-judgment examples (a `relevant` and
  an `irrelevant` verdict each, to sanity-check the judge's calibration),
  and the resulting recommendation for what the Phase 6 rerun should use.

## Reading a run back (`inspect_retrieval.py`)

A number in `FINDINGS.md` is not inspectable evidence unless the chunks
behind it are. This script renders, per question, a 10-cell grid —
`dense`, `dense_rerank`, `bm25`, `hybrid`, `hybrid_rerank`, each run on the
original question (`__orig`) and on the reformulated information need
(`__reform`) — with every chunk's full text, its judged relevance, the
options it was scored as supporting, gold hits inside k, and the Δ against
the same strategy's other query column.

- `--from-run <run>` is offline: it reads `results.jsonl` + `chunks.jsonl` +
  the verdict cache and issues zero calls. Rows written before the grid exist
  have their legacy ids (`reform_dense`) aliased onto grid cells and marked
  as such, and a row whose reformulation fell back prints the raw response and
  says the `__reform` column is a copy of `__orig`.
- `--question-id` / `--sample` retrieve live (and can grade with the cached
  rubric via `--judge`), then write the same artifact layout so the run is
  re-openable later.
- `--print-chunk <method>:<rank>` is the one-question-at-the-terminal path;
  `--out-name` keeps each inspection in its own directory instead of
  clobbering the last one.
- Artifact dir: `outputs/exploration/retrieval_tuning/<out-name>/` —
  `report.html`, `results.jsonl`, `chunks.jsonl`, `questions/q<id>.md`,
  `context.md`. The first `--from-run` run on a pre-sidecar run backfills
  `chunks.jsonl` from the index, so the index only has to be paid for once.

## What the first run (2026-09-29) actually showed

`FINDINGS.md` as committed reported `reform_dense` as a peer method and read
its +5pp recall@20 as a reformulation result. It was not one: the
reformulation fell back to the raw question on 18/20 rows, so `__reform` was a
byte-identical copy of `__orig` and the Δ was arithmetic on a lever that never
moved. The cause was a strict `json.loads(content)` on a completion the model
was free to wrap in prose; `parse_information_need()` in `reformulate.py` now
accepts fenced or padded output, demands a *non-empty* string (so
`{"information_need": ""}` stays a fallback instead of counting as a
"rewrite"), and keeps `raw_response` + `error` on the row so a future
fallback is diagnosable. It could not be this time: the 2026-09-29 rows store
only the boolean, so *why* those 18 failed is gone. That is the real lesson —
a fell-back query is an experiment result, not a detail, and it has to be
stored and surfaced. Re-rendered with the integrity block, the same run reads
`dense | +0pp | +0pp | +5pp | identical 18/20` and names all 18 questions,
which is what the run is actually worth.

BM25's `0% recall@5 / 1% rel_frac@5` on the same run is real — its candidate
lists are fully covered by verdicts (400 slots, 4 unjudged) — but it is not
"lexical retrieval is weak". Its top-5 is *degenerate*: across the 20
questions, only 59 distinct chunks fill the 100 top-5 slots, and one chunk
(`article-128082_187`, a child-abuse emergency-department narrative) is rank-1
for 6 of the 20 questions and inside the top-5 for 9, on questions ranging
from dermatology to umbilical embryology. A handful of boilerplate case-
writeup chunks win nearly every query. Hypothesis, untested: the BM25 corpus
is everything in the index including narrative case reports, and with
`b=0.75` length normalization over a mean-chunk-length baseline, a long chunk
that repeats general clinical vocabulary outranks a short precise one on any
vignette. `inspect_retrieval.py --sample 3` shows this in ten seconds; fixing
it means either a better tokenizer/field weighting or excluding narrative
boilerplate from the lexical field, which is Phase 7's problem, not this one.

The degeneracy is reproducible, not a one-run artifact. On the 2026-09-29
probe described below, whose first question is the same q1312, three of the
bm25 top-5 are the chunks that dominated the 20-question run: `article-17453_34`
(an ASA-Physical-Status example table) at rank 1, `article-128082_187` (the
Pennsylvania child-abuse case scenario) at rank 2, `article-40888_0` (a
polyarthralgias case study) at rank 5. Its sibling `article-128082_185` turns up
in the bm25 list of the other probe question too, a pill-esophagitis vignette.
Case-report boilerplate that wins on any clinical prose is a property of the
lexical index, not of the questions.

Re-rendering the run (`--rerender outputs/exploration/retrieval_tuning/20260929T120215Z`)
reproduces the committed table cell for cell — `dense` 40/55/65 became
`dense__orig` 40/55/65, `reform_dense` became `dense__reform` 40/55/70, bm25
stayed 0/0/10 — so the legacy-id aliasing moved no number; what it added is the
integrity block (`dense | +0pp | +0pp | +5pp | identical 18/20`), the `n=0`
rows for the four strategies that never retrieved a rewrite, and the alias list
itself. Every rate is still re-derivable from the 20 rows in `results.jsonl`
(`candidates` + `judgments`), and 18/20 of those rows carry
`reformulation_fallback: true` with no recorded cause — which is the gap
`ReformulationError.raw_response` exists to close on the next run.

The grid path has since been exercised end to end by a 2-question probe
(`outputs/exploration/retrieval_tuning/20260929T191914Z`, tagged
`probe=grid-smoke`): all ten cells filled, 0/2 reformulation fallbacks, the
chunk sidecar written, and `inspect_retrieval.py --from-run` on it replayed in
~6 s with no index load and no endpoint. That is evidence the harness measures
what it claims to, not evidence about any strategy — n=2. Note the harness
renders the tracked `FINDINGS.md` by default, so a probe writes its n=2 table
over the real one; the file was restored afterwards with
`--rerender outputs/exploration/retrieval_tuning/20260929T120215Z`, and no
probe should ever be the tracked findings.

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
