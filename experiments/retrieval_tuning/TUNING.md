# Retrieval tuning — the harness's method doc

This directory started as a side track and became the project (ADR-0007): the
harness it built is now `src/medical_rag/eval/`, the plan it fed is the R-track in
[`../../PLAN.md`](../../PLAN.md), and what is left here is prose — this file and
[`FINDINGS.md`](./FINDINGS.md), which is frozen evidence and is not rewritten.

Why it exists at all: Phase 6 came back underwhelming, and its own FINDINGS
diagnosed why the result is hard to trust — at `top_k=5`, the gold answer's *exact
wording* showed up in the retrieved excerpts for only 4/20 questions (8/40 rows).
Three-quarters of Phase 6's rows can't distinguish "the model ignored good context"
from "the context never had the answer." So validate the retrieval step itself
first, with a judge model grading whether retrieved chunks are actually useful for
answering the question, and use that signal to compare levers.

Runs against Phase 5's pinned 20-question sample:
`1312 2391 3998 4002 5209 5949 6205 6264 6727 6753 6771 7159 7502 7553 8966
9293 9473 9572 9597 10064` (via `select_by_ids` in `medical_rag.eval.runlog`).

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

The rubric and schema live in `src/medical_rag/eval/rubric.py` — one home, not one
per caller. They began inside the harness and `inspect_retrieval.py` grew a second
copy, which is exactly how a cached verdict and the sha stamped beside it come to
describe different wording. The harness now imports the rubric, stamps its sha, and
words no prompt of its own. It stays separate from `generation/prompt.py` because
grading *retrieval candidates* blind to the gold letter is a different contract from
the frozen answer/verification prompts — and R2's rubric-variant experiment needs to
hold two versions of this wording side by side without touching generation.

## Harness architecture — judge each chunk once, reuse everywhere

For each of the 20 questions:

1. Generate every method's candidate chunk_id list (see Levers, below).
2. Take the **union** of chunk_ids surfaced by any method for that question.
3. Issue **one** batched judge call over that union.
4. Cache verdicts to the shared
   `outputs/exploration/retrieval_tuning/judge_cache/judged_chunks.jsonl`,
   keyed by `(question_id, chunk_id, judge_prompt_sha)`. The sha is part of
   the key on purpose: a verdict is a fact about a chunk *under one rubric*,
   so editing `eval/rubric.py` invalidates every cached verdict instead of
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
   `LLMClient(config.models["model_a"])`, then retrieve on the rewritten
   query (`eval/rewrite.py`). It is the only lever here that spends
   completions, and its one honest result so far is +0/+0/+5pp with the
   rewrite falling back on 18/20 questions — so it sits in `PLAN.md`'s
   backlog rather than in a phase, and no `__reform` column in the first grid
   is evidence about it either way.

## The BM25 index

There was no lexical index when this was written; there is now, and it was
promoted into the package in R1 rather than staying here — the first grid made it
load-bearing, and the lexical leg's 0/0/10 made it the thing R4 has to fix. Two
facts about the artifact worth knowing before touching it:

- It is a 445 MB pickle of tokenized corpus, so it outlives the code that wrote it.
  Pickles embed a class's module path, which means the on-disk index still names
  `bm25_index.BM25Index`, the pre-move module; `lexical.py` resolves that legacy
  path and logs a warning instead of demanding a rebuild. R4 rebuilds anyway (it
  changes the tokenizer), which retires the shim.
- The pickle carries no corpus or tokenizer version. That is a real gap, not
  cosmetic: nothing on disk says which tokenizer produced it. R4 writes it.

- `retrieval/lexical.py` — pulls `(chunk_id, title, content)` for all 380,454 rows
  from the existing LanceDB table (`load_index()` + `.to_pandas()`),
  tokenizes with a simple lowercase regex splitter, builds
  `rank_bm25.BM25Okapi`, and pickles the chunk_id list + BM25 object to
  `data/index/statpearls_bm25.pkl` (gitignored, regenerable — same
  convention as the vector index). One-time build, cached; rebuilds only on
  an explicit `--rebuild-bm25` flag.
- `retrieval/fusion.py` — the RRF fusion function over two ranked chunk lists.
- New dependency: `rank_bm25` (pure Python), exact-pinned in
  `pyproject.toml` like the project's other dependencies.

## Harness script

`src/medical_rag/eval/harness.py`, run as `scripts/run_retrieval_grid.py` (the
script is a thin wrapper — the package module has the same CLI). It keeps
`scripts/exploration/`'s conventions (`--dry-run`, `--resume`, `--note KEY=VALUE`)
and reuses `RunWriter`, `error_row`/`result_row` and `EndpointCircuitBreaker` from
`eval/runlog.py` for the judge calls.

Outputs:

- `outputs/exploration/retrieval_tuning/<UTC-stamp>/` (gitignored,
  regenerable): `results.jsonl` (one row per question: the 10 candidate
  lists, the judge's verdicts for their union, the reformulation record
  including its `raw_response` and any parse error, `new_judgments`,
  queries, provenance), `chunks.jsonl` (the chunk text those ids point at —
  see `eval/store.py`, which is what lets a run be re-read without the
  380k-row index), and `context.md`. It also creates `chains/`, and leaves it
  empty: this harness never calls `RunWriter.write_chain`, because the exchange
  worth keeping is already on the row (`reformulation.raw_response`, the parse
  error) or in the shared judge cache (verdict + reason). Both run dirs on disk
  confirm it — `chains/` with zero files. Do not cite `chains/` as evidence for
  a retrieval-tuning run; cite the row. Per-question markdown dumps are
  `scripts/inspect_retrieval.py`'s job, not the harness's — the row already carries the
  question, options, and every query variant used.
- `outputs/exploration/retrieval_tuning/judge_cache/judged_chunks.jsonl`
  (gitignored, append-only, shared across runs; `.venv/bin/python -m
  medical_rag.eval.judge --stats` prints how much of it is still usable under the
  current rubric, and how much a rubric edit would invalidate).
- `experiments/retrieval_tuning/FINDINGS.md` (tracked, written by the
  harness run — not by hand, and re-writable from stored rows with
  `--rerender <run>`, which is free): the recall@k / relevant_fraction@k
  table per grid cell, the per-strategy Δ table, the reformulation
  integrity block, a few concrete chunk-judgment examples (a `relevant` and
  an `irrelevant` verdict each, to sanity-check the judge's calibration),
  and the recommendation the run produced.

## Reading a run back (`scripts/inspect_retrieval.py`)

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

## Serving it (`--serve`)

The reading surface and the interactive surface were in different windows: the
knobs lived in a terminal REPL that truncates chunk bodies to 110 characters (its
own renderer says "the HTML is where you read prose"), and the prose lived in a
`report.html` you reloaded by hand. Outside the REPL, every guess re-ran the script
and re-paid its startup before testing anything. Measured 2026-10-05 (MPS, warm
page cache): the loads are **~4 s** — dataset 0.9 s, index + embedder +
cross-encoder 1.3 s, the 445 MB BM25 pickle 1.7 s — and one five-cell `__orig` grid
at k=20 is **~10-12 s** (median 10.6 s, 5.5-19.7 s over n=9). So the load is the
smaller half, and it is the half that is pure repetition: `--serve` pays it once per
session instead of once per guess. (Two earlier claims here — "~40 s load", and the
`Inspector` docstring's "~2 s grid" — had the ratio inverted; the docstring now
carries the measured split, and a cold page cache is the only thing that makes the
pickle look expensive.)

`--serve` keeps that loaded state behind a localhost form:

```bash
# the full interactive grid (loads models + BM25, no endpoint until you click Judge)
.venv/bin/python scripts/inspect_retrieval.py --serve

# browse an already-judged run's verdicts with nothing loaded at all
.venv/bin/python scripts/inspect_retrieval.py --serve --no-retrieve \
    --from-run outputs/exploration/retrieval_tuning/grid_smoke_check

.venv/bin/python scripts/inspect_retrieval.py --serve --dry-run  # route table
```

`eval/lab.py` builds no second renderer: the page body is `report.render_main`, the
same function the static `report.html` is built from, and every route calls
`aretrieve_grid` / `ensure_verdicts` / `build_cells`. `--serve --dry-run` prints the
route table; `--port 0` binds a free port so two hypotheses can be compared in two
browsers, each with its own loaded index.

Constraints worth knowing before changing it:

- **Judging is a button, never a page load.** `Retrieve` and the knob fields make
  zero completions and read verdicts from `judge_cache/`; only **Judge unseen (N)**
  spends endpoint time on the shared `:8080`, and what it
  spends lands in the append-only cache, so a question costs once, ever. Re-grading
  re-renders the grid already in memory rather than re-inspecting, because
  re-inspecting re-runs the reformulator -- a paid completion to re-derive a query
  we already have, and a *different* string would silently swap the column being
  read.
- **One asyncio loop on its own thread**, because `LLMClient`'s `AsyncOpenAI` binds
  to whichever loop it first runs on; handler threads post coroutines to it and
  block. An `asyncio.Lock` serialises retrieval and judging -- the cross-encoder is
  MPS work and all three models share one oMLX process (PLAN.md, ADR-0005), so two
  tabs firing at once buys contention and a misread, not speed.
- **A rejected knob changes nothing.** `rerank_k` above `top_k` passes both bounds
  and is still nonsense, so validation is a full pass before any assignment; a
  half-applied pair would leave the running server retrieving with a combination it
  just refused.
- **One view per question per session.** `Inspector.inspect()` appends (right for
  the REPL); a page you can hit Refresh on must replace, because
  `render.aggregate_views` averages over the view list and five Retrieve clicks
  would report `n=5` for one question.
- Localhost, no auth. It renders full corpus text and can spend endpoint time; do
  not repoint it at `0.0.0.0`.

### The corpus ceiling probe

The box labelled *corpus ceiling probe* is the one thing here that answers a
question the judge harness cannot: the harness only grades chunks it already
retrieved, so it cannot separate **"the supporting chunk exists and retrieval
ranked it #60"** from **"StatPearls never contained it."** `dense__orig` sits at
40/55/65 recall@5/10/20, so roughly a third of the pinned 20 is in one of those two
states and `FINDINGS.md` is silent on which -- and only the first is winnable by
reformulation, k, or a reranker.

It runs the gold option's own wording two ways (`retrieval/ceiling.py`), spending no
completions: a literal scan over all 380,454 chunk bodies already in memory in the
BM25 pickle, and `retrieval.strategy.retrieve_base()` with that wording as the query. If a
query that *contains* the answer cannot surface a chunk that *contains* it, the
bottleneck is the index.

**It is a ceiling, not a lever.** It uses the gold answer, so it must never be
cited as a strategy's score -- every result carries that label into the rendered
panel and `probes.jsonl`, and `probes.jsonl` is never merged into `FINDINGS.md`.
This directory has already published a wrong number off an unlabelled column
(`reform_dense`, +5pp recall@20 from a rewrite that fell back on 18/20 questions).

And the bound is one-directional: **0 matches means "not stated in these words,"
not "not in StatPearls."** Phase 6 found 32/40 rows chose wording appearing nowhere
in their excerpts -- these models answer from understanding, so a paraphrased answer
can be fully supported by chunks a literal scan cannot see. Treat a miss as a hint
about wording, never as evidence of absence.

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
vignette. `scripts/inspect_retrieval.py --sample 3` shows this in a few grid
passes; fixing it means a better tokenizer, field weighting, or excluding
narrative boilerplate from the lexical field — which is **R4**, and R4's tracked
metric is the count of distinct chunks filling the top-5 (59/100 here), because
that number moves even when recall barely does.

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

## What this directory left open, and where it lives now

- **Is `judge_model` (`Qwen3.8-Flash-Next`) actually good at this grading task?**
  Written here as "worth a quick spot-check before trusting its verdicts at
  scale." That is no longer a suggestion: ADR-0008 made these verdicts the oracle,
  so R2 validates the judge — stratified hand-check with the boilerplate chunks as
  known-bad controls, a silent-drop audit on the batched calls, one rubric variant
  under its own sha — *before* any R4/R5 number gets quoted. PLAN.md open
  question 1.
- **The BM25 tokenizer is a naive splitter** — no stemming, no stopword removal,
  no field weighting. That stopped being a caveat and became R4's job.
- **All three models share one `:8080` oMLX process.** Judge calls stay sequential
  under the same `EndpointCircuitBreaker` discipline as Phase 6; concurrency is not
  free and R2's cache build inherits that. Constraint, not open question.

## What comes next

The Phase 6 rerun and Phase 7 this file used to point at are cut (ADR-0007). The
order is now R2 → R3 → R4 → R5 → R6, in `PLAN.md`. This file stays the method doc
for the harness those phases run; their numbers land in
`experiments/retrieval/R*/FINDINGS.md`, and `FINDINGS.md` in this directory stays
frozen — it is the evidence the direction change was made on.
