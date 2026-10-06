# Medical RAG — Retrieval Optimization Plan

A reproducible retrieval-optimization study: MedQA-USMLE questions
(`GBaker/MedQA-USMLE-4-options`, 1,273 test / `train` dev split) answered from
StatPearls — 380,454 chunks from 9,652 articles, embedded once with
`BAAI/bge-small-en-v1.5` under BGE's asymmetric query/passage convention, stored
in an exact-search LanceDB table (no ANN at this size), reranked with
`cross-encoder/ms-marco-MiniLM-L-6-v2`. Retrieval quality is measured **offline**:
`judge_model` grades whether each retrieved chunk supports the gold option, blind
to the letter, those verdicts are cached and versioned by rubric hash
([ADR-0008](./docs/adr/0008-judge-cache-as-ground-truth.md)), and
`scripts/eval_retrieval.py` scores a strategy against them with **zero endpoint
calls**. The LLM is a remote OpenAI-compatible endpoint on `:8080`; nothing here
runs a model locally, and the judge is the only model the mainline pays. Contracts
live in [`interfaces.md`](./interfaces.md).

**Why this is the project now.** Three facts, each measured rather than argued
(full version with `path:line` citations in [`refactor_plan.md`](./refactor_plan.md) §1-3):

- **The old study would have measured the wrong thing.** Phase 6 found the gold
  option's *wording* reaching the excerpts for only 4/20 questions at k=5, so
  three quarters of its rows could not tell "the model ignored the context" from
  "the context never had it" ([phase6 FINDINGS](./experiments/phase6/FINDINGS.md)).
  Judged semantically instead, the shipped funnel's recall is 45%
  ([retrieval_tuning FINDINGS](./experiments/retrieval_tuning/FINDINGS.md)).
  A generator-side result on that context is a retrieval measurement wearing a
  model label.
- **The cheapest lever was the un-examined one.** `dense_rerank`'s 45/45/45 across
  k=5/10/20 is not a plateau — the funnel hands the reranker 5 candidates and asks
  for 5, so it *cannot* rise. And the lexical leg is not weak, it is degenerate: 59
  distinct chunks fill 100 top-5 slots, one chunk is rank-1 for 6/20 questions.
- **The asymmetry is cost.** A retrieval hypothesis now costs index work plus
  arithmetic against cached verdicts; a generation hypothesis cost ~7 h of shared
  endpoint time per condition per arm. Fixing the input got both cheaper and more
  attributable. [ADR-0007](./docs/adr/0007-retrieval-first-direction-change.md)
  records cutting Phases 7-13 rather than deferring them.

**Where things stand.** Phases 1-6 are complete and frozen (their `FINDINGS.md`
files are evidence and are not rewritten). The grid has been run once, on 20
questions. **R1 is done**: the harness, metrics, judge cache, BM25, fusion and the
ceiling probe are now package code (`src/medical_rag/eval/`,
`src/medical_rag/retrieval/`), the deleted stubs are gone, and grid and shipped
path call one `retrieve_base()`. The next phase is **R2**, which is the one that
costs endpoint time — nothing downstream is believed until the oracle is validated.

## Constraints in force

Each of these has already cost a bad measurement, or will.

- **The harness only measures the union of strategies it ran.** A new lever's
  chunks are unjudged until that lever is in the grid, and an unjudged slot scores
  as *not supporting* — so a brand-new strategy's number is a floor. Add candidates
  to the grid; do not "measure" a new lever against yesterday's cache.
  `eval_retrieval.py` prints coverage next to recall and `--require-coverage` makes
  the floor fail loudly.
- **Never compare a `*_rerank` cell's recall@10/20 against an unreranked cell's.**
  The funnel caps it at `top_k_rerank` (see the 45/45/45 above).
- **Corpus and chunking are frozen for this track.** Verdicts are keyed
  `(question_id, chunk_id, judge_prompt_sha)`; changing chunk geometry changes
  every `chunk_id` and orphans the entire cache — a bill of roughly the whole R2
  judge spend. [ADR-0009](./docs/adr/0009-corpus-and-chunking-frozen.md).
- **Dev split only, and the pinned sample only grows by containment.** The 20
  questions Phases 5-6 measured: `1312 2391 3998 4002 5209 5949 6205 6264 6727
  6753 6771 7159 7502 7553 8966 9293 9473 9572 9597 10064`. R2's ~100 must
  contain them as a strict subset and a preflight must assert it — do **not** just
  draw `size=100, seed=1`, which will not.
- **The oracle is tracked and versioned.** `judge_prompt_sha` stays part of the
  cache key; R2 commits a verdict snapshot under `experiments/retrieval/evalset/`
  because a measurement nobody can read is not evidence.
- **Judge discipline is the old endpoint discipline.** `judge_model` shares `:8080`
  with the frozen arms: keep `EndpointCircuitBreaker` and resume, never re-run a
  batch to change a number (recompute from artifacts and say so), and record the
  judge's server-side sampling the way `model_b`'s was recorded — under ADR-0008
  those settings shape ground truth, not just a completion.
- **No number is a result unless `scripts/eval_retrieval.py` re-derives it.** That
  is `audit_run.py`'s successor role. `--rerender <run>` is the offline half of the
  same rule: a refactor that changes a cell is a bug, and this catches it.

## Current numbers to beat

Run `20260929T120215Z`, n=20, semantic recall@5/10/20
([source](./experiments/retrieval_tuning/FINDINGS.md)):

| strategy | @5 | @10 | @20 | read it as |
| --- | --- | --- | --- | --- |
| `dense` | 40 | 55 | 65 | the only column that climbs |
| `dense_rerank` | 45 | 45 | 45 | **capped-on-5**, not a plateau |
| `hybrid` | 30 | 40 | 55 | fusion is losing to dense alone |
| `hybrid_rerank` | 25 | 25 | 25 | capped, and on a worse list |
| `bm25` | 0 | 0 | 10 | degenerate index, see R4 |

`*_reform` columns from that run are **not** reformulation results: the rewrite
fell back to the raw question on 18/20 questions, so the column copies `__orig`.
n=20 means a 5 pp difference is one question — which is why R2 comes first.

## Phases

**R1 — Promote the harness.** *No new science.* Move the tuned tooling into the
package, delete the cut scaffolding, make the grid and the shipped path share one
code path, and introduce `retrieval.strategy.RetrievalStrategy` so a strategy means
the same thing in a findings table as in a call.
**Done.** Gate: `run_retrieval_grid.py --rerender outputs/exploration/retrieval_tuning/20260929T120215Z`
rebuilds the committed `FINDINGS.md` **byte-identical** (proving the move carried no
number), `.venv/bin/pytest` is green offline, and `sys.path.insert` is gone from
`src/`, `scripts/` and `tests/`.

**R2 — Trust the oracle, then enlarge it.** The judge is ground truth per ADR-0008
and unvalidated per `TUNING.md`; that inversion has to close before anything is
believed. Stratified hand-check of ~30 verdicts *including the known-bad boilerplate
chunks as controls*, reported per class rather than as one agreement percentage;
silent-drop audit proving `judgments == candidates` per batch at full prose length;
one rubric variant stored under its own sha to measure verdict stability; a test that
gold answer text cannot reach the prompt. Then grow the pinned sample 20 → ~100 by
containment and pay the judge calls once, resumable and circuit-breaker guarded.
*Deliverable:* `experiments/retrieval/R2/FINDINGS.md` + the tracked oracle snapshot.
**This phase spends real endpoint time — needs a go-ahead.**

**R3 — Corpus ceiling.** `retrieval/ceiling.py` already exists and spends nothing:
it pushes the gold option's own wording through a literal scan of all 380,454 chunk
bodies and through `retrieve_base()`. Run it over the enlarged sample to separate
*"exists, ranked #60"* from *"StatPearls never said it"*, and state the attainable
ceiling for R4/R5. It uses the gold answer, so it is an oracle that stays out of
every strategy table, and its bound is one-directional: 0 matches means *not in
these words*, never *not in StatPearls*. Free, and it decides whether R4/R5 are
chasing 25 points or 65.

**R4 — Fix the lexical index.** Tokenizer (medical terms, doses, numbers,
stopwords, stemming), field weighting (title / contents / body), and excluding
narrative case-report boilerplate. Give `retrieval/lexical.py` a real CLI — today it
is reachable only as someone else's `--rebuild-bm25` — and write corpus + tokenizer
version into the pickle so the 467 MB artifact stops being anonymous. *New tracked
metric:* distinct chunks filling the top-5 (59/100 today), because fixing degeneracy
has to be measurable even where recall barely moves.

**R5 — Fusion and funnel shape.** RRF `k`, per-list cutoffs, dense/lexical
weighting, rerank pool size, and where the funnel cuts — guided by the cap that made
`dense_rerank` flat. Paired deltas with bootstrap CIs, which are offline and free, so
a delta at n=100 is defensible rather than eyeballed.

**R6 — Freeze and deliver.** `scripts/eval_retrieval.py` is the single offline
command; the frozen `retrieval:` block cites the R4/R5 measurement behind each value;
the results doc is generated by the command rather than typed; index and BM25 pickle
provenance recorded. `README.md` finally describes what the repo does.

*Sequencing is not a suggestion.* Do not start R4 with an unvalidated judge, and do
not tune fusion before R4 stops returning the same five boilerplate chunks for every
question — tuning a fusion over a degenerate input list measures the degeneracy.

## Backlog, unscheduled

Deliberately not phases, so they cannot quietly become one:

- **Chunking / corpus shape.** The only lever that can beat R3's ceiling, and it
  costs the whole cache. See ADR-0009 before touching it; `data/chunking.py` waits
  here for it.
- **Reranker model, and `--rerank-with`.** A different cross-encoder is a bigger
  claim than an RRF `k`, and the `rerank_with` knob (which string the reranker sees)
  is a grid-only control that must never become a config field.
- **Query-side rewrite.** The only retrieval lever that costs completions, and its
  one honest result so far is +0/+0/+5 pp with 18/20 fallbacks. Code lives at
  `eval/rewrite.py`; `experiments/retrieval_tuning/TUNING.md` explains why it was
  not believed the first time.
- **The cut answer-accuracy study.** Phases 7-13, `model_a`/`model_b`, the runner,
  the statistics and the taxonomy — deleted, recoverable from git, with
  [ADR-0007](./docs/adr/0007-retrieval-first-direction-change.md) as the argument
  for the veto if it is ever revived.

## Open questions

Renumbered 1-6. The old numbering stops resolving on purpose: its load-bearing
referents (`interfaces.md:243`, `:417`, `conditions.yaml`, the stubs) were deleted
by R1. Mapping: old 4 → 6, old 5 → 2, old 7 → 5, old 16 → 4; old 3 and 15 went to
the backlog, old 6, 9, 12 and 14 were cut with the study they were about (their
"record every parameter that shaped the row" principle survives as the `context.md`
rule).

- **1 — How reliable is the judge, where does it fail, and does it fail equally
  across chunk classes?** R2 answers it. A single-model oracle errs systematically,
  not randomly: leniency about boilerplate inflates every strategy's lexical recall
  together and leaves the comparison looking clean. Decidable in R2.
- **2 — What is the significance test for a paired recall delta?** Per-question
  binary indicators, same questions both arms, so McNemar's exact test is the
  candidate; bootstrap CIs for the marginal. Decidable in R5, where the first delta
  worth defending is reported.
- **3 — What is the verdict-cache growth policy, and what selection bias does it
  encode?** The cache only contains chunks some strategy already retrieved, so it is
  blind to candidates no strategy has ever surfaced. Grow-on-demand keeps it cheap
  and keeps it biased; a random-pool sample per question would cost judge time to
  fix a bias nobody has yet sized.
- **4 — What does the endpoint actually apply to `judge_model`?** `LLMClient` sends
  only `max_completion_tokens`/`temperature`/`top_p`; per-model `top_k` and
  `repetition_penalty` live in `~/.omlx/model_settings.json`, outside git. For an
  answer that was one reproducibility bug (old question 16); for the oracle it is a
  validity bug. R2 records the values next to the snapshot; whether to send them
  deliberately stays open.
- **5 — Do we state the corpus-size drift?** 380,454 chunks vs MedRAG's documented
  301,202 snippets (NCBI's live archive grew since their snapshot). Decide whether
  `README.md` flags it as a caveat against comparing with published numbers. Cheap;
  should be yes, and R6 is when the README gets written anyway.
- **6 — Resume semantics for a ~100-question cache build.** Per-question commit is
  what `RunWriter` does today and it is enough, but the cache append is not
  transactional: a crash mid-line leaves a torn record that `load_cache` skips.
  Decide whether that is acceptable (cost: one re-asked batch) or whether the cache
  gets a rewrite-in-place compaction step.

## Decision record

Nine decisions that were argued out in earlier drafts live in
[`docs/adr/`](./docs/adr/README.md). The mapping below is what makes the old
question numbers in `interfaces.md` and the phase `FINDINGS.md` files still resolve.

| ADR | Decision | Was | Status |
| --- | --- | --- | --- |
| [0001](./docs/adr/0001-real-model-endpoints.md) | Real OpenAI-compatible endpoints (`:8080` oMLX), keys only ever env-var placeholders | old item 1 | Accepted |
| [0002](./docs/adr/0002-instruct-reasoning-model-pair.md) | Instruct/reasoning pair `model_a`/`model_b` | old item 2 | Partially superseded by 0007 |
| [0003](./docs/adr/0003-drop-seed-one-completion-per-question.md) | `seed` measured inert, so deleted; one completion per question | old item 10 | Accepted |
| [0004](./docs/adr/0004-thinking-mode-fixed-on.md) | Thinking mode fixed on, no toggle | old item 11 | Accepted |
| [0005](./docs/adr/0005-model-b-termination-endpoint-side.md) | `model_b` non-termination fixed endpoint-side | old item 13 | Accepted (recovery half unresolved) |
| [0006](./docs/adr/0006-token-ceilings-and-sampling-params.md) | Token ceilings and published sampling defaults | old item 8 | Scoped to frozen tooling by 0007 |
| [0007](./docs/adr/0007-retrieval-first-direction-change.md) | Retrieval-first; Phases 7-13 cut, not deferred | the direction change | Accepted |
| [0008](./docs/adr/0008-judge-cache-as-ground-truth.md) | The judge cache is the ground truth | `TUNING.md`'s open question | Accepted (R2 validates it) |
| [0009](./docs/adr/0009-corpus-and-chunking-frozen.md) | Corpus + chunking frozen; dead chunk keys deleted not wired | implicit until now | Accepted |

## Where the evidence lives

- [`experiments/retrieval_tuning/FINDINGS.md`](./experiments/retrieval_tuning/FINDINGS.md)
  — the one grid run, frozen. [`TUNING.md`](./experiments/retrieval_tuning/TUNING.md)
  is the harness's method doc.
- [`experiments/phase5/FINDINGS.md`](./experiments/phase5/FINDINGS.md) and
  [`experiments/phase6/FINDINGS.md`](./experiments/phase6/FINDINGS.md) — closed-book
  and raw-RAG baselines on the pinned 20, frozen. They are the *reason* for this
  direction, not results in it.
- `experiments/retrieval/R*/FINDINGS.md` — where R2-R6 land
  ([convention](./experiments/README.md)). Raw run dirs stay gitignored under
  `outputs/exploration/`.
- [`refactor_plan.md`](./refactor_plan.md) — the R1 refactor's own plan, kept as the
  audit trail for what moved and what was deleted.

## Related docs

- [file_layout.md](./file_layout.md) — the tree, module boundaries, the two import rules.
- [interfaces.md](./interfaces.md) — public contracts: signatures, invariants, stability.
- [environment.md](./environment.md) — runtimes, versions, env vars, setup.
- [docs/adr/README.md](./docs/adr/README.md) — decision record index (ADR-0001…0009).
