# Retrieval-first refactor plan

**Status: proposal, not yet executed.** Nothing in this file has been applied. `PLAN.md`,
`config/`, `src/`, `scripts/` and `tests/` are all still in their pre-refactor state as of
2026-10-05 (`00cd725`).

This is the working document for executing the refactor. `PLAN.md` stays the current
operating plan until R1 replaces it; this file is the bridge — it holds the reasoning,
the verified facts, and the phase definitions that `PLAN.md` will absorb.

- **Read first:** [`PLAN.md`](./PLAN.md) (current plan),
  [`experiments/retrieval_tuning/TUNING.md`](./experiments/retrieval_tuning/TUNING.md)
  (harness architecture),
  [`experiments/retrieval_tuning/FINDINGS.md`](./experiments/retrieval_tuning/FINDINGS.md)
  (the measurement this whole thing hangs on),
  [`experiments/phase6/FINDINGS.md`](./experiments/phase6/FINDINGS.md) (the diagnosis).
- **Standing conventions carried forward** from the last `PLAN.md` refactor
  (`prompt.md`, which is itself now a stale orphan — see §6): `PLAN.md` stays short, links
  beat duplicated prose, `experiments/**/FINDINGS.md` is read-only, no state-changing git
  without review.

---

## 1. Why the direction changed

The project was designed as a factorial study: model × retrieval × reformulation ×
verification, measured on answer accuracy. Two measurements killed that design, and both
say the same thing — the downstream factors cannot be estimated while retrieval is the
dominant source of variance.

1. **Most of Phase 6's rows never tested "model reads context."** At `top_k=5` the gold
   option's wording reached the excerpts for 4/20 questions (8/40 rows). Three quarters of
   the run cannot distinguish *the model ignored the context* from *the context never had
   it* (`phase6/FINDINGS.md` §2.2).
2. **The arms moved in opposite directions**, so no pooled accuracy number exists to be a
   finding: `model_a` 14/20 → 10/20, `model_b` 5/20 → 7/20. Pooling adds two opposite
   effects and reports the remainder (`phase6/FINDINGS.md` §2).
3. **Where context did change an answer, it often argued its way wrong.** 3 of 4 `rescued`
   rows land on the right letter with reasoning that rejects it, and `q9572` was talked out
   of a correct answer by an excerpt ranked #2. No prompt or verifier fixes a retrieval
   problem, and every one of them costs a completion per question to test.

Retrieval is the cheapest factor to measure (the judge grades chunks; no answer completions)
and currently the weakest link (`bm25__orig` recall@5 = 0%). So retrieval optimization
becomes the mainline, and the answer-accuracy study stops being scheduled.

---

## 2. Confirmed scope decisions

| # | Decision | What it rules out |
| --- | --- | --- |
| 1 | **Cut** the generation-side study: Phases 7–8 and 10–13 leave `PLAN.md`, and the stub modules + conditions 5–10 get deleted. | No frozen 2×2×2 design, no 1,273-question confirmatory run, no `RESULTS.md`, no failure taxonomy. |
| 2 | Cut at **stub depth**: the 7 stub modules and 2 stub scripts go; `closed_book.py`, `raw_rag.py`, `probe_models.py`, `audit_run.py`, the answer prompt builders and `parse_answer` stay as frozen Phases 5–6 evidence tooling. | Phases 5–6 stay re-runnable. `raw_rag.py` must remain importable (see §3, fact 1). |
| 3 | **Judge-only metrics.** Retrieval quality is `semantic_recall@k` / `relevant_fraction@k` from judge verdicts. | Zero answer-model completions per experiment. No per-arm (model A vs B) factor in the mainline. |
| 4 | The **judge cache remains the oracle**. | No hand-labeled gold relevance set, no external benchmark. This buys speed and costs auditability — constraints 4 and 5 in §7 exist to limit the damage. |
| 5 | **Promote proven code** into the package; the grid + inspector stay as scripts. `experiments/` becomes evidence-only. | `experiments/retrieval_tuning/` stops being where measurement code lives. |
| 6 | Eval sample grows **20 → ~100 questions**, with the pinned 20 as a strict subset. | The 20 stay the continuity anchor for Phase 5/6 comparisons; larger only if R2's cost estimate holds. |
| 7 | Scheduled levers: **lexical fixes + fusion/funnel tuning.** Chunking/corpus shape, reranker model, and query-side rewrite go to unscheduled backlog. | Chunking is backlog *and* §7 constraint 2 — its cost is not the experiment, it's the cache. |
| 8 | End state: **`scripts/eval_retrieval.py`** (offline, scores any config against the cache) + a frozen retrieval config block + a tracked results doc. | Not a package API driven from pytest; not "keep the grid CLI as-is." |

---

## 3. Verified facts this plan depends on

Confirmed by reading the code, not inferred. Re-verify any of these before acting on it
months later.

1. **The retrieval harness imports from a Phase 6 evidence script.**
   `experiments/retrieval_tuning/judge_harness.py:77` and
   `experiments/retrieval_tuning/inspect_retrieval.py:96` both do
   `from raw_rag import Preflight, build_retriever, preflight, resolve_index_dir`, which
   works only because of `sys.path` inserts at `judge_harness.py:64-65` and
   `inspect_retrieval.py:82-83`. **This has to be broken before anything moves.** Fix:
   give `build_retriever()` a home in `medical_rag.retrieval` (§6) and have `raw_rag.py`
   import it from there, which also keeps decision 2 intact.
2. **Six `sys.path.insert` sites, no `conftest.py` anywhere.**
   `judge_harness.py:64-65`, `inspect_retrieval.py:82-83`, `judge_cache.py:40-43`,
   `strategies.py:39-40`, `reformulate.py:44-45`, `serve.py:59-60`. Three modules import
   siblings with **no** path insert and only work because their importer already inserted
   the directory: `ceiling.py:47`, `render.py:37-38`. Those three break silently on a move.
   `tests/test_retrieval_tuning.py:42-45` inserts the directory and imports 9 modules bare,
   and its docstring at `:24-27` states they are *"not meant to be importable from `src/`"* —
   that docstring is exactly what decision 5 reverses, so it has to change, not just move.
3. **Dead config, and the trap that makes deleting it a code change.** `verifier.min_chunks`,
   `reformulator.max_retries`, `conditions`, and `retrieval.chunk_size`/`chunk_overlap` are
   read by nothing except their own validator and test fixtures. But `verifier` /
   `reformulator` / `conditions` are **required** fields on `ExperimentConfig`
   (`src/medical_rag/config.py:92-94`, no defaults) and `load_config()` hard-requires
   `conditions.yaml` to exist (`config.py:232`) — so deleting the YAML blocks without editing
   `config.py` breaks every caller, including both tuning CLIs. Affected test fixtures:
   `tests/test_config.py:18-25,59-60,117-118`, `tests/test_raw_rag.py:90-97`.
   `scripts/build_index.py` never calls `load_config()` at all.
4. **All 9 stubs are docstring-only** — no code, no `NotImplementedError`:
   `modules/reformulator.py`, `modules/verifier.py`, `experiment/conditions.py`,
   `experiment/runner.py`, `experiment/metrics.py`, `analysis/failure_taxonomy.py`,
   `analysis/report.py`, `scripts/run_pilot.py`, `scripts/run_experiment.py`.
   `tests/test_reformulator.py` and `tests/test_verifier.py` are 5 lines each with **zero
   test functions** — they collect and pass vacuously. `tests/test_pipeline.py:8-10` is the
   only real pipeline test (imports the package, asserts `__version__`).
5. **Real overlap is small.** BM25 (`bm25_index.py`), RRF fusion (`hybrid.py`), recall
   metrics (`metrics.py`), judge + verdict cache (`judge_prompt.py`, `judge_cache.py`) and
   the chunk sidecar (`chunk_store.py`) have **no** counterpart in `src/` — promotion is a
   move, not a merge. `strategies.py:175-183` already calls `retriever.retrieve()` /
   `rerank()` rather than reimplementing, which is why the grid is trustworthy. Two genuine
   duplicates to reconcile: the numbered-chunk-block f-string at
   `judge_prompt.py:80-82` is identical to `generation/prompt.py:123-125`, and
   `ceiling.py`'s containment scan is a second normalization of
   `_common.chunk_match_rank` (`_common.py:757-787`).
6. **`_common.py` is the shared spine.** Imported by `raw_rag.py:73`, `closed_book.py:64`,
   and across the tree by `judge_harness.py:67`, `inspect_retrieval.py:85`,
   `judge_cache.py:45`; loaded by path in `tests/test_exploration_common.py:29`. Its own
   docstring at `:6` already says it's meant to be promoted into the package.
7. **The `*_rerank` cells are capped by construction.** `top_k_rerank: 5` means every
   `*_rerank` cell's recalled list is at most 5 chunks, so `dense_rerank__orig`'s
   45/45/45 is a *ceiling-on-5*, not a plateau. Do not read it as "reranking costs recall at
   higher k" — it means the funnel shape is a first-order lever (R5).
8. **Current numbers to beat** (`retrieval_tuning/FINDINGS.md`, run `20260929T120215Z`,
   n=20): `dense__orig` 40/55/65 recall@5/10/20, `dense_rerank__orig` 45/45/45,
   `hybrid__orig` 30/40/55, `hybrid_rerank__orig` 25/25/25, `bm25__orig` 0/0/10.
   `dense__reform` 40/55/70 is **not** a reformulation result — the rewrite fell back on
   18/20 questions, so that column is a copy of `__orig`.
9. **The lexical index is degenerate, reproducibly.** 59 distinct chunks fill the 100 top-5
   slots across 20 questions; `article-128082_187` (a child-abuse ED narrative) is rank-1
   for 6/20 and top-5 for 9/20 across dermatology-to-embryology vignettes, with
   `article-17453_34` (an ASA-Physical-Status example table), `article-40888_0` and
   `article-128082_185` recurring on the probe questions. That's a property of the lexical
   index, not of the questions.
10. **Index/build facts:** 380,454 chunks from 9,652 articles (NCBI's live corpus, grown past
    MedRAG's documented 301,202), `BAAI/bge-small-en-v1.5` with BGE's asymmetric
    query/passage convention, exact-search LanceDB (no ANN at this size),
    `cross-encoder/ms-marco-MiniLM-L-6-v2`. Retrieval is cheap — median 0.27 s/question;
    generation is what costs time.

---

## 4. New `PLAN.md` shape

Keep the sections that earn their place today (constraints-in-force, artifact convention,
open questions, decision record). Rewrite context and phases.

| Section | Content |
| --- | --- |
| Context | One paragraph. A StatPearls retrieval-optimization project for MedQA-USMLE: the LLM is a remote OpenAI-compatible endpoint used as a *judge*; retrieval quality is measured offline against a versioned judge-verdict cache; the corpus is 380,454 chunks behind exact-search LanceDB. |
| Why this direction | 3 bullets + links, zero narrative (§1 above, compressed). |
| Constraints in force | §7 below, rewritten. |
| Phases | R1–R6 (§5), with per-phase findings links. |
| Backlog, unscheduled | Chunking/corpus shape (carries constraint 2), reranker model + `--rerank-with`, query-side rewrite (the only lever that costs completions, and its one honest result is +0/+0/+5pp with 18/20 fallbacks), and the cut answer-accuracy study. |
| Open questions | Pruned and renumbered 1–6, with a mapping note — their old load-bearing referents (`interfaces.md`, `conditions.yaml`, the stubs) are being deleted, so the old numbering stops resolving. Survivors: judge reliability at scale; the paired-delta significance test over per-question recall indicators; verdict-cache growth and selection-bias policy; `judge_model`'s unreproducible server-side sampling (old Q16, now applies to the judge); the corpus-size-drift caveat (old Q7); resume semantics for a 100-question cache build. |
| Decision record | Keep the ADR-0001…0006 table, add the new ADRs, mark what the direction change supersedes. |
| Checklist | Links to `phase5`/`phase6` FINDINGS and `retrieval_tuning/FINDINGS.md` as **history**, plus `experiments/retrieval/R*/FINDINGS.md`. |

**Numbering.** New work uses an **R-prefix**. `Phase 7` and `Phase 8` are cited throughout
`phase6/FINDINGS.md` §4–5 and `interfaces.md` as *reformulation* and *verification*; reusing
those numbers for retrieval work would silently reassign every historical reference. A
one-line mapping note in `PLAN.md` is the whole cost.

The pinned 20 question ids stay in `PLAN.md` — they remain a strict subset of the R2 eval set.

---

## 5. Phase definitions

### R1 — Promote the harness (no new science)

Move code, delete the cut scaffolding, make grid and production share one code path.

- Break the `raw_rag.py` dependency first (§3 fact 1), then move per §6.
- Introduce `medical_rag.retrieval.strategy.RetrievalStrategy` built from config — dense
  top-k, lexical top-k, RRF params, rerank k, rerank query variant — used by *both* the grid
  and the shipped path, so the grid can never measure a pipeline that doesn't ship.
- `scripts/exploration/_common.py` keeps its name and re-exports the promoted helpers so
  `closed_book.py`, `raw_rag.py` and their tests keep working untouched (decision 2).
- **Gate:** `--rerender outputs/exploration/retrieval_tuning/20260929T120215Z` into a temp
  path `diff`s cell-for-cell against the committed `FINDINGS.md` **without overwriting it**.
  This is free, offline, and already proven to reproduce once — it is the regression test for
  the move itself. Plus: `pytest` green with zero `sys.path.insert` in `tests/`.

### R2 — Trust the oracle, then enlarge it

Decision 4 makes the judge ground truth, and `TUNING.md` still lists its reliability as an
open question. That inversion has to be closed before anything else is believed.

- Stratified hand-check of ~30 verdicts, deliberately including the §3 fact-9 boilerplate
  chunks as known-bad controls, reported per class rather than as one agreement %.
- **Silent-drop audit** of `judge_cache.ensure_verdicts` at `--judge-batch 24` with full
  chunk prose: a batch that truncates or returns fewer judgments than candidates would
  poison every downstream number *quietly*. Verify `judgments == candidates` per call.
- One rubric variant stored under its own `judge_prompt_sha` (the cache is already keyed by
  it, so verdict stability across rubrics is free to store and cheap to compare).
- Confirm by test, not by reading, that gold answer text never reaches the judge prompt.
- Then grow the pinned sample 20 → ~100: the 20 stay a strict subset, the extra ~80 come
  from a deterministic draw over the remainder, and preflight asserts the containment — the
  pattern `raw_rag.py` already uses when it aborts unless `select_sample(pool=10178,
  size=20, seed=1)` still returns exactly the pinned list.
  **Do not** just call `select_sample(size=100, seed=1)` — that will not contain the 20.
- Pay the judge calls once. **Estimate, not measurement:** ~4–5 batched calls per new
  question for unseen chunk unions ⇒ a few hundred calls on the shared `:8080`, resumable,
  circuit-breaker guarded.
- Deliverable includes the tracked oracle snapshot from §7 constraint 4.

### R3 — Corpus ceiling first

`ceiling.py` already exists and spends zero completions: it runs the gold option's own
wording through a literal scan of all 380,454 chunk bodies and through
`strategies.retrieve_base()`. Run it over the enlarged sample to split *"exists, ranked #60"*
from *"StatPearls never said it"*, and state the attainable ceiling for R4/R5.

It is a ceiling, never a lever — it uses the gold answer, so it stays out of every strategy
table, exactly as `TUNING.md` demands after this directory already published a wrong number
off an unlabelled column. And the bound is one-directional: 0 matches means "not in these
words," not "not in StatPearls."

### R4 — Fix the lexical index

Tokenizer (medical terms, doses, numbers, stopwords, stemming), field weighting
(title / contents / body), and excluding narrative case-report boilerplate. Give
`bm25_index.py` a real CLI — it has no argparse and no `__main__`, reachable today only as
`--rebuild-bm25` from the other two scripts — and write corpus + tokenizer version into the
pickle so the 467 MB artifact isn't anonymous.

**New tracked metric:** distinct chunks filling the top-5 (today 59/100, one chunk rank-1 on
6/20). Fixing degeneracy is measurable even where recall barely moves.

### R5 — Fusion and funnel shape

RRF `k`, per-list cutoffs, dense/lexical weighting, rerank pool size, and where the funnel
cuts — guided by fact 7: `dense__orig` climbs 40 → 55 → 65 across k while
`dense_rerank__orig` is capped at 45 by a 5-chunk funnel. Report paired deltas with
bootstrap CIs (offline, free) so a delta at n=100 is defensible rather than eyeballed.

### R6 — Freeze and deliver

`scripts/eval_retrieval.py` becomes the single offline command (zero endpoint calls), the
frozen retrieval config block cites its R4/R5 measurement per value, the tracked results doc
is generated from the command rather than typed, and index + BM25 pickle provenance is
recorded. `README.md` finally describes what the repo does.

---

## 6. Scaffolding change list

**New package code**

| Target | Source | Note |
| --- | --- | --- |
| `src/medical_rag/retrieval/lexical.py` | `experiments/retrieval_tuning/bm25_index.py` | net-new, no `src/` counterpart |
| `src/medical_rag/retrieval/fusion.py` | `hybrid.py` | RRF, pure |
| `src/medical_rag/retrieval/ceiling.py` | `ceiling.py` | adopt `_common.chunk_match_rank`'s normalization instead of a second one |
| `src/medical_rag/retrieval/strategy.py` | new (absorbs `strategies.py`'s candidate builders) | `RetrievalStrategy` + `build_retriever()` — the fact-1 fix |
| `src/medical_rag/eval/metrics.py` | `metrics.py` | recall@k, relevant_fraction, first_relevant_rank |
| `src/medical_rag/eval/judge.py` | `judge_prompt.py` + `judge_cache.py` | rubric sha stays part of the cache key |
| `src/medical_rag/eval/store.py` | `chunk_store.py` | the sidecar that lets a run be read without the 380k index |
| `src/medical_rag/eval/runlog.py` | `_common.py`'s `RunWriter`, `EndpointCircuitBreaker`, `result_row`/`error_row` | `_common.py` re-exports, stays importable |
| `src/medical_rag/eval/report.py` | `render.py`'s 3 table functions only | see below |
| `scripts/eval_retrieval.py` | new | the R6 end state |
| `scripts/inspect_retrieval.py`, grid runner | `inspect_retrieval.py`, `judge_harness.py` | CLIs belong in `scripts/`, matching the repo's existing split |

`render.py` is ~1,200 lines and moves as a unit — **do not rewrite it**. Only the
table/summary functions split into `eval/report.py`; the HTML grid view stays with the
inspector. `serve.py` drives `Inspector` and needs its imports updated, nothing else.
`reformulate.py` stays in `experiments/retrieval_tuning/` as the unscheduled
costs-completions lever.

**Delete**

- Stub modules + their now-empty subpackages: `src/medical_rag/modules/`
  (`reformulator.py`, `verifier.py`), `src/medical_rag/experiment/` (`conditions.py`,
  `runner.py`, `metrics.py`), `src/medical_rag/analysis/` (`failure_taxonomy.py`, `report.py`).
- `scripts/run_pilot.py`, `scripts/run_experiment.py`.
- `tests/test_reformulator.py`, `tests/test_verifier.py` (vacuous, §3 fact 4).
- `config/conditions.yaml`; the `verifier` and `reformulator` blocks and
  `retrieval.chunk_size`/`chunk_overlap` from `config/default.yaml`; the matching
  `VerifierConfig` / `ReformulatorConfig` / `ConditionConfig` models and
  `check_condition_models_exist` validator in `config.py`; the `conditions.yaml` existence
  requirement at `config.py:232`.
- `prompt.md` at the root — tracked, zero repo references, and its instruction set
  ("Only edit/create Markdown. Do not touch code, config, or test files") now actively
  contradicts the repo's task. Recoverable from git.

**Keep, and label clearly**

- `closed_book.py`, `raw_rag.py`, `probe_models.py`, `audit_run.py`, `SYSTEM_PROMPT`,
  `build_answer_prompt`, `build_verification_prompt`, `parse_answer` → documented in
  `interfaces.md` as *frozen Phases 5–6 evidence tooling, not mainline*.
- `config/models.yaml` keeps `model_a` / `model_b` (the frozen tooling needs them) with a
  comment that the mainline uses `judge_model` only.
- `data/chunking.py` — unused today, kept as the natural home of the backlog chunking lever.

**Docs to update**

`PLAN.md` (rewritten, §4) · `interfaces.md` (drop cut contracts, add strategy/eval
contracts, fix its open-question line refs at `:243`/`:417`) · `file_layout.md` (new tree —
it's the file that currently justifies the old layout) · `README.md` (what the repo does +
the one eval command) · `experiments/README.md` (R-phase convention, drop the Phase-10/13
rules) · `experiments/retrieval_tuning/TUNING.md` (retitle from *"side track, not a phase"*
to the mainline harness doc; correct the stale "~2 s" claim its own docstring still makes;
drop the "commit `docs/adr/`" to-do, which landed in `a0fb068`) · `docs/adr/README.md` +
new ADRs · `pyproject.toml` description string only.

**New ADRs:** 0007 retrieval-first direction change (supersedes the factorial consequence of
0002) · 0008 judge cache as ground truth · 0009 corpus + chunking frozen for this track.
Edits to 0002/0006 are `Status:` lines only — no body rewrites, that's what Superseded means.

---

## 7. Constraints to carry into the new `PLAN.md`

Each of these has already cost a bad measurement or will cost one.

1. **The harness only measures the union of strategies it ran.** A new lever's chunks are
   unjudged until that lever is in the grid, so R4/R5 must add their candidate lists *to the
   grid*, not evaluate against the existing cache — otherwise the cache stays permanently
   blind to the exact candidate pool you're trying to prove.
2. **Corpus + chunking are frozen for this track.** Verdicts are keyed
   `(question_id, chunk_id, judge_prompt_sha)`. Changing chunking changes `chunk_id` and
   invalidates the entire cache — cost ≈ the whole R2 judge spend. So it becomes an explicit
   freeze decision in R6, not a silent dependency the backlog trips over.
3. **Never compare a `*_rerank` cell's recall@10/20 against an unreranked cell's.** The
   funnel caps it (§3 fact 7).
4. **The oracle must be versioned.** `judge_cache/judged_chunks.jsonl` lives in gitignored
   `outputs/`, which contradicts this repo's own rule that evidence has to be tracked to be
   reviewable. R2 commits a compact snapshot — ids, relevance, `supports_options`, reason,
   no chunk prose — under `experiments/retrieval/evalset/`. Live cache stays
   regenerable-but-expensive; the snapshot is the oracle.
5. **Judge discipline is the old endpoint discipline.** `judge_model` shares `:8080` with
   `model_a`/`model_b`; keep the circuit breaker and resume rules, and record the judge's
   endpoint-side sampling settings the way `PLAN.md` already demands for `model_b` — under
   decision 4, those settings shape the ground truth, so old open question 16 now applies to
   the judge.
6. **Dev split only**, and no number in `outputs/` becomes a result unless
   `scripts/eval_retrieval.py` re-derives it. That's the successor to `audit_run.py`'s role.

---

## 8. Verification

- `.venv/bin/pytest` — offline (the single `live` test skips without
  `MEDICAL_RAG_LIVE_TESTS=1`), and zero `sys.path.insert` anywhere under `tests/`.
- New tests: `tests/test_retrieval_strategy.py` (strategy composition over a fake
  retriever — dense-only, dense+rerank, lexical, hybrid, each funnel shape) and
  `tests/test_eval_metrics.py` (`gold_hit`, `recall@k`, `relevant_fraction@k` against
  fixtures of known value). Retarget `tests/test_retrieval_tuning.py` to package imports —
  it already covers grid shape, candidate assembly, cache keying/eviction, the chunk sidecar
  and legacy-run re-rendering, so it moves rather than being rewritten. Drop the
  condition-validator cases from `test_config.py`; drop the deleted blocks from
  `test_raw_rag.py`'s fixtures.
- **Golden re-render** (R1's gate): re-render run `20260929T120215Z` to a temp path and
  `diff` against the committed `FINDINGS.md`. Must be identical. The file must not be
  overwritten by the check — `TUNING.md` records that a 2-question probe once wrote its n=2
  table over the real findings.
- Frozen tooling still runs: `raw_rag.py --dry-run` and `closed_book.py --dry-run` pass
  preflight (proves both the import fix and that the pinned-20 assertion still resolves);
  `inspect_retrieval.py --serve --no-retrieve --from-run
  outputs/exploration/retrieval_tuning/grid_smoke_check` works with nothing loaded;
  `eval_retrieval.py --dry-run` prints the resolved strategy from clean config.
- `git status` reviewed together before anything is staged. No staging, committing, or
  reverting on my own.

---

## 9. Calls I made without asking — veto any of them

1. **`data/chunking.py` survives** despite having no callers, because it's the natural home
   of the backlog chunking lever. Delete it if you'd rather keep the tree honest.
2. **`retrieval.chunk_size` / `chunk_overlap` get deleted rather than wired up.** The
   alternative was making `scripts/build_index.py` actually read them, but the vendored
   MedRAG section chunker never used them — StatPearls is chunked by article section, not by
   a 512-token window. Making them live would be a different lie. Actual corpus shape gets
   documented as provenance instead, and a scheduled chunking phase re-adds a real knob.
3. **`prompt.md` gets deleted** (see §6). It's the only file in the repo whose contents
   would mislead a future agent about what this repo is for.
4. **`PLAN.md`'s claim that `docs/adr/` is untracked gets dropped** as stale — it landed in
   `a0fb068`.
5. **Delete-don't-comment-out** for the cut stubs, per decision 1. If you'd rather keep them
   as compiling no-ops behind a "deferred" note, that's a one-line change to this plan's §6.

---

## 10. Sequencing

R1 is mechanical and unblocks everything, including the fact-1 import fix that makes any
later move safe. R2 is the one that costs real endpoint time and is the only thing standing
between this project and trustworthy numbers, since decision 4 made an unvalidated judge the
oracle. R3 is free and tells you whether R4/R5's ceiling is 65% or 90%. R4/R5 are the actual
science. R6 is paperwork that a previous phase has repeatedly skipped.

Do not start R4 while the judge is unverified, and do not start R5's fusion tuning before R4
stopped returning the same five boilerplate chunks for every question — tuning a fusion over
a degenerate input list measures the degeneracy.
