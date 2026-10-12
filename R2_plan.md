# R2 — Trust the oracle, then enlarge it: execution plan

> **Status: draft for review.** Executes the R2 section of [`PLAN.md`](./PLAN.md)
> (with [ADR-0008](./docs/adr/0008-judge-cache-as-ground-truth.md) as the gate it
> closes). Grounded against the tree at `e1ac072`; every "current state" number in
> §2 was re-measured on this checkout on 2026-11-10 with the command shown, not
> copied from prose. Nothing in this plan spends endpoint time before gate **G1**
> (§A5) passes and you approve the §C quote.
>
> One-line summary: **A** validates the oracle (leakage tests, drop audit, stratified
> hand-check with the boilerplate controls, rubric variant under its own sha) —
> offline plus ~20 min of endpoint; **B** grows the pinned sample 20 → 100 by
> containment into a tracked file; **C** prices the build from `--dry-run`; **D** pays
> the judge once, resumable and circuit-breaker guarded; **E** closes out the oracle
> (cache verify/compact, tracked snapshot, n=100 scoring); **F** writes
> `experiments/retrieval/R2/FINDINGS.md` and updates the tree docs.

## 0. Where PLAN.md leaves R2, and what this plan adds

PLAN.md's R2 section is directionally complete; this plan adds the operational
detail it names but does not spec:

1. **The go-ahead was undefined.** PLAN.md says R2 "needs a go-ahead" and to price
   from `--dry-run`, but gives no pass/fail criteria for the judge validation and no
   branch for failing it. This plan defines gate **G1** (§A5) with explicit
   thresholds and a rubric-v2 fallback that is built before the spend, so a failed
   validation costs minutes of endpoint time, not the ~4 h build.
2. **Containment was asserted, not mechanised.** "Do not just draw `size=100,
   seed=1`" is correct — `select_sample()` hash-ranking is explicitly *not*
   containment-preserving (`eval/runlog.py`'s own docstring warns a new question can
   rank inside the cut). §B builds the sample as `pinned ∪ hash-ranked remainder`
   so containment holds by construction, persists it as a tracked JSON, and tests
   the committed artifact.
3. **The hand-check had no artifact.** "Stratified hand-check of ~30 verdicts" needs
   strata definitions, a rendered reviewable file, a blind two-pass procedure, and a
   per-class agreement computation. §A3 specs all four; the artifact is committed.
4. **The rubric variant implies a small refactor.** `eval/rubric.py` currently holds
   one module-level wording, and `ensure_verdicts(judge_sha=…)` can stamp an
   arbitrary sha over unchanged wording — a sha-lie. §A4 introduces `RubricSpec`
   (version + wording → sha) with a default whose sha is asserted to stay
   `21e27467277a`, so the 926 cached verdicts survive the refactor byte-for-byte.

5. **Open questions 4 and 6 are settled here** (they are R2's to settle): OQ6's torn
   record policy = append-only stays + `judge.py --verify` + one post-build
   compaction (§E1, D5); OQ4's endpoint-side sampling is recorded next to the
   snapshot and frozen for the track (§2, D6) — with the constraint PLAN.md omits:
   **the 926 existing verdicts were graded under today's server-side settings, so
   changing them mid-track splits the oracle into two graders.**
6. **A trap made explicit.** The grid's `--findings-path` defaults to the *frozen*
   `experiments/retrieval_tuning/FINDINGS.md` — the exact overwrite TUNING.md
   documents (a probe once clobbered it and had to be restored). Every command in
   this plan sets `--findings-path` explicitly.

## 1. Scope

**In scope.** Judge validation; sample enlargement by containment; one grid run over
the enlarged sample whose only purpose is to pay the oracle for the union of the ten
existing cells; cache hygiene; tracked snapshot; n=100 scoring with
`--require-coverage`; the R2 findings file.

**Out of scope (each is another phase's job, and touching them here is how R2
becomes R4):**

- No retrieval lever changes: `config/default.yaml` is untouched (`top_k_retrieve:
  20`, `top_k_rerank: 5`, `bm25_top_k: 20`, `rrf_k: 60`, `strategy: dense_rerank`).
  R2's grid runs the shipped knobs so the only variable is n.
- No new strategies or levers (invariant 1: a lever not in the grid has no verdicts;
  and unjudged slots score as *not supporting*, so a new lever measured against
  yesterday's cache is a floor, not a result).
- No corpus or chunking changes (ADR-0009: changing chunk geometry orphans every
  `chunk_id` and bills roughly the whole R2 judge spend again).
- `SAMPLE_QUESTION_IDS` (the pinned 20) is **not** edited — Phase 5/6 comparability
  and `eval/inspector.py` defaults depend on it. The enlargement is a *new* tracked
  sample beside it.
- `JUDGE_PROMPT_VERSION` stays `2026-09-29.1` unless gate G1 fails and the variant
  is adopted (then, and only then, a new default wording + sha lands, by design).
- No significance-test machinery (OQ2 belongs to R5), no fusion tuning (R5), no
  lexical index work (R4), no ceiling run (R3 — free, and it runs on whatever sample
  R2 leaves behind).
- `README.md` and the corpus-drift note (OQ5) stay with R6.

## 2. Current measured state (verified 2026-11-10, this checkout)

| # | Fact | Value | How verified |
| --- | --- | --- | --- |
| 1 | Verdict cache | `outputs/exploration/retrieval_tuning/judge_cache/judged_chunks.jsonl`: **926 verdicts, 0 malformed** | `.venv/bin/python -m medical_rag.eval.judge --stats` |
| 2 | Rubric identity | current sha `21e27467277a` (`JUDGE_PROMPT_VERSION = "2026-09-29.1"`); 926/926 usable | same |
| 3 | Verdicts by source | 819 grid run `20260929T120215Z` + 56 smoke + 12 inspect + 39 live | same |
| 4 | Verdicts by relevance | 698 irrelevant / 162 partial / 66 relevant | same |
| 5 | The four known-bad controls | `article-128082_187` (12 records), `article-17453_34` (9), `article-40888_0` (10), `article-128082_185` (6) — **all judged `irrelevant`, `supports_options=[]`** | grep over the cache |
| 6 | Judge model | `Qwen3.8-Flash-Next-oQ6e-mtp` @ `http://localhost:8080/v1`; client sends `temperature=1.0`, `top_p=0.95`, `max_tokens=4096` | `config/models.yaml` (`judge_model`) |
| 7 | Judge server-side sampling | `~/.omlx/model_settings.json` key `Qwen3.8-Flash-Next-oQ6e-mtp`: `temperature 0.7, top_p 0.8, top_k 20, min_p 0.0, repetition_penalty 1.0, presence_penalty 1.5, force_sampling false, max_context_window 262144`. Request-level values win where sent; the rest apply. The file also holds an `oQ4e` sibling entry — only the exact `api_model_name` key governs | grep over `~/.omlx/model_settings.json` |
| 8 | Dry-run cost line (n=20) | "planned calls, worst case (empty cache): 20 reformulation + ~60 judge batches (58ish chunks/question at 24 per batch). Cached pairs cost nothing." | `.venv/bin/python scripts/run_retrieval_grid.py --dry-run` |
| 9 | Measured per-question judge cost | 819 graded chunks / 20 questions ≈ **41 chunks/question** at `--judge-batch 24` (union over 10 cells) | cache source_run tally ÷ the run's 20 result rows |
| 10 | Sample universe | dev split = `train`, 10,178 questions; test split untouched; pinned 20 = Phase 5's hash-ranked draw (seed 1) | `config/default.yaml`, `experiments/phase5/FINDINGS.md`, `eval/runlog.py` |
| 11 | Test suite | 191 tests collected; green offline (R1 gate claim) | `.venv/bin/python -m pytest --collect-only -q` |
| 12 | The findings trap | grid `--findings-path` defaults to the frozen `experiments/retrieval_tuning/FINDINGS.md` | `parse_args()` in `src/medical_rag/eval/harness.py` |

Reading of #5 for the hand-check: the judge has so far been *hard*, not lenient, on
the four boilerplate controls (37/37 `irrelevant`). The controls therefore anchor
two questions at once — does a human agree `irrelevant` is right (no over-hardness),
and does any *other* boilerplate chunk carry a `relevant`/`partial` verdict in the
cache (the leniency PLAN.md fears). The strata in §A3 are built to answer both.

## 3. Decisions for review (D1–D8)

Each has a recommendation; **none blocks Stage A** (validation is offline except
A4's ~20 min). Edit any wording/number in place before execution starts.

- **D1 — sample builder.** `pinned-20 ∪ hash-ranked 80` with seed 2, persisted to a
  tracked JSON, preflighted for containment. *Recommend: adopt* (§B).
- **D2 — grid shape for the build.** Full 10 cells incl. `*_reform`: +~100 short
  model_a calls (the rewrite bug is fixed — 0/2 fallbacks in the grid smoke), and
  reform unions genuinely enlarge the candidate set the oracle must pay for.
  *Recommend: keep.* `--no-reform` (5 cells, 0 reform calls) is the fallback if
  `:8080` is contended; decide at Stage C, not mid-run.
- **D3 — rubric variant wording.** Shown in §A4; tightened on exactly the failure
  mode PLAN.md fears (boilerplate leniency). *Recommend: adopt as drafted; edit
  before any variant spend.*
- **D4 — gate G1 thresholds.** Controls 100% human-confirmed `irrelevant`; per-class
  relevance agreement ≥ 80%; ≤ 1 systematic one-directional error per class (judge
  says relevant where human says irrelevant). *Recommend: adopt as written; loosening
  any of these deserves a recorded reason.*
- **D5 — torn records (OQ6).** Append-only per-question commits stay (they are the
  resume story); add `judge.py --verify` + one post-build compaction pass; a tear
  costs ≤ one batch re-ask (~24 chunks ≈ a couple of minutes). *Recommend: adopt.*
- **D6 — endpoint sampling (OQ4).** Record-only, never change mid-track. The 926
  existing verdicts were graded under today's server-side settings; changing them
  splits the oracle into two graders the cache key cannot distinguish. If sampling
  is ever changed deliberately, that is a new-oracle event: bump
  `JUDGE_PROMPT_VERSION` in the same commit. *Recommend: adopt.*
- **D7 — hand-check size and strata.** ~30 records across five strata (table §A3),
  seeded and committed. *Recommend: adopt; strata sizes are the knob to argue about.*
- **D8 — snapshot layout.** `verdicts_<sha>.jsonl` + `snapshot_metadata.json` +
  `sample_v2.json` under `experiments/retrieval/evalset/` — matches the contract
  already written in `experiments/README.md`. *Recommend: adopt.*

## 4. Execution stages

### Stage 0 — entry gates (offline, ~10 min)

```bash
.venv/bin/python -m pytest -m "not live" -q                     # 191 passed expected
.venv/bin/python -m medical_rag.eval.judge --stats              # the §2 #1–#4 lines
.venv/bin/python scripts/eval_retrieval.py --dry-run            # sample + coverage
.venv/bin/python scripts/run_retrieval_grid.py --dry-run        # cost line (n=20)
git status --short                                              # clean tree
```

Acceptance: all outputs match §2. If the cache moved since this plan was written,
re-read §2 and re-price Stage C before anything else. Effort: none beyond running
them.

### Stage A — validate the oracle (offline + ~20 min endpoint in A4) — before any large spend

**A1. Gold-leakage tests** (~45 min, offline; PLAN.md's "a test that gold answer
text cannot reach the prompt")

- Where: `tests/test_retrieval_tuning.py`, judge section (next to the existing
  cache-key tests).
- Test 1 — prompt invariant to the gold letter: same question, `answer_idx = A` vs
  `answer_idx = B` → byte-identical `build_judge_prompt(...)` output *and* system
  prompt.
- Test 2 — canary containment: make the correct option's text a canary string; strip
  the `Options:` section from the prompt; assert the canary appears nowhere else in
  prompt or system prompt.
- Note: option *text* is in the prompt by design (the judge sees all four options —
  that is the point of `supports_options`); what must not leak is *which option is
  correct*. Both tests target exactly that, per the contract already stated in
  `eval/rubric.py` ("never the correct answer").
- Acceptance: tests exist and are green against the current wording with **no code
  change**. If either fails, that is a stop-everything oracle bug — halt before A2.

**A2. Silent-drop audit** (~1.5 h, offline; ADR-0008 "judge-drops = silent vote-shape
change")

- Code: `eval/judge.py` gains `audit_run(run_dir)` + `--audit-run RUN_DIR`: for each
  non-error row of `results.jsonl`, `set(row["judgments"])` must equal the union of
  `row["candidates"]` over all methods present; print per-question shortfalls and a
  total.
- Tests (at **full prose length**, per the ADR's clause — batch 24 × ~8,000-char
  bodies):
  1. fake client returns one judgment per id parsed from the prompt (reuse the
     `FakeJudgeClient` pattern) → `judged == n`, `missing_judgment == 0`.
  2. client returns only 20 of 24 ids → `missing_judgment == 4`, the four are *not*
     in the cache, `problems` non-empty (extends the existing dropped-`c5` test).
- Run the audit over the two existing run dirs — `outputs/exploration/
  retrieval_tuning/20260929T120215Z` (the 819-chunk grid run) and the grid smoke
  `20260929T191914Z` — and record the counts in FINDINGS.
- Acceptance: zero unaccounted drops in the existing runs; both tests green.

**A3. Stratified hand-check** (~2 h tooling + 1–2 h human review, offline)

- New module `src/medical_rag/eval/evalset.py` (+ thin `scripts/evalset.py`,
  `python -m medical_rag.eval.evalset` also works). Functions:
  `build_hand_check()` (seeded stratified selection over the cache),
  `render_hand_check()` (writes md + CSV; judge verdicts hidden in pass 1),
  `score_hand_check()` (reveals, computes the per-class agreement table).
- Chunk prose comes from `eval/store.resolve_chunks` against the LanceDB table
  (the 20-question run predates sidecars — §3 of experiments/README).
- Strata (~30 records total, D7; selection seeded and committed with the artifact):

| stratum | definition | n |
| --- | --- | --- |
| S1 controls | the four known-bad chunks (`article-128082_187`, `article-17453_34`, `article-40888_0`, `article-128082_185`), 2 records each, spread over ≥ 5 questions | 8 |
| S2 relevant | cached verdict `relevant` | 8 |
| S3 partial | cached verdict `partial` | 8 |
| S4 irrelevant | random `irrelevant`, excluding S1 chunks | 4 |
| S5 letter-discrimination | `supports_options` non-empty *without* the gold letter | 2 |

- Procedure (blind two-pass): pass 1 — the human reads chunk prose + question +
  options (+ gold letter at the bottom) and fills `human_relevance` +
  `human_supports` in the CSV, judge verdicts not visible; pass 2 —
  `score_hand_check` reveals the judge columns and emits the per-class table:
  relevance agreement (3-way, with the relevant↔partial confusion shown
  separately), `supports_options` exact-set match, and gold-support correctness
  (does the chunk genuinely evidence the gold option → judge's
  `supports_options[gold]` presence correct?).
- Artifact: `experiments/retrieval/R2/hand_check.md` + filled CSV, committed.
- Acceptance: per-class table exists; **no pooled percentage anywhere** (constraint
  in force); S2/S3 sampling is checked for boilerplate that earned a non-irrelevant
  verdict (the leniency hunt §2 describes).

**A4. Rubric variant under its own sha** (~1 h code + ~20 min endpoint)

- Refactor `eval/rubric.py`: `RubricSpec(version, system_prompt, json_instruction)`
  with `sha()`; `DEFAULT_RUBRIC` carries the exact current wording and a test
  asserts its sha stays `21e27467277a` (the rename must invalidate nothing);
  `build_judge_prompt()` / `judge_prompt_sha()` become thin delegating shims so
  existing imports keep working.
- `ensure_verdicts(..., rubric: RubricSpec = DEFAULT_RUBRIC)` replaces the loose
  `judge_sha` override (stamping an arbitrary sha over unchanged wording would be a
  sha-lie); records gain the rubric sha they were graded under.
- Variant `r2-stability-a` (D3), tightened on the feared failure mode:
  *"relevant requires a finding, quantity, mechanism, or guideline statement that
  discriminates between the options; case narrative, administrative, or reporting
  boilerplate is irrelevant even when it mentions option-related terms"* + the
  reason must name the discriminating sentence.
- Spend: re-grade the hand-check pair set under the variant sha (≤ ~20 calls,
  batched per question). Verdicts land in the same cache keyed by their own sha;
  `load_cache` filters by sha so nothing is polluted (test asserts default-sha
  loading ignores them).
- Measure: raw agreement + Cohen's kappa (`statsmodels`, already pinned) between
  v1 and v2 on the identical pairs, with the human as third rater on S1–S5.
- Acceptance: variant sha ≠ default; stability numbers computed; nothing in `eval/`
  or the grid reads the variant unless gate G1 adopts it.

**A5. Gate G1 — the oracle verdict** (decision point; no code)

- **PASS** — every control record human-confirmed `irrelevant`; per-class relevance
  agreement ≥ 80% (S2–S5); no class shows > 1 systematic one-directional error
  (judge-relevant where human says irrelevant) → proceed to Stage B/C/D under sha
  `21e27467277a`; A4's numbers are reported as rubric-sensitivity evidence.
- **FAIL + variant passes the same gate** → adopt the variant as `DEFAULT_RUBRIC`
  (bump `JUDGE_PROMPT_VERSION`; the old 926 become stale-sha — ignored, never
  deleted), re-price Stage C under the new sha, proceed.
- **FAIL both** → **stop.** No R2 build spend. FINDINGS records the failure classes;
  R3+ is blocked; escalating (different judge model or a rubric redesign) is an ADR.
- If G1 has not been evaluated, Stage D does not start. This is what PLAN.md's
  "needs a go-ahead" resolves into: **G1 passing + the §C quote approved.**

### Stage B — enlarge the sample 20 → 100 by containment (~1–1.5 h, offline)

- Builder in `eval/evalset.py`: `build_sample(questions, pinned, total=100,
  seed=2)` = pinned ∪ first 80 of the remainder ranked by
  `sha256(f"{seed}:{question_id}")` ascending — containment **by construction**,
  and deliberately *not* `select_sample(seed, 100)`, which can rank a new question
  inside the cut (its own docstring says so).
- Output `experiments/retrieval/evalset/sample_v2.json`: ids, `split: train`, n,
  seed, selection-method string, pinned subset echoed, created date, git rev.
- Preflight (function + CLI + test): |ids| == 100, no dups, pinned 20 ⊆ ids, every
  id resolves via `select_by_ids(load_medqa(dev_split), ids)`.
- Wiring: `--question-ids-file PATH` on `run_retrieval_grid.py`/`harness.py` and
  `eval_retrieval.py` (JSON list or newline-separated ids). The harness runs the
  same preflight before any endpoint call when the file is used;
  `eval_retrieval.py` warns if a scored set drops pinned ids. (The harness already
  takes `--question-ids`; the *file* is what makes the sample tracked and
  reproducible.)
- `SAMPLE_QUESTION_IDS` stays exactly as pinned.
- Test: `tests/test_evalset.py::test_sample_v2_contains_the_pinned_20` reads the
  committed file, so the tracked artifact itself is what is tested.

### Stage C — price the spend (~10 min, offline; quote, don't guess)

```bash
.venv/bin/python scripts/run_retrieval_grid.py --dry-run \
  --question-ids-file experiments/retrieval/evalset/sample_v2.json
```

- Record in FINDINGS: the printed worst-case line, plus the projection from the
  measured rate — 80 unseen questions × ~41 chunks ≈ **~3.3k graded chunks ≈ ~140
  batches at 24 ≈ 3–4.5 h sequential** (20-q pace: 819 chunks in 45–60 min);
  worst-case unions 58/q → ~4.6k → ~4.5–6 h; plus ~100 short model_a reform calls
  (or 0 with `--no-reform`, D2). The pinned 20 re-cost nothing.
- This printed line is the number the go-ahead approves; re-run it if anything
  upstream changed.

### Stage D — the cache build (endpoint spend; precondition: G1 passed + §C quote approved)

1. **Smoke first** (~10 min):
   ```bash
   .venv/bin/python scripts/run_retrieval_grid.py \
     --question-ids-file experiments/retrieval/evalset/sample_v2.json --limit 2 \
     --findings-path /tmp/r2_smoke_FINDINGS.md --note probe=r2-smoke
   ```
   Confirms: unions judged, 0 drops, reform fallback count, findings written to the
   *explicit* path.
2. **The build** (run in tmux/background; resumable):
   ```bash
   set -a; source .env; set +a
   .venv/bin/python scripts/run_retrieval_grid.py \
     --question-ids-file experiments/retrieval/evalset/sample_v2.json \
     --findings-path /tmp/r2_FINDINGS_AUTO.md \
     --note phase=R2 --note sample=v2-n100 --note judge_sha=21e27467277a
   ```
   - On interruption: re-invoke with `--resume outputs/exploration/retrieval_tuning/<stamp>`;
     the append-only cache + `EndpointCircuitBreaker` do the rest. Cached pairs
     (pinned 20 included) cost nothing.
   - Discipline: never re-run a batch to change a number. If a batch shows
     `missing_judgment > 0`, re-ask only that batch, or drop `--judge-batch` to 12
     for the remainder and record the change in the run's `context.md` (batch size
     is harness plumbing, not the oracle's identity — the sha is).
   - Raw run dir stays under `outputs/exploration/retrieval_tuning/<stamp>`/
     (gitignored); only FINDINGS, the sample, and the snapshot get tracked.
3. Acceptance: run completes with 0 error rows (or each error row understood and
   recorded); cache delta within the §C quote ± a stated tolerance; no frozen file
   touched.

### Stage E — close out the oracle (~45 min, offline)

1. **Cache verify + one-shot compaction** (settles OQ6, D5):
   `python -m medical_rag.eval.judge --verify` (new) reports malformed lines,
   duplicate keys, stale-sha counts. If malformed > 0: `--compact` rewrites a clean
   file (dedupe keep-last), original kept as `.bak`, then re-verify. The
   append-only per-question commit stays — the accepted cost of a tear is one
   re-asked batch.
2. **Snapshot export** (D8):
   `python -m medical_rag.eval.evalset export --out experiments/retrieval/evalset/`
   writes `verdicts_<sha>.jsonl` (question_id, chunk_id, relevance,
   supports_options, reason, judged_at, source_run, model — **no prose**) and
   `snapshot_metadata.json` (git rev; full rubric text + sha; judge model +
   `api_model_name`; client-sent sampling from `models.yaml`; **server-side sampling
   pasted from `~/.omlx/model_settings.json`** incl. the `force_sampling` note (D6);
   dataset + split; corpus/index provenance — 380,454 chunks, index dir + index sha
   from the run's `context.md`; sample file + its sha; cache stats;
   dry-run-predicted vs actual reconciliation; one rendered example prompt).
   `--check` asserts snapshot == current-sha slice of the cache.
3. **Scoring** (the only numbers FINDINGS may print — from this script, not by hand):
   ```bash
   .venv/bin/python scripts/eval_retrieval.py \
     --question-ids-file experiments/retrieval/evalset/sample_v2.json \
     --require-coverage 1.0 \
     --json outputs/exploration/retrieval/R2/n100_summary.json
   ```
   plus the same command without the file for the n=20 continuity row. Keep the
   n=20 table's "read it as" column semantics (rerank caps etc.) so the two
   tables read as one.

### Stage F — findings and docs (~1.5 h, offline)

- `experiments/retrieval/R2/FINDINGS.md` on the `experiments/TEMPLATE.md` spine
  (numbers / failures with chunk ids / decisions / open questions settled / cost):
  per-class hand-check table + the G1 decision; drop-audit counts over both runs;
  rubric-variant stability (agreement + kappa); the n=100 table beside the n=20
  table; OQ4's recorded sampling values; the OQ6 decision; the cost section
  (calls, batches, wall clock, the §C dry-run line quoted verbatim).
- Update `file_layout.md` (new `eval/evalset.py`, `tests/test_evalset.py`, the
  `experiments/retrieval/evalset/` files, the R2 dir) and `interfaces.md`
  (`eval.evalset` section; `eval.judge` new flags; `--question-ids-file`;
  `RubricSpec`).
- Optional (user's call): annotate PLAN.md's R2 section with the gate outcome and
  OQ1/OQ4/OQ6 pointers.
- Final gates: `.venv/bin/python -m pytest -m "not live" -q` green; `git status`
  shows only intended files; the frozen `experiments/retrieval_tuning/FINDINGS.md`
  untouched (`git diff --stat` proves it).

## 5. Cost budget

| Stage | Endpoint calls | Wall clock | Notes |
| --- | --- | --- | --- |
| 0, A1–A3, B, C, E2–E3, F | 0 | ~7–9 h offline total | A3 includes 1–2 h of *human* review time |
| A4 variant re-grade | ≤ ~20 judge calls | ~15–20 min | shared `:8080` |
| D1 smoke (2 q) | ~80–120 chunks ≈ 4–6 batches + 2 reform | ~10 min | |
| D2 build (80 unseen q) | ~3.3k–4.6k chunks ≈ 140–190 batches + ~80 reform | **~3.5–6.5 h sequential** | resumable; pinned 20 free |

Total endpoint: ~3.4k–4.8k judge completions + ~100 (or 0) reform calls, all on the
shared `:8080` at `--judge-batch 24`, sequential, interruptible at any point.

## 6. Risks

| # | Risk | Mitigation / contingency |
| --- | --- | --- |
| 1 | Judge fails the hand-check gate | Rubric-v2 branch is pre-built (A4); worst case: stop before any build spend, R3+ blocked, ADR escalation |
| 2 | `:8080` contention or crash mid-build | `--resume` + append-only cache + `EndpointCircuitBreaker`; per-question commits cap the loss at one batch |
| 3 | Structured-output truncation at batch 24 on full-prose prompts | A2's drop-audit tests run at full prose length; live drops → re-ask batch or `--judge-batch 12` for the remainder (recorded in `context.md`) |
| 4 | Dataset re-download changes question ids | Sample preflight + `select_by_ids` fail loudly; tracked sample file pins the ids |
| 5 | Server-side `presence_penalty 1.5` distorting long JSON lists | Recorded (D6) and *measured* (A2 audit + A3 human pass) rather than changed; mid-track change is forbidden — it would split the oracle |
| 6 | Variant verdicts polluting the cache | Impossible by sha keying; `load_cache` filters; test asserts |
| 7 | Overwriting the frozen `retrieval_tuning/FINDINGS.md` | Every command sets `--findings-path` explicitly (the TUNING.md precedent) |
| 8 | Snapshot drifting from the live cache | `evalset export --check` re-reads and asserts equality; snapshot is regenerated, never hand-edited |
| 9 | n=100 coverage < 1.0 (interrupted run leaves unjudged slots) | `--require-coverage 1.0` fails loudly; resume finishes the job |
| 10 | Rubric refactor accidentally changing the default sha | Test asserts `DEFAULT_RUBRIC.sha() == 21e27467277a` — the 926 verdicts' identity is guarded |

## 7. Rollback / revert story

- Everything except Stage D is offline and revertable by ordinary git means.
- The cache is append-only: a partial build is resumed or simply abandoned —
  verdicts under the current sha remain valid forever, by design.
- Stage D's outputs live under gitignored `outputs/`; deleting a run dir loses
  nothing the cache hasn't already kept.
- Snapshot, sample, and FINDINGS are new files; removing them restores the pre-R2
  tree. `SAMPLE_QUESTION_IDS`, config knobs, and the frozen
  `experiments/retrieval_tuning/FINDINGS.md` are untouched by this plan.
- If G1 fails *after* a build (only possible if the build ran out of order — it must
  not), the sha design contains the damage: a new default sha re-keys everything
  without deleting history.

## 8. Files touched

**Created**

- `src/medical_rag/eval/evalset.py` — sample builder + preflight, hand-check
  build/render/score, snapshot export/`--check`
- `scripts/evalset.py` — thin wrapper per tree convention
- `tests/test_evalset.py` — containment, preflight, snapshot↔cache consistency
- `tests/test_retrieval_tuning.py` — extended (no new file): leakage tests, drop
  audit tests, `RubricSpec` sha-preservation + variant-keying tests
- `experiments/retrieval/evalset/sample_v2.json` — the tracked n=100 sample
- `experiments/retrieval/evalset/verdicts_<sha>.jsonl` — oracle snapshot
- `experiments/retrieval/evalset/snapshot_metadata.json` — provenance record
- `experiments/retrieval/R2/FINDINGS.md` — the deliverable
- `experiments/retrieval/R2/hand_check.md` (+ filled CSV) — committed review artifact

**Modified**

- `src/medical_rag/eval/rubric.py` — `RubricSpec` refactor, default sha preserved
- `src/medical_rag/eval/judge.py` — rubric parameter, `--verify` / `--compact` /
  `--audit-run`
- `src/medical_rag/eval/harness.py` — `--question-ids-file` + preflight
- `scripts/eval_retrieval.py` — `--question-ids-file`
- `file_layout.md`, `interfaces.md`
- (optional, user's call) `PLAN.md` — R2 section annotation

## 9. Open questions — status after R2

- **OQ1 (judge trust)**: answered by A3/A5 — the plan produces the evidence and the
  recorded gate decision.
- **OQ4 (judge sampling)**: settled as *record-and-freeze* (D6); values in §2 #7 and
  in `snapshot_metadata.json`.
- **OQ6 (torn records)**: settled as append-only + verify + one post-build
  compaction (D5).
- **OQ2 (significance)**: untouched — R5's machinery on the enlarged sample.
- **OQ3 (lexical index burn)**: untouched (R4); noted that the R2 build grows the
  judge cache on demand, per current policy.
- **OQ5 (corpus drift note)**: deferred to R6 per PLAN.md.

## 10. Reviewer checklist

- [ ] D1–D8 confirmed or amended (edits in place are welcome before execution)
- [ ] Who performs the A3 human pass — by design, the reviewer; ~30 records, 1–2 h
- [ ] Approve the §C dry-run quote **after** G1 passes (the build's precondition)
- [ ] Confirm the smoke-then-build order for Stage D and the resumability rules