# Phase 5 — Closed-book smoke test findings

> **Status: COMPLETE as of run `20260928T020841Z` (§8): both arms, 40/40 rows,
> zero errors, zero truncations.** §1–§7 describe the first run
> (`20260927T162758Z`) and stand as the record of what broke: `model_b` would not
> terminate (2 of its 3 completions looped to the 16,384-token ceiling and
> returned `content_chars=0`) and its endpoint died mid-run, so 17 of its 20 rows
> never got a completion. The fix landed **endpoint-side, not in this repo** — a
> 4,096-token thinking budget plus `top_k=40` in `~/.omlx/model_settings.json`,
> on an oMLX server that now serves both arms — and it turned each of those
> 4.5-minute non-answers into a 9–58 s `finish_reason="stop"` with a parsable
> `ANSWER:` line. **§8 is the measurement Phase 6 and Phase 10 should cite**; the
> `model_b` column below is not a measurement of the model at all. Exploratory
> evidence, dev split only — not a result (see `../README.md`).

- **Run 1 (history, §1–§7) — date / git rev:** run `20260927T162758Z` @ `be8797b`.
  The harness itself was untracked when the run executed (`scripts/exploration/`,
  `tests/test_exploration_common.py` and this directory are all still `??` in
  `git status`), so `be8797b` does **not** identify the code that produced these
  rows. Commit them before the next run.
- **Run 1 — command:** `.venv/bin/python scripts/exploration/closed_book.py --sample-size 20`
  (default checks `main` + `seed`; both models; models sequential, not
  `--concurrent-models`)
- **Run 1 — raw artifacts:** `outputs/exploration/phase5/20260927T162758Z/`
  (`results.jsonl`, `context.md`, `seed_check.md`, 31 chains). The stdout
  transcript was written to `/tmp/phase5_full.log` and has been copied to
  `outputs/exploration/phase5/20260927T162758Z/run.log` — `/tmp` would not have
  survived the week.
- **Run 2 (the measurement, §8) — raw artifacts:**
  `outputs/exploration/phase5/20260928T020841Z/` (`results.jsonl`, `context.md`,
  40 chains, `run.log`). A **fresh dir rather than `--resume`**, because both the
  ceiling (16,384 → 8,096) and the endpoint changed between the runs; §8's
  `model_a` drift — 5 of 20 answers moved with nothing edited in this repo — is
  the evidence that the two runs' rows do not belong in one `results.jsonl`.
- **Sample — pin these ids, Phase 6 must reuse exactly them** (also in
  `context.md`; `--sample-seed` is the script default):
  `1312 2391 3998 4002 5209 5949 6205 6264 6727 6753 6771 7159 7502 7553 8966
  9293 9473 9572 9597 10064`. Gold letters C=8 A=5 D=4 B=3 — the C-heavy
  imbalance is the sample's, not a model's, and n=20 cannot carry a per-letter
  accuracy claim.

## 1. Question this phase asked

Three decisions, plus one artifact:

1. **Is each model's `max_new_tokens` right?** (Open questions item 8.) Turn
   "1,024 is probably enough / 16,384 is probably generous" into counts of
   `finish_reason="length"` and recovery firings.
2. **Does `seed` actually bite?** (Open questions item 10.) The 3-seed design and
   every CI built on it is fiction if the endpoint ignores `seed`.
3. **Do closed-book answers parse, and is either model biased toward one option
   letter?** A letter-biased model confounds every later delta.
4. **Artifact:** the pinned 20-question sample Phase 6 reuses, so "the same 20
   questions" is a fact rather than a hope.

## 2. Numbers (run 1 — the `model_b` column is not a measurement; see §8)

`n` counts rows in `tag="main"`. model_b's 17 "no completion" rows are the
endpoint dying (§7), not model behaviour; they are excluded from every
per-completion statistic below — but not from the `n` column, because a row that
never answered is exactly what an accuracy denominator must see.

| model | n | completions | correct | unanswered | `finish_reason="length"` | recovery fired / worked | median tok/s | median wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `model_a` (1,024 tok cap) | 20 | 20 | 10 (50%) | 0 | **0 / 20** | 0 / 0 | 73.2 | 5.3 |
| `model_b` (16,384 tok cap) | 20 | 3 | 0 | 2 (of 3) | **2 / 3** | 2 / **0** | 63.4 | 264.6 |

Completion tokens per question:

| model | min | median | max | cap | headroom at the max |
| --- | --- | --- | --- | --- | --- |
| `model_a` | 239 | 352 | 475 | 1,024 | 549 tokens (worst case used 46% of budget) |
| `model_b` | 626 | 16,896 | 16,896 | 16,384 | **none — two rows pinned at the cap** |

`model_b`'s 16,896 = 16,384 (main call) **+ 512 (recovery call)**: the recovery
call hit *its* cap too, and both rows carry `content_chars: 0` — after spending
16,896 tokens the model had produced **no answer text at all**, only chain of
thought (78,458 and 68,686 reasoning chars). See §3.

Answer-letter distribution of **errors**:

| model | A | B | C | D | unparsed |
| --- | --- | --- | --- | --- | --- |
| `model_a` (10 wrong) | 2 | 2 | 2 | 4 | 0 |
| `model_b` (3 wrong) | 0 | 0 | 0 | 1 | 2 |

`model_a`'s mild D lean (4/10 errors, against D being 4/20 of the gold) is not
readable at n=10 and needs re-measuring on Phase 9's pilot, where n is 50. The
`model_b` row is meaningless at 3 completions.

**Neither model's accuracy here is comparable to the other's.** `model_a`
completed 20/20, `model_b` 3/20. Do not carry "50% vs 0%" anywhere: the
difference is the run falling over, not capability (§7).

## 3. Failures worth reading

Chains copied to [`./chains/`](chains/) (31 files, 264 KB — every chain the run
produced, not just the failures; README rule 4).

- **`1312` — `model_b`, unanswered (gold `B`).**
  Chain: [`chains/model_b__1312.md`](chains/model_b__1312.md) — 78 KB, 550 lines.
  What to notice: the chain degenerates into **1,034 occurrences of the word
  "Alternatively"**, cycling between re-picking option A and option B:
  > "Alternatively, maybe the answer is A) Right upper quadrant (RUQ) ultrasound.
  > Because in cholangitis, especially in the upper abdomen, an ultrasound can
  > show signs of obstruction or stones. Alternatively, maybe ERCP is better.
  > Alternatively, maybe option B is more thorough. Alternatively, maybe option A
  > is sufficient. Alternatively, maybe option B is more thorough. Alternatively,
  > maybe optio"

  — and it stops there, mid-word, at the token cap. This is not long reasoning;
  it is a repetition attractor. The model reasoned coherently earlier in the
  chain and then lost the ability to commit.
- **`2391` — `model_b`, unanswered (gold `B`).**
  Chain: [`chains/model_b__2391.md`](chains/model_b__2391.md) — 69 KB, 349
  "Alternatively" plus 18 "Wait," restarts. Same attractor, same ending
  (`content_chars: 0`). Two of three `model_b` completions falling into the *same*
  degenerate loop is a mode, not a coincidence.
- **`3998` — `model_b`, wrong (gold `C`, answered `D`); `model_a` got this one
  right (answered `C`).**
  Chain: [`chains/model_b__3998.md`](chains/model_b__3998.md) — 2.4 KB, 626
  tokens, `finish_reason="stop"`, 9.9 s. The only substantive `model_b` answer in
  the run, worth reading for the contrast: when `model_b` works it is *brief*
  (2.4 KB vs. the loops' 69–78 KB) and it misses on clinical judgement, not on
  format. The pairing PLAN's Phase 6 asks for ("right closed-book, wrong once
  context arrived") cannot be built for `model_b` until its arm actually runs.
- **`1312` / `2391` — `model_a`, both wrong (gold `B`; answered `D` and `C`).**
  Chains [`chains/model_a__1312.md`](chains/model_a__1312.md),
  [`chains/model_a__2391.md`](chains/model_a__2391.md). Both models miss both
  questions, for opposite reasons: `model_a` commits fast to a plausible-but-wrong
  option in ~350 tokens; `model_b` fails to commit at all. Phase 12's taxonomy
  needs a slot for both — one that only has "wrong answer" files these two
  together and loses the distinction.
- **The `model_a` seed-check draws** (see §5 item 10) — the chains *are* the
  evidence. All five chains per question are byte-identical:

  | question | draws (main, `seed-same-1/2`, `seed-diff-1/2`) | body sha256 |
  | --- | --- | --- |
  | `1312` | 5 | all `df81fd97558ee6da…` |
  | `2391` | 5 | all `558f8580fb1dd303…` |

  (Hash over the chain body, i.e. the file minus its 2 header lines, so the
  differing seed stamp in the header does not hide behind an equal hash.)

## 4. Decisions this produced

| Decision | Evidence | Status / where it lands |
| --- | --- | --- |
| `model_a.max_new_tokens` stays **1,024** — do not raise it | 0/20 `finish_reason="length"`; worst case 475 tok = 46% of budget; median 352 | **Adopted.** Cited by the Phase 10 freeze; `config/models.yaml` comment updated to cite this file |
| `model_b.max_new_tokens` = 16,384 is **not** a validated ceiling | 2/3 completions pinned at the cap, both with `content_chars: 0` after 16,896 tok | **Adopted as a correction.** `config/models.yaml` claimed 16,384 was "the measured worst case at which it still terminates"; this run falsifies it. The comment now says unverified and points here. The *value* is unchanged — sizing is meaningless until item 13 is fixed |
| Raising `model_b`'s cap is **not** the fix | Both failures consumed 100% of a 16K budget while producing no answer; a 32K cap buys 4.5 more minutes of the same loop | **Adopted.** Do not "solve" this by editing the number |
| `seed` must not be trusted as a replicate on the `model_a` endpoint | 5/5 draws byte-identical (§3) | **Proposed → Phase 10.** Either drive repetition with `temperature` or report seed variance as "unsupported by this endpoint" |
| Phase 9 may adopt `_common.py`'s row shape | Pinning, `--resume`, per-row `params` snapshot and chain capture all worked under a real failure | **Adopted in spirit;** PLAN's Phase 9 already says "adopt `_common.py`'s row shape if it proved itself" — it did, except for error rows (§5 item 9) |
| Recovery must become loop-aware; harness needs a connection-error circuit breaker | Recovery 0/2, because it replays the loop (§7); 17 rows burned after the endpoint died | **Proposed → §7 punch list**, not yet code |

## 5. Open questions settled / moved

- **§6 item 8 (`max_new_tokens` sizing): partly settled.** `model_a`: settled at
  1,024 (0/20 length, 549 tokens of headroom at the worst case). `model_b`:
  **still open**, and it cannot be closed by measuring more — two of three
  completions ate the entire budget without answering, so no ceiling is large
  enough for a model that does not terminate. Blocked on new item 13 (§7); the
  script's `--caps` contingency pass is the tool that *will* close it once the
  loops stop.
- **§6 item 10 (does `seed` bite): settled for `model_a` — it does not.** Same
  prompt, `temperature=0.7` > 0: two draws at `seed=0` and two at `seed=100/101`
  produced byte-identical 375- and 396-token outputs on both questions (5 draws,
  2 distinct hashes). The seed dimension currently contributes **zero** variance
  on that endpoint. `model_b`: **not measured** — its 8 seed-check draws never
  ran because the endpoint died; re-run with `--checks seed`.
  Lead on *why* (not confirmed): `mlx_lm.server` takes the per-request `seed`
  (`server.py:1191`, threaded in at `:1404`) and only seeds the global MLX PRNG
  in the non-batched path (`mx.random.seed(args.seed)` at `:956-957`), while
  generation runs on a separate `generation_stream` (`:690-691`) — MLX PRNG state
  is per-stream. One two-request probe against a freshly started server
  (`scripts/probe_models.py --checks latency --seed 0` vs `--seed 1`) would
  confirm or kill this; either way the study conclusion for `model_a` stands.
- **§6 item 9 (`QuestionResult` snapshots active params): new counterexample.**
  `README.md` rule 5 says a row must be self-describing. The 17 `model_b` error
  rows violate it: `RunWriter.error_row()` emits 13 fields and drops `params`,
  `seed`, `finish_reason`, prompt/completion tokens and prompt size, so an error
  row cannot be told apart from an error row produced by different settings.
  Phase 9 should carry `params`/`seed` into error rows as well as result rows.
- **§6 item 12 (concurrency / timeouts in the runner): concrete motivation
  recorded.** A dead endpoint cost one 224 s hung request plus 16 refusals in 7 s
  (§7). Whatever semaphore Phase 9 builds also needs to stop a run from politely
  walking its remaining queue after the endpoint is gone.

## 6. Cost of this run

Wall clock **15 min 24 s** (16:27:58 → 16:43:22 UTC). Summed per-row wall is
924 s; `model_b` accounts for 771 s of it (84%) for **3 answers**.

| arm | completions | completion tokens | prompt tokens | logged wall |
| --- | --- | --- | --- | --- |
| `model_a` (20 main + 8 seed-check draws) | 28 | 10,049 | 8,321 | 148 s |
| `model_b` (3 main + 2 recovery calls; 17 errored) | 5 | 34,418 | 823 | 771 s |

**33,792 tokens — 98% of everything `model_b` spent — went into two loops that
produced no answer.** (2 rows × 16,896 tokens, the 16,384 main call plus its 512
token rescue.) That is the invoice for item 13 still being open.

Throughput: `model_a` median 73.2 tok/s (range 29.5–76.3), `model_b` 63.4 — the
"~60–78 tok/s" figure `config/models.yaml:23` assumes when it sizes `timeout`
held for both arms. What it does *not* cover is termination: at the observed loop
rate a single question costs 4.4 min and 16.4K tokens, so the Phase 13 estimate
(1,273 questions × conditions × 3 seeds) is unbounded until item 13 is fixed.
Plan Phase 13's cost only from a post-fix Phase 5.

## 7. What blocked the run, and what a completing run needs

> **Status (2026-09-28): every item below landed, and run 2 (§8) proves it —
> 40/40 rows, 0 errors.** `completed_keys()` now ignores error rows, `error_row()`
> carries `params` plus the cost fields, and `EndpointCircuitBreaker` abandons an
> arm after 3 consecutive transport failures; the `model_b` non-termination
> blocker was cleared **endpoint-side**, not in this repo. One item did *not*
> land: loop-aware recovery. With 0/40 recoveries firing it stayed unreachable, so
> `_recover_answer` replaying a loop to itself stands as an open hazard (§8.4),
> not a fix. The procedure below is kept verbatim as the record; its
> `mlx_lm.server` command and its "strip the error rows" workaround are both
> obsolete — the workaround is now what `completed_keys()` does unprompted, so a
> plain `--resume` needs no surgery.


Timeline, reconstructed from `results.jsonl` timestamps (each stamped at
completion) and `run.log`:

| UTC | event |
| --- | --- |
| 16:28:03–16:30:26 | `model_a` arm: 28 calls, all fine, 148 s |
| 16:34:56 | `model_b`/`1312` completes at the cap (270 s, loop) |
| 16:39:21 | `model_b`/`2391` completes at the cap (265 s, loop) |
| 16:39:31 | `model_b`/`3998` completes normally (9.9 s) |
| 16:43:15 | `model_b`/`4002` fails after **224 s hung mid-generation** — the `model_b` server on `:8082` died underneath an in-flight request |
| 16:43:15–16:43:22 | remaining **16 questions fail in 7 s total** (0.4–0.5 s each, connection refused) |
| 16:43:22 | run exits `1`. No traceback, no user cancel — it finished walking the queue |

Three separate defects, only one of which is the model's:

1. **`model_b` non-termination (→ new PLAN Open questions item 13). The
   blocker.** Repetition attractor: 2/3 completions looped on "Alternatively"
   until the cap and emitted no answer text. `model_b` runs at its own published
   `temperature=0.6`/`top_p=0.95`, so this is not a bad-parameter artefact of our
   config. Candidate levers, checked against the server actually serving this
   model (`mlx_lm/server.py:1174-1192`): it accepts **per-request
   `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `top_k`,
   `min_p`, `xtc_probability`, `stop`** — and `LLMClient._complete()` currently
   sends only `max_completion_tokens`, `temperature`, `top_p`, `seed`. Server-side
   `repetition_penalty` defaults to **0.0 (off)**. The fix work is therefore an
   `extra_body` passthrough (same shape as the thinking-mode kwarg already used)
   plus a measured sweep — which ceiling/penalty combination makes `model_b`
   commit — not a config number edit. Phase 4 left this passthrough as a named
   follow-up; item 13 is where it gets paid.
2. **`_recover_answer` makes looping failures unrecoverable.** `llm.py:297-328`
   replays the abandoned chain as an assistant turn ("the rescue continues the
   chain it abandoned"), which for a loop means feeding 78 KB of "Alternatively"
   back to the model as its own voice and giving it 512 more tokens. It worked in
   Phase 4 because those chains were *budget-starved*, not degenerate; here it
   went **0/2** and cost 1,024 extra tokens. Needs to recognise a loop (e.g.
   `content_chars == 0` *and* repetitive tail — cheap n-gram check) and either
   truncate to a question-only re-prompt or mark the row unrecoverable and stop
   paying for it.
3. **Nothing stops a run when an endpoint dies.** The client converted the death
   into a per-row `LLMError` and the script dutifully continued for 16 more rows.
   Cost: one 224 s hang + 7 s of refusals + 17 rows that look like model failures
   in the summary. Needs a circuit breaker — N consecutive `APIConnectionError`s
   ⇒ abort that model arm (non-zero exit, clear message) instead of draining the
   queue — and `error_row()` must keep `params`/`seed`/`finish_reason` (§5 item
   9) so an aborted row is still self-describing.

**How to finish Phase 5** (after the `model_b` fix, in this order):

```sh
# 0. bring :8082 back with its output captured, or the next death is invisible
mlx_lm.server --model /Volumes/Data/mlx/DeepSeek-R1-Distill-Qwen-7B-8bit \
  --port 8082 > outputs/model_b.log 2>&1 &
.venv/bin/python scripts/probe_models.py            # pre-flight; exits non-zero if down

# 1a. TRAP: plain `--resume` would skip the 17 questions that never ran.
#     RunWriter.completed_keys() (_common.py:217-220) keys on
#     (model, question_id, tag) and never checks that the row has a completion,
#     so the 17 connection-error rows count as "done". Strip them first:
.venv/bin/python - <<'PY'
import json, pathlib
p = pathlib.Path("outputs/exploration/phase5/20260927T162758Z/results.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
keep = [r for r in rows if "error" not in r]
p.with_suffix(".jsonl.bak").write_text(p.read_text())
p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
print(f"kept {len(keep)} of {len(rows)} rows; dropped {len(rows) - len(keep)} error rows")
PY

# 1b. resume the same dir — now fills exactly the 17 missing model_b main rows
#     and its 8 seed-check draws, and leaves model_a's 28 rows untouched
.venv/bin/python scripts/exploration/closed_book.py --sample-size 20 \
  --resume outputs/exploration/phase5/20260927T162758Z

# 2. then close item 8 with the contingency pass this script already has
.venv/bin/python scripts/exploration/closed_book.py --checks caps \
  --caps 1024 2048 4096 --cap-questions 5 --models model_b
```

The completed-key check is worth fixing properly in `_common.py` (treat a row as
done only if it has a `finish_reason`, or re-run rows carrying `error`) — it is a
one-line change and Phase 9's runner needs the same rule, since Phase 13's
resumability claim in PLAN is otherwise quietly wrong in the same way.

If the fix changes `model_b`'s sampling params, prefer a **fresh** run dir over
`--resume` so the old looping rows don't sit in the same `results.jsonl` as the
fixed ones — and re-note in this file which run dir is the one Phase 10 cites.
Either way, then: flip this file's status banner to "complete", fill `model_b`'s
row in §2, and mark items 8/13 settled (item 10 is settled below, without
`model_b`).

## Resolution (2026-09-27) — `seed` removed from the pipeline

This file's `model_a` seed-check result (§3/§5: 5 byte-identical draws across
`seed=0/100/101`) settled that `seed` is inert on that endpoint. Rather than
chase the same measurement on `model_b` — whose endpoint died before its half
of the seed check ran, and non-termination (item 13) would confound the result
anyway — the project dropped the repeated-seed-draws design outright: `seed`
is no longer a field on `ModelConfig`, `ConditionConfig`, `LLMClient`, or
`QuestionResult`, and the experiment now runs one completion per question per
condition instead of three per seed (see `PLAN.md`, open question 10). The
`main` findings above and this file's raw artifacts are unaffected and remain
the historical record; only the seed-check follow-up work is now moot.

## 8. Run 2 (`20260928T020841Z`) — the completed measurement

Both arms ran to completion: **40/40 rows, 0 errors, 0 `finish_reason="length"`,
0 recoveries fired, 9.3 min of GPU wall for 40 completions (37,527 completion
tokens).** Phase 5's question is now answered for both models.

### 8.1 What changed between the runs — one of the three is not in this repo

1. **The backend moved, and quietly.** Both arms now point at one **oMLX** process
   on `:8080`; `GET /v1/models` advertises **bare ids** (`Qwen2.5-7B-Instruct-8bit`,
   `DeepSeek-R1-Distill-Qwen-7B-8bit`). `models.yaml` still carried
   `/Volumes/Data/mlx/...` paths, which oMLX answers with `HTTP 404 Model ... not
   found` — and the harness's own preflight caught it before the first request
   fired, which is the only reason this cost seconds instead of 40 error rows. The
   two `mlx_lm.server` instances on `:8081`/`:8082` that `environment.md` still
   describes are gone; `models.yaml`'s header now says so. One process serving
   both arms also means the arms are no longer independent endpoints, so
   `--concurrent-models` would confound per-arm tok/s — they ran sequentially.
2. **The non-termination fix is endpoint-side.** `~/.omlx/model_settings.json`
   for `DeepSeek-R1-Distill-Qwen-7B-8bit` now carries
   `thinking_budget_enabled: true`, `thinking_budget_tokens: 4096`,
   `enable_thinking: true`, `top_k: 40`. `LLMClient` sends only
   `max_completion_tokens`/`temperature`/`top_p`, so **nothing in git produced
   this fix** — it is recorded in the run's `context.md` via the new `--note`
   channel. Consequence for item 13(a): the `extra_body` passthrough is *not*
   needed to make `model_b` terminate. It stays worth doing the first time a
   phase needs to *vary* one of those knobs per condition, which Phase 6+ does
   not currently plan to.
3. **In this repo:** `api_model_name` paths → bare ids for both models,
   `model_b`'s ceiling 16,384 → 8,096, and §7's three harness fixes
   (`completed_keys()` no longer counts error rows, `error_row()` carries
   `params`/`question`/cost fields, `EndpointCircuitBreaker` aborts an arm after
   3 consecutive transport failures) — plus `--note`. 72 tests pass, 8 of them new.
   **Provenance caveat, the same one §"Run 1" reproaches run 1 for:** run 2 executed
   at HEAD `a9fb100` *plus an uncommitted working tree* containing everything in
   this numbered list, so no rev identifies the code that produced these rows.
   Commit before Phase 6 runs, or the ambiguity recurs a second time.

### 8.2 Numbers (n=20 per arm, `tag="main"`)

| model | n | completions | correct | unanswered | `finish=length` | recovery fired/worked | median tok/s | median wall s | tokens median/max (cap) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `model_a` (1,024) | 20 | 20 | **14 (70%)**, Wilson 95% [48%, 85%] | 0 | 0 | 0 / 0 | 72.7 | 5.2 | 377 / 512 — 50% of cap |
| `model_b` (8,096) | 20 | 20 | **5 (25%)**, Wilson 95% [11%, 47%] | 0 | 0 | 0 / 0 | 73.5 | 19.1 | 1,327 / 4,237 — 52% of cap |

Wrong-answer letters: `model_a` D×2 C×1 B×2 A×1 (n=6); `model_b` C×5 B×5 D×3 A×2
(n=15). Against gold C×8 A×5 D×4 B×3 — no letter signal is extractable from these
at n=20, and neither arm produced an unparsable row.

**The loop is gone, measured on the two questions that produced it:**

| question | run 1 | run 2 |
| --- | --- | --- |
| q1312 | `length`, 16,896 tok, `content_chars=0`, "Alternatively"×1,034, 264 s | `stop`, 1,091 tok, "Alternatively"×3, `ANSWER: C`, 14.6 s |
| q2391 | `length`, 16,896 tok, `content_chars=0`, "Alternatively"×349 | `stop`, 1,954 tok, "Alternatively"×2, `ANSWER: C`, 26.3 s |

Worst chain across all 20 run-2 `model_b` completions: "Alternatively"×20 at
14 KB (`q10064`) vs ×1,034 at 76 KB before. Both questions above are *wrong*
(gold B) — that is now legitimate Phase 5 evidence rather than plumbing failure.

### 8.3 What this settles

* **Item 8 closes for both arms**, with headroom measured on real MedQA
  questions: the worst case used 50% (`model_a`) and 52% (`model_b`) of its
  ceiling, 0/40 truncated. The `--caps` sweep is no longer needed to justify the
  ceilings at this n. The sizing *rule* changed in kind, though: `model_b`'s
  ceiling is now structurally ~2× the endpoint's 4,096-token thinking budget, so
  **the budget — not `max_new_tokens` — is what bounds a chain.**
* **Item 13 closes** on termination, and by a decoding constraint rather than a
  bigger ceiling — exactly the shape the item guessed at ("raising
  `max_new_tokens` is not the fix"). 264 s of non-answer per question became
  14.6–26.3 s of answer.
* **Item 12's circuit-breaker half and the `--resume` trap are fixed**, with tests
  that fail if either regresses. Phase 9's resumability claim inherits the
  corrected `completed_keys()` rule rather than run 1's broken one.

### 8.4 What this does not settle

* **`model_a`'s 70% is not an improvement over run 1's 50%.** Same pinned 20
  questions, same `temperature`/`top_p` in `models.yaml`, same code path — and yet
  **5/20 answers moved (q10064, 5949, 6205, 6727, 9473) and 0/20 outputs were
  byte-identical.** What changed is the endpoint's per-model sampling
  (`top_k=20`, `top_p=0.8`, `repetition_penalty=1.05`), which lives outside git.
  Two consequences: the two runs' rows must not be pooled, and **Phase 10's freeze
  is not reproducible unless the endpoint's per-model settings are recorded next to
  the repo's `params`.** That is item 9's argument, upgraded from "a row's params
  may be incomplete" to "parameters no artifact in this repo names moved accuracy
  by 20 points on n=20."
* **The budget's tail is unmeasured, and Phase 13 will hit it.** q9572 reached
  4,237 tokens (~3,900 of it reasoning) — within 5% of the 4,096 budget — and its
  chain still concludes coherently ("In conclusion, despite the normal amylase…"),
  so nothing was cut off at n=20. At n=1,273 the budget *will* bind, and what
  happens then — answer forced, chain truncated, or loop reinstated — is unknown.
  Before Phase 13, either re-run the pinned 20 with the budget raised and compare,
  or adopt the budget as part of the frozen definition of `model_b` and say so.
* **Item 13(b) — the recovery path feeding a loop back to itself — is unexercised,
  not fixed.** Recovery fired 0/40 times, because nothing came back unanswered, so
  `_recover_answer`'s loop-replay hazard (`llm.py:297-328`) is still in the code
  and still untested against a genuine loop. The endpoint fix routed around it; it
  did not remove it.
* **`model_b` 25% vs `model_a` 70%** is the first plumbing-clean contrast between
  the arms, and at n=20 the intervals ([11%, 47%] vs [48%, 85%]) only just clear
  each other. One hypothesis worth a cheap test before it hardens into an
  assumption: forcing a commitment at 4,096 reasoning tokens may be *what costs*
  the reasoning model, since its chains show it deliberating past the point of
  sufficiency ("I'm going to have to make a decision… Alternatively, maybe the
  answer is D"). Not a finding — a 20-question A/B against a larger budget would
  settle it.





