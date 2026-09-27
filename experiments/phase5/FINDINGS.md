# Phase 5 — Closed-book smoke test findings

> **Status: PARTIAL. Phase 5 is not complete and this is not the phase's final
> word.** The harness (`scripts/exploration/closed_book.py`) works — sample
> pinning, per-row param snapshots, chain capture, `--resume`, and the seed check
> all behaved. What did not work is the thing the phase exists to measure:
> `model_b` does not reliably reach an answer inside its budget, and its endpoint
> died mid-run, so 17 of its 20 rows never got a completion. `model_a`'s half is
> measured and reported here; `model_b`'s is not. §7 is the punch list a
> completing run needs. Exploratory evidence, dev split only — not a result (see
> `../README.md`).

- **Date / git rev:** run `20260927T162758Z` @ `be8797b`. The harness itself was
  untracked when the run executed (`scripts/exploration/`,
  `tests/test_exploration_common.py` and this directory are all still `??` in
  `git status`), so `be8797b` does **not** identify the code that produced these
  rows. Commit them before the next run.
- **Command:** `.venv/bin/python scripts/exploration/closed_book.py --sample-size 20`
  (default checks `main` + `seed`; both models; models sequential, not
  `--concurrent-models`)
- **Raw artifacts:** `outputs/exploration/phase5/20260927T162758Z/`
  (`results.jsonl`, `context.md`, `seed_check.md`, 31 chains). The stdout
  transcript was written to `/tmp/phase5_full.log` and has been copied to
  `outputs/exploration/phase5/20260927T162758Z/run.log` — `/tmp` would not have
  survived the week.
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

## 2. Numbers

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
row in §2, and mark items 8/10/13 settled.




