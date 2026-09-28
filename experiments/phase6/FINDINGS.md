# Phase 6 — Raw RAG end-to-end findings

> **Status: COMPLETE as of run `20260928T143811Z` — 40/40 rows, 0 errors, 0
> unanswered, 0 `finish_reason="length"`, both arms paired to Phase 5 on 20/20
> questions each, and both arms verified to have seen byte-identical context per
> question.** The pooled delta is *not* the finding: retrieval moved the two arms
> in **opposite directions** — `model_a` 14/20 → 10/20, `model_b` 5/20 → 7/20 —
> so any pooled accuracy number here cancels a real effect against its antonym.
> The other finding is that at `top_k=5` the **gold option's wording was in the
> prompt for only 8/40 rows**, which means most of the 40 rows are not
> "model vs context" comparisons at all, and Phase 7 must measure retrieval
> recall before it sweeps prompts. Exploratory evidence, dev split only — not a
> result (see `../README.md`).

- **Date / git rev:** run `20260928T143811Z` @ `6079c13`, **dirty working tree**:
  `raw_rag.py`, `tests/test_raw_rag.py`, `audit_run.py` and this directory were
  still untracked when the run executed, and `_common.py` / `closed_book.py` /
  `PLAN.md` were modified. `6079c13` therefore identifies *around* the code, not
  the code — the provenance caveat `phase5/FINDINGS.md` reproaches run 1 for,
  recurring once. Commit before Phase 7.
- **Command:** `.venv/bin/python -u scripts/exploration/raw_rag.py` (defaults:
  20 pinned questions, `--baseline-dir outputs/exploration/phase5/20260928T020841Z`,
  models sequential — see §5 for why not `--concurrent-models`)
- **Raw artifacts:** `outputs/exploration/phase6/20260928T143811Z/` —
  `results.jsonl` (40 rows), `pairing.md`, `prompts.md` (the exact string sent,
  per row), `context.md`, `chains/` (40), and `run.log` (the stdout transcript,
  copied out of `/tmp` for the reason `phase5/FINDINGS.md` gives: `/tmp` does not
  survive the week).
- **Audit:** `scripts/exploration/audit_run.py` recomputes every number quoted
  below from the artifacts (`.venv/bin/python scripts/exploration/audit_run.py
  outputs/exploration/phase6/20260928T143811Z`) — **13 PASS, 0 FAIL** at the time
  of writing. Read it before quoting anything here.
- **Sample — the same 20 ids Phase 5 pinned** (`1312 2391 3998 4002 5209 5949
  6205 6264 6727 6753 6771 7159 7502 7553 8966 9293 9473 9572 9597 10064`;
  gold C×8 A×5 D×4 B×3). The run re-derives them with
  `select_sample(pool=10178, size=20, seed=1)` and **fails the preflight** if the
  derived set is not exactly this list — `run.log`: *"pin OK: select_sample(pool=10178,
  size=20, seed=1) still returns exactly these ids"*. That check is the difference
  between "the same 20 questions" being a fact and being a hope.

## 1. Question this phase asked

Phase 5 established what each model does with nothing but the question. This
phase asks what changes when the same model is handed five StatPearls excerpts for
the same questions, and what that costs:

1. **Does the pairing actually hold?** Same ids, same questions, and — the part a
   test cannot check — *the same context in both arms*, so a later "condition B
   helped" claim is not secretly "condition B saw different text".
2. **Does retrieval move accuracy, and in which direction per arm?** Reported as
   transitions (`rescued` / `distracted` / `lost`), never as one pooled number.
3. **What does context cost** — prompt tokens, wall clock, chain length, and
   whether Phase 5's token ceilings still hold once ~800 tokens of excerpts are
   bolted onto the front.
4. **Is the retrieval stage doing anything at all** — does the reranker reorder,
   and is the answer's wording even present in what gets retrieved?

## 2. Numbers (n=20 per arm, `tag="main"`)

The closed-book column is `phase5/FINDINGS.md` §8.2 re-read from that run's
`results.jsonl` by the pairing, so both columns are the same 20 questions and the
same endpoint.

| arm | model | closed-book | + RAG | rescued | distracted | lost | both right | both wrong | `finish=length` | recovery | median tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `model_a` | `Qwen2.5-7B-Instruct-8bit` | **14/20 (70%)** | **10/20 (50%)** | 1 | 5 | 0 | 9 | 5 | 0/20 | 0/20 | 58.9 |
| `model_b` | `DeepSeek-R1-Distill-Qwen-7B-8bit` | **5/20 (25%)** | **7/20 (35%)** | 3 | 1 | 0 | 4 | 12 | 0/20 | 0/20 | 64.4 |

Pooled 19/40 → 17/40. The audit asserts the identity that makes those columns
consistent — `closed_right − (distracted + lost) + rescued == rag_right`
(19 − 6 + 4 = 17) — and that `both_wrong` / `both_right` are *not* counted as
RAG-caused loss. `lost` is 0 in both arms: RAG never turned a wrong answer into a
differently-wrong one, so every wrong row was either already wrong closed-book
(17) or was a flip from right (6).

At n=20 per arm the Wilson intervals overlap heavily (Phase 5: `model_a`
[48%, 85%], `model_b` [11%, 47%]). **Nothing in this table is an accuracy
estimate.** What it is: 10 of 40 rows changed letter, they split 4 right / 6
wrong, and *which arm they split in* is the signal — the arm that knew the
material lost ground to the context, the arm that did not gained some. Phase 7
must report transitions per arm; a pooled accuracy number here is an artifact of
adding two opposite effects.

Wrong-answer letters (gold is C×8 A×5 D×4 B×3): `model_a` A×2 B×2 C×3 D×3
(n=10), `model_b` A×2 B×4 C×2 D×5 (n=13). `model_b`'s lean toward D is the lean
Phase 5 already measured (D×3 of 15 then); no arm collapsed onto one letter, and
no row was unparsable in either arm.

### 2.1 What the five excerpts did to the request

Paired per (model, question) against the same model's Phase 5 row — same
endpoint, same day, both runs sequential (`model_b`'s `reasoning_chars` is the
thinking block; `model_a` does not emit one):

| arm | prompt tokens (median) | completion tokens (median) | reasoning chars (median) | median wall | max wall |
| --- | --- | --- | --- | --- | --- |
| `model_a` | 285 → **1,094** (+822) | 378 → 342 | — | 5.2s → 6.1s | 12.0s |
| `model_b` | 278 → **1,088** (+822) | 1,328 → **830** | 5,182 → 3,562 | 19.1s → **13.4s** | 58.2s |

Both "+822" figures are the median of the *per-question paired* deltas (same model,
same question, ± context), not the difference of the two medians — and they hide a
2.3× spread: the excerpts added between **518 and 1,215** prompt tokens depending on
the question, because chunk length varies. The completion medians are the audit's
`statistics.median` of 20 rows, which lands on `.5` for both arms (377.5, 1,327.5);
Phase 5 §8.2's table truncates the same medians to 377 and 1,327. Same data, two
roundings — the audit is the one to cite.

**Context made `model_b` shorter and faster, not longer.** Handed five excerpts,
the reasoning model stopped auditing its own memory and answered in 62% of the
tokens at 70% of the wall time; `model_a`, which barely reasons, just paid the
prefill and got a little slower. 0/40 rows hit `finish_reason="length"` with ~800
tokens of excerpts bolted on, so **Phase 5 §8.3's ceilings survive the context
bump for both arms** — item 8 stays closed.

Retrieval is free at this scale: **median 0.27 s/question** (0.213 s vector
search over 380,454 chunks on MPS + 0.043 s cross-encoder over the 20 candidates),
**7.0 s across the whole run**. The 1.74 s worst question is the *reranker's* cold
start (1.44 s), not the vector search, which never exceeded 0.293 s.
`prompt_tokens` ranged 820–1,710 across all 40 rows.

### 2.2 Whether the retrieval stage is earning its keep

| measurement | value | reading |
| --- | --- | --- |
| Index / embedder / reranker | 380,454 chunks; `bge-small-en-v1.5`; `ms-marco-MiniLM-L-6-v2`; retrieve 20 → keep 5 | config defaults, untouched |
| Gold option's wording present in the 5 excerpts | **8/40 rows (4/20 questions)**; ranks {1:×2, 2:×2, 4:×2, 5:×2} | three-quarters of the rows never had the answer *named* in their context — they are not tests of a model reading context |
| Chosen answer's wording absent from the context | **32/40 (80%)** | including correct answers: these models answer from understanding, not string-matching, so a name-match verifier would reject most *correct* rows too |
| Reranker promoted a chunk over its vector rank | **154/200 chunks** | the cross-encoder does most of the ordering |
| Pre-rerank cosine span (top1 − last of the 20) < 0.05 | **34/40 rows** | first-stage scores are nearly flat on this corpus — which is exactly *why* the reranker matters, and why `vector_top1` values (0.749–0.849) must not be read as confidence |
| Both arms saw identical `retrieved_chunk_ids` per question | **20/20 questions, PASS** | the shared-retrieval invariant is measured, not assumed |

## 3. Failures worth reading

All eight chains below are copied into [`chains/`](chains/). The headline that the
accuracy table cannot carry: **3 of the 4 `rescued` rows land on the right letter
with wrong or self-contradicting reasoning, and 2 of the 6 `distracted` rows
explicitly cite the excerpts for a claim the excerpts never make (both of them
`q4002`).** Accuracy counts the letter; these chains say what the letter was
worth.

- **`q4002` — `model_a`, distracted (gold `B`, closed-book `B`, answered `A`
  "pigmented casts").** Chain:
  [`chains/model_a__4002.md`](chains/model_a__4002.md). It writes *"The clinical
  reference excerpts mention that emphysematous cystitis is associated with gas
  formation in the bladder wall"* — true — and then *"the most likely finding on
  urinalysis would be pigmented casts. These are associated with conditions
  involving gas formation or hematuria"* — invented. The excerpts name neither the
  gold option nor the chosen one (`answer_chunk_rank=-`, `gold_chunk_rank=-`), so
  the context supplied a *topic to bridge from*, not an answer.

- **`q4002` — `model_b`, distracted (gold `B`, closed-book `B`, answered `D`
  "waxy casts").** Chain: [`chains/model_b__4002.md`](chains/model_b__4002.md).
  Same five chunks as `model_a` (parity verified), **different wrong letter.**
  *"In the urinalysis, emphysematous cystitis is associated with waxy casts …
  Pseudomonas can produce slimy mucus that leads to these casts."* Nothing in the
  context says that. The chain also misreads every safety-critical number it
  touches — Hb 14.0 "quite low", K⁺ 5.1 "very low", FeNa 2.1% "really low" — so
  the excerpt did not correct a misread, it gave the misread somewhere to go.
  **This pair is the phase's cleanest result: identical context, two independent
  confabulations. The distraction is not in the retrieved text; it is in what each
  model does with an authoritative-looking near-miss.**

- **`q9572` — `model_a`, distracted (gold `C` gallstone disease, closed-book `C`,
  answered `A` acalculous cholecystitis).** Chain:
  [`chains/model_a__9572.md`](chains/model_a__9572.md). The *only* distractor
  where the chosen option's wording is actually in the context
  (`answer_chunk_rank=2`). The model rules out the gold answer because *"the
  absence of visualized gallstones on ultrasound … suggests that this is not the
  primary cause"*, missing that the stem says the ultrasound failed **because of
  her body habitus**. Closed-book it picked the right letter; with a StatPearls
  excerpt on acalculous cholecystitis ranked #2 in front of it, it talked itself
  out of the right answer. This is the case a "answer only from the excerpts"
  instruction is meant to catch — Phase 7's to test.

- **`q10064` — `model_a`, distracted (gold `A` optic glioma, closed-book `A`,
  answered `D` giant cell astrocytoma).** Chain:
  [`chains/model_a__10064.md`](chains/model_a__10064.md). Freckling in the skin
  folds + bilateral iris nodules + multiple soft painless nodules is NF1; the chain
  concludes *"characteristic of tuberous sclerosis complex"* and drops the optic
  finding it had itself listed. Same syndrome-mislabel as Phase 5's closed-book
  errors — context neither caused nor fixed it, it just re-rolled the same defect.

- **`q5209` — `model_a`, rescued (gold `A` normal cerebrum, closed-book `B`).**
  Chain: [`chains/model_a__5209.md`](chains/model_a__5209.md). Read the last
  paragraph: *"Normal cerebrum (A) is unlikely given the patient's symptoms"*,
  then *"brain imaging is most likely to show no specific atrophy"*, then
  **`ANSWER: A`**. Right letter, reasoning that rejects it. The only `model_a`
  rescue in the run is not evidence of retrieval comprehension.

- **`q9473` — `model_b`, rescued (gold `C` PPD before anti-TNF).** Chain:
  [`chains/model_b__9473.md`](chains/model_b__9473.md). Diagnoses "sacroiliac
  joint dysfunction" instead of ankylosing spondylitis, spends four paragraphs
  looking for a FABER test among the options, and settles on C because *"the PPD
  test is for TB, which is a possible cause of chronic pain."* The real rationale
  (screen latent TB before a biologic) never appears. **A rationale-reading judge
  should have failed this row; a letter-checking metric scores it as a win.**
- **`q7553` / `q7159` — `model_b`, rescued.** Chains:
  [`chains/model_b__7553.md`](chains/model_b__7553.md),
  [`chains/model_b__7159.md`](chains/model_b__7159.md). `q7553` is the one chain
  in the run that shows the mechanism Phase 7 is betting on: it walks the options
  under a heading *"Looking at the clinical references"* and cites excerpt content
  per option (still misreading Hb 16.3 as "low but not anemic" and calling topical
  nifedipine "a local anesthetic" — reasoning defects no retrieval fixes). That is
  1 of 4 rescues with context actually used as evidence.

The two remaining `distracted` rows, `q5949` (gold `D`, closed-book `D`, answered
`B`) and `q9597` (gold `A`, closed-book `A`, answered `C`), are `model_a` flips
with `answer_chunk_rank=-` — the model named an alternative no excerpt mentions,
having been told to weigh excerpts that did not cover it. Their chains stay in
`outputs/exploration/phase6/20260928T143811Z/chains/` rather than being copied
here: they show the same shape as `q4002`'s `model_a` row without adding detail.

## 4. Decisions this produced

Stated as commits-to-come, so the Phase 10 freeze can cite them.

| Decision | Evidence | Where it lands |
| --- | --- | --- |
| Report **per-arm transitions**, never pooled RAG-vs-closed-book accuracy | `model_a` −4 rows, `model_b` +2 rows, pooled −2 rows: pooling adds two opposite effects and calls the remainder a finding | Phase 7+ reporting convention; `raw_rag.py` already prints per-arm tables and `pairing.md` |
| **Phase 7 opens with a retrieval-recall measurement**, not a prompt sweep: gold option wording reached the top-5 for 4/20 questions | 8/40 rows with gold in context; three quarters of this run's rows never tested context reading | first condition of Phase 7 — it is an offline recount of the `chunks` already in `results.jsonl`, at k=5/10/20 |
| **Keep the cross-encoder**, keep recording `vector_score` *and* `rerank_score` per chunk | 154/200 chunks promoted by the reranker; pre-rerank cosine span < 0.05 on 34/40 rows, so the first stage cannot order these on its own | retriever config at the freeze; both scores already per-row |
| **Token ceilings unchanged** — Phase 5 item 8 stays closed | 0/40 `finish_reason="length"` with +822 prompt tokens; `model_b`'s median completion *fell* 1,328 → 830 | `config/models.yaml` at the freeze |
| **The Phase 8 judge reads the rationale, not the letter** | 3 of 4 rescues argue against or outside their own answer (`q5209` calls its answer "unlikely" then emits it; `q9473` invents a PPD rationale) | Phase 8 verifier design |
| Add an **"answer only from the excerpts, say which one"** condition | `q9572` (context ranked #2 argued it out of a correct answer) and the `q4002` pair (identical context, two invented bridges) | Phase 7 condition list |
| **`audit_run.py` gates every phase's numbers** before they leave the repo | Its first drafts caught two real bugs in itself — comparing `base_answer` across arms, and counting `both_wrong` as RAG loss — either of which would have published a wrong reconciliation | `experiments/README.md` rule + this script |

## 5. Open questions settled / moved

- **Pairing integrity: settled.** The run re-derives the pinned sample and aborts
  if it drifts (*"pin OK"*, `run.log`), and the audit re-verifies per question that
  both arms received the same `retrieved_chunk_ids`. Note the invariant is
  *context* parity only — the two arms' closed-book answers legitimately differ,
  because each has its own Phase 5 row.
- **"Does raw RAG help?": moved, and narrowed.** Not answerable at n=20 pooled.
  The defensible claim is the split: the arm with 70% closed-book accuracy lost 4
  rows to context, the arm with 25% gained 2. Phase 7 tests whether a stricter
  prompt moves the split; Phase 13 measures it at power.
- **Retrieval recall: still open, and now first in line.** At k=5 the gold option's
  wording appeared in 4/20 questions, so most rows cannot distinguish "the model
  ignored the context" from "the context never had it". Phase 7 item 1.
- **Groundedness checking: moved to Phase 8, with a constraint attached.** 32/40
  rows — most of them *correct* — chose wording that appears nowhere in the
  excerpts, so a support-check built on string overlap would reject correct answers
  and pass invented bridges like `q4002`'s. The judge has to be semantic.
- **`tok_s` is not comparable across conditions, and Phase 9 must not treat it as
  decode speed.** It is `completion_tokens / wall_s`, and `wall_s` includes prefill:
  prompt medians went 285 → 1,094 while reported tok/s fell 72.7 → 58.9
  (`model_a`) and 73.5 → 64.4 (`model_b`) on the same endpoint, same day, both
  runs sequential. The prompt quadrupled; that is the obvious cause but nothing in
  the harness separates prefill from decode. Either fix prompt length when
  comparing tok/s or record the two phases separately — before the Phase 9
  concurrency sweep publishes any per-arm number.
- **Concurrency: settled as a constraint, not a measurement.** Both arms are served
  by **one** oMLX process — `run.log`'s preflight prints the same 12-model list for
  both endpoints on `:8080` — so `--concurrent-models` would confound per-arm
  tok/s. This run went sequentially, which is also why its latencies are usable
  for Phase 13 planning at all.

## 6. Cost of this run

- **Wall clock: ~8.5 min** for 40 completions plus index load — first row stamped
  `14:38:18Z`, last `14:46:44Z`, run started `14:38:11Z` (`context.md`). Of that,
  505 s was generation (128 s `model_a`, 377 s `model_b`) and **7.0 s** was
  retrieval across all 20 questions (median 0.27 s each; the 1.74 s first question
  is the cross-encoder's cold start).
- **Tokens: 77,791 total** — prompt 46,816 (23,478 + 23,338), completion 30,975
  (6,951 + 24,024). The prompt half is now the larger half: at +822 tokens of
  excerpts per request, a 20-question two-arm RAG run costs 40 × ~1.1k prompt
  tokens before it generates a single token.
- **Disk: 1.1 MB** for the run dir — `run.log` 540 KB (the bulk), `results.jsonl`
  173 KB, 40 chains 212 KB, `prompts.md` 138 KB, `context.md` + `pairing.md` 5 KB.
  `outputs/` is gitignored (`.gitignore:3`), so none of it is committed and a
  Phase 7 run cannot accidentally check in a corpus-scale artifact; the tracked
  footprint is `experiments/phase6/` at 60 KB (this file plus 8 chains).
- **The harness's own worst-case bound was ~9× too pessimistic** and should be
  read as a tripwire, not a budget. `run.log:23-24` printed *"worst case model_a:
  0.4–0.2 min/question … model_b: 3.4–1.7 min/question … a bound, not an
  estimate"* assuming every completion ran to its ceiling; run sequentially that is
  **38–76 min** for this run's 40 completions, against **8.5 min realised**. The
  realised medians (6.1 s and 13.4 s/question) are what Phase 13 should budget
  with; keep the bound only to catch a run that has started looping.
