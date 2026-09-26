# Phase N — <short name> findings

> Copy to `experiments/phaseN/FINDINGS.md`. Delete this banner and any section
> that genuinely does not apply — an empty section is worse than an absent one.
> Exploratory evidence, dev split only. Not a result (see `../README.md`).

- **Date / git rev:** `<UTC stamp>` @ `<git rev-parse --short HEAD>`
- **Command:** `` `.venv/bin/python scripts/exploration/<script>.py --sample-size 20` ``
- **Raw artifacts:** `outputs/exploration/phaseN/<stamp>/`
- **Sample:** 20 dev-split questions, ids pinned in `context.md` (list them here
  too if later phases must reuse exactly this sample)

## 1. Question this phase asked

One or two sentences. What decision was this evidence supposed to inform? If
nothing here would change a decision, the phase was not worth running.

## 2. Numbers

| model | n | correct | unanswered | `finish_reason="length"` | recovery fired | median tok/s | median wall s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `model_a` | | | | | | | |
| `model_b` | | | | | | | |

Answer-letter distribution of **errors** (a model biased toward one letter
confounds every delta in the study — this table is how you notice):

| model | A | B | C | D |
| --- | --- | --- | --- | --- |
| `model_a` | | | | |
| `model_b` | | | | |

## 3. Failures worth reading

One entry per case a later reader should look at. Name the id, paste the chain
out of `outputs/` into `./chains/`, and say what to notice in it — do not make
someone reconstruct the run to learn why an answer was wrong.

- **`<question_id>` — `model_b`, wrong (gold `C`, answered `B`).**
  Chain: [`chains/model_b__<question_id>.md`](chains/model_b__<question_id>.md).
  What it shows: <the misread cue, the truncation, the ignored excerpt, ...>.

## 4. Decisions this produced

State them as commits-to-come, so the Phase 10 freeze can cite them.

| Decision | Evidence | Where it lands |
| --- | --- | --- |
| e.g. `model_b.max_new_tokens` 16384 → `<n>` | 0/40 `finish_reason="length"` at 16384 | `config/models.yaml` at the freeze |

## 5. Open questions settled / moved

- §6 item `<k>`: **settled** — <answer in one line>, or **still open because**
  <what is missing and which phase will get it>.

## 6. Cost of this run

Wall clock, per-model request counts, tokens spent. Needed to plan the pilot and
the Phase 13 run, and impossible to recover later if not written down now.
