# ADR-0005: `model_b` non-termination resolved endpoint-side

## Status
Accepted (2026-09-28, run `20260928T020841Z`). Partially superseding: the
`extra_body` decoding-constraint work this item originally called for is no longer
required to make `model_b` answer.

## Context
Phase 5's first run (1, §1–§7) found `model_b` falling into a repetition
attractor: 2 of its 3 completions looped on "Alternatively" (×1,034 and ×349)
until they consumed all 16,384 tokens and returned `content_chars=0` — chain of
thought, no answer. The third completion took 626 tokens and 9.9 s, so this was a
distinct failure mode and not "the model thinks long". Per-request knobs
(`repetition_penalty`, `top_k`, `min_p`, …) were available on the then-current
`mlx_lm.server` but unsent by `LLMClient._complete()`, so the obvious fix was an
`extra_body` passthrough plus a penalty × ceiling sweep. Raising
`max_new_tokens` was ruled out early: a 32K ceiling buys more minutes of the same
loop, not an answer.

## Decision
Adopt the fix as it landed — **endpoint-side, not in this repo**. One oMLX server
on `:8080` now applies `enable_thinking: true`, `thinking_budget_enabled: true`,
`thinking_budget_tokens: 4096` and `top_k: 40` to `model_b` from
`~/.omlx/model_settings.json`. No `extra_body` decoding passthrough is added for
termination; the repo sends only `max_completion_tokens` / `temperature` /
`top_p`. The repo-side harness fixes that the same run needed (circuit breaker,
`completed_keys()` ignoring error rows, `error_row()` carrying `params`) stay.

## Consequences
Easier: all 20 `model_b` rows finish `stop`; the two questions that had looped
closed in 14.6 s and 26.3 s with parsable answers, and Phase 5's per-model
comparison is usable. `PLAN.md` states the resulting ceiling once
([ADR-0006](./0006-token-ceilings-and-sampling-params.md)).

Harder — three sub-problems resolved unevenly, and the asymmetry is the residue:

- **(a) solved, but outside git.** Nothing in this repo names the parameters that
  decide whether the model answers at all, so the `extra_body` passthrough is now
  needed for *reproducibility* rather than termination: a run's `context.md` must
  record the endpoint's per-model settings (`--note`) and Phase 10 must name them
  in the freeze.
- **(b) NOT solved.** Recovery fired 0/40 rows, so `_recover_answer`'s loop
  replay is unexercised, not fixed — it would still feed a degenerate chain back
  to the model as its own prior turn. This stays an open question in `PLAN.md`.
- **(c) solved and tested.** `EndpointCircuitBreaker` (3 consecutive transport
  failures ⇒ abandon the arm) and the `completed_keys()` fix that plain `--resume`
  depends on, both with tests that fail on regression.

## Evidence
`../../experiments/phase5/FINDINGS.md` §7 (what blocked run 1 and the punch list),
§8.1 (what changed between the runs), §8.3 (what it settles), §8.4 (what it does
not). The looping chains themselves:
`../../experiments/phase5/chains/model_b__1312.md` and
`../../experiments/phase5/chains/model_b__2391.md`.
