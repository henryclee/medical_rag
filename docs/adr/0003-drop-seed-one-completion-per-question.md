# ADR-0003: `seed` removed; one completion per question per condition

## Status
Accepted (2026-09-27). Supersedes the 3-seed repeated-draws design in the
original spec.

## Context
The design specified three completions per question per condition, one per seed,
to estimate within-question variance. Phase 5 measured the premise and found it
false: on the `model_a` endpoint, five draws across `seed=0`, `seed=100` and
`seed=101` at `temperature=0.7` were byte-identical (chain-body sha256 equal,
differing only in the header stamp), so the seed dimension contributed zero
variance. `model_b`'s half of the check never ran — its endpoint died, and its
non-termination (ADR-0005) would have confounded the result anyway.
Alternatives considered: drive repetition with `temperature` instead, or keep the
draws and report seed variance as "unsupported by this endpoint".

## Decision
Drop the repeated-draws design outright. `ModelConfig.seed`,
`ConditionConfig.seeds`, the `seed` parameter on `LLMClient.agenerate` /
`agenerate_structured`, and `QuestionResult.seed` are removed — not deprecated,
not defaulted to `None` — and `seed` is never sent to an endpoint. The experiment
runs **one completion per question per condition**.

## Consequences
Easier: roughly a third of the completions and wall clock of the Phase 13 run;
`interfaces.md`'s `QuestionResult` and runner notes already reflect the single
completion; and Phase 4's finding that passing `seed` disables `mlx_lm.server`'s
batching stops mattering, which is why Phase 9's concurrency sweep is re-measured
without it (open question 12 in `PLAN.md`).

Harder: the study can no longer make a within-question replicate claim.
Variance comes only from the frozen condition set and the bootstrap CIs, so the
primary test has to carry the uncertainty (open question 5).

## Evidence
`../../experiments/phase5/FINDINGS.md` §3 (identical-hash table), §5 (the
measurement and the MLX per-stream PRNG lead), and §"Resolution (2026-09-27) —
`seed` removed from the pipeline"; the batching penalty in
`../../experiments/phase4/FINDINGS.md` §5.
