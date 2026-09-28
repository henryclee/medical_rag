# ADR-0001: Real model endpoints chosen

## Status
Accepted (Phase 4).

## Context
`PLAN.md` carried the LLM endpoints as a placeholder, so `LLMClient` could not be
verified end to end and the Phase 5 smoke test had nothing to call. The binding
constraint is architectural: generation is never local — the pipeline only ever
makes HTTP calls to an OpenAI-compatible server. The alternative considered was a
hosted API; it was rejected in favour of locally served MLX models, which the
study can re-run repeatedly at no marginal cost (see `environment.md` for how the
server is started).

## Decision
`config/models.yaml` names two locally served, OpenAI-compatible models
(`model_a`, `model_b`), with keys read from the gitignored `.env` via
`ModelConfig.api_key_env`. `LLMClient.agenerate()` / `agenerate_structured()`
were verified live against them.

## Consequences
Easier: Phases 4+ became testable against a real endpoint rather than a stub, and
`scripts/probe_models.py` became a usable pre-flight that exits non-zero when an
arm is down.

Harder: the endpoint became part of the configuration. Behaviour that the server
adds on its own — a chain of thought in a non-standard field, reasoning-only
completions when the budget runs out, per-model sampling the client never sends —
is now a variable the study has to control for. That is what later forced
[ADR-0005](./0005-model-b-termination-endpoint-side.md).

## Evidence
`../../experiments/phase4/FINDINGS.md` §2 (live verification) and §6 (decision
table). Current addresses, model ids and env vars: `../../environment.md`.
