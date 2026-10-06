# ADR-0002: Instruct / reasoning model pair

## Status
Accepted (Phase 4). Partially superseded by
[ADR-0007](./0007-retrieval-first-direction-change.md): the pair, its per-arm
settings and Phases 5-6's per-arm evidence all stand, but the model factor no
longer has a study to feed -- the factorial that needed it was cut.

## Context
The design's `model` factor was a placeholder. If the two arms turned out to be
the same model, or two capability tiers of one family, that factor would be
degenerate and every interaction term built on it meaningless. Phase 4 probed
both candidates on closed-book and retrieval-augmented questions to establish
that they differ in kind, not just in score.

## Decision
`models.yaml` names an instruct/reasoning pair: `model_a` =
`Qwen2.5-7B-Instruct-8bit`, `model_b` = `DeepSeek-R1-Distill-Qwen-7B-8bit`.
Per-model `max_new_tokens`, `temperature` and `top_p` differ on purpose (see
[ADR-0006](./0006-token-ceilings-and-sampling-params.md)).

## Consequences
Easier: the model factor is real, and Phase 6's per-arm split (context moved the
two arms in opposite directions) is exactly the kind of effect the axis exists to
find.

Harder: the arms differ on several axes at once — chain length, cost per
completion, token ceiling, and now the endpoint's thinking budget — so all
reporting is per-arm and pooled accuracy is discarded
(`../../experiments/phase6/FINDINGS.md` §4), and any model-vs-model contrast is
partly confounded by settings that live outside git (ADR-0005).

## Evidence
`../../experiments/phase4/FINDINGS.md` §3 (the probe) and §6 (decision table);
the per-arm consequence in `../../experiments/phase6/FINDINGS.md` §2 and §4.
