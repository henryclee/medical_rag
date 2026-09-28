# ADR-0004: Thinking mode fixed on — no toggle

## Status
Accepted. (Phase 4's decision table recorded this as "deferred to Phase 10";
`PLAN.md`'s open question 11 later resolved it without a toggle. This ADR records
the later decision.)

## Context
Phase 4 measured `extra_body={"chat_template_kwargs": {"thinking": False}}` as
accepted by `model_b` and measurably faster (one identical request: 28.1 s / 465
completion tokens → 5.5 s / 231), and deliberately did *not* adopt it, because
whether the model thinks is part of what the study measures — a design lever, not
a performance tweak. The question then was whether to carry thinking on/off as a
condition factor. It cannot be one symmetrically: `chat_template_kwargs.thinking`
is a hybrid-reasoning template feature, and `model_a` (Qwen2.5-7B-Instruct) has
no thinking mode to toggle, so an on/off comparison would no-op or behave
unpredictably for one arm while being real for the other.

## Decision
No toggle and no `ModelConfig` field. Thinking mode stays **on** — its default,
and Phase 4's already-live configuration — for both models in every condition.

## Consequences
Easier: one fewer condition axis, one fewer Phase 5 measurement, and no
half-arm factor to explain in `RESULTS.md`.

Harder: whether thinking helps remains unmeasured, and it is *not* in the study.
The depth of `model_b`'s thinking is now governed by the endpoint's
`thinking_budget_tokens` (ADR-0005) — a knob in `~/.omlx/model_settings.json`,
outside git — which Phase 10 must record for the freeze to be reproducible, and
whose tail is still unmeasured (open question 15).

## Evidence
`../../experiments/phase4/FINDINGS.md` §4 (the on/off measurement) and §6
(the "design lever, not perf tweak" note); the budget consequence in
`../../experiments/phase5/FINDINGS.md` §8.1.
