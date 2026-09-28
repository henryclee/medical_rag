# ADR-0006: Final token ceilings and sampling parameters

## Status
Accepted. Resolves the contradiction between `PLAN.md`'s Phase 5 status line
(both arms settled, `model_b` 16,384 → 8,096) and its open-questions item 8
(`model_b`'s 16,384 "not validated"), which had been carried side by side.

## Context
Phase 5 asked whether each model's `max_new_tokens` was right. Run 1 gave a
clean negative result and a misleading one at the same time: `model_a` never came
close to 1,024, while 2 of 3 `model_b` completions pinned the 16,384 cap with
`content_chars=0` — which falsified `models.yaml`'s comment that 16,384 was "the
measured worst case at which it still terminates", but said nothing about the
right ceiling, because no ceiling is large enough for a model that does not
terminate (see [ADR-0005](./0005-model-b-termination-endpoint-side.md)). After
the endpoint-side thinking budget landed, run 2 measured both arms properly: worst
case used 50% (`model_a`) and 52% (`model_b`) of ceiling with 0/40
`finish_reason="length"`. Phase 6 then re-checked the ceilings with ~822 tokens of
excerpts bolted onto the prompt: still 0/40 truncated, and `model_b`'s median
completion got *shorter* with context (1,328 → 830 tokens).

## Decision
`PLAN.md` states one pair of caps: **`model_a` 1,024, `model_b` 8,096** (as
already set in `config/models.yaml`). Do not raise either to "fix" a
non-terminating run. The sizing *rule* changed in kind: `model_b`'s ceiling is
now structurally ~2× the endpoint's 4,096-token thinking budget, so the budget —
not `max_new_tokens` — is what bounds a chain. The `--caps` contingency sweep is
optional, not required, at this n.

Sampling parameters, same item: each model runs at its own published defaults
rather than the spec's placeholder `0.9` for both — `model_a` `temperature: 0.7` /
`top_p: 0.8`, read from its downloaded `generation_config.json` (`top_k: 20` and
`repetition_penalty: 1.05` are published alongside them but are not fields on
`ModelConfig`, which is open question 16 in `PLAN.md`), and `model_b`
`temperature: 0.6` / `top_p: 0.95`, DeepSeek-R1-Distill's recommended values.

## Consequences
Easier: `timeout` sizing follows from the ceiling (~60–80 tok/s ⇒ 8,096 tokens is
~2 min, so 300 s leaves headroom), Phase 13's cost estimate becomes bounded, and
the plan no longer contradicts itself about which number is current.

Harder: the ceiling's validity is contingent on an out-of-repo setting. If the
4,096 thinking budget is raised or removed, 8,096 is no longer a measured value
and has to be re-derived. The budget's tail is unmeasured — one chain reached
4,237 tokens, within 5% of the budget — and at n=1,273 it will bind; that risk is
tracked as an open question in `PLAN.md`, not resolved here.

## Evidence
`../../experiments/phase5/FINDINGS.md` §8.2 (per-arm table), §8.3 ("item 8 closes
for both arms"), §8.4 (unmeasured tail); `../../experiments/phase6/FINDINGS.md`
§2.1 (ceilings survive the context bump) and §4 ("Token ceilings unchanged").
