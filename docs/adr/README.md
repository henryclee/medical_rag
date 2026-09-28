# ADR index

Architectural decision records for this project: the resolved half of
`PLAN.md`'s "Open questions" list. `PLAN.md` carries only what changes the next
edit, test, or command; the debates that produced those constraints live here.

| ADR | Decision | Was | Status |
| --- | --- | --- | --- |
| [0001](./0001-real-model-endpoints.md) | Real model endpoints chosen | item 1 | Accepted |
| [0002](./0002-instruct-reasoning-model-pair.md) | Instruct / reasoning model pair | item 2 | Accepted |
| [0003](./0003-drop-seed-one-completion-per-question.md) | `seed` removed; one completion per question per condition | item 10 | Accepted |
| [0004](./0004-thinking-mode-fixed-on.md) | Thinking mode fixed on, no toggle | item 11 | Accepted |
| [0005](./0005-model-b-termination-endpoint-side.md) | `model_b` non-termination fixed endpoint-side | item 13 | Accepted (recovery half unresolved) |
| [0006](./0006-token-ceilings-and-sampling-params.md) | Token ceilings `model_a` 1,024 / `model_b` 8,096 | item 8 | Accepted (passthrough follow-up unresolved) |

Open questions 3–9, 12, 14 and 15 are **not** decided yet. They stay in
`PLAN.md` and are frozen at Phase 10; this directory stays empty of them until
then.
