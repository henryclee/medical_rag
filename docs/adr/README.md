# ADR index

Architectural decision records for this project: the resolved half of
`PLAN.md`'s "Open questions" list. `PLAN.md` carries only what changes the next
edit, test, or command; the debates that produced those constraints live here.

| ADR | Decision | Was | Status |
| --- | --- | --- | --- |
| [0001](./0001-real-model-endpoints.md) | Real model endpoints chosen | item 1 | Accepted |
| [0002](./0002-instruct-reasoning-model-pair.md) | Instruct / reasoning model pair | item 2 | Accepted; consequence partly superseded by 0007 |
| [0003](./0003-drop-seed-one-completion-per-question.md) | `seed` removed; one completion per question per condition | item 10 | Accepted |
| [0004](./0004-thinking-mode-fixed-on.md) | Thinking mode fixed on, no toggle | item 11 | Accepted |
| [0005](./0005-model-b-termination-endpoint-side.md) | `model_b` non-termination fixed endpoint-side | item 13 | Accepted (recovery half unresolved) |
| [0006](./0006-token-ceilings-and-sampling-params.md) | Token ceilings `model_a` 1,024 / `model_b` 8,096 | item 8 | Accepted; scoped to frozen tooling by 0007 |
| [0007](./0007-retrieval-first-direction-change.md) | Retrieval-first; the answer-accuracy study (Phases 7-13) is cut | the direction change | Accepted |
| [0008](./0008-judge-cache-as-ground-truth.md) | The judge verdict cache is the measurement oracle | `TUNING.md`'s judge-reliability question | Accepted; R2 must validate it |
| [0009](./0009-corpus-and-chunking-frozen.md) | Corpus + chunking frozen; dead chunk keys deleted | implicit constraint | Accepted, with expiry |

0007-0009 came from the direction change described in
[`../../refactor_plan.md`](../../refactor_plan.md); 0002 and 0006 were edited in
their `Status:` sections only, since rewriting a superseded argument would destroy
the thing an ADR exists to preserve.

Open questions 1-6 in `PLAN.md` are **not** decided yet. They are numbered fresh —
the old numbering pointed at `interfaces.md` line numbers, `conditions.yaml` and
stub modules that R1 deleted — and they get settled inside R2-R6 rather than at a
Phase-10 design freeze that no longer exists.
