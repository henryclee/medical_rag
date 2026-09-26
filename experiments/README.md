# Exploratory findings

This directory is **tracked on purpose** — unlike `data/` and `outputs/`, which
are gitignored. It holds the distilled evidence from the exploratory phases
(5–9), and it is the input to the Phase 10 design freeze.

The reasoning is in `PLAN.md` §5 ("The exploratory-artifact convention"), in one
line: the question *"why did `model_b` get this one wrong?"* gets asked later,
when the config that produced the answer may already have been edited — so the
answer has to be in version control, not in a terminal scrollback.

## What lives where

| Path | Tracked? | Contents |
| --- | --- | --- |
| `experiments/phaseN/FINDINGS.md` | yes | The numbers, the failures *with the chain that shows why*, the decision each produced, the open questions it settled or moved. |
| `experiments/phaseN/chains/` | yes | The few chains a finding cites. Copy them out of `outputs/` here — `outputs/` is regenerable and will be deleted. |
| `outputs/exploration/phaseN/<stamp>/` | no | Raw: `results.jsonl`, every `chains/<model>__<question_id>.md`, `context.md` (git rev, resolved config, sampled ids). |
| `outputs/runs/` | no | Pilot and confirmatory results. **Nothing in here is ever cited as a result.** |

## Rules

1. **Dev split only.** `run_pilot.py` and `scripts/exploration/*` read
   `config.dev_split`. The test split belongs to Phase 13.
2. **n=20 is plumbing evidence.** Report it as "the path works and here is what
   it cost", never as an accuracy estimate, and never in `RESULTS.md`.
3. **Cite, don't summarize.** A finding that says "`model_b` was worse" is not
   reviewable. Name the `question_id` and link the chain.
4. **Every chain, not just the wrong ones.** `model_b`'s chains run 2.9k–4.8k
   characters, so a whole phase is ~200 KB. Failures-only retention hides the
   base rate you need to interpret the failures.
5. **A row must be self-describing.** Each `results.jsonl` row carries the
   `params` / resolved `min_chunks` / `max_retries` active for it, because
   `conditions.yaml` will be edited after the row is written.

## Starting a new one

```sh
mkdir -p experiments/phase5
cp experiments/TEMPLATE.md experiments/phase5/FINDINGS.md
```
