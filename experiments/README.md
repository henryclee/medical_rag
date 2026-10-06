# Exploratory findings

This directory is **tracked on purpose** -- unlike `data/` and `outputs/`, which
are gitignored. It holds the distilled evidence, and it is the only part of a
measurement that survives a rebuild, a new machine, or a deleted `.venv`.

The reasoning, in one line: the question *"why did this strategy lose here?"* gets
asked later, when the config and even the code that produced the answer may
already have been edited -- so the answer has to be in version control, not in a
terminal scrollback or a gitignored run directory.

## What lives where

| Path | Tracked? | Contents |
| --- | --- | --- |
| `experiments/retrieval/R<n>/FINDINGS.md` | yes | One R-phase's numbers, the questions that failed *with the chunk that shows why*, the decision each produced, and what it settled or moved. |
| `experiments/retrieval/evalset/` | yes (from R2) | The oracle: a compact snapshot of the judge verdicts -- ids, relevance, `supports_options`, reason, no chunk prose. The live cache under `outputs/` is regenerable but expensive; this snapshot is what makes the ground truth reviewable. |
| `experiments/retrieval_tuning/` | yes | Frozen: `TUNING.md` (method) and `FINDINGS.md` (the one measured grid). Its code now lives in `src/medical_rag/`. |
| `experiments/phase4/`, `phase5/`, `phase6/` | yes | Frozen evidence from the generator-side phases, including the chains each finding cites. |
| `outputs/exploration/<phase>/<stamp>/` | no | Raw: `results.jsonl`, `chunks.jsonl`, `report.html`, `context.md` (git rev, resolved config, resolved question ids). |
| `outputs/exploration/retrieval_tuning/judge_cache/` | no | The live verdict cache. Precise: regenerating it costs endpoint time. |

## Rules

1. **Numbers come from a command, not from a keystroke.** Every figure in a
   `FINDINGS.md` must be re-derivable by
   `.venv/bin/python scripts/eval_retrieval.py` (the successor to
   `audit_run.py`'s role). If you computed it in a notebook, the command is part
   of the finding.
2. **Dev split only.** Every entry point reads `config.dev_split`. The test split
   stays untouched -- it is the only thing that can still tell you whether any of
   this generalized.
3. **A lever that was not run is not a lever that failed.** A cell with `n = 0`
   is blank, never zero, and a delta over an empty cell is meaningless. Related:
   the harness only judges the union of the candidate lists it actually
   retrieved, so a new lever has to be added to the grid before it is measurable
   at all -- scoring it against the existing cache silently scores a pool nobody
   graded.
4. **Never compare a `*_rerank` cell's `recall@10`/`@20` against an unreranked
   cell's.** The funnel caps it: `top_k_rerank: 5` means that list has five
   chunks in it, whatever k says.
5. **n=20 is plumbing evidence.** Report it as "the path works and here is what
   it cost", never as an estimate of anything.
6. **Cite, don't summarize.** "Hybrid did worse" is not reviewable. Name the
   `question_id` and the `chunk_id`.
7. **Written findings are frozen.** Never rewrite a `FINDINGS.md` to match new
   code -- it is evidence about the state that produced it. `--rerender` exists so
   a refactor can *prove* it moved no number; run it to a temp path, because the
   default target is the tracked file.
8. **Judging is endpoint discipline.** `judge_model` shares the `:8080` endpoint
   with everything else: keep the circuit breaker and resume behaviour, and
   record the endpoint-side sampling settings a run actually ran under. Under
   ADR-0008 those settings shape the ground truth, so an unrecorded one is worse
   here than it ever was for an answer model.

## Starting a new one

```sh
mkdir -p experiments/retrieval/R2
cp experiments/TEMPLATE.md experiments/retrieval/R2/FINDINGS.md
```

`TEMPLATE.md` was written for the per-arm generation phases; keep its
"numbers / failures / decisions / open questions" spine and drop the sections
that assumed two answer models.
