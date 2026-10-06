# Medical RAG -- Retrieval Optimization

## Overview

Retrieval for medical board-exam questions: MedQA-USMLE
(`GBaker/MedQA-USMLE-4-options`, 1,273 test questions, `train` used as the dev
split) answered from StatPearls -- 380,454 chunks from 9,652 articles, embedded
with `BAAI/bge-small-en-v1.5`, stored in an exact-search LanceDB table, reranked
with `cross-encoder/ms-marco-MiniLM-L-6-v2`.

Retrieval quality is measured **offline**. `judge_model` grades whether each
retrieved chunk supports the gold option -- blind to the letter -- and those
verdicts are cached and versioned by rubric hash. Scoring a new retrieval
strategy is therefore index work plus arithmetic: **zero endpoint calls per
hypothesis**. That asymmetry is the whole reason retrieval is the thing being
iterated on; a generation-side hypothesis costs hours of shared endpoint time
for one number. Answer accuracy is deliberately not measured here -- see
[PLAN.md](./PLAN.md) and
[ADR-0007](./docs/adr/0007-retrieval-first-direction-change.md).

Nothing runs a model locally. Every LLM is a remote OpenAI-compatible endpoint,
and the judge is the only model the mainline path pays for.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Commands below are written as `.venv/bin/python` so they work from a cold shell
without activating anything.

## Endpoints and keys

`config/models.yaml` names the endpoint, served model id, sampling parameters and
API-key variable for each model; one oMLX server on `:8080` serves them all. Keys
live in a gitignored `.env` at the repo root (copy `.env.example` and fill it in):

```bash
set -a; source .env; set +a     # optional for Python entry points
```

`load_config()` calls `config.load_env()`, which merges the repo root's `.env`
into the environment before any client reads a key. It only fills gaps, so
anything you `export` still wins, and an empty `KEY=` in the file stays empty
(you get the `LLMError`, not a 401). Keep using the prefix for raw `curl`
checks, which have no config loader.

## The command that matters

```bash
.venv/bin/python scripts/eval_retrieval.py --dry-run    # resolved config + cache state, loads nothing
.venv/bin/python scripts/eval_retrieval.py              # all five strategies, pinned sample
.venv/bin/python scripts/eval_retrieval.py --strategy hybrid --k 5 10 20
.venv/bin/python scripts/eval_retrieval.py --require-coverage 1.0   # gate: fail if anything is unjudged
```

It prints cache **coverage** as loudly as recall, on purpose. A candidate slot
with no cached verdict counts as *not* supporting the gold option, so a 40%
recall over slots that are 60% unjudged is a measurement of the cache, not of
retrieval. New strategies need the grid (`scripts/run_retrieval_grid.py`) to
judge their candidates before a comparison against an older, better-judged
strategy means anything.

## Rebuilding a findings table offline

```bash
.venv/bin/python scripts/run_retrieval_grid.py \
    --rerender outputs/exploration/retrieval_tuning/20260929T120215Z \
    --findings-path /tmp/FINDINGS.md
diff /tmp/FINDINGS.md experiments/retrieval_tuning/FINDINGS.md   # must be empty
```

Zero endpoint calls, and this diff is the gate every refactor of this code has to
pass: the table must come out cell-for-cell identical to the committed one.
**Redirect `--findings-path` when you are checking** -- its default is the
tracked `experiments/retrieval_tuning/FINDINGS.md`, which `--rerender`
overwrites. That is not hypothetical: an n=2 probe once wrote its table over the
real findings (`experiments/retrieval_tuning/TUNING.md` records it).

## Where the numbers are

The one measured grid (`experiments/retrieval_tuning/FINDINGS.md`, run
`20260929T120215Z`, n=20, dev split) reports `semantic_recall@5/10/20`:
`dense` 40/55/65, `dense_rerank` 45/45/45, `hybrid` 30/40/55,
`hybrid_rerank` 25/25/25, `bm25` 0/0/10. Three caveats are load-bearing:

- The `*_rerank` columns are capped by `top_k_rerank: 5` -- the funnel hands the
  reranker 5 candidates and asks for 5, so their recall cannot rise with k.
  45/45/45 is a ceiling, not a plateau, and never compare them against an
  unreranked column at the same k.
- The `__reform` column is not a reformulation result: the rewrite fell back to
  the raw question on 18/20 questions, so it is a copy of `__orig`.
- n=20 on the dev split. That is plumbing evidence, not an estimate.

## The oracle

`outputs/exploration/retrieval_tuning/judge_cache/judged_chunks.jsonl` -- one
verdict per `(question_id, chunk_id)`, stamped with the rubric hash. It is
gitignored because it is regenerable, but regenerating it costs real endpoint
time, so treat it as precious:

```bash
.venv/bin/python -m medical_rag.eval.judge --stats    # how much is usable under the current rubric
```

Editing the rubric (`src/medical_rag/eval/rubric.py`) invalidates every cached
verdict, by design: a verdict is a fact about a chunk *under one rubric*. R2
commits a compact tracked snapshot under `experiments/retrieval/evalset/` so the
oracle itself is reviewable rather than machine-local.

## Index and tests

```bash
.venv/bin/python scripts/build_index.py --corpus statpearls   # idempotent; downloads on first run
.venv/bin/pytest                                               # offline by default; `live` tests skip
MEDICAL_RAG_LIVE_TESTS=1 .venv/bin/pytest                      # also hits the real endpoint
```

The BM25 pickle (`data/index/statpearls_bm25.pkl`) is built on demand by
`scripts/eval_retrieval.py`; pass `--rebuild-bm25` after changing anything that
affects tokenization.

## Docs

[PLAN.md](./PLAN.md) -- the R-phase plan and the constraints that each cost a bad
measurement. [experiments/retrieval_tuning/TUNING.md](./experiments/retrieval_tuning/TUNING.md)
-- the method. [experiments/README.md](./experiments/README.md) -- what is tracked
as evidence and why. [interfaces.md](./interfaces.md) -- contracts.
[file_layout.md](./file_layout.md) -- the tree.
[refactor_plan.md](./refactor_plan.md) -- why the tree looks like this now.
[docs/adr/](./docs/adr/README.md) -- decisions argued out once so they stop being
re-litigated.
