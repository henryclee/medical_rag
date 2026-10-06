# File Layout

Part of the plan (./PLAN.md). Update this file when the tree changes; PLAN.md shouldn't need to change for that.

The rule that prunes this tree: it keeps the code that measures retrieval and the
code that is cited as evidence, and nothing that only aspired to be either. The
generator-side scaffolding is deleted rather than stubbed -- `modules/`,
`experiment/`, `analysis/`, `run_pilot.py`, `run_experiment.py`,
`config/conditions.yaml`, `tests/test_reformulator.py`,
`tests/test_verifier.py`. A stub directory is how a plan pretends it has an
implementation, and this repo has already paid for one plausible table computed
by a lever that never ran (`__reform`, 18/20 fallbacks).

```
config/default.yaml            Benchmark, splits, output dir, and the `retrieval:` block that IS the experiment: corpus, embedding/reranker model, `strategy`, `top_k_retrieve`/`top_k_rerank`/`bm25_top_k`, `rrf_k`. Every field is read, by `RetrievalStrategy.from_config()`; the comments say which measurement picked the value. `chunk_size`/`chunk_overlap` are gone -- StatPearls is chunked by the vendored MedRAG section algorithm, never by a token window, so those keys moved nothing.
config/models.yaml             Per-model endpoint config (base_url, api_model_name, api_key_env, sampling params, ceilings) against one oMLX server on :8080. The mainline reads `judge_model` only; `model_a`/`model_b` remain because the frozen Phase 5-6 scripts read them.
pyproject.toml                 Package metadata, exact-pinned dependencies, pytest config.
requirements.lock.txt          Full `pip freeze` snapshot of the environment.
.gitignore                     Excludes .venv/, data/, outputs/, caches, .env -- and `prompt.md`, which is worth knowing about: a root file by that name is invisible to `git status`, so deleting one looks like a no-op. Deliberately NOT excluded: experiments/, because exploratory evidence has to be versioned to be reviewable.
.env / .env.example            `.env` is gitignored and holds the endpoint keys; `.env.example` is tracked and lists the variable names with no secrets. `load_config()` merges `.env` into `os.environ` (an exported value wins; an empty `KEY=` stays empty), so `source` is only needed for raw `curl` checks.
README.md                      What the repo does now, and the one command that scores a retrieval hypothesis.
PLAN.md                        The operating plan: R1-R6, the constraints that each cost a bad measurement, the backlog, and the R1 audit trail -- what moved, what was deleted, and the judgement calls made without asking. Read that trail before asking "why is this here?".
interfaces.md                  Public contracts: signatures, invariants, stability.
environment.md                 Runtimes, versions, setup. Its endpoint section is stale (two `mlx_lm.server` instances on :8081/:8082); `config/models.yaml`'s header is authoritative -- one oMLX on :8080.

src/medical_rag/__init__.py            Package version (__version__ = "0.1.0").
src/medical_rag/paths.py               REPO_ROOT, written down once. Per-module `Path(__file__).parents[n]` arithmetic is exactly what silently pointed at the wrong directory when a module moved a level.
src/medical_rag/config.py              Pydantic models + `load_config()`/`load_env()`. Merges two files (default.yaml, models.yaml). Its validator refuses a retrieval block that cannot be measured: an unknown `strategy` name, or a rerank depth wider than the candidate list it reorders.
src/medical_rag/text.py                `normalize()` / `contains()` / `chunk_match_rank()` -- one definition of "the same text", which the ceiling probe, the audit script and the gold-wording counts all need to agree on or they report three different recall numbers.
src/medical_rag/data/__init__.py       Data subpackage docstring.
src/medical_rag/data/load_medqa.py     MedQA-USMLE loader + MedQAQuestion model.
src/medical_rag/data/load_statpearls.py StatPearls loader (NCBI download + MedRAG-algorithm chunking) + StatPearlsChunk model.
src/medical_rag/data/chunking.py       Generic token-based chunker, unused today; kept as the natural home of the backlog's chunking lever.
src/medical_rag/retrieval/__init__.py  Retrieval subpackage docstring.
src/medical_rag/retrieval/embedder.py  Embedder: wraps SentenceTransformer, BGE asymmetric query/passage convention.
src/medical_rag/retrieval/index.py     LanceDB table build/load helpers (exact search; no ANN at this corpus size).
src/medical_rag/retrieval/retriever.py Retriever: vector search + optional cross-encoder reranking; RetrievedChunk model.
src/medical_rag/retrieval/strategy.py  The seam between measuring and shipping: BASE_STRATEGIES, METHOD_GRID, `retrieve_base()` (one fan-out -> all five ranked lists), `retrieve_grid()`/`aretrieve_grid()`, `RetrievalStrategy` (config -> the funnel both the grid and a production call run), `build_retriever()`, `resolve_index_dir()`.
src/medical_rag/retrieval/lexical.py   BM25Index + `load_or_build_bm25_index()`, pickling to `data/index/statpearls_bm25.pkl`. That file is big enough that `eval_retrieval.py` scores all five strategies in one pass rather than re-loading it per hypothesis; rebuilding it is `--rebuild-bm25`, deliberately not automatic.
src/medical_rag/retrieval/fusion.py    `rrf_fuse()`. Pure, and the only place RRF's smoothing default is named.
src/medical_rag/retrieval/ceiling.py   The corpus-ceiling probe. It reads the gold option's own wording, so it is an upper bound and never a lever -- it must not appear in a strategy table.
src/medical_rag/generation/__init__.py Generation subpackage docstring.
src/medical_rag/generation/llm.py      LLMClient: OpenAI-compatible async client, LLMError, structured output.
src/medical_rag/generation/prompt.py   Prompt builders + parse_answer(). The answer/verification builders are frozen Phase 5-6 tooling; `build_reformulation_prompt()` is the one still reachable from `eval/rewrite.py`.
src/medical_rag/generation/preflight.py `Preflight`/`preflight()`: endpoint reachability and "does the advertised model id exist", shared by the grid and the frozen scripts.

src/medical_rag/eval/__init__.py       The offline measurement surface: everything that answers "did retrieval help?" without asking an answer model anything.
src/medical_rag/eval/metrics.py        recall@k, relevant_fraction@k, first-relevant-rank, `aggregate()`. Pure.
src/medical_rag/eval/rubric.py         The judging prompt and `judge_prompt_sha()`. Editing it invalidates every cached verdict on purpose: a verdict is a fact about a chunk under one rubric.
src/medical_rag/eval/judge.py          The verdict cache and the only code that spends endpoint time; `python -m medical_rag.eval.judge --stats` reports how much of it is usable under the current rubric.
src/medical_rag/eval/store.py          The `chunks.jsonl` sidecar -- stores the text a run retrieved so re-reading a run does not need the 380k-row index.
src/medical_rag/eval/report.py         One grid renderer behind three outputs: terminal text, `report.html`, and the markdown of `FINDINGS.md`.
src/medical_rag/eval/runlog.py         Pinned samples (`select_sample`/`select_by_ids`), the row schema (`result_row`/`error_row`), the run-dir writer, the endpoint circuit breaker. Formerly `scripts/exploration/_common.py`.
src/medical_rag/eval/harness.py        The grid runner: retrieve -> judge the union of candidates -> metrics -> FINDINGS.md. `--rerender` rebuilds a report from a stored run offline.
src/medical_rag/eval/inspector.py      Read a run back (`--from-run`, zero calls, zero models) or inspect live retrieval (`--question-id`, `--repl`).
src/medical_rag/eval/lab.py            The localhost knob panel: serves `Inspector` so the index and embedder load once per session instead of once per guess.
src/medical_rag/eval/rewrite.py        Query reformulation. Importable and unscheduled -- it is the one lever that still costs completions.

scripts/eval_retrieval.py        THE command: score a strategy against the verdict cache, zero endpoint calls, and prints cache coverage as loudly as recall.
scripts/run_retrieval_grid.py    Thin wrapper over `eval.harness.main()` -- the grid, and `--rerender` for the offline rebuild.
scripts/inspect_retrieval.py     Thin wrapper over `eval.inspector.main()`.
scripts/build_index.py           CLI: build the LanceDB retrieval index from a corpus.
scripts/probe_models.py          CLI: live check of the served models -- cost/latency/throughput, `finish_reason`, the ANSWER-recovery path. Frozen tooling.
scripts/exploration/closed_book.py  Phase 5: closed-book probe over the pinned sample, saving every chain of thought. Frozen.
scripts/exploration/raw_rag.py      Phase 6: the same with retrieval + reranking wired in. Frozen.
scripts/exploration/audit_run.py    Re-derives a phase's numbers from its `results.jsonl` and PASS/FAILs them against that phase's FINDINGS.md. Frozen; `eval_retrieval.py` is its successor for the R-phases.

tests/test_pipeline.py           Smoke: package imports and exposes __version__.
tests/test_config.py             load_config(), the .env merge rules, and the unmeasurable-funnel rejections.
tests/test_data_loading.py       MedQA loader schema/count (live) + StatPearls chunker logic (fixture, no network).
tests/test_retrieval.py          Embedder shape, index build/save/load round-trip, retrieve() relevance, rerank() reordering.
tests/test_retrieval_strategy.py That config -> strategy loses no depth, that one fan-out yields all five lists, and that `retrieve()` is the same path the grid scores.
tests/test_retrieval_tuning.py   The measurement machinery: grid shape, candidate assembly, cache keying/eviction, chunk sidecar, and that a legacy run re-renders without inventing comparisons.
tests/test_generation.py         Prompt builders, parse_answer tiers, LLMClient over httpx.MockTransport, + one `live` endpoint test.
tests/test_exploration_common.py Tests `eval/runlog` -- the filename still says `_common.py` because that is the module it grew up testing.
tests/test_raw_rag.py            The frozen Phase 6 comparison logic: outcome classification, baseline pairing, drift detection. Loaded by path; `scripts/` is not an importable package.

data/                          Gitignored. Cached corpora (data/statpearls/), the built index (data/index/statpearls/), and the BM25 pickle (data/index/statpearls_bm25.pkl).
experiments/                   Tracked (deliberately NOT gitignored). Distilled evidence -- see experiments/README.md.
experiments/phase5/            Frozen: closed-book FINDINGS + the chains it cites.
experiments/phase6/            Frozen: raw-RAG FINDINGS + the chains it cites.
experiments/phase4/            Frozen: endpoint/prompt findings.
experiments/retrieval_tuning/  Tracked, and now documentation only: TUNING.md (the method) and FINDINGS.md (written by `eval.harness`, never by hand). The code that produced them moved into `src/medical_rag/`; a run directory plus the cache reproduces every cell.
experiments/retrieval/         Does not exist yet. R-phase output starts at R2: `R<n>/FINDINGS.md`, plus the tracked oracle snapshot under `evalset/`.
experiments/TEMPLATE.md        Skeleton a new FINDINGS.md is copied from (written for the generation phases; its per-arm sections no longer apply).
outputs/exploration/retrieval_tuning/ Gitignored. `<UTC-stamp>/` grid runs and `inspect_<stamp>/` inspector dirs, plus the oracle cache at `judge_cache/judged_chunks.jsonl`.
outputs/exploration/phaseN/    Gitignored. Raw run dirs from `scripts/exploration/` (results.jsonl, chains/, context.md).
outputs/probes/                Gitignored. probe_models.py rows + chains.
```

## What moved in R1, and why it was not left where it was

The name-by-name map and the deletion list are `PLAN.md`'s
**R1 audit trail**; this section is only the reasoning, because "why is this here?"
is a different question from "where did it go?".

`scripts/exploration/_common.py` is gone, not shimmed. It was the shared spine of
every measurement script and it lived behind a `sys.path.insert`, with
`tests/test_exploration_common.py` loading it by file path to prove nothing
better than "this import works by accident". A shim would only keep the pretence
that `scripts/` owns code the package depends on. The harness, inspector,
renderer, verdict cache, rubric, metrics, sidecar store, BM25 index, RRF fusion
and ceiling probe moved into the package for the same reason: they are what this
project produces, not scaffolding around something else.

`experiments/retrieval_tuning/` keeps only prose. That is the honest inventory --
the directory's code is reproducible from the package, and its numbers are
reproducible from a run directory plus the cache.

## Two conventions that hold it together

**Nothing under `src/` imports from `scripts/`.** One direction only. When the
grid lived in `experiments/retrieval_tuning/` and the shared helpers in
`scripts/exploration/`, package code and script code needed each other, and the
only way to make that work was `sys.path` surgery at import time -- which works
until a module moves a level, and then fails somewhere far from the edit. If
`src/` needs something, that thing is in `src/`.

**`scripts/*.py` are thin.** Each is argument parsing plus a call to a package
`main()`, and `python -m medical_rag.eval.harness` is the identical command. The
wrapper exists so `ls scripts/` shows the study's commands without first
learning the package layout; the logic never lives there, because a script is not
importable and therefore not testable. `scripts/exploration/` is the exception
that is honest about itself: those are frozen Phase 5-6 commands, loaded by path
in tests when they are tested at all.
