# File Layout

Part of the plan (./PLAN.md). Update this file when the tree changes; plan.md shouldn't need to change for that.

```
config/default.yaml            Benchmark name, output dir, and retrieval defaults (corpus, embedding/reranker model, chunk/top-k sizes).
config/models.yaml             Per-model remote endpoint config (base_url, api_model_name, api_key_env, sampling params, timeout/max_retries, answer_recovery_max_tokens) for model_a/model_b — two genuinely different served models on two local mlx_lm.server endpoints (section 6, item 2).
config/conditions.yaml         Experiment conditions for the current study; provisional until the Phase 10 design freeze, named and ID-ordered to match the delta formulas metrics.py will use.
pyproject.toml                 Package metadata, exact-pinned dependencies, pytest config.
requirements.lock.txt          Full `pip freeze` snapshot of the environment.
.gitignore                     Excludes .venv/, data/, outputs/, caches, and .env — and deliberately NOT experiments/, which is tracked because exploratory evidence has to be versioned to be reviewable (section 5's artifact convention).
.env                           Gitignored. Holds MODEL_A_API_KEY / MODEL_B_API_KEY for the served endpoints; `source` it before any endpoint-calling command.
.env.example                   Template listing those variable names with no secrets.
README.md                      Overview/Installation/Usage (Usage still a stub).
PLAN.md                        This file.

src/medical_rag/__init__.py            Package version (__version__ = "0.1.0").
src/medical_rag/config.py              Pydantic config models + load_config().
src/medical_rag/data/__init__.py       Data subpackage docstring.
src/medical_rag/data/load_medqa.py     MedQA-USMLE loader + MedQAQuestion model.
src/medical_rag/data/load_statpearls.py StatPearls loader (NCBI download + MedRAG-algorithm chunking) + StatPearlsChunk model.
src/medical_rag/data/chunking.py       Generic token-based chunker for any future non-pre-chunked corpus (unused by StatPearls).
src/medical_rag/retrieval/__init__.py  Retrieval subpackage docstring.
src/medical_rag/retrieval/embedder.py  Embedder: wraps SentenceTransformer, BGE asymmetric query/passage convention.
src/medical_rag/retrieval/index.py     LanceDB table build/load helpers.
src/medical_rag/retrieval/retriever.py Retriever: vector search + optional cross-encoder reranking; RetrievedChunk model.
src/medical_rag/modules/__init__.py    Pipeline-modules subpackage docstring.
src/medical_rag/modules/reformulator.py  [STUB] Query reformulation module.
src/medical_rag/modules/verifier.py      [STUB] Answer/context verification module.
src/medical_rag/generation/__init__.py Generation subpackage docstring.
src/medical_rag/generation/llm.py      LLMClient: OpenAI-compatible async client, LLMError, structured output. IMPLEMENTED.
src/medical_rag/generation/prompt.py   Prompt templates (answer/reformulation/verification) + parse_answer(). IMPLEMENTED.
src/medical_rag/experiment/__init__.py Experiment subpackage docstring.
src/medical_rag/experiment/conditions.py [STUB] Condition -> pipeline assembly.
src/medical_rag/experiment/runner.py     [STUB] Main experiment loop + per-question result logging.
src/medical_rag/experiment/metrics.py    [STUB] Accuracy, deltas, significance testing.
src/medical_rag/analysis/__init__.py   Analysis subpackage docstring.
src/medical_rag/analysis/failure_taxonomy.py [STUB] Failure categorization for incorrect answers.
src/medical_rag/analysis/report.py           [STUB] Summary tables + figures + markdown report.

scripts/build_index.py         CLI: build the LanceDB retrieval index from a corpus. IMPLEMENTED.
scripts/probe_models.py        CLI: repeatable live check of the served models — per-model cost/latency/throughput, `finish_reason`, the one-shot ANSWER-recovery path, and a saved copy of every chain of thought. Exits non-zero on endpoint or recovery failure. IMPLEMENTED.
scripts/exploration/closed_book.py   Closed-book probe over a pinned dev-split sample, saving every chain of thought. [Phase 5, planned]
scripts/exploration/raw_rag.py       Same, with retrieval + reranking wired in. [Phase 6, planned]
scripts/exploration/_common.py       Shared row/chain writer for the two above; promoted into experiment/runner.py at Phase 9 if it earns it. [Phase 5, planned]
scripts/run_pilot.py           [STUB] CLI: small-scale validation run.
scripts/run_experiment.py      [STUB] CLI: full experiment run.

tests/__init__.py
tests/test_pipeline.py         Smoke test: package imports and has __version__.
tests/test_config.py           load_config() success + undefined-model-reference rejection.
tests/test_data_loading.py     MedQA loader schema/count (live) + StatPearls chunker logic (fixture, no network).
tests/test_retrieval.py        Embedder shape, index build/save/load round-trip, retrieve() relevance, rerank() reordering.
tests/test_generation.py       Prompt builders, parse_answer tiers, LLMClient over httpx.MockTransport, + one `live` endpoint test.
tests/test_reformulator.py     [STUB] Placeholder for reformulator tests.
tests/test_verifier.py         [STUB] Placeholder for verifier tests.

data/                          Gitignored. Cached corpora (data/statpearls/) and the built index (data/index/statpearls/).
experiments/                   Tracked (deliberately NOT gitignored). Per-phase FINDINGS.md plus the failure chains each phase chose to keep — Phase 10's evidence base. [Phase 5 onwards]
experiments/README.md          The convention in one page: what is tracked, the rules, how to start a new FINDINGS file.
experiments/TEMPLATE.md        Skeleton that experiments/phaseN/FINDINGS.md is copied from.
outputs/probes/                Gitignored. probe_models.py rows + chains; in use since Phase 4.
outputs/exploration/           Gitignored. Raw run dirs from scripts/exploration/ (results.jsonl, chains/<stamp>/, context.md).
outputs/runs/                  Gitignored. Pilot and confirmatory results (Phases 9/13) — the only outputs/ tree RESULTS.md may cite.
```