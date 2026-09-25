# Medical RAG Experiment — Project Plan

## 1. Context

This project is a reproducible pipeline for running retrieval-augmented generation (RAG) experiments on medical board-exam questions (MedQA-USMLE): it can evaluate how interventions — retrieval, reranking, query reformulation, and answer verification — affect accuracy, and whether that effect depends on which LLM is answering. The pipeline itself is the artifact; the current study (built around these levers) is one instantiation of it. Which levers are active and what the condition set looks like are provisional until the design freeze (Phase 10). Generation is never local: every model is served remotely behind an OpenAI-compatible endpoint, and the pipeline only ever makes HTTP calls to it. Retrieval runs against StatPearls, a clinical reference corpus, embedded and indexed once and then queried repeatedly across the experiment.

As of this document, the project is a pip-installable `src`-layout package (`medical_rag`) with a working configuration layer, both data loaders, and a fully functional retrieval subsystem — including a real, populated vector index built from the actual StatPearls corpus. Everything from query reformulation and answer verification through the experiment runner, statistics, and report generation is still a stub. The end state is a repo where `python scripts/build_index.py` and `python scripts/run_experiment.py` reproduce the study from scratch, producing `outputs/runs/summary.csv`, `outputs/runs/summary.json`, figures under `outputs/figures/`, and a `RESULTS.md` reporting effect sizes and confidence intervals for the contrasts actually run.

The experiment design described above — the condition set, prompt wording, `min_chunks`, the reformulator's retry policy, seed count, and the primary significance test — is provisional. Phases 4-9 exist to explore and pressure-test that design, running against a held-out dev split rather than the test split. A design freeze (Phase 10) follows Phase 9: the design is finalized there, before the confirmatory run against the test split in Phase 13.

## 2. Environment

- **Python**: 3.13.4, in a project-local `.venv` (`requires-python = ">=3.11"` in `pyproject.toml`).
- **Runtime dependencies** (exact-pinned in `pyproject.toml`):
  `pydantic-ai==2.48.0`, `openai==3.19.0`, `httpx==0.28.1`, `lancedb==0.39.0`, `sentence-transformers==6.1.0`, `datasets==5.0.1`, `pandas==3.0.6`, `loguru==0.7.3`, `statsmodels==0.15.0`, `matplotlib==3.11.2`, `pyyaml==6.0.3`.
  `sentence-transformers` pulls in `torch==2.14.0` and `transformers==5.17.0` transitively (CPU/MPS only — no CUDA, no `bitsandbytes`, no local model loading of the LLMs under test).
- **Dev dependencies**: `pytest==9.1.1`, `pytest-asyncio==1.4.0` (installed via the `[dev]` extra, never separately).
- **Running tests**: `.venv/bin/pytest` from the repo root (`testpaths = ["tests"]`, `asyncio_mode = "auto"` are set in `pyproject.toml`, so no extra flags are needed).
- **Running the pipeline end to end** (current state — retrieval only):
  1. `.venv/bin/python scripts/build_index.py --corpus statpearls` — idempotent; downloads/caches the StatPearls corpus under `data/statpearls/` on first run (~1.9GB from NCBI, one-time), then embeds it and writes a LanceDB table to `data/index/statpearls/`. Pass `--force` to rebuild.
  2. Once Phases 4–13 (below) are implemented, this will extend to `.venv/bin/python scripts/run_pilot.py` (small-scale validation on the dev split) and `.venv/bin/python scripts/run_experiment.py` (the full confirmatory run over the test split, using the frozen condition set).
- **Why LanceDB, not FAISS**: the original design called for FAISS. In practice, `faiss-cpu` and `torch` both bundle their own copy of the OpenMP runtime; loading both in one process and querying an index at real corpus scale (~380k vectors) reliably segfaults on this machine (confirmed via `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`, reproduced with synthetic data at the same scale, and unfixed by the standard `KMP_DUPLICATE_LIB_OK`/single-thread workarounds). LanceDB is an embedded, disk-backed vector store with a Rust core (no OpenMP dependency), verified crash-free at full scale before the codebase was switched over. It also defaults to exact brute-force kNN search whenever no ANN index is explicitly created on a table, which is what preserves the original design's exact-search semantics (equivalent to FAISS's `IndexFlatIP`) — nothing approximate was introduced.
- **Why `BAAI/bge-small-en-v1.5`**: the embedding model specified in the original design — a small, fast, retrieval-tuned model that supports an asymmetric query/passage convention (see `Embedder` below), which the codebase implements explicitly.
- **Why `cross-encoder/ms-marco-MiniLM-L-6-v2`**: the reranker specified in the original design — a standard, general-purpose passage-relevance cross-encoder, applied to the top-`k` candidates LanceDB returns when the active condition's `params` calls for reranking (see `ConditionConfig`/`Pipeline` in section 4 — reranking is a lever, not an unconditional step).
- **Device selection note**: on this machine, `torch`'s CPU backend produces NaN output from the cross-encoder's forward pass (confirmed by inspecting intermediate hidden states); MPS does not. `Embedder`'s default device order is therefore `cuda` > `mps` > `cpu`, and both the embedding model and the reranker (which shares the embedder's device) run on MPS here.

## 3. File layout

```
config/default.yaml            Benchmark name, output dir, and retrieval defaults (corpus, embedding/reranker model, chunk/top-k sizes).
config/models.yaml             Per-model remote endpoint config (base_url, api_model_name, api_key_env, sampling params) for model_a/model_b — placeholders, not yet pointed at real endpoints.
config/conditions.yaml         Experiment conditions for the current study; provisional until the Phase 10 design freeze, named and ID-ordered to match the delta formulas metrics.py will use.
pyproject.toml                 Package metadata, exact-pinned dependencies, pytest config.
requirements.lock.txt          Full `pip freeze` snapshot of the environment.
.gitignore                     Excludes .venv/, data/, outputs/, caches.
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
src/medical_rag/generation/llm.py      [STUB] Remote OpenAI-compatible LLM client.
src/medical_rag/generation/prompt.py   [STUB] Prompt templates (answer/reformulation/verification).
src/medical_rag/experiment/__init__.py Experiment subpackage docstring.
src/medical_rag/experiment/conditions.py [STUB] Condition -> pipeline assembly.
src/medical_rag/experiment/runner.py     [STUB] Main experiment loop + per-question result logging.
src/medical_rag/experiment/metrics.py    [STUB] Accuracy, deltas, significance testing.
src/medical_rag/analysis/__init__.py   Analysis subpackage docstring.
src/medical_rag/analysis/failure_taxonomy.py [STUB] Failure categorization for incorrect answers.
src/medical_rag/analysis/report.py           [STUB] Summary tables + figures + markdown report.

scripts/build_index.py         CLI: build the LanceDB retrieval index from a corpus. IMPLEMENTED.
scripts/run_pilot.py           [STUB] CLI: small-scale validation run.
scripts/run_experiment.py      [STUB] CLI: full experiment run.

tests/__init__.py
tests/test_pipeline.py         Smoke test: package imports and has __version__.
tests/test_config.py           load_config() success + undefined-model-reference rejection.
tests/test_data_loading.py     MedQA loader schema/count (live) + StatPearls chunker logic (fixture, no network).
tests/test_retrieval.py        Embedder shape, index build/save/load round-trip, retrieve() relevance, rerank() reordering.
tests/test_reformulator.py     [STUB] Placeholder for reformulator tests.
tests/test_verifier.py         [STUB] Placeholder for verifier tests.

data/                          Gitignored. Cached corpora (data/statpearls/) and the built index (data/index/statpearls/).
outputs/                       Gitignored. Reserved for experiment run results (not yet used).
```

## 4. Interfaces

Status markers: **✅ implemented and tested** / **🔲 planned, not yet written** (signatures below for planned modules are the intended design and may be refined during that phase).

### `medical_rag.config` ✅

```python
class ModelConfig(BaseModel):
    name: str
    base_url: str
    api_model_name: str
    api_key_env: str        # name of an env var to read the API key from; no secrets stored in YAML
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int

class RetrievalConfig(BaseModel):
    corpus: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k_retrieve: int
    top_k_rerank: int
    reranker_model: str

class VerifierConfig(BaseModel):
    min_chunks: int

class ReformulatorConfig(BaseModel):
    max_retries: int

class ConditionConfig(BaseModel):
    id: str
    name: str
    model: str               # must be a key in ExperimentConfig.models
    retrieval: bool
    reformulation: bool
    verification: bool
    seeds: list[int]
    params: dict[str, Any] = {}   # free-form lever parameters discovered useful
                                   # during exploration (e.g. {"reranking": true}),
                                   # so a new lever never needs its own typed field

class ExperimentConfig(BaseModel):
    models: dict[str, ModelConfig]
    retrieval: RetrievalConfig
    verifier: VerifierConfig
    reformulator: ReformulatorConfig
    conditions: list[ConditionConfig]
    benchmark: str
    dev_split: str            # split Phases 5-9 run exploration against
    test_split: str           # split Phase 13's confirmatory run uses
    output_dir: str
    # @model_validator(mode="after"): raises ValueError if any ConditionConfig.model
    # is not a key in `models`.

def load_config(path: str | Path = "config/default.yaml") -> ExperimentConfig
    # Reads default.yaml, models.yaml, conditions.yaml from the same directory as
    # `path`, shallow-merges their top-level keys, and validates the result.
```

### `medical_rag.data.load_medqa` ✅

```python
EXPECTED_TEST_COUNT: int = 1273
DATASET_ID: str = "GBaker/MedQA-USMLE-4-options"

class MedQAQuestion(BaseModel):
    id: str
    question: str
    options: dict[str, str]      # keys are option letters, e.g. "A".."D"
    answer_idx: str               # a key into `options`
    answer_text: str
    meta_info: str | None = None

def load_medqa(split: str = "test") -> list[MedQAQuestion]
    # Loads via `datasets.load_dataset(DATASET_ID, split=split)`. Logs a WARNING
    # (does not raise) if split == "test" and len(result) != EXPECTED_TEST_COUNT.
```

### `medical_rag.data.load_statpearls` ✅

```python
STATPEARLS_TARBALL_URL: str = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"
EXPECTED_SNIPPET_COUNT: int = 301_202   # MedRAG's documented count; NCBI's live corpus has since grown

class StatPearlsChunk(BaseModel):
    chunk_id: str
    title: str
    content: str
    contents: str            # title + content concatenated (MedRAG's own retrieval convention)
    source: str = "statpearls"

def load_statpearls(cache_dir: str | Path = "data/statpearls") -> list[StatPearlsChunk]
    # Returns the cached corpus (data/statpearls/statpearls_corpus.jsonl) if present;
    # otherwise downloads the NCBI tarball, extracts .nxml articles, chunks them via
    # the module's private, vendored port of MedRAG's src/data/statpearls.py
    # algorithm, and caches the result before returning. Logs a WARNING (does not
    # raise) if the resulting count != EXPECTED_SNIPPET_COUNT.

# Private helpers (implementation detail, vendored from MedRAG, not part of the
# public interface): _download_tarball, _extract_nxml_files,
# _extract_article_snippets, _concat, _extract_text, _is_subtitle,
# _ends_with_ending_punctuation.
```

### `medical_rag.data.chunking` ✅

```python
def chunk_document(text: str, tokenizer: Any, chunk_size: int, overlap: int) -> list[str]
    # `tokenizer` must expose .encode(text, add_special_tokens=False) -> list[int]
    # and .decode(token_ids) -> str (e.g. a HF tokenizer). Returns overlapping
    # windows of at most `chunk_size` tokens, striding by (chunk_size - overlap).

def chunk_corpus(documents: list[dict], tokenizer: Any, chunk_size: int, overlap: int) -> list[dict]
    # `documents`: list of {"content": str, ...arbitrary extra keys...}.
    # Returns a flat list of {**original_doc, "content": <chunk piece>, "chunk_index": int},
    # one entry per resulting chunk, preserving all of the original document's other keys.
```

### `medical_rag.retrieval.embedder` ✅

```python
BGE_QUERY_INSTRUCTION: str = "Represent this sentence for searching relevant passages: "

def _default_device() -> str    # "cuda" | "mps" | "cpu", in that preference order

class Embedder:
    device: str
    model: SentenceTransformer

    def __init__(self, model_name: str, device: str | None = None) -> None
        # device defaults to _default_device() when omitted.

    def embed_texts(self, texts: list[str], batch_size: int = 128) -> np.ndarray
        # shape (len(texts), model_dim), dtype float32, L2-normalized.
        # No query instruction prefix -- use for corpus/passage text.

    def embed_query(self, query: str) -> np.ndarray
        # shape (model_dim,), dtype float32, L2-normalized.
        # Prepends BGE_QUERY_INSTRUCTION before embedding.
```

### `medical_rag.retrieval.index` ✅

```python
TABLE_NAME: str = "chunks"

def build_index(chunks: list[dict], embeddings: np.ndarray, path: str | Path) -> lancedb.table.Table
    # `chunks[i]` must have "chunk_id", "title", "content" keys (extra keys ignored).
    # `embeddings.shape == (len(chunks), dim)`, dtype float32.
    # Connects to (creating if needed) a LanceDB database at `path` and
    # (over)writes a table named TABLE_NAME with columns:
    #   chunk_id: string, title: string, content: string,
    #   vector: fixed_size_list<float32>[dim]

def load_index(path: str | Path) -> lancedb.table.Table
    # Opens the existing TABLE_NAME table at `path` for querying.
```

### `medical_rag.retrieval.retriever` ✅

```python
class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    title: str
    score: float          # similarity; higher = more relevant, regardless of source

class Retriever:
    embedder: Embedder
    table: lancedb.table.Table
    reranker_model: str

    def __init__(self, embedder: Embedder, table: lancedb.table.Table, reranker_model: str) -> None

    @property
    def cross_encoder(self) -> CrossEncoder
        # Lazily constructed on first access, on embedder.device.

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]
        # Embeds `query` (with BGE prefix), does exact cosine-similarity search
        # against `table`, returns up to `top_k` RetrievedChunk sorted by score desc.

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]
        # Scores each (query, chunk.content) pair with the cross-encoder, returns
        # the top `top_k` re-sorted by that score (chunk.score is overwritten).
        # Returns [] if `chunks` is empty.
```

### `scripts/build_index.py` ✅ (CLI, not a library interface)

```
--corpus {statpearls}          default: statpearls
--output-dir PATH              default: data/index/<corpus>
--data-cache-dir PATH          default: data/<corpus>
--embedding-model NAME          default: BAAI/bge-small-en-v1.5
--batch-size INT                default: 128
--force                         rebuild even if the index directory already exists
```

### `medical_rag.generation.llm` 🔲 planned

```python
class LLMClient:
    config: ModelConfig
    client: openai.AsyncOpenAI      # base_url=config.base_url, api_key=os.environ[config.api_key_env]

    def __init__(self, config: ModelConfig) -> None

    async def agenerate(
        self,
        prompt: str,
        *,
        seed: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str
        # Per-call kwargs override the ModelConfig defaults. Every call logs
        # (loguru) the full prompt, full response, model name, seed, and a
        # timestamp, per the "log everything" requirement in the original spec.
```

### `medical_rag.generation.prompt` 🔲 planned

```python
def build_answer_prompt(
    question: str,
    options: dict[str, str],
    context_chunks: list[RetrievedChunk] | None,
) -> str
    # context_chunks=None (or []) renders the closed-book variant of the prompt.

def build_reformulation_prompt(question: str) -> str

def build_verification_prompt(question: str, chunks: list[RetrievedChunk]) -> str
    # Batched: all candidate chunks go into ONE prompt (one LLM call per
    # question, not one per chunk) -- see Phase 2 design-review note in
    # section 6 on cost/latency at the scale of many conditions x seeds x
    # questions.
```

### `medical_rag.modules.reformulator` 🔲 planned

```python
class ReformulationResult(BaseModel):
    information_need: str

class Reformulator:
    def __init__(self, llm_client: LLMClient, max_retries: int = 2) -> None
        # max_retries is normally sourced from ReformulatorConfig.max_retries
        # via build_pipeline, not left at this Python default.

    async def reformulate(self, question: str, answer_options: dict[str, str]) -> str
        # Uses pydantic-ai structured output (ReformulationResult). Validates
        # non-empty and that the result does not contain any answer_options
        # value verbatim. On repeated validation failure (after max_retries),
        # falls back to the original `question` and logs a WARNING with the
        # question id. The retry count now has a config home
        # (ReformulatorConfig.max_retries); its frozen value is still a
        # Phase 10 decision (section 6, item 3).
```

### `medical_rag.modules.verifier` 🔲 planned

```python
class ChunkVerdict(BaseModel):
    chunk_id: str
    keep: bool
    reason: str

class VerifierResult(BaseModel):
    verdicts: list[ChunkVerdict]

class Verifier:
    def __init__(self, llm_client: LLMClient, min_chunks: int = 2) -> None
        # min_chunks is normally sourced from VerifierConfig.min_chunks via
        # build_pipeline, not left at this Python default.

    async def verify(self, question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]
        # ONE batched LLM call scores all candidate chunks at once (VerifierResult),
        # not one call per chunk. Filters to chunk_id in {v.chunk_id for v in
        # verdicts if v.keep}. If fewer than min_chunks survive, falls back to the
        # top `min_chunks` of the original `chunks` by score and logs a WARNING
        # with the question id (the "no silent fallbacks" rule from the spec).
```

### `medical_rag.experiment.conditions` 🔲 planned

```python
@dataclass
class Pipeline:
    condition: ConditionConfig
    llm_client: LLMClient
    retriever: Retriever | None        # None when condition.retrieval is False
    reformulator: Reformulator | None  # None when condition.reformulation is False
    verifier: Verifier | None          # None when condition.verification is False
    # Reranking is not a separate component: when retriever is not None, callers
    # always call retriever.retrieve(); they call retriever.rerank() afterward
    # only when condition.params.get("reranking") is truthy -- reranking is read
    # from params like any other exploratory lever, not a dedicated field.

def build_pipeline(condition: ConditionConfig, components: dict[str, Any]) -> Pipeline
    # `components` holds shared singletons built once per run:
    #   {"llm_clients": dict[str, LLMClient], "retriever": Retriever,
    #    "reformulator": Reformulator, "verifier": Verifier}
    # build_pipeline wires the subset each condition actually uses.
```

### `medical_rag.experiment.runner` 🔲 planned

```python
class QuestionResult(BaseModel):
    condition_id: str
    seed: int
    question_id: str
    question: str
    reformulated_query: str | None
    retrieved_chunk_ids: list[str]
    verifier_decisions: list[ChunkVerdict] | None
    raw_model_output: str
    parsed_answer: str | None    # None if the answer letter couldn't be parsed
    correct_answer: str
    is_correct: bool
    timestamp: str                # ISO 8601

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, components: dict[str, Any]) -> None

    async def run_condition(
        self, condition: ConditionConfig, dataset: list[MedQAQuestion]
    ) -> list[QuestionResult]
        # Writes outputs/runs/{condition.id}/seed_{seed}.jsonl incrementally, one
        # line per question, per seed. Resumable: on start, reads any existing
        # file for (condition, seed) and skips question_ids already present
        # (exact resume-manifest format: open question, see section 6).

    async def run_all(
        self, conditions: list[ConditionConfig], dataset: list[MedQAQuestion]
    ) -> dict[str, list[QuestionResult]]
        # Keyed by condition.id.
```

### `medical_rag.experiment.metrics` 🔲 planned

```python
def compute_accuracy(results: list[QuestionResult]) -> float

def bootstrap_ci(
    results_a: list[QuestionResult],
    results_b: list[QuestionResult],
    n_resamples: int = 1000,
) -> tuple[float, float, float]
    # Returns (point_estimate_delta, ci_low, ci_high) for acc(b) - acc(a).

def mcnemar_test(
    results_a: list[QuestionResult], results_b: list[QuestionResult]
) -> tuple[float, float]
    # Returns (statistic, p_value); paired on question_id. Primary significance
    # test per the Phase 2 design review (replaces the original spec's
    # bootstrap + paired-t-test + McNemar + factorial-ANOVA combination --
    # exact final choice of primary method is an open question, section 6).

def summarize_results(results_by_condition: dict[str, list[QuestionResult]]) -> pd.DataFrame
```

### `medical_rag.analysis.failure_taxonomy` 🔲 planned

```python
FailureCategory = Literal[
    "distraction", "contradiction", "context_mismatch", "over_reliance",
    "reasoning_failure", "format_failure", "dataset_error",
]

def classify_failure(result: QuestionResult) -> FailureCategory

def build_failure_taxonomy(results: list[QuestionResult]) -> pd.DataFrame
```

### `medical_rag.analysis.report` 🔲 planned

```python
def generate_summary_table(results_by_condition: dict[str, list[QuestionResult]]) -> pd.DataFrame

def plot_condition_comparison(summary: pd.DataFrame) -> matplotlib.figure.Figure

def generate_report(results_by_condition: dict[str, list[QuestionResult]], output_dir: str | Path) -> None
    # Writes summary.csv, summary.json, and figures under output_dir.
```

### `scripts/run_pilot.py` / `scripts/run_experiment.py` 🔲 planned (CLIs)

```
run_pilot.py:    --sample-size INT (default 50)  --conditions LIST[str] (default: all)  --output-dir PATH
run_experiment.py: --config PATH (default config/default.yaml)  --output-dir PATH  --dry-run (validate config only, no LLM calls)
```

`run_pilot.py` loads questions from `config.dev_split`; `run_experiment.py` loads questions from `config.test_split`. Neither takes a `--split` override -- the dev/test separation is enforced by which script you run, not a flag either could get passed by mistake.

## 5. Phases

Phases 4-9 are exploratory. The `🔲 planned` interfaces in section 4 for the modules those phases touch are provisional and may change based on what those phases find.

**Phase 1 — Scaffold.** The project is a `src`-layout, pip-installable package (`pip install -e ".[dev]"`) named `medical_rag`, with a subpackage per pipeline stage (`data`, `retrieval`, `modules`, `generation`, `experiment`, `analysis`) and top-level `config/`, `scripts/`, `tests/`, gitignored `data/` and `outputs/`. `pyproject.toml` declares exact-pinned runtime and dev dependencies and configures pytest. A smoke test (`tests/test_pipeline.py`) confirms the package imports and exposes `__version__`.

**COMPLETED**

**Phase 2 — Configuration and data loading.** `medical_rag.config.load_config()` loads and validates three YAML files (`config/default.yaml`, `models.yaml`, `conditions.yaml`) into a single `ExperimentConfig`, failing loudly if any condition references an undefined model. `load_medqa()` loads the 1,273-question MedQA-USMLE test set from Hugging Face (`GBaker/MedQA-USMLE-4-options`) into typed `MedQAQuestion` records. `load_statpearls()` builds the StatPearls corpus directly from its authoritative source — NCBI's raw NLM-XML archive plus a vendored, faithful port of MedRAG's own chunking algorithm — rather than relying on the official (empty) `MedRAG/statpearls` Hugging Face repo or unverified community mirrors, and caches the result locally after the first (~1.9GB) build.

**COMPLETED**

**Phase 3 — Retrieval.** `Embedder` wraps `BAAI/bge-small-en-v1.5` with BGE's asymmetric query/passage convention (an instruction prefix on queries only). `medical_rag.retrieval.index` builds and loads an exact-search LanceDB vector table. `Retriever` combines the two, plus a `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker, into `retrieve()` + `rerank()`. `scripts/build_index.py` drives the whole load → embed → index → save pipeline from the command line. A real index exists on disk at `data/index/statpearls/`: 380,454 chunks from 9,652 StatPearls articles (NCBI's live corpus has grown since MedRAG's 301,202-snippet paper snapshot), embedded and indexed, manually verified against real MedQA questions to return topically relevant results.

**COMPLETED**

**Phase 4 — Generation client + prompts.** Implement `LLMClient` and the three prompt builders per section 4. Files: `src/medical_rag/generation/llm.py`, `src/medical_rag/generation/prompt.py`, `tests/test_generation.py` (new). *Verify*: unit tests for the prompt builders (string content contains question/options/context as expected); `LLMClient.agenerate()` returns a non-empty string against a real endpoint once `config/models.yaml` is filled in with actual values (see section 6, item 1).

**Phase 5 — Closed-book generation smoke test.** Run `LLMClient` + `build_answer_prompt(context_chunks=None)` over 20 real MedQA questions from the dev split. No new production files — a throwaway script or REPL session. *Verify*: every response parses to a valid option letter via whatever `ANSWER: [letter]` parsing logic is chosen; report rough accuracy.

**Phase 6 — Raw RAG end-to-end.** Wires `Retriever` (Phase 3) + `LLMClient` (Phase 4) together manually, against the dev split. No new production files. *Verify*: run on the same 20 dev-split questions with retrieval on; confirm retrieved chunks appear in the rendered prompt and answers still parse.

**Phase 7 — Reformulator.** Implement `Reformulator` per section 4, including the retry/fallback policy (decide exact retry count, section 6 item 2). Files: `src/medical_rag/modules/reformulator.py`, `tests/test_reformulator.py`. *Verify*: unit tests with a stub `LLMClient` cover both the success path and the fallback-to-original-question path, asserting the fallback is logged.

**Phase 8 — Verifier.** Implement `Verifier` per section 4 as a single batched call per question. Files: `src/medical_rag/modules/verifier.py`, `tests/test_verifier.py`. *Verify*: unit tests with a stub `LLMClient` cover the normal filtering path and the `min_chunks` fallback path, asserting the fallback is logged with the question id.

**Phase 9 — Experiment runner + pilot.** Implement `Pipeline`/`build_pipeline`, `ExperimentRunner`, and the resumable JSONL logging scheme (decide exact resume mechanism, section 6 item 3), running against the dev split. Files: `src/medical_rag/experiment/conditions.py`, `src/medical_rag/experiment/runner.py`, `scripts/run_pilot.py`, `tests/test_pipeline_integration.py` (new). *Verify*: `python scripts/run_pilot.py --sample-size 50` completes without error across the currently-defined conditions at 1 seed each, and re-running it after an interrupt does not re-call the LLM for already-logged questions.

**Phase 10 — Design freeze.** Based on Phases 5-9, decide the final condition set (including which levers are active per condition), prompt wording, `min_chunks`, the reformulator's retry policy, seed count, and primary significance test. Update `config/conditions.yaml` and section 4 of this plan to reflect the choices, and note what was dropped and why. Files: `config/conditions.yaml`, `PLAN.md` (section 4). *Verify*: `config/conditions.yaml` and section 4's `🔲 planned` interfaces match the choices made here, and this phase's entry (or a note alongside it) records what was dropped and why.

**Phase 11 — Metrics.** Implement `compute_accuracy`, `bootstrap_ci`, `mcnemar_test` (or the finally-chosen primary significance method, section 6 item 4), `summarize_results`. Files: `src/medical_rag/experiment/metrics.py`, `tests/test_metrics.py`. *Verify*: unit tests against small synthetic `QuestionResult` fixtures with known accuracy/delta values.

**Phase 12 — Failure taxonomy + report.** Implement classification heuristics (decide the exact heuristic set and whether manual labeling is in scope, section 6 item 5) and report generation. Files: `src/medical_rag/analysis/failure_taxonomy.py`, `src/medical_rag/analysis/report.py`, `tests/test_analysis.py`. *Verify*: `generate_report()` run against Phase 9's pilot results produces a non-empty `summary.csv`, `summary.json`, and at least one figure under a temp `outputs/` directory.

**Phase 13 — Full experiment run.** Files: `scripts/run_experiment.py`, `README.md` (fill in Usage), `RESULTS.md` (new). *Verify*: `python scripts/run_experiment.py` completes the full run over the test split (1,273 questions × the frozen condition set × 3 seeds), resumable if interrupted, produces `outputs/runs/summary.csv`, `outputs/runs/summary.json`, and `outputs/figures/`, and `RESULTS.md` reports effect sizes and confidence intervals for the contrasts actually run.

## 6. Open questions

Items 2-6 and 8 are decisions to make at the design freeze (Phase 10), not blockers before Phase 13. Items 1 and 7 stand as-is: 1 is a real blocker, 7 is a real caveat.

1. **Real model endpoints.** `config/models.yaml` still has placeholder `base_url`/`api_model_name` values for `model_a`/`model_b`. Phase 4 (generation) cannot be verified against a real endpoint until these are filled in — need the actual served URLs, model identifiers, and which env vars hold their API keys.
2. **What "two models" means.** The original design frames the study as testing whether a *model capability profile* moderates RAG effectiveness, but nothing yet specifies whether `model_a`/`model_b` are meant to be two capability tiers of the same family (e.g. a small vs. large Qwen), two different model families, or something else. This affects how the eventual `RESULTS.md` framing/conclusions should be written.
3. **Reformulator retry policy.** How many validation retries before falling back to the original question, and should that fallback count toward any per-run failure-rate metric?
4. **Resume/checkpoint mechanism.** `ExperimentRunner.run_condition()` needs to skip already-logged `(condition, seed, question_id)` triples on restart — worth deciding now whether that's done by re-reading the existing JSONL file's question ids at startup (simple, but O(n) reread each restart) or a separate lightweight manifest/index file.
5. **Primary significance test.** Phase 2's design review recommended replacing the original spec's four-method statistics stack (bootstrap CI + paired t-test + McNemar + factorial ANOVA) with one primary method plus bootstrap CIs as secondary presentation, tentatively McNemar's test. Worth confirming McNemar (simple, pairwise, matches binary per-question outcomes) over a mixed-effects logistic regression (handles the full factorial and question-level random effects in one model, but heavier to implement and explain) before Phase 11.
6. **Failure taxonomy scope.** The original spec calls for "a combination of heuristics and manual review" with a CLI to export a sample for manual labeling. Worth deciding whether manual labeling is in scope for this codebase at all, or whether `analysis/failure_taxonomy.py` should ship heuristics-only and treat manual review as an out-of-band step the user does separately.
7. **StatPearls corpus size drift.** The built corpus has 380,454 snippets vs. MedRAG's documented 301,202 (NCBI's live Bookshelf archive has grown since their paper snapshot). Worth deciding whether `RESULTS.md`/`README.md` should note this explicitly as a caveat against direct comparison to MedRAG's published numbers.
8. **Sampling temperature and seed count.** 3 seeds at temperature > 0 was decided in Phase 2, but the actual `temperature`/`top_p` values in `config/models.yaml` (0.7/0.9) are still placeholders carried over from the original spec's example — confirm these are the intended values for the real experiment before Phase 13's full run.
9. **`QuestionResult` doesn't snapshot active params.** `QuestionResult` (section 4) has no field echoing which `params` (or resolved `min_chunks`/`max_retries`) were active for that logged row — analysis has to join back against `conditions.yaml` by `condition_id`, which is only reliable if `conditions.yaml` is never edited after a run that produced already-logged results. Worth deciding at Phase 9 whether `QuestionResult` should snapshot these values directly for a self-contained, tamper-proof log record.
