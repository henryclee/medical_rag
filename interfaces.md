# Interfaces

Part of the [plan](./PLAN.md). This file defines contracts, not implementations.

## Source of truth

- If interfaces exist in code, code wins.
- If they don't exist yet, this file is the source of truth.

## Conventions

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
    timeout: float = 120.0  # added in Phase 4: per-request HTTP timeout for LLMClient
    max_retries: int = 4    # added in Phase 4: openai-SDK transport retries (429/5xx/connection)

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

def load_env(config_path: str | Path = "config/default.yaml", *,
             env_path: str | Path | None = None) -> Path | None
    # Merges the gitignored .env into os.environ and returns the file read (None if
    # there is none, which is not an error). Searches cwd and its ancestors, then
    # the repo root `config_path` implies. Fill-only: never overwrites an exported
    # variable, never exports an empty value (so LLMError still fires), never
    # expands `$` inside a key, and never logs values -- names only.

def load_config(path: str | Path = "config/default.yaml", *,
                env_path: str | Path | None = None,
                load_env_file: bool = True) -> ExperimentConfig
    # Reads default.yaml, models.yaml, conditions.yaml from the same directory as
    # `path`, shallow-merges their top-level keys, and validates the result.
    # Calls load_env() first (unless load_env_file=False), so an api_key_env whose
    # value lives only in .env is populated before any LLMClient reads it.
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

### `medical_rag.generation.llm` ✅ implemented

```python
class LLMError(RuntimeError): ...   # endpoint failure, or a response with nothing usable

class LLMClient:
    config: ModelConfig
    client: openai.AsyncOpenAI      # base_url=config.base_url, api_key=os.environ[config.api_key_env],
                                    # timeout=config.timeout, max_retries=config.max_retries

    def __init__(
        self,
        config: ModelConfig,
        *,
        http_client: httpx.AsyncClient | None = None,   # test seam: pass an httpx.MockTransport client
    ) -> None
        # Raises LLMError naming config.api_key_env when neither os.environ nor the
        # .env that config.load_env() merged supplies it.

    async def agenerate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        system_prompt: str | None = SYSTEM_PROMPT,   # pass None to send no system turn
    ) -> str
        # Per-call kwargs override the ModelConfig defaults (note: 0.0 is a real
        # value, not "unset"). Sends max_completion_tokens, temperature, top_p.
        # `seed` is deliberately not a parameter: measured inert on this endpoint
        # (experiments/phase5/FINDINGS.md), and passing it disabled server-side
        # batching, so it is never sent (see PLAN.md open question 10).
        # Every call logs the full prompt, full response, model name, and a
        # timestamp (DEBUG carries prompt/response text, INFO carries a one-line
        # summary with finish_reason, elapsed, and token counts), per the
        # "log everything" requirement in the original spec.
        # Warns when finish_reason == "length"; raises LLMError when content is
        # empty/absent (this endpoint emits reasoning-only completions when the
        # token budget runs out) or when the endpoint errors after retries.

    async def agenerate_structured(
        self,
        prompt: str,
        output_type: type[ModelT],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        system_prompt: str | None = SYSTEM_PROMPT,
    ) -> ModelT
        # pydantic-ai Agent + PromptedOutput (JSON demanded in the prompt, parsed
        # from content) over OpenAIProvider(openai_client=self.client) -- reuses
        # the same base_url/key/timeout/retries as agenerate(). PromptedOutput
        # rather than ToolOutput because this endpoint's tool calling is broken.
        # Any validation or transport failure surfaces as LLMError.

    async def aclose(self) -> None
    async def __aenter__ / __aexit__      # async context manager; closes the session
```

Note: there is no synchronous `generate()`. The whole pipeline is async, and a
sync wrapper here would only invite blocking the event loop in Phase 9's
concurrent runner.

### `medical_rag.generation.prompt` ✅ implemented

```python
SYSTEM_PROMPT: str                 # "You are a medical expert answering USMLE-style MCQs."
ANSWER_FORMAT_INSTRUCTION: str     # think step by step, end with exactly "ANSWER: X"

def format_options(options: dict[str, str]) -> str
    # letter-sorted "A. text" lines.

def format_context(chunks: Sequence[RetrievedChunk]) -> str
    # numbered "[n] title" + content blocks -- the "[n]" markers Phase 6 checks
    # for when confirming chunks reached the prompt.

def build_answer_prompt(
    question: str,
    options: dict[str, str],
    context_chunks: list[RetrievedChunk] | None,
) -> str
    # context_chunks=None (or []) renders the closed-book variant of the prompt.
    # The RAG variant appends an adherence instruction ("base your answer
    # primarily on the excerpts above; if they do not contain the answer, use
    # your best clinical judgment") -- the wording that Phase 10 will revisit,
    # since how strongly the model is bound to the excerpts is a first-order
    # driver of whether RAG helps or hurts.

def build_reformulation_prompt(question: str) -> str
    # Deliberately takes the question ONLY: leaking option wording into the
    # reformulated query would let retrieval fetch the answer instead of the
    # information need. Demands {"information_need": "..."} JSON.

def build_verification_prompt(question: str, chunks: list[RetrievedChunk]) -> str
    # Batched: all candidate chunks go into ONE prompt (one LLM call per
    # question, not one per chunk) -- see Phase 2 design-review note in
    # section 6 on cost/latency at the scale of many conditions x questions.
    # Each block prints its chunk_id verbatim so the Verifier can
    # join verdicts back without fuzzy matching; demands
    # {"verdicts": [{"chunk_id", "keep", "reason"}]} JSON.

def parse_answer(text: str, valid_letters: set[str] | None = None) -> str | None
    # Returns an upper-case option letter, or None when nothing parseable was
    # found (callers log parsed_answer=None / is_correct=False; never guesses).
    # Three preference tiers, each tried newest-match-first so a decoy "ANSWER:"
    # quoted inside the chain of thought cannot beat the final answer:
    #   1. an "ANSWER: X" marker (tolerates markdown/parentheses/trailing prose)
    #   2. prose ("the answer is X" / "would be X")
    #   3. a bare letter standing alone on the last line (capped at 40 chars)
    # Letters outside valid_letters are rejected, so a model that answers "E" to
    # an A-D question counts as unparseable rather than wrong-but-scored.
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
        # Writes outputs/runs/{condition.id}/results.jsonl incrementally, one
        # line per question (one completion per question per condition --
        # PLAN.md open question 10 dropped the repeated-seed-draws design).
        # Resumable: on start, reads any existing file for `condition` and
        # skips question_ids already present (exact resume-manifest format:
        # open question, see section 6).

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