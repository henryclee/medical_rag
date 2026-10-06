# Interfaces

Part of the [plan](./PLAN.md). This file defines contracts, not implementations.

## Source of truth

- If interfaces exist in code, code wins.
- If they don't exist yet, this file is the source of truth.

## Conventions

Status markers: **[implemented]** in the package and covered by `tests/` -- **[frozen]**
Phases 5-6 evidence tooling: it runs, it is cited by a `FINDINGS.md`, and it is not
the mainline -- nothing new builds on it -- **[unscheduled]** importable and working,
but no phase will run it until the backlog says so.

Two structural rules the file assumes, because every signature below depends on them:
nothing under `src/` imports from `scripts/` (the tree is arranged so the measurement
code has no dependency on a CLI), and `scripts/*.py` are thin wrappers over a package
`main()` so `ls scripts/` still shows the study's commands.

---

## Invariants

These three are the ones a contributor breaks without noticing. Each has already
produced a number this repo published and then had to walk back.

**1. A strategy name must be in `BASE_STRATEGIES`, and both the grid and the shipped
path go through `retrieve_base()`.** The harness only judges the union of chunks the
strategies it ran actually surfaced, so a new lever that is not in the grid has no
verdicts and scores zero -- which reads as "the lever does not help" rather than "the
lever was never measured". Adding a lever means adding it to `retrieve_base()`'s fan-out;
measuring it against the existing cache measures the cache.

**2. A `*_rerank` list is capped by `rerank_k`, so its `recall@10` and `recall@20` are
capped-on-5 and must never be compared against an unreranked cell at the same k.** With
the shipped funnel (`top_k_retrieve: 20` -> `top_k_rerank: 5`) a reranked cell's list is
at most 5 chunks long, so `dense_rerank`'s 45/45/45 across k=5/10/20 is a ceiling on 5,
not a plateau. `RetrievalStrategy.__post_init__` rejects the config typo that would
produce this silently (`rerank_k > top_k` for a reranking strategy), but no code can
stop a reader from comparing two columns that were never comparable.

**3. A verdict is keyed `(question_id, chunk_id, judge_prompt_sha)`.** Chunk *content*
is not in the key, so two things orphan the cache: editing the rubric in
`eval/rubric.py` (which is the point -- a verdict is a fact about a chunk under one
rubric) and re-chunking the corpus (which changes `chunk_id` and invalidates every
verdict at once, costing the whole judge spend again). The second is why the corpus and
its chunking are frozen for this track; see `PLAN.md`'s constraints.

---

## `medical_rag.config` [implemented]

```python
class ModelConfig(BaseModel):
    name: str
    base_url: str
    api_model_name: str
    api_key_env: str        # name of an env var to read the API key from; no secrets in YAML
    max_new_tokens: int
    temperature: float
    top_p: float
    timeout: float = 120.0            # per-request HTTP timeout for LLMClient
    max_retries: int = 4              # openai-SDK transport retries (429/5xx/connection)
    answer_recovery_max_tokens: int = 0
    # Recovery budget for the one-shot rescue when a completion carries no parsable
    # ANSWER: line. 0 disables the rescue, so a caller can measure the raw
    # unanswered rate. Only the frozen answer runs use it; the judge never parses.

class RetrievalConfig(BaseModel):
    corpus: str
    embedding_model: str
    strategy: str = "dense_rerank"   # must be a member of strategy.BASE_STRATEGIES
    top_k_retrieve: int
    top_k_rerank: int
    reranker_model: str
    bm25_top_k: int = 20             # lexical depth, deliberately separate from top_k
    rrf_k: int = 60                  # RRF smoothing; 60 is fusion.RRF_K

class ExperimentConfig(BaseModel):
    models: dict[str, ModelConfig]
    retrieval: RetrievalConfig
    benchmark: str
    dev_split: str            # every measurement run uses this split, never the test split
    test_split: str
    output_dir: str
    # @model_validator(mode="after") -> check_retrieval_config_is_measurable:
    # builds RetrievalStrategy.from_config(self) and lets its ValueError propagate, so
    # a config that names a strategy nobody measures, or a funnel wider than the list
    # it reorders, fails at load instead of after an afternoon of judge calls.

def load_env(config_path: str | Path = "config/default.yaml", *,
             env_path: str | Path | None = None) -> Path | None
    # Merges the gitignored .env into os.environ and returns the file read (None if
    # there is none, which is not an error). Searches cwd and its ancestors, then the
    # repo root `config_path` implies. Fill-only: never overwrites an exported
    # variable, never exports an empty value (so LLMError still fires), never expands
    # `$` inside a key, and never logs values -- names only.

def load_config(path: str | Path = "config/default.yaml", *,
                env_path: str | Path | None = None,
                load_env_file: bool = True) -> ExperimentConfig
    # Reads default.yaml and models.yaml from the directory holding `path`,
    # shallow-merges their top-level keys, and validates the result. Two files:
    # `conditions.yaml` is gone with the generator-side study, and a stale copy in a
    # working tree is ignored rather than fatal. Calls load_env() first unless
    # load_env_file=False, so an api_key_env whose value lives only in .env is
    # populated before any LLMClient reads it.
```

`dev_split` is `train` and `test_split` is `test` in `config/default.yaml`. Every
retrieval number in this repo comes from `dev_split`; `test_split` is what a
confirmatory run would use and is not read by anything in the retrieval track.

## `medical_rag.paths` [implemented]

```python
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
```

The repo root, derived once. Modules that moved into the package used to compute it as
`Path(__file__).parents[n]`, where `n` was correct for the directory they were written
in -- a move keeps the code and changes the depth, which would have pointed
`DEFAULT_JUDGE_CACHE` inside `src/` and made every run "find" an empty cache, paying
twice while reporting a reuse rate of zero. Nothing else in the package counts parent
directories.

## `medical_rag.text` [implemented]

```python
def normalize(text: str) -> str
    # Casefold, drop typographic quotes/dashes, collapse whitespace.

def contains(needle: str, haystack: str) -> bool
    # Normalized substring test. False on an empty needle, which is not a match.

def chunk_match_rank(needle: str, chunks: Sequence[Any]) -> int | None
    # 1-based index of the first chunk whose `.content` contains `needle`, else None.
```

One implementation of "is this phrase inside that chunk?". Read `chunk_match_rank` with
both of its errors in mind: it under-counts (an excerpt explaining the mechanism without
the gold phrase reads as absent) and over-counts (an excerpt naming the gold option in
order to rule it out reads as present). Under-counting is the commoner error at this
corpus size, so a low hit rate is evidence about the metric before it is evidence about
the retriever, and it is never accuracy.

`retrieval.ceiling`'s corpus scan deliberately does **not** use this: it needs byte
offsets into the original body to render its excerpt window, and normalizing 380k bodies
per probe multiplies a scan R3 runs hundreds of times. The consequence is documented
there -- a typographic variant reads as a miss, so the probe's ceiling is a lower bound.

## `medical_rag.data.load_medqa` [implemented]

```python
EXPECTED_TEST_COUNT: int = 1273
DATASET_ID: str = "GBaker/MedQA-USMLE-4-options"

class MedQAQuestion(BaseModel):
    id: str
    question: str
    options: dict[str, str]      # keys are option letters, e.g. "A".."D"
    answer_idx: str              # a key into `options`
    answer_text: str
    meta_info: str | None = None

def load_medqa(split: str = "test") -> list[MedQAQuestion]
    # Loads via `datasets.load_dataset(DATASET_ID, split=split)`. Logs a WARNING
    # (does not raise) if split == "test" and len(result) != EXPECTED_TEST_COUNT.
    # The dev split this repo measures against is `train` (10,178 questions).
```

## `medical_rag.data.load_statpearls` [implemented]

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
```

The built corpus is 380,454 chunks from 9,652 articles. Chunking is by article section
(MedRAG's algorithm), not by a token window -- which is why `RetrievalConfig` carries no
chunk geometry: the two keys it used to have were read by nothing, and a knob that moves
nothing is a lie in a config file.

## `medical_rag.data.chunking` [unscheduled]

```python
def chunk_document(text: str, tokenizer: Any, chunk_size: int, overlap: int) -> list[str]
    # `tokenizer` must expose .encode(text, add_special_tokens=False) -> list[int]
    # and .decode(token_ids) -> str. Returns overlapping windows of at most
    # `chunk_size` tokens, striding by (chunk_size - overlap).

def chunk_corpus(documents: list[dict], tokenizer: Any, chunk_size: int, overlap: int) -> list[dict]
    # Returns a flat list of {**original_doc, "content": <piece>, "chunk_index": int}.
```

No caller today. It survives as the natural home of the backlog's chunking lever,
because re-chunking is the one experiment that invalidates the whole verdict cache
(invariant 3) and should therefore start from code that already exists rather than from
an argument about where to put it.

## `medical_rag.retrieval.embedder` [implemented]

```python
BGE_QUERY_INSTRUCTION: str = "Represent this sentence for searching relevant passages: "

class Embedder:
    device: str
    model: SentenceTransformer

    def __init__(self, model_name: str, device: str | None = None) -> None
        # device defaults to cuda | mps | cpu, in that preference order.

    def embed_texts(self, texts: list[str], batch_size: int = 128) -> np.ndarray
        # (len(texts), dim) float32, L2-normalized. No query prefix -- corpus/passage text.

    def embed_query(self, query: str) -> np.ndarray
        # (dim,) float32, L2-normalized. Prepends BGE_QUERY_INSTRUCTION.
```

## `medical_rag.retrieval.index` [implemented]

```python
TABLE_NAME: str = "chunks"

def build_index(chunks: list[dict], embeddings: np.ndarray, path: str | Path) -> lancedb.table.Table
    # (Over)writes TABLE_NAME with chunk_id/title/content plus vector
    # fixed_size_list<float32>[dim]. Exact search only: no ANN index at this size.

def load_index(path: str | Path) -> lancedb.table.Table
```

## `medical_rag.retrieval.retriever` [implemented]

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

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]
        # Embeds `query` (with the BGE prefix), exact cosine search, top_k by score desc.

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]
        # Cross-encoder over (query, chunk.content); returns top_k re-sorted and
        # OVERWRITES chunk.score with the reranker logit. Returns [] on empty input.
```

`rerank()` truncates and overwrites `score`. Anything that needs the pre-rerank ranking
or the cosine score must capture it first -- `retrieve_base()` keeps the dense list
precisely so fusion can be fed the untruncated input, and a row that logs only
post-rerank chunks cannot distinguish a bad retriever from a bad reranker.

## `medical_rag.retrieval.strategy` [implemented]

The single definition of what retrieval does. The grid and the shipped path both call
`retrieve_base()`, so a strategy means the same thing in a findings table as it does in
a call -- before this existed the composition lived in the harness while `raw_rag.py`
retrieved however it happened to, which is how a measured table drifts from the thing
anything would run.

```python
BASE_STRATEGIES: tuple[str, ...] = ("dense", "dense_rerank", "bm25", "hybrid", "hybrid_rerank")
QUERY_VARIANTS: tuple[str, ...] = ("orig", "reform")
METHOD_GRID: tuple[str, ...]          # "<strategy>__<variant>", 10 ids, variant-major
PRERANK_SOURCE: dict[str, str] = {"dense_rerank": "dense", "hybrid_rerank": "hybrid"}
LEGACY_METHOD_ALIASES: dict[str, str]  # pre-grid ids (`reform_dense`) -> grid ids

def method_id(base: str, variant: str) -> str          # rejects an off-grid pair
def split_method(method: str) -> tuple[str, str]
def base_of(method: str) -> str
def variant_of(method: str) -> str
def grid_for(variants: Sequence[str]) -> tuple[str, ...]
def canonical_method(name: str) -> tuple[str, bool]    # (grid id, was_legacy)

def retrieve_base(
    query: str, retriever: Any, bm25_index: Any, *,
    top_k: int, rerank_k: int,
    bm25_top_k: int | None = None,     # defaults to top_k
    rrf_k: int | None = None,          # None defers to fusion.RRF_K rather than restating it
    rerank_query: str | None = None,   # the string the cross-encoder sees
) -> dict[str, list[RetrievedChunk]]
    # The five lists for ONE query string: dense and BM25 are each retrieved once and
    # reused as the inputs to rerank and fusion. Keys are BASE_STRATEGIES.

@dataclass
class GridResult:
    lists: dict[str, list[RetrievedChunk]]      # keyed by METHOD_GRID id
    variant_queries: dict[str, str]             # what each variant actually searched on
    reused_variant: str | None                  # "reform" when the rewrite == the question
    rerank_with: str = "question"
    def methods(self) -> tuple[str, ...]
    def all_chunks(self) -> dict[str, RetrievedChunk]

def retrieve_grid(
    question_text: str, retriever: Any, bm25_index: Any, *,
    reformulated_text: str | None = None,       # None/identical -> 5-cell orig-only grid
    top_k: int, rerank_k: int,
    bm25_top_k: int | None = None, rrf_k: int | None = None,
    rerank_with: str = "question",
) -> GridResult
    # Blocks. `rerank_with="question"` (the default) keeps orig-vs-reform a
    # single-variable change: only the embedded string differs, the reranker's does not.

async def aretrieve_grid(question_text, retriever, bm25_index, **kwargs) -> GridResult
    # retrieve_grid() in a worker thread. One grid is seconds of MPS work that would
    # otherwise stall the loop talking to the judge.

def prerank_map(method: str, lists: dict[str, list[RetrievedChunk]]) -> dict[str, int]
    # chunk_id -> rank in the pre-rerank source list, for a "#7 -> #1" movement marker.
    # Empty when there is no recorded source, which is the caller's cue to print a
    # plain rank rather than a self-agreeing "#1 -> #1".

def resolve_index_dir(config: ExperimentConfig, index_dir: str | Path | None = None) -> Path
    # `--index-dir`, else data/index/<corpus>. One function so the path printed in
    # context.md cannot disagree with the table actually queried.

def build_retriever(config: ExperimentConfig, index_dir: str | Path | None = None
                   ) -> tuple[Retriever, dict[str, Any]]
    # Loads the existing index; never rebuilds (a silently rebuilt index is a second,
    # invisible variable under a run whose whole claim is a delta). Raises SystemExit
    # with the build command if the index is absent. The dict is the static per-run
    # fields: index_dir, index_rows, embedding_model, reranker_model, device.

@dataclass(frozen=True)
class RetrievalStrategy:
    name: str = "dense_rerank"
    top_k: int = 20
    rerank_k: int = 5
    bm25_top_k: int = 20
    rrf_k: int | None = None

    def __post_init__(self) -> None
        # ValueError when: name is not in BASE_STRATEGIES ("unknown retrieval
        # strategy"); top_k < 1; name ends in "_rerank" and rerank_k > top_k
        # ("exceeds top_k" -- invariant 2).
        # `rerank_k` is NOT constrained for a non-reranking strategy, because it is
        # inert there and rejecting it would block scoring BM25 at deeper k.

    @classmethod
    def from_config(cls, config: ExperimentConfig, name: str | None = None) -> RetrievalStrategy
        # The `name` argument is the CLI's `--strategy` override: it changes which list
        # is read out, never the funnel's depths.

    def lists(self, query: str, retriever: Any, bm25_index: Any) -> dict[str, list[RetrievedChunk]]
    def retrieve(self, query: str, retriever: Any, bm25_index: Any) -> list[RetrievedChunk]
        # `lists(...)[self.name]` -- i.e. one full fan-out per call. Deliberate: a
        # per-strategy shortcut would mean the shipped path no longer shares the grid's
        # code. Score several strategies through one `lists()` call, not `retrieve()`
        # per strategy.
```

`rerank_with` is absent from `RetrievalStrategy` on purpose. It exists to isolate the
embedding step from the reranker's string when two query variants are compared; with one
query there is one string, and a config field that only sometimes applies is a field
that lies.

## `medical_rag.retrieval.lexical` [implemented]

```python
DEFAULT_BM25_PATH = Path("data/index/statpearls_bm25.pkl")

def tokenize(text: str) -> list[str]      # r"[a-z0-9]+" over text.lower()

@dataclass
class BM25Index:
    chunk_ids: list[str]
    titles: list[str]
    contents: list[str]
    bm25: BM25Okapi
    def search(self, query: str, top_k: int) -> list[RetrievedChunk]
        # `score` is the BM25 score -- not comparable to a cosine, and never compared
        # against one. The list attributes are public because `ceiling.py` scans them.

def build_bm25_index(index_dir: str | Path, out_path: str | Path = DEFAULT_BM25_PATH) -> BM25Index
def load_bm25_index(path: str | Path = DEFAULT_BM25_PATH) -> BM25Index
def load_or_build_bm25_index(index_dir: str | Path, path: str | Path = DEFAULT_BM25_PATH,
                             *, rebuild: bool = False) -> BM25Index
```

The tokenizer is the naive one it has always been, and that is a finding rather than an
oversight: measured `bm25` recall was 0/0/10, with 59 distinct chunks filling the 100
top-5 slots and one boilerplate chunk ranking first for 6 of 20 questions. R4 is the
phase that changes `tokenize()` and adds field weighting; until then the pickle is
byte-compatible with the numbers in `FINDINGS.md`. The pickle carries no version stamp
today, which is also R4's job.

## `medical_rag.retrieval.fusion` [implemented]

```python
RRF_K: int = 60

def rrf_fuse(*ranked_lists: list[RetrievedChunk], top_k: int, k: int = RRF_K) -> list[RetrievedChunk]
    # sum(1 / (k + rank)). Returns copies with `score` replaced by the fused score;
    # never mutates its inputs, so the same chunk objects are safe to share across
    # cells. A chunk absent from a list contributes nothing to it rather than being
    # penalized with a fabricated worst-case rank.
```

## `medical_rag.retrieval.ceiling` [implemented]

An oracle, never a lever: the probe queries with the gold answer's own wording, so its
number goes in a ceiling statement and never in a strategy table. `ORACLE_LABEL` is
attached to every `CeilingResult` and carried into the render for exactly that reason --
this directory already published a wrong number off an unlabelled column.

```python
ORACLE_LABEL: str          # "...a diagnostic ceiling, NOT a retrieval lever..."
MIN_NEEDLE_CHARS = 3       # below this a "needle" is a common word, not a finding
GENERIC_MATCH_COUNT = 5_000

@dataclass
class ContainmentHit:
    chunk_id: str; title: str; offset: int; excerpt: str

@dataclass
class Scan:
    matches: int           # census: EVERY matching chunk
    ids: set[str]          # what a strategy had to hit
    hits: list[ContainmentHit]   # sample: the first `limit`, for a human to read
    truncated: bool

@dataclass
class CeilingResult:
    query: str; corpus_chunks: int; matches: int
    hits: list[ContainmentHit]; truncated: bool
    lists: dict[str, list[RetrievedChunk]]
    best_rank: dict[str, int | None]   # per strategy: best rank of a containing chunk
    surfaced: dict[str, int]
    warning: str; label: str = ORACLE_LABEL; elapsed_ms: int
    def to_dict(self) -> dict[str, Any]     # row-shaped, `"oracle": True`

def containment_scan(index: Any, needle: str, *, limit: int = 20, window: int = 90) -> Scan
    # Regex over the BM25 pickle's in-memory `contents`, not a LanceDB LIKE: the
    # bodies are already loaded and a 380k-row .to_pandas() per question is the cost
    # this module exists to avoid. Raises ValueError below MIN_NEEDLE_CHARS.

def best_ranks(lists: dict[str, Sequence[RetrievedChunk]], containing: set[str]
              ) -> tuple[dict[str, int | None], dict[str, int]]
    # Empty `containing` -> every strategy None, not an empty dict: "did not surface"
    # for all of them, rather than a blank row that reads as a missing measurement.

def probe(query_text: str, retriever: Any, bm25_index: Any, *,
          top_k: int, rerank_k: int, bm25_top_k: int | None = None,
          limit: int = 20) -> CeilingResult
    # The scan and the retrieval over the SAME string, scored against every matching
    # chunk (Scan.ids), not the displayed sample.
```

The bound is one-directional: zero matches means "not stated in these words", never
"not in StatPearls".

## `medical_rag.eval` [implemented]

The measurement surface. Nothing here asks an answer model anything: the oracle is a
judge model grading candidate chunks, its verdicts are cached, and every metric is
arithmetic over that cache. That asymmetry is why retrieval hypotheses cost index work
while an answer-accuracy run of the same sample costs hours of generation.

### `eval.metrics`

```python
K_VALUES: tuple[int, ...] = (5, 10, 20)
RELEVANT_LEVELS = {"relevant", "partial"}
BEYOND_K = 99   # rank scored for a question with no relevant chunk in the window, so
                # mean_first_relevant_rank cannot improve by surfacing fewer chunks

def gold_hit(chunk_ids: Sequence[str], judgments: dict[str, Any], gold: str) -> bool
    # Any chunk in the window whose supports_options contains the gold letter.
    # Recall counts support, not `relevance` -- topically-related-but-useless is what
    # relevant_fraction@k measures, and conflating them hides the failure that matters.

def relevant_fraction(chunk_ids: Sequence[str], judgments: dict[str, Any]) -> float
def first_relevant_rank(chunk_ids: Sequence[str], judgments: dict[str, Any]) -> int | None
    # 1-based; `partial` deliberately does not count.

def cell_metrics(chunk_ids, judgments, gold, ks=K_VALUES) -> dict[str, Any]
    # Single-question row: n_chunks, per_k{recall, relevant_fraction},
    # first_relevant_rank, gold_positions.

def aggregate(rows: Sequence[dict[str, Any]], methods: Sequence[str],
              ks: Sequence[int] = K_VALUES) -> dict[str, Any]
    # {"n_questions": int, "methods": {method: {"n", "per_k": {k: {"semantic_recall",
    #  "relevant_fraction"}}, "mean_first_relevant_rank", "n_first_relevant"}}}
    # Skips rows carrying "error". A method a row does not carry lowers that method's
    # `n` instead of counting as a miss: a missing column is not a measured zero.
```

Every formula takes the gold letter *as data*. The judge never sees it -- that is what
makes one verdict reusable across every strategy instead of re-graded per condition.

### `eval.rubric`

```python
JUDGE_PROMPT_VERSION = "2026-09-29.1"   # a version, not a date to read; see below
RELEVANT_LEVELS = {"relevant", "partial"}

class ChunkJudgment(BaseModel):
    chunk_id: str
    relevance: Literal["relevant", "partial", "irrelevant"]
    supports_options: list[str]
    reason: str

class JudgeVerdict(BaseModel):
    judgments: list[ChunkJudgment]

JUDGE_SYSTEM_PROMPT: str
JUDGE_JSON_INSTRUCTION: str

def judge_prompt_sha() -> str
    # sha256[:12] over (JUDGE_PROMPT_VERSION, JUDGE_SYSTEM_PROMPT,
    # JUDGE_JSON_INSTRUCTION) -- the RUBRIC, not a rendered prompt. Question text and
    # options are already keyed by question_id and chunk_id identifies content, so
    # hashing the rendered prompt would key verdicts to their own inputs and never
    # reuse. Only a changed rubric invalidates a verdict, which is the only thing that
    # should.

def build_judge_prompt(question: MedQAQuestion, chunks: Sequence[RetrievedChunk]) -> str
    # One batched, gold-blind prompt. Renders `chunk_id=` and `title=` verbatim so
    # verdicts join back without fuzzy matching.
```

There is exactly one copy of this wording in the repo. Editing it invalidates every
cached verdict on purpose -- roughly an hour of endpoint time for the 20-question
sample -- which is why the harness stamps the sha and never words a prompt of its own.

### `eval.judge`

```python
DEFAULT_JUDGE_CACHE = REPO_ROOT / "outputs/exploration/retrieval_tuning/judge_cache/judged_chunks.jsonl"
DEFAULT_BATCH = 24    # the 20-question run used 40 and the model began dropping items

def verdict_key(question_id: str, chunk_id: str) -> tuple[str, str]

def load_cache(path=DEFAULT_JUDGE_CACHE, *, judge_sha: str | None = None
              ) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, int]]
    # (verdicts, counts) with counts = {loaded, stale_prompt, malformed, missing_file}.
    # A record whose sha is not the current rubric's is IGNORED, not deleted: a rubric
    # can be reverted, and "how many verdicts would this edit invalidate" is worth more
    # than a tidy file. Torn lines are skipped -- the appender is not transactional.

def append_records(path, records: Sequence[dict[str, Any]]) -> None
def import_run(run_dir, cache_path=DEFAULT_JUDGE_CACHE, *, judge_sha=None) -> dict[str, int]
def stats(cache_path=DEFAULT_JUDGE_CACHE) -> dict[str, Any]

@dataclass
class JudgeOutcome:
    verdicts: dict[str, dict[str, Any]]
    cached: int; judged: int; missing_judgment: int; batches: int
    new_chunk_ids: list[str]; problems: list[str]

async def ensure_verdicts(question, chunks, cache, *, client,
                          cache_path=DEFAULT_JUDGE_CACHE,
                          breaker: EndpointCircuitBreaker | None = None,
                          batch: int = DEFAULT_BATCH,
                          judge_sha: str | None = None,
                          source_run: str | None = None) -> JudgeOutcome
    # The only function in the package that spends endpoint time on measurement. Cache
    # first, judge only what is missing, append before re-raising an LLMError so a run
    # that dies on question 12 does not re-pay the first 11. `missing_judgment` counts
    # chunks the judge skipped -- a partial verdict set must be reported, never scored
    # as if the gap were `irrelevant`.

def main(argv=None) -> int      # --cache / --import-run (repeatable) / --stats
```

Verdict record: `question_id, chunk_id, relevance, supports_options, reason,
judge_prompt_sha, model, judged_at, source_run`. `supports_options` is upper-cased and
sorted on write so a join against a gold letter is exact.

### `eval.store`

```python
SIDECAR_NAME = "chunks.jsonl"

def write_chunks(path, chunks: Iterable[RetrievedChunk]) -> Path
def append_chunks(path, chunks: Iterable[RetrievedChunk]) -> Path
def read_chunks(path) -> dict[str, RetrievedChunk]
def fetch_from_table(table, chunk_ids: Sequence[str], *, batch: int = 200) -> dict[str, RetrievedChunk]
    # Filter-only LanceDB query, batched: each query scans 380k rows, so one query per
    # id would make opening an old run take minutes for 40 strings. Logs the ids that
    # are not in the index (stale ids from a rebuilt corpus) instead of dropping them.

def resolve_chunks(chunk_ids, *, run_dir=None, table=None) -> tuple[dict[str, RetrievedChunk], str]
    # (chunks, source). Sidecar first, then LanceDB, then nothing -- and `source` is
    # rendered into the report so a body missing because a 5-day-old run has no sidecar
    # is visibly an artifact gap, not a retrieval bug.
```

### `eval.report`

One grid behind the terminal dump, `report.html`, and the findings table, so the number
in the browser and the number in the report cannot disagree.

```python
@dataclass class ChunkView:     # rank, chunk_id, title, content, score, relevance,
                                # supports_options, reason, gold_hit, prerank,
                                # judged, body_available
@dataclass class CellView:      # method, base, variant, chunks, metrics, reused,
                                # legacy, unjudged ; .label
@dataclass class QuestionView:  # question_id, question, options, gold_letter, split,
                                # meta, queries, reform, cells, provenance, notes, ks
                                # ; .gold_text, .cell(method), .variants(), .methods()

def build_cells(lists, judgments, gold, *, ks=K_VALUES, legacy=()) -> list[CellView]
    # A chunk absent from `judgments` renders as `?` and increments `unjudged` rather
    # than defaulting to `irrelevant`: the difference is whether a cell reads as low
    # recall or as unmeasured, and only the second one is honest.

def build_question_view(question: MedQAQuestion, grid: GridResult,
                        judgments: Mapping[str, Mapping[str, Any]], *,
                        split: str = "", reform: Mapping[str, Any] | None = None,
                        ks: Sequence[int] = K_VALUES,
                        chunk_lookup: Mapping[str, RetrievedChunk] | None = None,
                        provenance: Mapping[str, Any] | None = None,
                        notes: Mapping[str, str] | None = None) -> QuestionView
    # Live mode: the grid this process just retrieved. `split` is a parameter because
    # MedQAQuestion does not carry one -- the loader takes it, the record does not
    # return it.

def view_from_row(row: Mapping[str, Any], question: MedQAQuestion | None,
                  chunk_lookup: Mapping[str, RetrievedChunk], *,
                  ks: Sequence[int] = K_VALUES,
                  provenance: Mapping[str, Any] | None = None,
                  notes: Mapping[str, str] | None = None,
                  chunk_source: str = "") -> QuestionView
    # Read mode: rebuild the grid from a stored row at zero cost. Legacy ids are
    # aliased onto the grid and flagged, so the page says "this column is the old
    # broken one" instead of presenting it as a measured reform result.
def render_terminal(view, *, max_chunks: int | None = 10) -> str
def render_markdown(view: QuestionView) -> str
def aggregate_views(views: Sequence[QuestionView]) -> dict[str, dict[str, Any]]
    # Counts QUESTIONS, not clicks: the served page replaces a re-inspected view.
def render_main(views, *, meta=None) -> str
    # The inside of <main>: provenance table, fallback banner, summary, sections. The
    # static report and the live page both render their body from here -- two renderers
    # is how a viewer drifts from the table it exists to explain.
def render_html(views, *, title="Retrieval inspection", meta=None,
                live: bool = False, toolbar: str = "") -> str
    # live=True adds the toolbar and ~80 lines of dependency-free JS. Left off by
    # default so a written report still opens from file:// with the network off.
def render_toolbar(state: Mapping[str, Any]) -> str
def render_ceiling(result: "CeilingResult") -> str
```

### `eval.runlog`

The run plumbing Phases 5-6 logged through, promoted from
`scripts/exploration/_common.py` with no shim left behind -- a shim would only keep the
pretence that `scripts/` owns this code.

```python
DEFAULT_RUN_ROOT = "outputs/exploration"
DEFAULT_SAMPLE_SEED = 1
SELECTION_METHOD = "sha256(seed:question_id) ascending, first k"

def select_sample(questions, sample_size: int, sample_seed: int = DEFAULT_SAMPLE_SEED) -> list[MedQAQuestion]
    # Deterministic in (seed, question_id) only, so the draw survives load order and
    # interpreter version. Raises on sample_size <= 0 or larger than the split. A grown
    # pool redraws borderline members, which is why callers pin ids instead -- and why
    # growing the sample to ~100 must NOT be done by re-drawing at size=100.

def select_by_ids(questions, ids: Iterable[str]) -> list[MedQAQuestion]
    # The pinned ids are the contract. Raises KeyError naming EVERY missing id at once:
    # raising on the first, or returning the ones that resolved, would shrink a paired
    # table without saying so.

def gold_distribution(questions) -> Counter
def warn_on_degenerate_sample(questions) -> Counter
def git_rev() -> tuple[str, bool]                       # rev, dirty
def parse_notes(notes: Sequence[str]) -> dict[str, str]  # repeatable --note KEY=VALUE
def missing_chunk_evidence(prompt: str, chunks: Sequence[Any]) -> list[str]

class RunWriter:
    def __init__(self, phase: str, *, root=DEFAULT_RUN_ROOT, run_dir=None,
                 save_chains: bool = True) -> None
        # run_dir reopens an existing directory, which is what makes --resume attach to
        # a half-finished run instead of starting a second one. Never writes
        # outputs/runs/: exploratory rows must not be mistaken for pilot results.
    dir: Path; results_path: Path; chains_dir: Path | None
    @property stamp: str
    def write_row(self, row) -> dict
    def completed_keys(self) -> set[tuple[str, str, str]]
        # Only cells that EARNED a completion. An error_row is "attempted and failed",
        # never "done" -- treating it as done cost Phase 5's first run 17 silent no-ops.
    def load_rows(self) -> list[dict]
    def write_chain(self, model, tag, header, body) -> Path | None
    def write_context(self, ...) -> None      # git rev, config, ids, baseline, notes
    def close(self) -> None                   # also a context manager

def params_snapshot(model, **overrides) -> dict[str, Any]
def result_row(*, model, question, tag, condition_id, split, prompt, result,
               overrides=None, retrieval=None) -> dict[str, Any]
def error_row(*, model, question, tag, condition_id, split, exc, wall_s,
              prompt="") -> dict[str, Any]
    # Same keys as result_row wherever it can, `params` included. finish_reason/tok_s
    # stay None and completion_tokens null -- that is not zero, which would claim a
    # measured-empty generation.

class EndpointCircuitBreaker:
    def __init__(self, model_name: str, threshold: int = 3) -> None
    @property tripped: bool
    def record_success(self) -> None
    def record_failure(self, exc: BaseException) -> None
    def abort_problem(self, skipped: int) -> str

def percentile(values, q: float) -> float | None
def letter_distribution(rows, *, of_errors_only: bool = True) -> Counter
def summarize(rows, *, tag: str | None = "main") -> str
def retrieval_stats(rows, *, tag: str | None = "main") -> dict[str, Any]
```

`error_row`'s and `result_row`'s field names come from `QuestionResult`'s shape so a row
written in 2026 is still readable by an analysis script written later without a
translation layer, and so an error row is never the row that silently lacks `params`.

### `eval.harness` -- the grid runner

CLI: `scripts/run_retrieval_grid.py`, or `python -m medical_rag.eval.harness`.

```python
PHASE = "retrieval_tuning"; CONDITION_ID = "retrieval_tuning_judge"
SAMPLE_QUESTION_IDS: list[str]     # the pinned 20; PLAN.md's list, kept identical
K_VALUES = (5, 10, 20)
METHOD_NAMES = METHOD_GRID

def compute_metrics(rows) -> dict[str, Any]          # delegates to eval.metrics.aggregate
def load_run_rows(run_dir: Path) -> list[dict[str, Any]]
def canonicalize_rows(rows) -> list[dict[str, Any]]  # legacy ids -> grid ids, flagged
def render_findings(...) -> str
def rerender(args) -> int                            # offline rebuild of FINDINGS.md
def main() -> None
```

Flags: `--config --split --question-ids --sample-seed --limit --index-dir
--bm25-index --rebuild-bm25 --top-k --rerank-k --bm25-top-k --rerank-with
{question,reform} --no-reform --reform-prompt FILE --judge-cache --judge-batch
--output-dir --findings-path --resume RUN_DIR --rerender RUN_DIR --dry-run
--note KEY=VALUE`.

`--findings-path` defaults to the tracked `experiments/retrieval_tuning/FINDINGS.md`.
Any `--rerender` that is not a deliberate re-publication must redirect it to a temp
path: a 2-question probe once wrote its n=2 table over the real n=20 findings, and the
file is evidence.

`--rerender` is the regression gate every refactor of this code has to pass -- it
rebuilds the table from a run's `results.jsonl` with no index, no BM25 pickle and no
endpoint, and its output must `diff` empty against the committed file.

### `eval.inspector` -- read a run, or inspect live

CLI: `scripts/inspect_retrieval.py`, or `python -m medical_rag.eval.inspector`.

```python
PHASE = "retrieval_tuning"; CONDITION_ID = "retrieval_inspection"; K_DEFAULT = (5, 10, 20)

def read_rows(run_dir: Path) -> list[dict[str, Any]]
def views_from_run(...) -> list[QuestionView]          # zero LLM calls, zero models
def row_from_view(view: QuestionView) -> dict[str, Any]
    # Harness-shaped, so `judge --import-run` and grep work on an inspection dir.
    # Carries `question`/`options` on purpose: their absence is what made the first
    # run's rows unreadable by anything but the dataset that produced them.
def dedupe_chunks(views) -> list[RetrievedChunk]
def write_artifacts(out_dir, views, meta) -> dict[str, Path]
def prepare(args) -> Ready                             # resolve mode -> loaded or not
class Inspector:
    def __init__(self, args: argparse.Namespace, config: ExperimentConfig) -> None
    async def open(self, *, retrieval: bool, judge: bool) -> None
    async def close(self) -> None
    def meta(self, extra=None) -> dict[str, Any]        # index sha, rubric sha, knobs
    async def inspect(...) -> QuestionView
def main() -> None
```

Three modes with deliberately different costs: **read** (`--from-run`) pays nothing;
**live** (`--question-id`, `--all`, `--repl`) pays the index+model load once per process
and one judge batch per unseen chunk; **served** (`--serve`, see `eval.lab`) is the same
loaded state behind a browser. (No load-time figure here on purpose: the repo has
carried three different ones -- 2 s, 12 s, 40 s -- for the same load, and a number that
depends on warm caches and device is not part of a contract.) Flags: `--question-id
--all --repl --serve --from-run --split --open --host --port --no-retrieve`, grid knobs
`--top-k --rerank-k --bm25-top-k --rerank-with {question,reform} --k --index-dir
--bm25-index --rebuild-bm25`, reformulation `--no-reform --query TEXT --reform-prompt
FILE`, judging `--no-judge --judge-cache --judge-batch`, output `--output-dir --out-name
--no-html --max-chunks --dry-run --note`. `--query` hand-supplies an information need
and skips the reformulation call, so no fallback can silently flatten the reform
column. `--serve --dry-run` prints the route table and loads nothing; `--port 0` binds
a free port so two hypotheses can sit in two browsers.

### `eval.lab` -- the localhost knob panel

```python
TIMEOUTS = {"inspect": 180.0, "knobs": 180.0, "judge": 1_800.0, "probe": 600.0, "chunk": 300.0}
K_LIMITS = {"top_k": (1, 200), "rerank_k": (1, 200), "bm25_top_k": (1, 1000)}

class HttpError(Exception):
    def __init__(self, status: int, message: str) -> None

@dataclass
class SessionItem:
    question: MedQAQuestion | None
    view: QuestionView
    def require_question(self) -> MedQAQuestion   # 409 when the row stored no text

class Backend:
    def __init__(self, insp, pool: dict[str, MedQAQuestion], browse: Sequence[str] = ())
    def ingest(self, question, view) -> None       # REPLACE, not append
    def submit(self, coro, *, timeout=None) -> Any
    # + the route coroutines behind the nine endpoints, the knob validator, and _sync_views

def serve(insp, pool, *, host="127.0.0.1", port=0, retrieval=True, judge=True,
          browse=(), on_open=None, on_port=None) -> int
    # Blocks until Ctrl-C; returns the bound port. `insp.open()` runs on the backend's
    # loop because AsyncOpenAI binds to the loop it first runs on -- opening it on the
    # CLI's loop would raise "attached to a different loop" on the first handler call.
    # `on_open` runs after open() and before the port listens, so the first request
    # already has a grid and no handler can race the seed.
```

Nine routes (`GET /`, `/api/state`, `/api/chunk?id=`, `POST /api/inspect`, `/api/judge`,
`/api/knobs`, `/api/probe`, `/api/report`). Errors come back as `text/plain` with a real
status because the client prints them into the status line; returning 200 with an error
string inside it would leave a stale grid on screen looking current. `ingest()` replaces
because `aggregate_views` averages over the view list -- appending would report n=2 for
one question.

Localhost only: it renders full corpus text, and `/api/judge` spends endpoint time on
the shared server. Never repoint it at `0.0.0.0`.

### `eval.rewrite` [unscheduled]

Query reformulation, kept importable because the grid's `__reform` columns are built
with it and the golden re-render needs them. It is the only lever that still costs
completions, and its one honest result so far is +0/+0/+5pp with 18/20 fallbacks, so it
sits in the backlog rather than the plan.

```python
REFORMULATION_SYSTEM_PROMPT: str

class Reformulation(BaseModel):        # information_need: str (min_length=1)
class ReformulationResult(BaseModel):
    query: str                 # always usable; on failure it is the original question
    prompt: str
    raw_response: str | None = None
    fallback: bool = False     # whether that usability cost anything
    error: str | None = None   # ... and what
    method: str = "structured"
    model: str | None = None
    prompt_source: str = "production build_reformulation_prompt()"

def parse_information_need(raw: str) -> str | None
    # Tolerant: fenced JSON, prose-wrapped, bare strings. None when the content is too
    # thin to be an information need -- a one-word "answer" is not one.

def load_template(path: str | Path | None = None) -> str | None
    # No file is ever loaded implicitly: an experiment should not start diverging from
    # production because someone left a file behind. `--reform-prompt` is the opt-in.

def build_prompt(question: str, template: str | None = None) -> str
    # template=None -> production's build_reformulation_prompt(). A template without
    # {question} gets the question appended rather than emitting a fluent rewrite of
    # nothing.

async def reformulate_query(question: MedQAQuestion, client: LLMClient,
                            *, template: str | None = None) -> ReformulationResult
    # Two attempts, then an EXPLICIT fallback recorded on the result (`fallback=True`)
    # rather than only logged -- a fallback that only exists in a log line is how
    # `reform_dense` looked like a method for a whole release. Transport errors
    # re-raise so the caller's circuit breaker counts them: an outage must not be
    # reported as "the rewrite was bad".

def manual_result(query: str, *, model: str | None = None) -> ReformulationResult
    # A hand-written `--query`, labelled so nothing downstream can claim a prompt ran.
```

## `medical_rag.generation.llm` [implemented]

Mainline for the retrieval track: the judge rides this client, and so does
`agenerate_structured` for verdicts.

```python
class LLMError(RuntimeError): ...

@dataclass(slots=True)
class GenerationResult:
    content: str = ""            # what scoring and the judge must use
    reasoning: str = ""          # logs/qualitative only: R1 chains restate passages,
                                 # so scoring them inflates every support measure
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    elapsed_s: float = 0.0
    recovery_attempted: bool = False
    recovery_content: str = ""
    recovered: bool = False
    @property truncated: bool        # finish_reason == "length"
    @property parsed_answer: str | None
    @property unanswered: bool       # no parsable answer after recovery -> incorrect

class LLMClient:
    config: ModelConfig
    client: openai.AsyncOpenAI

    def __init__(self, config: ModelConfig, *,
                 http_client: httpx.AsyncClient | None = None) -> None
        # http_client is the test seam (httpx.MockTransport). Raises LLMError naming
        # config.api_key_env when neither os.environ nor load_env()'s .env supplies it.

    async def agenerate(self, prompt, *, max_new_tokens=None, temperature=None,
                        top_p=None, system_prompt=SYSTEM_PROMPT) -> GenerationResult
    async def agenerate_structured(self, prompt, output_type: type[ModelT], *,
                                   max_new_tokens=None, temperature=None, top_p=None,
                                   system_prompt=SYSTEM_PROMPT) -> ModelT
    async def aclose(self) -> None      # async context manager too
```

Per-call kwargs override `ModelConfig` (0.0 is a value, not "unset"). `seed` is
deliberately not a parameter: measured inert on this endpoint and passing it disabled
server-side batching, so it is never sent. `agenerate_structured` runs pydantic-ai's
`PromptedOutput` over `OpenAIProvider(openai_client=self.client)` -- same
base_url/key/timeout/retries as `agenerate`, and `PromptedOutput` rather than
`ToolOutput` because this endpoint's tool calling is broken. Every call logs prompt,
response, model and timestamp; only names of env vars are ever logged, never values.

## `medical_rag.generation.preflight` [implemented]

```python
class Preflight(Enum):
    PROCEED = "proceed"
    BLOCK = "block"

async def preflight(model: ModelConfig, client: LLMClient) -> tuple[Preflight, list[str]]
    # Returns a decision plus the lines to print instead of logging or raising: it runs
    # once per arm before ~20 completions, and the two failure modes want different
    # handling -- a dead endpoint blocks the run, a misnamed api_model_name invalidates
    # only that arm.
```

Load-bearing for the judge as well as for the answer runs: if `:8080` has been
repointed at another model, the verdict cache silently changes oracle mid-run while
every record keeps naming the model `models.yaml` claims.

## `medical_rag.generation.prompt` [frozen]

Written for the answer-accuracy study. The answer-accuracy study is cut; these stay
because `raw_rag.py` and `closed_book.py` are cited evidence and must still run, and
because `format_options` is used by the judge prompt.

```python
SYSTEM_PROMPT: str
ANSWER_FORMAT_INSTRUCTION: str
ANSWER_RECOVERY_INSTRUCTION: str

def format_options(options: dict[str, str]) -> str        # letter-sorted "A. text"
def format_context(chunks: Sequence[RetrievedChunk]) -> str
def build_answer_prompt(question, options, context_chunks: list[RetrievedChunk] | None) -> str
    # None/[] renders the closed-book variant. Used by closed_book.py and raw_rag.py.
def build_reformulation_prompt(question: str) -> str
    # Takes the question ONLY: leaking option wording into the query would let
    # retrieval fetch the answer instead of the information need. This is the builder
    # the mainline `eval.rewrite` still calls by default.
def build_verification_prompt(question, chunks) -> str
    # Frozen with the Verifier it served: nothing calls it.
def parse_answer(text: str, valid_letters: set[str] | None = None) -> str | None
    # Three tiers, newest-match-first, so a decoy "ANSWER:" quoted inside a chain of
    # thought cannot beat the final answer. Letters outside valid_letters are
    # unparseable rather than wrong-but-scored.
```

## CLI entry points

`scripts/*.py` are thin wrappers over a package `main()`; the module and the script are
the identical command. They exist so `ls scripts/` shows the study's commands without
first learning the package layout.

```
.venv/bin/python scripts/eval_retrieval.py         # score strategies against the verdict
                                                   # cache: zero endpoint calls
    --config PATH | --strategy NAME (repeatable, choices=BASE_STRATEGIES)
    --question-id ID (repeatable; default the pinned 20) | --k K...
    --judge-cache PATH | --index-dir PATH | --bm25-index PATH | --rebuild-bm25
    --json PATH | --require-coverage FRACTION | --dry-run

.venv/bin/python scripts/run_retrieval_grid.py     # retrieve -> judge -> metrics -> FINDINGS.md
    (see eval.harness for the flag list; --dry-run prints the plan)

.venv/bin/python scripts/inspect_retrieval.py      # read a run, or inspect live/served
    (see eval.inspector for the flag list; --serve --dry-run prints the route table)

.venv/bin/python -m medical_rag.eval.judge         # verdict-cache maintenance
    --cache PATH | --import-run RUN_DIR (repeatable) | --stats

.venv/bin/python scripts/build_index.py            # build the LanceDB table
    --corpus statpearls | --output-dir PATH (data/index/<corpus>)
    --data-cache-dir PATH (data/<corpus>) | --embedding-model BAAI/bge-small-en-v1.5
    --batch-size 128 | --force
```

`eval_retrieval.py --dry-run` answers "what would this score, and against how many
verdicts" from config alone -- no dataset, no index, no endpoint. Its
`--require-coverage` exists because a new strategy's chunks are unjudged by definition:
scoring 40% recall over slots that are 60% unjudged is a measurement of the cache, not
of retrieval, so unjudged slots count as non-supporting and the coverage line prints next
to every number.

## Frozen Phases 5-6 tooling [frozen]

The generator-side study -- the 2x2 of model x retrieval, reformulation and verification
conditions, the failure taxonomy, the confirmatory run -- is cut. Its phases, its stubs
(`modules/`, `experiment/`, `analysis/`), `config/conditions.yaml` and its two runner
scripts are deleted rather than commented out; `PLAN.md`'s backlog carries the lever list
and the reason they are not scheduled. What remains here does not serve that study -- it
is the evidence the retrieval track cites, and it has to keep running for the citations
to be checkable.

```
scripts/exploration/closed_book.py    Phase 5: closed-book answers on the pinned 20.
scripts/exploration/raw_rag.py        Phase 6: RAG end-to-end + the closed-book pairing.
    load_baseline(run_dir) -> dict[(model, question_id), row]
    endpoint_drift(...) -> list[str]          # pairs only what it can pair
    outcome_of(rag, base) -> str              # rescued / distracted / both_wrong / ...
    pin_check(...)                            # re-derives the seed draw and prints the diff
    retrieve_pass(...)                        # captures cosine + pre-rerank rank BEFORE rerank
scripts/exploration/audit_run.py      re-derives every number a FINDINGS.md claims and
                                      exits non-zero on a mismatch.
scripts/probe_models.py               endpoint/cost/recovery pre-flight.
```

`raw_rag.py` imports `Preflight`/`preflight`, `build_retriever`/`resolve_index_dir` and
`chunk_match_rank` back out of the package. That direction is the point: before R1 the
retrieval harness reached its measurement machinery *through* this Phase 6 script, so
moving either broke the other and a `sys.path.insert` was the only thing holding the
arrangement together.

Its `--dry-run` currently exits 1 by design when the paired baseline has no rows for
every model in `config/models.yaml` -- `judge_model` joined the config after Phase 5 ran
and the Phase 5 baseline never served it. That is the pairing guard working, not a
broken script: pass `--models model_a model_b` to re-check the Phase 6 comparison.
