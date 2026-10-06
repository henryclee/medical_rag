"""Configuration loading for the medical RAG experiment.

Loads and merges YAML configuration from config/default.yaml and
config/models.yaml into typed, validated pydantic models consumed by the rest
of the pipeline. Every model is served remotely behind an OpenAI-compatible
endpoint -- there is no local model loading or quantization config here.

The condition set this used to carry -- `conditions.yaml`, its
model x reformulation x verification factorial, and the `verifier` /
`reformulator` blocks -- is gone. The study it described was cut when retrieval
became the mainline, and config keys no code reads are worse than absent ones:
they read as parameters that were chosen.

`load_config()` also fills `os.environ` from the repo's gitignored `.env`
(see `load_env()`), because `LLMClient` reads every endpoint key with
`os.environ.get(ModelConfig.api_key_env)` and `.env.example`, the README, and
that client's own `LLMError` all tell the user to put the keys in `.env`.
Nothing used to read that file, so the promise was shell-only: a run launched
without the `set -a; source .env; set +a` prefix died with "unset or empty"
while the key sat in a correct `.env` -- and because the harness builds the
judge client first, that surfaced as a `judge_model` error naming a key that
was fine.
"""

import os
from pathlib import Path

import yaml
from dotenv import dotenv_values, find_dotenv
from loguru import logger
from pydantic import BaseModel, model_validator

# The gitignored secrets file `.env.example` templates and `api_key_env` names
# variables into. Single constant so the docs and the lookup cannot drift.
ENV_FILENAME = ".env"


class ModelConfig(BaseModel):
    """An OpenAI-compatible remote endpoint serving one LLM."""

    name: str
    base_url: str
    api_model_name: str
    api_key_env: str
    max_new_tokens: int
    temperature: float
    top_p: float
    # HTTP transport policy for LLMClient; defaulted so existing configs need no
    # change and so the retry/timeout rule lives in config rather than in code.
    timeout: float = 120.0
    max_retries: int = 4
    # One-shot rescue when the first completion carries no parsable ANSWER: line
    # -- typically a reasoning model that spent the whole budget on its chain of
    # thought (measured: DeepSeek-R1-Distill-Qwen-7B rambles past 3,072 tokens on
    # ~2/3 of closed-book questions). LLMClient replays what the model produced
    # as an assistant turn and asks it to close, using this many tokens. The
    # final fallback is an unanswered row, scored incorrect. 0 disables the
    # rescue, so a caller can measure the raw unanswered rate.
    answer_recovery_max_tokens: int = 0


class RetrievalConfig(BaseModel):
    """What to retrieve, from where, and how much of it survives the funnel.

    Every field here is read, by `retrieval.strategy.RetrievalStrategy.from_config`.
    The two chunking keys this class used to carry (`chunk_size: 512`,
    `chunk_overlap: 64`) were read by nothing -- StatPearls is chunked by the
    vendored MedRAG section algorithm, never by a token window -- so they are
    deleted rather than wired up, and a future chunking lever adds a knob that
    actually moves something.
    """

    corpus: str
    embedding_model: str
    # One of the five names the grid measured (`strategy.BASE_STRATEGIES`), so a
    # row of FINDINGS.md and a production call cannot drift into meaning two
    # different things. Validated at load, below.
    strategy: str = "dense_rerank"
    top_k_retrieve: int
    top_k_rerank: int
    reranker_model: str
    bm25_top_k: int = 20
    # RRF smoothing constant: 60 is the value every measured `hybrid` number in
    # this repo used. Tuning it is a lever (R5), not a default to fiddle with.
    rrf_k: int = 60


class ExperimentConfig(BaseModel):
    models: dict[str, ModelConfig]
    retrieval: RetrievalConfig
    benchmark: str
    dev_split: str  # every measurement run uses this split, never the test split
    test_split: str
    output_dir: str

    @model_validator(mode="after")
    def check_retrieval_config_is_measurable(self) -> "ExperimentConfig":
        """Reject an unmeasurable retrieval config at load, before anything is paid for.

        Reuses `RetrievalStrategy`'s own checks rather than restating them: the
        name must be one the grid measures, the funnel may not exceed the
        candidate list it reorders, `top_k` must be positive. A `default.yaml`
        that names `rrf_rerank`, or reranks 10 chunks out of a 5-chunk funnel,
        is a typo -- and after an afternoon of judge calls is the expensive
        place to discover a typo.
        """
        from medical_rag.retrieval.strategy import RetrievalStrategy  # local: strategy imports config

        RetrievalStrategy.from_config(self)
        return self


def _env_file_candidates(config_dir: Path) -> list[Path]:
    """Existing `.env` files to read, first-found wins, most specific first.

    Two roots are searched because the repo's commands are documented from the
    repo root (`README.md`, `file_layout.md`) but every script takes a
    `--config` path can be run from anywhere:

    1. the current directory and each ancestor of it -- so a harness launched
       from `experiments/retrieval_tuning/` still reaches the repo root's file;
    2. the repo root the config path itself implies
       (`<repo>/config/default.yaml` -> `<repo>/.env`), then the config
       directory, for `cd /tmp && ... --config /repo/config/default.yaml`.
    """
    candidates: list[Path] = []

    walked = find_dotenv(ENV_FILENAME, raise_error_if_not_found=False, usecwd=True)
    if walked:
        candidates.append(Path(walked))
    candidates.extend((config_dir.parent / ENV_FILENAME, config_dir / ENV_FILENAME))

    seen: set[Path] = set()
    found: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and resolved.is_file():
            seen.add(resolved)
            found.append(resolved)
    return found


def load_env(
    config_path: str | Path = "config/default.yaml",
    *,
    env_path: str | Path | None = None,
) -> Path | None:
    """Merge the repo's gitignored `.env` into `os.environ`; return the file read.

    Returns `None` when there is no file to read, which is not an error: keys
    may legitimately come from the shell, CI secrets, or a wrapper instead.

    `env_path` pins one file (used by tests and by callers that have already
    located their own) instead of searching; `config_path` only supplies the
    repo root the search is anchored on.

    Reads and writes are deliberately conservative:

    * Values are parsed with `dotenv_values()` and assigned by hand rather than
      handed to `load_dotenv()`, so the two rules below are visible here rather
      than implicit in a dependency's defaults.
    * **Existing variables win.** A name already in `os.environ` -- exported by
      the shell, `source`d with the documented `set -a; source .env; set +a`, or
      set per-command (`MODEL_A_API_KEY=... .venv/bin/python ...`) -- is never
      overwritten, so this only ever fills gaps and cannot break the workflow
      the docs describe.
    * **Empty values are skipped.** A half-filled `JUDGE_MODEL_API_KEY=` stays
      empty so `LLMClient` still raises its "unset or empty" error instead of
      sending an unusable `Authorization` header and coming back as a confusing
      HTTP 401.
    * **No `$` expansion** (`interpolate=False`). API keys are opaque, and an
      expansion pass would silently rewrite a literal `$` in a key from the
      environment -- usually to nothing.

    Only variable *names* are ever logged; values never are, matching
    `eval/runlog.py`'s context.md rule that `api_key_env` is recorded as a name.
    """
    paths = [Path(env_path)] if env_path is not None else _env_file_candidates(
        Path(config_path).resolve().parent
    )

    for path in paths:
        if not path.is_file():
            continue
        values = dotenv_values(path, interpolate=False)
        exported = sorted(
            name for name, value in values.items() if value and name not in os.environ
        )
        for name in exported:
            os.environ[name] = values[name] or ""
        logger.bind(kind="config").debug(
            "loaded {} from {}: {} of {} variable(s) set from file ({})",
            ENV_FILENAME,
            path,
            len(exported),
            len(values),
            ", ".join(exported) or "none -- all already exported",
        )
        return path

    logger.bind(kind="config").debug(
        "no {} found near {}; relying on exported environment variables",
        ENV_FILENAME,
        Path(config_path).resolve().parent,
    )
    return None


def load_config(
    path: str | Path = "config/default.yaml",
    *,
    env_path: str | Path | None = None,
    load_env_file: bool = True,
) -> ExperimentConfig:
    """Load and merge default.yaml and models.yaml.

    `path` points at default.yaml; models.yaml is read from the same directory.
    Raises on any missing file, invalid YAML, or a retrieval config that cannot
    be measured (see `ExperimentConfig`'s validator).

    Loads `.env` into `os.environ` first (`load_env()`), so an `api_key_env`
    whose value lives in that file is populated before any `LLMClient` reads
    it. `env_path` pins the file; `load_env_file=False` skips the whole step
    for callers that manage their own environment (and for tests).
    """
    default_path = Path(path)
    config_dir = default_path.parent

    if load_env_file:
        load_env(default_path, env_path=env_path)

    merged: dict = {}
    for filename in ("default.yaml", "models.yaml"):
        file_path = config_dir / filename
        with file_path.open() as f:
            data = yaml.safe_load(f)
        if data:
            merged.update(data)

    return ExperimentConfig(**merged)
