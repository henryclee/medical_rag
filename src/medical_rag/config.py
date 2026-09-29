"""Configuration loading for the medical RAG experiment.

Loads and merges YAML configuration from config/default.yaml,
config/models.yaml, and config/conditions.yaml into typed, validated
pydantic models consumed by the rest of the pipeline. Every model is served
remotely behind an OpenAI-compatible endpoint -- there is no local model
loading or quantization config here.

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
from typing import Any

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
    model: str
    retrieval: bool
    reformulation: bool
    verification: bool
    # Free-form lever parameters discovered useful during exploration (e.g.
    # {"reranking": true}), so a new exploratory lever never needs its own
    # typed field.
    params: dict[str, Any] = {}


class ExperimentConfig(BaseModel):
    models: dict[str, ModelConfig]
    retrieval: RetrievalConfig
    verifier: VerifierConfig
    reformulator: ReformulatorConfig
    conditions: list[ConditionConfig]
    benchmark: str
    dev_split: str  # split Phases 5-9 run exploration against, not the test split
    test_split: str  # split Phase 13's confirmatory run uses
    output_dir: str

    @model_validator(mode="after")
    def check_condition_models_exist(self) -> "ExperimentConfig":
        for condition in self.conditions:
            if condition.model not in self.models:
                raise ValueError(
                    f"condition '{condition.id}' references undefined model "
                    f"'{condition.model}'; defined models: {sorted(self.models)}"
                )
        return self


def _env_file_candidates(config_dir: Path) -> list[Path]:
    """Existing `.env` files to read, first-found wins, most specific first.

    Two roots are searched because the repo's commands are documented from the
    repo root (`README.md`, `file_layout.md`) but every script takes a
    `--config` path and the exploration harnesses get run from their own
    directory (`experiments/retrieval_tuning/judge_harness.py`):

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
    `_common.py`'s context.md rule that `api_key_env` is recorded as a name.
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
    """Load and merge default.yaml, models.yaml, and conditions.yaml.

    `path` points at default.yaml; models.yaml and conditions.yaml are read
    from the same directory. Raises on any missing file, invalid YAML, or
    a condition that references an undefined model.

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
    for filename in ("default.yaml", "models.yaml", "conditions.yaml"):
        file_path = config_dir / filename
        with file_path.open() as f:
            data = yaml.safe_load(f)
        if data:
            merged.update(data)

    return ExperimentConfig(**merged)
