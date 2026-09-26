"""Configuration loading for the medical RAG experiment.

Loads and merges YAML configuration from config/default.yaml,
config/models.yaml, and config/conditions.yaml into typed, validated
pydantic models consumed by the rest of the pipeline. Every model is served
remotely behind an OpenAI-compatible endpoint -- there is no local model
loading or quantization config here.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, model_validator


class ModelConfig(BaseModel):
    """An OpenAI-compatible remote endpoint serving one LLM."""

    name: str
    base_url: str
    api_model_name: str
    api_key_env: str
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
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
    seeds: list[int]
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


def load_config(path: str | Path = "config/default.yaml") -> ExperimentConfig:
    """Load and merge default.yaml, models.yaml, and conditions.yaml.

    `path` points at default.yaml; models.yaml and conditions.yaml are read
    from the same directory. Raises on any missing file, invalid YAML, or
    a condition that references an undefined model.
    """
    default_path = Path(path)
    config_dir = default_path.parent

    merged: dict = {}
    for filename in ("default.yaml", "models.yaml", "conditions.yaml"):
        file_path = config_dir / filename
        with file_path.open() as f:
            data = yaml.safe_load(f)
        if data:
            merged.update(data)

    return ExperimentConfig(**merged)
