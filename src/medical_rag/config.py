"""Configuration loading for the medical RAG experiment.

Loads and merges YAML configuration from config/default.yaml,
config/models.yaml, and config/conditions.yaml into typed, validated
pydantic models consumed by the rest of the pipeline. Every model is served
remotely behind an OpenAI-compatible endpoint -- there is no local model
loading or quantization config here.
"""

from pathlib import Path

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


class RetrievalConfig(BaseModel):
    corpus: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k_retrieve: int
    top_k_rerank: int
    reranker_model: str


class ConditionConfig(BaseModel):
    id: str
    name: str
    model: str
    retrieval: bool
    reformulation: bool
    verification: bool
    seeds: list[int]


class ExperimentConfig(BaseModel):
    models: dict[str, ModelConfig]
    retrieval: RetrievalConfig
    conditions: list[ConditionConfig]
    benchmark: str
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
