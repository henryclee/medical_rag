"""Tests for medical_rag.config."""

import pytest

from medical_rag.config import load_config


def test_load_config_succeeds():
    config = load_config("config/default.yaml")
    assert config.benchmark == "medqa_usmle"
    assert len(config.conditions) == 10
    assert set(config.models) == {"model_a", "model_b"}


def test_load_config_rejects_undefined_model(tmp_path):
    (tmp_path / "models.yaml").write_text(
        "models:\n"
        "  model_a:\n"
        "    name: model_a\n"
        "    base_url: http://localhost\n"
        "    api_model_name: m\n"
        "    api_key_env: KEY\n"
        "    max_new_tokens: 10\n"
        "    temperature: 0.0\n"
        "    top_p: 1.0\n"
        "    seed: 0\n"
    )
    (tmp_path / "conditions.yaml").write_text(
        "conditions:\n"
        "  - id: \"1\"\n"
        "    name: bad\n"
        "    model: does_not_exist\n"
        "    retrieval: false\n"
        "    reformulation: false\n"
        "    verification: false\n"
        "    seeds: [0]\n"
    )
    (tmp_path / "default.yaml").write_text(
        "benchmark: x\n"
        "output_dir: outputs\n"
        "retrieval:\n"
        "  corpus: statpearls\n"
        "  embedding_model: m\n"
        "  chunk_size: 1\n"
        "  chunk_overlap: 0\n"
        "  top_k_retrieve: 1\n"
        "  top_k_rerank: 1\n"
        "  reranker_model: m\n"
    )

    with pytest.raises(ValueError, match="undefined model"):
        load_config(tmp_path / "default.yaml")
