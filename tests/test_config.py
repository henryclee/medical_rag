"""Tests for medical_rag.config."""

import os

import pytest

from medical_rag.config import load_config, load_env
from medical_rag.generation.llm import LLMClient, LLMError


def test_load_config_succeeds():
    # load_env_file=False: a test must not read -- and leak into os.environ --
    # whatever secrets the developer's real .env happens to hold.
    config = load_config("config/default.yaml", load_env_file=False)
    assert config.benchmark == "medqa_usmle"
    assert config.dev_split == "train"
    assert config.test_split == "test"
    assert config.verifier.min_chunks == 2
    assert config.reformulator.max_retries == 2
    assert len(config.conditions) == 10
    # judge_model joined in the retrieval-tuning side-track (models.yaml): the
    # larger model that grades retrieved chunks.
    assert set(config.models) == {"model_a", "model_b", "judge_model"}

    by_id = {c.id: c for c in config.conditions}
    assert by_id["1"].params == {}
    assert by_id["3"].params == {"reranking": True}


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
    )
    (tmp_path / "conditions.yaml").write_text(
        "conditions:\n"
        "  - id: \"1\"\n"
        "    name: bad\n"
        "    model: does_not_exist\n"
        "    retrieval: false\n"
        "    reformulation: false\n"
        "    verification: false\n"
    )
    (tmp_path / "default.yaml").write_text(
        "benchmark: x\n"
        "dev_split: train\n"
        "test_split: test\n"
        "output_dir: outputs\n"
        "retrieval:\n"
        "  corpus: statpearls\n"
        "  embedding_model: m\n"
        "  chunk_size: 1\n"
        "  chunk_overlap: 0\n"
        "  top_k_retrieve: 1\n"
        "  top_k_rerank: 1\n"
        "  reranker_model: m\n"
        "verifier:\n"
        "  min_chunks: 2\n"
        "reformulator:\n"
        "  max_retries: 2\n"
    )

    with pytest.raises(ValueError, match="undefined model"):
        load_config(tmp_path / "default.yaml", load_env_file=False)


# --- .env loading (config.load_env) ------------------------------------------
#
# The regression these cover: `.env.example`, the README, and LLMClient's own
# LLMError all told the user to put endpoint keys in `.env`, but nothing read
# that file -- `LLMClient` only ever looked at `os.environ`. A run launched
# without `set -a; source .env; set +a` therefore died with "unset or empty"
# naming `JUDGE_MODEL_API_KEY` (the harness builds that client first) while the
# key sat correctly written in the repo root's `.env`.

_MINIMAL_MODELS_YAML = (
    "models:\n"
    "  model_a:\n"
    "    name: model_a\n"
    "    base_url: http://localhost\n"
    "    api_model_name: m\n"
    "    api_key_env: MODEL_A_API_KEY\n"
    "    max_new_tokens: 10\n"
    "    temperature: 0.0\n"
    "    top_p: 1.0\n"
)


def _write_config(tmp_path, models_yaml: str = _MINIMAL_MODELS_YAML):
    """Write the three config files `load_config` merges; return default.yaml."""
    (tmp_path / "models.yaml").write_text(models_yaml)
    (tmp_path / "conditions.yaml").write_text(
        "conditions:\n"
        "  - id: \"1\"\n"
        "    name: base\n"
        "    model: model_a\n"
        "    retrieval: false\n"
        "    reformulation: false\n"
        "    verification: false\n"
    )
    default = tmp_path / "default.yaml"
    default.write_text(
        "benchmark: x\n"
        "dev_split: train\n"
        "test_split: test\n"
        "output_dir: outputs\n"
        "retrieval:\n"
        "  corpus: statpearls\n"
        "  embedding_model: m\n"
        "  chunk_size: 1\n"
        "  chunk_overlap: 0\n"
        "  top_k_retrieve: 1\n"
        "  top_k_rerank: 1\n"
        "  reranker_model: m\n"
        "verifier:\n"
        "  min_chunks: 2\n"
        "reformulator:\n"
        "  max_retries: 2\n"
    )
    return default


def test_load_env_fills_gaps_and_never_overwrites_exported_values(tmp_path, monkeypatch):
    # delenv first so monkeypatch restores "unset" at teardown: load_env writes
    # to os.environ directly, which monkeypatch cannot undo on its own.
    monkeypatch.delenv("MODEL_A_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_B_API_KEY", raising=False)
    monkeypatch.delenv("JUDGE_MODEL_API_KEY", raising=False)
    monkeypatch.setenv("MODEL_A_API_KEY", "from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MODEL_A_API_KEY=from-file\n"
        "MODEL_B_API_KEY=from-file\n"
        "JUDGE_MODEL_API_KEY=\n"
    )

    assert load_env(env_path=env_file) == env_file

    assert os.environ["MODEL_A_API_KEY"] == "from-shell"   # exported wins
    assert os.environ["MODEL_B_API_KEY"] == "from-file"    # gap filled
    # An empty value stays absent rather than becoming an unusable "" key that
    # would surface later as an HTTP 401.
    assert "JUDGE_MODEL_API_KEY" not in os.environ


def test_load_env_does_not_expand_dollars_in_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_A_API_KEY", raising=False)
    monkeypatch.setenv("NOT_A_SECRET", "expanded")
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_A_API_KEY=sk-$NOT_A_SECRET-literal\n")

    load_env(env_path=env_file)

    assert os.environ["MODEL_A_API_KEY"] == "sk-$NOT_A_SECRET-literal"


def test_load_env_returns_none_when_there_is_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_env(tmp_path / "config" / "default.yaml") is None


def test_load_env_finds_the_repo_root_file_from_a_subdirectory(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_A_API_KEY", raising=False)
    (tmp_path / ".env").write_text("MODEL_A_API_KEY=from-file\n")
    nested = tmp_path / "experiments" / "retrieval_tuning"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    assert load_env(tmp_path / "config" / "default.yaml") == (tmp_path / ".env").resolve()
    assert os.environ["MODEL_A_API_KEY"] == "from-file"


def test_load_config_populates_env_so_llmclient_stops_raising(tmp_path, monkeypatch):
    """The reported failure, end to end: key in .env only, no shell export."""
    monkeypatch.delenv("MODEL_A_API_KEY", raising=False)
    default = _write_config(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_A_API_KEY=from-file\n")

    config = load_config(default, env_path=env_file)

    assert os.environ["MODEL_A_API_KEY"] == "from-file"
    client = LLMClient(config.models["model_a"])   # used to raise LLMError here
    assert client.client.api_key == "from-file"


async def test_llmclient_still_raises_when_neither_env_nor_file_has_the_key(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MODEL_A_API_KEY", raising=False)
    config = load_config(_write_config(tmp_path), load_env_file=False)

    with pytest.raises(LLMError, match="MODEL_A_API_KEY"):
        LLMClient(config.models["model_a"])
