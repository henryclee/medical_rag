"""Tests for medical_rag.config -- i.e. for the one frozen retrieval funnel that
every R-phase number in this repo is defined by.

The retrieval track retired the generator-side study (ADR-0007), so the checks
that used to live here -- "the 10 conditions exist", "every condition names a
defined model" -- were deleted rather than updated. What replaced them is the
constraint that actually bites now: `config/default.yaml` is what R6 freezes, and
`scripts/eval_retrieval.py` scores the strategy resolved out of it, so a config
that loads but cannot be *measured* (an unknown strategy name, a rerank depth
wider than the candidate list it draws from) has to fail at load. Both of those
mistakes produce a plausible table instead of an error, which is the failure mode
this repo has now paid for twice.
"""

import os

import pytest

from medical_rag.config import load_config, load_env
from medical_rag.generation.llm import LLMClient, LLMError
from medical_rag.retrieval.strategy import BASE_STRATEGIES, RetrievalStrategy


def test_load_config_succeeds():
    # load_env_file=False: a test must not read -- and leak into os.environ --
    # whatever secrets the developer's real .env happens to hold.
    config = load_config("config/default.yaml", load_env_file=False)
    assert config.benchmark == "medqa_usmle"
    assert config.dev_split == "train"
    assert config.test_split == "test"
    # judge_model joined in the retrieval-tuning side-track (models.yaml): the
    # larger model that grades retrieved chunks. It is now the only model the
    # mainline path uses at all -- model_a/model_b belong to frozen Phases 5-6.
    assert set(config.models) == {"model_a", "model_b", "judge_model"}

    retrieval = config.retrieval
    assert (retrieval.corpus, retrieval.embedding_model, retrieval.reranker_model) == (
        "statpearls",
        "BAAI/bge-small-en-v1.5",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    # The funnel R6 freezes. 20 -> 5 is also why every `*_rerank` cell in the
    # first grid read 45/45/45: recall@20 over a 5-chunk list cannot rise.
    assert retrieval.strategy == "dense_rerank"
    assert (retrieval.top_k_retrieve, retrieval.top_k_rerank) == (20, 5)
    assert (retrieval.bm25_top_k, retrieval.rrf_k) == (20, 60)

    # The config you read is the config the evaluator runs -- the validator below
    # means `load_config` already proved this resolves to a real strategy.
    strategy = RetrievalStrategy.from_config(config)
    assert (strategy.name, strategy.top_k, strategy.rerank_k) == ("dense_rerank", 20, 5)
    assert (strategy.bm25_top_k, strategy.rrf_k) == (20, 60)


def _retrieval_block(**overrides) -> str:
    """A `retrieval:` block with the real field names, one of them broken."""
    fields = {
        "corpus": "statpearls",
        "embedding_model": "m",
        "strategy": "dense_rerank",
        "top_k_retrieve": 20,
        "top_k_rerank": 5,
        "reranker_model": "m",
        "bm25_top_k": 20,
        "rrf_k": 60,
    }
    fields.update(overrides)
    return "".join(f"  {key}: {value}\n" for key, value in fields.items())


@pytest.mark.parametrize(
    "overrides, message",
    [
        # A strategy the fan-out never computes would silently score nothing:
        # `retrieve_base` returns five lists, and `RetrievalStrategy.retrieve()`
        # would KeyError on the sixth name at the first question of a paid run.
        ({"strategy": "hybrid_rerank_v2"}, "unknown retrieval strategy"),
        # The cap that the first run's table hid: reranking 5 candidates to
        # depth 20 returns those same 5, so recall@20 is recall@5 wearing a
        # longer label. Reject it at load, not in a footnote.
        ({"top_k_rerank": 40}, "exceeds top_k"),
        ({"top_k_retrieve": 0}, "top_k must be"),
    ],
)
def test_load_config_refuses_a_funnel_it_cannot_measure(tmp_path, overrides, message):
    default = _write_config(tmp_path, retrieval_block=_retrieval_block(**overrides))

    with pytest.raises(ValueError, match=message):
        load_config(default, load_env_file=False)


def test_a_strategy_that_never_reranks_may_ask_for_more_than_it_retrieves(tmp_path):
    # The other side of that validator: `top_k_rerank` is inert for `bm25`, so
    # the check is scoped to the strategies that actually rerank. Over-rejecting
    # would block R4, which scores BM25 at depths the rerank cells never reach.
    default = _write_config(
        tmp_path,
        retrieval_block=_retrieval_block(strategy="bm25", top_k_retrieve=20, top_k_rerank=40),
    )

    config = load_config(default, load_env_file=False)

    strategy = RetrievalStrategy.from_config(config)
    assert strategy.name in BASE_STRATEGIES
    assert strategy.rerank_k == 40


def test_load_config_ignores_a_leftover_conditions_yaml(tmp_path):
    # The generator-side study is cut, but a reader with an old working tree has
    # `conditions.yaml` next to their `default.yaml`. It must be inert -- not
    # merged, not an error -- rather than half-loaded.
    default = _write_config(tmp_path)
    (tmp_path / "conditions.yaml").write_text(
        "conditions:\n"
        "  - id: \"1\"\n"
        "    name: a condition from the cut factorial\n"
        "    model: does_not_exist\n"
        "    retrieval: false\n"
    )

    config = load_config(default, load_env_file=False)

    assert not hasattr(config, "conditions")
    assert config.retrieval.strategy == "dense_rerank"



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


_DEFAULT_RETRIEVAL_BLOCK = (
    "  corpus: statpearls\n"
    "  embedding_model: m\n"
    "  strategy: dense_rerank\n"
    "  top_k_retrieve: 20\n"
    "  top_k_rerank: 5\n"
    "  reranker_model: m\n"
    "  bm25_top_k: 20\n"
    "  rrf_k: 60\n"
)


def _write_config(
    tmp_path,
    models_yaml: str = _MINIMAL_MODELS_YAML,
    retrieval_block: str = _DEFAULT_RETRIEVAL_BLOCK,
):
    """Write the two files `load_config` merges; return default.yaml.

    Two, not three: `conditions.yaml` stopped being merged when the generator-side
    study was cut, and chunk geometry is not a knob while the corpus is frozen.
    """
    (tmp_path / "models.yaml").write_text(models_yaml)
    default = tmp_path / "default.yaml"
    default.write_text(
        "benchmark: x\n"
        "dev_split: train\n"
        "test_split: test\n"
        "output_dir: outputs\n"
        "retrieval:\n"
        f"{retrieval_block}"
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
