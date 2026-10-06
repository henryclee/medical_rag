"""Tests for `retrieval.strategy.RetrievalStrategy` -- the one definition of
"what retrieval does", shared by the measuring harness and the shipped path.

This type exists because the two used to be separate. The grid composed its five
strategies in the harness while `raw_rag.py` retrieved however it happened to, so
a findings table could report a `hybrid` that nothing downstream would ever run --
the same drift as the first run's `reform_dense` column, one level up. These
assertions are therefore about *identity* rather than arithmetic
(`test_retrieval_tuning.py` already covers the fan-out's numbers):

* `config/default.yaml` -> `RetrievalStrategy` loses no field, and `--strategy`
  overrides the name without quietly resetting the funnel's depths
* `retrieve()` returns the ranking the grid would have scored -- the same
  fan-out, not a strategy-specific shortcut -- and the cost that implies is
  pinned, so the trade stays a decision rather than an accident
* one query through `lists()` costs one dense pass and one lexical pass no matter
  how many strategies are read out of it, which is why R4/R5 can compare many
  hypotheses for the price of loading the index once
* `bm25_top_k` reaches the lexical leg rather than being clamped to `top_k`,
  which is the knob R4 exists to turn
* `rrf_k=None` defers to `fusion.rrf_fuse()`'s default instead of restating it
"""

from typing import Any

import pytest

from medical_rag.config import ExperimentConfig, RetrievalConfig
from medical_rag.retrieval import strategy as strategy_module
from medical_rag.retrieval.retriever import RetrievedChunk
from medical_rag.retrieval.strategy import BASE_STRATEGIES, RetrievalStrategy


class FakeRetriever:
    """Records calls so "retrieved once per query" is an assertion, not a hope."""

    def __init__(self) -> None:
        self.retrieve_calls: list[tuple[str, int]] = []
        self.rerank_calls: list[tuple[str, int, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.retrieve_calls.append((query, top_k))
        return [_chunk(f"dense-{i}") for i in range(top_k)]

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        self.rerank_calls.append((query, len(chunks), top_k))
        return chunks[:top_k]


class FakeBM25:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.search_calls.append((query, top_k))
        return [_chunk(f"bm25-{i}") for i in range(top_k)]


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, title=f"t {chunk_id}", content=f"c {chunk_id}", score=1.0
    )


def _config(**overrides: Any) -> ExperimentConfig:
    fields: dict[str, Any] = {
        "models": {},
        "retrieval": RetrievalConfig(
            corpus="statpearls",
            embedding_model="embed-model",
            strategy="dense_rerank",
            top_k_retrieve=6,
            top_k_rerank=4,
            reranker_model="rerank-model",
            bm25_top_k=9,
            rrf_k=25,
        ),
        "benchmark": "medqa",
        "dev_split": "train",
        "test_split": "test",
        "output_dir": "outputs",
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def test_config_reaches_the_strategy_without_losing_a_depth():
    strategy = RetrievalStrategy.from_config(_config())

    assert (strategy.name, strategy.top_k, strategy.rerank_k) == ("dense_rerank", 6, 4)
    assert (strategy.bm25_top_k, strategy.rrf_k) == (9, 25)


def test_overriding_the_name_does_not_reset_the_funnel():
    # `--strategy hybrid` chooses which column to read out; it must not silently
    # return the depths to their defaults, or an ad-hoc comparison would score a
    # different funnel than the one `default.yaml` claims.
    strategy = RetrievalStrategy.from_config(_config(), "hybrid")

    assert strategy.name == "hybrid"
    assert (strategy.top_k, strategy.rerank_k, strategy.bm25_top_k) == (6, 4, 9)


def test_a_configured_rrf_k_reaches_the_fusion_leg():
    depths: list[dict[str, Any]] = []

    def spy(*lists: Any, **kwargs: Any) -> list[RetrievedChunk]:
        depths.append(kwargs)
        return list(lists[0])

    original = strategy_module.rrf_fuse
    strategy_module.rrf_fuse = spy
    try:
        RetrievalStrategy.from_config(_config()).lists("q", FakeRetriever(), FakeBM25())
    finally:
        strategy_module.rrf_fuse = original

    assert depths and depths[0]["k"] == 25


def test_rrf_k_none_defers_to_the_fusion_default_instead_of_restating_it():
    # Two places naming a default is how the grid and the module disagree about
    # what "unconfigured" means; `None` means `fusion` decides, so there is one
    # answer to look up.
    depths: list[dict[str, Any]] = []

    def spy(*lists: Any, **kwargs: Any) -> list[RetrievedChunk]:
        depths.append(kwargs)
        return list(lists[0])

    original = strategy_module.rrf_fuse
    strategy_module.rrf_fuse = spy
    try:
        RetrievalStrategy(name="hybrid", top_k=6, rerank_k=4, rrf_k=None).lists(
            "q", FakeRetriever(), FakeBM25()
        )
    finally:
        strategy_module.rrf_fuse = original

    assert depths and "k" not in depths[0]


@pytest.mark.parametrize("name", list(BASE_STRATEGIES))
def test_one_fan_out_yields_every_ranked_list(name: str):
    # The cost model behind "try another lever for free": one query through the
    # fan-out yields all five ranked lists, so comparing strategies against a
    # cached verdict set is arithmetic. If a future change re-retrieved per
    # strategy, the offline evaluator would silently become 5x the index work.
    retriever, bm25 = FakeRetriever(), FakeBM25()

    lists = RetrievalStrategy(name=name, top_k=6, rerank_k=4, bm25_top_k=9).lists(
        "q", retriever, bm25
    )

    assert set(lists) == set(BASE_STRATEGIES)
    assert retriever.retrieve_calls == [("q", 6)], "one dense pass"
    assert bm25.search_calls == [("q", 9)], "the lexical leg gets bm25_top_k, not top_k"
    assert retriever.rerank_calls == [("q", 6, 4), ("q", len(lists["hybrid"]), 4)]


@pytest.mark.parametrize("name", list(BASE_STRATEGIES))
def test_retrieve_returns_the_ranking_the_grid_would_score(name: str):
    # Same call path, so a reported column and a shipped call cannot mean two
    # pipelines -- the drift this type was introduced to close.
    #
    # Note the *cost*, which is easy to miss from the outside: `retrieve()` runs
    # the whole fan-out and returns one list, so a single-strategy production call
    # pays for two reranks and a BM25 search it will not use. That is deliberate
    # -- a per-strategy shortcut would mean the shipped path no longer shares the
    # grid's code, which is the worse failure -- but it is why scoring several
    # strategies goes through one `lists()` call rather than `retrieve()` per
    # strategy. This assertion pins the doubling so a future optimisation is a
    # conscious change rather than an accident.
    retriever, bm25 = FakeRetriever(), FakeBM25()
    strategy = RetrievalStrategy(name=name, top_k=6, rerank_k=4, bm25_top_k=9)

    ranked = strategy.retrieve("q", retriever, bm25)

    assert ranked == strategy.lists("q", retriever, bm25)[name]
    assert len(retriever.retrieve_calls) == 2


def test_the_funnel_depths_are_the_lists_the_names_claim():
    # `hybrid_rerank` reranks the *fused* list. Read from the wrong input it is
    # just `dense_rerank` with a new label -- which is how a reranker looks like a
    # winner when it only re-ordered a list it was handed twice.
    lists = RetrievalStrategy(name="hybrid", top_k=6, rerank_k=4, bm25_top_k=9).lists(
        "q", FakeRetriever(), FakeBM25()
    )

    assert len(lists["dense"]) == 6
    assert len(lists["bm25"]) == 9
    assert len(lists["hybrid"]) == 6
    assert len(lists["dense_rerank"]) == 4
    assert len(lists["hybrid_rerank"]) == 4
    assert lists["dense_rerank"] != lists["hybrid_rerank"]


def test_a_strategy_name_outside_the_measured_set_is_rejected_at_construction():
    # `load_config` validates this too; the direct check is here because
    # `eval_retrieval.py` also builds strategies from `--strategy` at the REPL,
    # and a typo like `hybrid_reeank` would otherwise surface as a KeyError
    # halfway through scoring.
    with pytest.raises(ValueError, match="unknown retrieval strategy"):
        RetrievalStrategy(name="hybrid_reeank")
