"""Tests for scripts/exploration/raw_rag.py -- the comparison Phase 6 reports on.

Phase 6 spends 40 completions to produce one number: the closed-book -> RAG
transition on 20 questions Phase 5 already answered. Every way that number can be
wrong produces a *plausible table* rather than an error, so these tests are about
the comparison, not the generation:

* `outcome_of` -- `rescued` and `distracted` are the two findings the phase
  exists to count; if the answerability transitions collapse into them, a
  formatting change reads as a retrieval effect.
* `load_baseline` / `endpoint_drift` -- pairing with the wrong run (or the same
  model at a different endpoint) leaves the row count intact and the delta
  meaningless. Phase 5 §8.4 measured a 50% -> 70% swing from exactly that.
* `pin_check` -- "the same 20 questions" is a claim about a list of ids, and the
  only honest version of it is recomputing the seed's draw and printing the diff.
* `retrieve_pass` -- `Retriever.rerank()` overwrites `score` with the
  cross-encoder logit, so the cosine and the pre-rerank rank have to be captured
  before it runs or `moved`/`promoted` describe an artifact.

`run_question()` itself is not tested: it is glue around these helpers and the
live smoke run (`--limit 2`) exercises it against a real endpoint.

Loaded by path -- `scripts/` is deliberately not an importable package -- but it
needs no `sys.path` hack any more: since R1 its helpers (`eval.runlog`,
`generation.preflight`, `retrieval.strategy`) are ordinary package imports.
"""

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from medical_rag.config import ExperimentConfig, ModelConfig, RetrievalConfig
from medical_rag.data.load_medqa import MedQAQuestion
from medical_rag.retrieval.retriever import RetrievedChunk

_MODULE_NAME = "exploration_raw_rag"
_ROOT = Path(__file__).resolve().parents[1]
_PATH = _ROOT / "scripts" / "exploration" / "raw_rag.py"


@pytest.fixture(scope="module")
def raw_rag():
    """Load `raw_rag.py` once per module, by file path."""
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _model(key: str = "model_a", **overrides) -> ModelConfig:
    fields = {
        "name": key,
        "base_url": f"http://127.0.0.1:{8081 if key == 'model_a' else 8082}/v1",
        "api_model_name": f"Served-{key.upper()}",
        "api_key_env": "IRRELEVANT_API_KEY",
        "max_new_tokens": 1024,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    fields.update(overrides)
    return ModelConfig(**fields)


def _config(**overrides) -> ExperimentConfig:
    fields: dict[str, Any] = {
        "models": {"model_a": _model(), "model_b": _model("model_b")},
        "retrieval": RetrievalConfig(
            corpus="statpearls",
            embedding_model="embed-model",
            strategy="dense_rerank",
            top_k_retrieve=10,
            top_k_rerank=3,
            reranker_model="rerank-model",
        ),
        "benchmark": "medqa",
        "dev_split": "dev",
        "test_split": "test",
        "output_dir": "outputs",
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def _question(index: str = "1", answer_idx: str = "A") -> MedQAQuestion:
    options = {
        "A": "oral corticosteroids",
        "B": "ibuprofen",
        "C": "sarcoidosis",
        "D": "watchful waiting",
    }
    return MedQAQuestion(
        id=index,
        question=f"Question {index}?",
        options=options,
        answer_idx=answer_idx,
        answer_text=options[answer_idx],
    )


def _row(
    model: str = "model_a",
    question_id: str = "1",
    parsed_answer: str | None = "A",
    *,
    answer_idx: str = "A",
    **overrides: Any,
) -> dict[str, Any]:
    """A row shaped like `result_row()` output, with the Phase 6 additions."""
    row: dict[str, Any] = {
        "tag": "main",
        "model": model,
        "question_id": question_id,
        "parsed_answer": parsed_answer,
        "correct_answer": answer_idx,
        "is_correct": parsed_answer == answer_idx,
        "unanswered": parsed_answer is None,
        "endpoint": _model(model).base_url,
        "api_model": _model(model).api_model_name,
        "retrieved_chunk_ids": ["c1", "c2"],
    }
    row.update(overrides)
    return row


def _write_baseline(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    run = tmp_path / "phase5_closed_book" / "20260928T020841Z"
    run.mkdir(parents=True)
    (run / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return run


# --- the transition table ---------------------------------------------------


@pytest.mark.parametrize(
    ("rag_correct", "base_correct", "expected"),
    [
        (True, False, "rescued"),
        (False, True, "distracted"),
        (True, True, "both_right"),
        (False, False, "both_wrong"),
    ],
)
def test_outcome_of_the_four_answered_transitions(raw_rag, rag_correct, base_correct, expected):
    rag = _row(parsed_answer="C" if rag_correct else "B", is_correct=rag_correct)
    base = _row(parsed_answer="C" if base_correct else "B", is_correct=base_correct)
    assert raw_rag.outcome_of(rag, base) == expected


def test_outcome_of_keeps_answerability_separate_from_correctness(raw_rag):
    """A model that never answered closed-book is a different event than one wrong.

    Folding `newly_answered` into `rescued` would let `model_b`'s Phase 5
    formatting failures (the unanswered rows) be counted as retrieval wins.
    """
    answered = _row(parsed_answer="A", is_correct=True)
    silent = _row(parsed_answer=None, is_correct=False)
    assert raw_rag.outcome_of(answered, silent) == "newly_answered"
    assert raw_rag.outcome_of(silent, answered) == "newly_unanswered"
    assert raw_rag.outcome_of(silent, silent) == "both_unanswered"


def test_outcome_of_reports_a_missing_pair_rather_than_skipping_it(raw_rag):
    assert raw_rag.outcome_of(_row(), None) == "unpaired"


# --- the baseline it pairs against ------------------------------------------


def test_load_baseline_keeps_only_usable_main_rows(raw_rag, tmp_path):
    run = _write_baseline(
        tmp_path,
        [
            _row("model_a", "1"),
            _row("model_a", "2", tag="retry"),  # not the scored attempt
            _row("model_a", "3", error="timeout"),  # an attempt, not an outcome
            _row("model_b", "1"),
        ],
    )

    baseline = raw_rag.load_baseline(run)

    assert set(baseline) == {("model_a", "1"), ("model_b", "1")}


def test_load_baseline_refuses_a_directory_with_nothing_to_pair(raw_rag, tmp_path):
    run = _write_baseline(tmp_path, [_row(tag="retry"), _row(question_id="9", error="boom")])

    with pytest.raises(SystemExit, match="no usable main-tag rows"):
        raw_rag.load_baseline(run)


def test_load_baseline_accepts_a_results_file_directly(raw_rag, tmp_path):
    run = _write_baseline(tmp_path, [_row()])

    assert raw_rag.load_baseline(run / "results.jsonl") == {("model_a", "1"): _row()}


def test_endpoint_drift_is_silent_when_nothing_moved(raw_rag):
    baseline = {
        ("model_a", "1"): _row(),
        ("model_b", "1"): _row("model_b"),
    }

    assert raw_rag.endpoint_drift(baseline, _config(), ["model_a", "model_b"]) == []


def test_endpoint_drift_flags_the_endpoint_swap_that_moved_model_ten_points(raw_rag):
    """Phase 5 §8.4: same model, same 20 questions, different port -> 50% -> 70%."""
    baseline = {("model_a", "1"): _row(endpoint="http://127.0.0.1:9099/v1")}
    config = _config(models={"model_a": _model(base_url="http://127.0.0.1:8081/v1")})

    problems = raw_rag.endpoint_drift(baseline, config, ["model_a"])

    assert len(problems) == 1
    assert "measure the endpoint" in problems[0]
    assert "127.0.0.1:9099" in problems[0]


def test_endpoint_drift_flags_an_arm_the_baseline_never_answered(raw_rag):
    baseline = {("model_a", "1"): _row()}

    problems = raw_rag.endpoint_drift(baseline, _config(), ["model_a", "model_b"])

    assert len(problems) == 1
    assert "no rows for model 'model_b'" in problems[0]


def test_endpoint_drift_flags_a_baseline_that_is_not_one_run(raw_rag):
    baseline = {
        ("model_a", "1"): _row(),
        ("model_a", "2"): _row(endpoint="http://127.0.0.1:9099/v1"),
    }

    problems = raw_rag.endpoint_drift(baseline, _config(), ["model_a"])

    assert any("spans 2 endpoints" in problem for problem in problems)


# --- the pinned sample ------------------------------------------------------


def test_pin_check_confirms_a_draw_that_still_reproduces(raw_rag):
    """The positive branch, built from `select_sample` rather than a hardcoded list.

    The real pin against the real dev pool is checked by `--dry-run`, which loads
    the dataset; asserting it here would make the suite depend on the HF cache.
    """
    pool = [_question(str(index)) for index in range(200)]
    drawn = [question.id for question in raw_rag.select_sample(pool, 20, 1)]

    held, lines = raw_rag.pin_check(pool, drawn, 1)

    assert held, lines
    assert lines[0].startswith("pin OK")


def test_pin_check_prints_the_diff_when_the_draw_has_drifted(raw_rag):
    """The run still uses the pinned ids; what the check buys is the warning.

    A grown pool is the drift `select_sample()`'s docstring concedes it cannot
    prevent, so this is the case the print exists for.
    """
    pool = [_question(str(index)) for index in range(200)]
    pinned = [question.id for question in raw_rag.select_sample(pool, 20, 1)]
    grown = pool + [_question("999999")]

    held, lines = raw_rag.pin_check(grown, pinned, sample_seed=7)

    assert not held
    assert any(line.startswith("PIN DRIFT") for line in lines)
    assert any("added:" in line for line in lines)
    assert any("dropped:" in line for line in lines)
    assert any("FINDINGS.md" in line for line in lines)


def test_answer_chunk_rank_reads_prompt_order_not_vector_order(raw_rag):
    chunks = [
        RetrievedChunk(chunk_id="c2", title="T2", content="Dose ibuprofen at 400 mg.", score=9.0),
        RetrievedChunk(chunk_id="c1", title="T1", content="Unrelated excerpt.", score=8.0),
    ]
    question = _question()

    assert raw_rag.answer_chunk_rank(question, "B", chunks) == 1
    assert raw_rag.answer_chunk_rank(question, "C", chunks) is None
    assert raw_rag.answer_chunk_rank(question, None, chunks) is None


# --- the retrieval pass -----------------------------------------------------


class _StubRetriever:
    """`Retriever`'s call shape without torch, LanceDB, or a reranker download.

    `rerank()` reproduces the one behaviour `retrieve_pass()` exists to work
    around: it returns *copies* whose `score` has been overwritten with an
    unbounded cross-encoder logit, and it truncates to `top_k`.
    """

    def __init__(self, candidates: list[RetrievedChunk], kept: list[tuple[str, float]]):
        self._candidates = candidates
        self._kept = kept
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append(("retrieve", top_k))
        return list(self._candidates)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        self.calls.append(("rerank", top_k))
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        return [
            by_id[chunk_id].model_copy(update={"score": score})
            for chunk_id, score in self._kept[:top_k]
        ]


def _candidates() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(chunk_id=f"c{i}", title=f"Title {i}", content=f"Body {i}.", score=score)
        for i, score in [(1, 0.80), (2, 0.70), (3, 0.60), (4, 0.50)]
    ]


def _pass(raw_rag, kept: list[tuple[str, float]], top_k_retrieve: int = 4, top_k_rerank: int = 2):
    retriever = _StubRetriever(_candidates(), kept)
    static = {
        "index_dir": "data/index/statpearls",
        "index_rows": 1234,
        "device": "mps",
        "embedding_model": "embed-model",
        "reranker_model": "rerank-model",
    }
    chunks, context, meta = raw_rag.retrieve_pass(
        retriever, _question(), top_k_retrieve, top_k_rerank, static
    )
    return retriever, chunks, context, meta


def test_retrieve_pass_asks_for_the_k_it_records(raw_rag):
    retriever, _chunks, _context, meta = _pass(raw_rag, [("c3", 4.0), ("c1", 2.0)])

    assert retriever.calls == [("retrieve", 4), ("rerank", 2)]
    assert meta["params"]["top_k_retrieve"] == 4
    assert meta["params"]["top_k_rerank"] == 2
    assert meta["params"]["candidates"] == 4
    assert meta["params"]["kept"] == 2
    assert meta["params"]["index_rows"] == 1234  # the static half survives


def test_retrieve_pass_keeps_the_cosine_the_reranker_overwrote(raw_rag):
    """The trap: `rerank()` returns copies with `score` replaced by a logit.

    If `vector_score` were read off the kept list it would be the logit again,
    and `summarize_retrieval()`'s two score rows would be the same number.
    """
    _retriever, _chunks, _context, meta = _pass(raw_rag, [("c3", 4.0), ("c1", 2.0)])

    by_id = {chunk["chunk_id"]: chunk for chunk in meta["chunks"]}
    assert by_id["c3"]["vector_score"] == 0.60  # cosine, captured pre-rerank
    assert by_id["c3"]["rerank_score"] == 4.0  # logit, from the kept copy
    assert by_id["c1"]["vector_rank"] == 1
    assert by_id["c3"]["vector_rank"] == 3
    assert meta["params"]["vector_top1_score"] == 0.80
    assert meta["params"]["vector_last_score"] == 0.50
    assert meta["params"]["rerank_top_score"] == 4.0


def test_retrieve_pass_numbers_the_prompt_in_final_order(raw_rag):
    _retriever, chunks, context, meta = _pass(raw_rag, [("c3", 4.0), ("c1", 2.0)])

    assert [chunk["prompt_rank"] for chunk in meta["chunks"]] == [1, 2]
    assert [chunk.chunk_id for chunk in chunks] == ["c3", "c1"]
    assert meta["chunk_ids"] == ["c3", "c1"]
    assert meta["context_chars"] == len(context)
    prompt = raw_rag.build_answer_prompt("Question 1?", _question().options, chunks)
    assert raw_rag.missing_chunk_evidence(prompt, chunks) == []


def test_retrieve_pass_churn_survives_into_the_summary(raw_rag):
    """`moved`/`promoted` are the reranking claim, and they read off `vector_rank`."""
    _retriever, _chunks, _context, meta = _pass(raw_rag, [("c3", 4.0), ("c1", 2.0)])
    row = {
        "tag": "main",
        "question_id": "1",
        "retrieved_chunk_ids": meta["chunk_ids"],
        "chunks": meta["chunks"],
        "retrieval": meta["params"],
        "context_chars": meta["context_chars"],
        "prompt_chars": 1000,
    }

    summary = raw_rag.summarize_retrieval([row])

    assert "2 chunk(s) moved" in summary
    assert "1 entered the prompt from past the top-k_rerank cut" in summary
    assert "1/1 pass(es) changed at all" in summary


def test_prompt_artifact_prints_the_string_and_both_scores(raw_rag):
    _retriever, chunks, _context, meta = _pass(raw_rag, [("c3", 4.0), ("c1", 2.0)])
    prompt = raw_rag.build_answer_prompt("Question 1?", _question().options, chunks)

    artifact = raw_rag.prompt_artifact(_question(), prompt, meta)

    assert artifact.startswith("### q1 -- CONDITION phase6_raw_rag")
    assert f"```\n{prompt}\n```" in artifact
    assert '[1] c3 "Title 3" vector_rank=3 vector=0.6000 rerank=4.0000' in artifact
    assert '[2] c1 "Title 1" vector_rank=1 vector=0.8000 rerank=2.0000' in artifact
    assert "data/index/statpearls" in artifact and "1234 rows" in artifact


def test_prompt_artifact_dashes_a_score_that_does_not_exist(raw_rag):
    """A missing score must not print as 0.0000 -- that reads as a measured zero."""
    meta = {
        "chunk_ids": ["c1"],
        "chunks": [
            {
                "chunk_id": "c1",
                "title": "T",
                "prompt_rank": 1,
                "vector_rank": 1,
                "vector_score": None,
                "rerank_score": None,
            }
        ],
        "context_chars": 0,
        "params": {
            "retrieve_s": 0.0,
            "rerank_s": 0.0,
            "index_dir": "x",
            "index_rows": 0,
            "device": "cpu",
            "top_k_retrieve": 1,
            "top_k_rerank": 1,
        },
    }

    artifact = raw_rag.prompt_artifact(_question(), "PROMPT", meta)

    assert "vector=- rerank=-" in artifact
    assert "0.0000" not in artifact


# --- pairing.md -------------------------------------------------------------


def test_pairing_markdown_tallies_transitions_and_keeps_the_unpaired_visible(raw_rag):
    rows = [
        _row("model_a", "1", parsed_answer="C", is_correct=True, outcome="rescued",
             base_answer="B", base_correct=False, answer_chunk_rank=2, gold_chunk_rank=1),
        _row("model_a", "2", parsed_answer="A", is_correct=False, outcome="both_wrong",
             base_answer="A", base_correct=False),
        _row("model_a", "3", parsed_answer="A", is_correct=True, outcome="unpaired",
             base_answer="-", base_correct=None),
        _row("model_b", "1", tag="retry"),  # an attempt, never part of a table
    ]

    table = raw_rag.pairing_markdown(rows, ["model_a", "model_b"])

    assert "## `model_a` (3 question(s))" in table
    assert "| rescued | 1 |" in table
    assert "closed-book correct 0 -> RAG correct 2 of 3" in table
    assert "| 1 | A | B | C | rescued | 2 | 1 | 2 |" in table
    assert "| 2 | A | A | A | both_wrong | - | - | 2 |" in table
    assert "| 3 | A | - | A | unpaired | - | - | 2 |" in table
    assert "## `model_b`" not in table


# --- preflight and the index it reads ---------------------------------------


class _Client:
    """Just enough `LLMClient` for `preflight()`: `.client.models.list()`."""

    def __init__(self, ids: list[str] | None = None, error: Exception | None = None):
        self._ids, self._error = ids or [], error

    async def _list(self):
        if self._error:
            raise self._error
        return SimpleNamespace(data=[SimpleNamespace(id=model_id) for model_id in self._ids])

    @property
    def client(self):
        return SimpleNamespace(models=SimpleNamespace(list=self._list))


def test_preflight_proceeds_when_the_endpoint_serves_the_named_model(raw_rag):
    decision, lines = asyncio.run(
        raw_rag.preflight(_model(), _Client(["Other-Model", "Served-MODEL_A"]))
    )

    assert decision is raw_rag.Preflight.PROCEED
    assert "serves Served-MODEL_A" in lines[0]


def test_preflight_blocks_a_model_the_endpoint_does_not_serve(raw_rag):
    """Not a warning: rows from a different model look identical and mean else."""
    decision, lines = asyncio.run(raw_rag.preflight(_model(), _Client(["Something-Else"])))

    assert decision is raw_rag.Preflight.BLOCK
    assert "Served-MODEL_A" in lines[0]


def test_preflight_blocks_a_dead_endpoint(raw_rag):
    decision, lines = asyncio.run(
        raw_rag.preflight(_model(), _Client(error=ConnectionError("connection refused")))
    )

    assert decision is raw_rag.Preflight.BLOCK
    assert "not answering" in lines[0]


def test_resolve_index_dir_names_the_table_the_plan_prints(raw_rag):
    assert raw_rag.resolve_index_dir(_config()) == Path("data/index/statpearls")
    assert raw_rag.resolve_index_dir(_config(), "elsewhere/table") == Path("elsewhere/table")


def test_dry_run_reports_and_writes_no_run_directory(raw_rag, tmp_path, capsys, monkeypatch):
    """`--dry-run` must report and stop -- no run directory, no stamped name.

    Two bugs this pins shut: a dry run that mints a run directory tells the next
    reader a run happened, and because `RunWriter` disambiguates by appending
    `-1`, a second dry run would leave two of them behind. `--dry-run` is the
    command run before spending 40 completions, so "it wrote nothing" has to be
    checked, not assumed. The dataset and the id lookup are stubbed -- what is
    under test is the dry run's side effects, not whether the pinned ids resolve.
    """
    monkeypatch.setattr(raw_rag, "load_medqa", lambda split: [_question("1")])
    monkeypatch.setattr(
        raw_rag,
        "parse_args",
        lambda: argparse.Namespace(
            config="config/default.yaml",
            models=["model_a"],
            split=None,
            question_ids=["1"],
            sample_seed=raw_rag.DEFAULT_SAMPLE_SEED,
            limit=None,
            index_dir=None,
            top_k_retrieve=None,
            top_k_rerank=None,
            baseline_dir=raw_rag.DEFAULT_BASELINE_DIR,
            concurrent_questions=1,
            concurrent_models=False,
            output_dir=str(tmp_path),
            no_chains=False,
            resume=None,
            dry_run=True,
            note=[],
        ),
    )

    with pytest.raises(SystemExit) as exit_info:
        raw_rag.main()

    out = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert not list(tmp_path.glob("*")), "dry run wrote into the output root"
    assert "no index load, no preflight, no requests" in out
    assert "pin OK" in out
