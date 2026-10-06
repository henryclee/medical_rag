"""Tests for the retrieval-tuning tooling (`experiments/retrieval_tuning/`).

These modules exist because a number in `FINDINGS.md` was wrong in a way no test
would have caught: `reform_dense` looked like a method, its query was the
unmodified question on 18 of 20 rows, and the reported delta was real arithmetic
on a lever that never moved. So the tests here target *that* class of failure -- a
plausible result computed from the wrong input -- rather than arithmetic:

* the grid is 10 cells and `orig`/`reform` differ by exactly one variable, so a
  delta between them is attributable (`retrieve_grid`, `rerank_with`)
* a reformulation that cannot be parsed is *recorded* as a fallback instead of
  silently returning the question (`parse_information_need`, `reformulate_query`)
* the judge cache cannot mix rubric generations, and an inspection that hits it
  makes zero calls (`load_cache`, `ensure_verdicts`, `judge_prompt_sha`)
* a chunk with no verdict is rendered as unmeasured, never as `irrelevant`
  (`build_cells`, `metrics`)
* the ceiling probe cannot be mistaken for a retrieval lever -- it carries its
  oracle label into the render, and a literal miss reads as "not stated in these
  words", never as "absent from the corpus" (`ceiling`, `render_ceiling`)
* the served page renders from the same `render_main` as the written report, spends
  no completion and no re-retrieval to re-grade a grid, keeps one view per question
  so `aggregate_views` counts questions rather than clicks, and answers a bad knob
  with a 400 rather than a 200 holding a traceback (`serve`, `render`)

The modules under test moved into the package in R1 -- `PLAN.md`'s audit trail has
the full old-name map; these long-standing test files keep their original short
names for them through the aliases below, because the assertions are about
behaviour and renaming 1300 lines of references would only bury the diff. The
mapping, if you are chasing a name:

    strategies    -> medical_rag.retrieval.strategy
    render        -> medical_rag.eval.report
    chunk_store   -> medical_rag.eval.store
    judge_cache   -> medical_rag.eval.judge
    judge_prompt  -> medical_rag.eval.rubric
    reformulate   -> medical_rag.eval.rewrite
    serve         -> medical_rag.eval.lab
"""

import asyncio
import json
import re
import threading
import time
import types
from pathlib import Path

import pytest

from medical_rag.data.load_medqa import MedQAQuestion
from medical_rag.retrieval import ceiling
from medical_rag.retrieval import strategy as strategies
from medical_rag.retrieval.retriever import RetrievedChunk
from medical_rag.eval import judge as judge_cache
from medical_rag.eval import metrics
from medical_rag.eval import report as render
from medical_rag.eval import rewrite as reformulate
from medical_rag.eval import rubric as judge_prompt
from medical_rag.eval import store as chunk_store
from medical_rag.eval import lab as serve


def chunk(chunk_id: str, *, body: str | None = None, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        title=f"title of {chunk_id}",
        content=body if body is not None else f"body of {chunk_id}",
        score=score,
    )


class FakeRetriever:
    """Deterministic stand-in for `Retriever`.

    Chunk ids encode which query produced them, so a test can assert the reform
    column retrieved on the *rewrite* -- and catch the first run's failure mode of
    retrieving the question twice.
    """

    def __init__(self) -> None:
        self.retrieve_calls: list[tuple[str, int]] = []
        self.rerank_calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.retrieve_calls.append((query, k))
        tag = "r" if "REWRITE" in query else "o"
        return [chunk(f"c{tag}{i}", score=1.0 / i) for i in range(1, k + 1)]

    def rerank(self, query: str, chunks, k: int) -> list[RetrievedChunk]:
        self.rerank_calls.append((query, len(chunks)))
        return list(reversed(list(chunks)[:k]))


class FakeBM25:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        tag = "r" if "REWRITE" in query else "o"
        return [chunk(f"b{tag}{i}", score=float(k - i)) for i in range(1, k + 1)]


def question(answer: str = "B", text: str = "A patient has a thing. What is it?") -> MedQAQuestion:
    return MedQAQuestion(
        id="42",
        question=text,
        options={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        answer_idx=answer,
        answer_text="beta",
        meta_info=None,
    )


# --- the grid ------------------------------------------------------------------


def test_grid_is_five_strategies_times_two_query_variants():
    assert len(strategies.METHOD_GRID) == 10
    assert len(set(strategies.METHOD_GRID)) == 10
    assert set(strategies.grid_for(["orig"])) == {
        "dense__orig", "dense_rerank__orig", "bm25__orig", "hybrid__orig", "hybrid_rerank__orig",
    }
    with pytest.raises(ValueError):
        strategies.method_id("dense", "paraphrase")
    with pytest.raises(ValueError):
        strategies.split_method("reform_dense")


def test_legacy_method_ids_alias_onto_the_grid_and_are_flagged():
    # `--from-run` on the 2026-09-29 run depends on this: its rows predate the grid.
    assert strategies.canonical_method("reform_dense") == ("dense__reform", True)
    assert strategies.canonical_method("dense") == ("dense__orig", True)
    assert strategies.canonical_method("hybrid__reform") == ("hybrid__reform", False)
    with pytest.raises(ValueError):
        strategies.canonical_method("qqq")


def test_retrieve_grid_respects_each_depth_knob():
    retriever, bm25 = FakeRetriever(), FakeBM25()
    grid = strategies.retrieve_grid(
        "question", retriever, bm25, reformulated_text="REWRITE thing",
        top_k=10, rerank_k=4, bm25_top_k=25,
    )
    assert len(grid.lists) == 10
    assert len(grid.lists["dense__orig"]) == 10
    assert len(grid.lists["bm25__orig"]) == 25, "bm25_top_k must be able to exceed top_k"
    assert len(grid.lists["hybrid__orig"]) == 10
    assert len(grid.lists["dense_rerank__orig"]) == 4
    assert len(grid.lists["hybrid_rerank__reform"]) == 4


def test_reform_column_retrieves_on_the_rewrite_and_orig_on_the_question():
    retriever, bm25 = FakeRetriever(), FakeBM25()
    grid = strategies.retrieve_grid(
        "question", retriever, bm25, reformulated_text="REWRITE thing", top_k=5, rerank_k=3
    )
    assert grid.lists["dense__orig"][0].chunk_id == "co1"
    assert grid.lists["dense__reform"][0].chunk_id == "cr1"
    assert set(retriever.retrieve_calls) == {("question", 5), ("REWRITE thing", 5)}
    assert bm25.calls == [("question", 5), ("REWRITE thing", 5)]
    assert grid.variant_queries == {"orig": "question", "reform": "REWRITE thing"}


def test_reranker_default_string_keeps_orig_and_reform_one_variable_apart():
    # The reform column must differ from orig in the *embedding* query only, or a
    # delta between them is not attributable to reformulation.
    retriever = FakeRetriever()
    strategies.retrieve_grid(
        "the question", retriever, FakeBM25(), reformulated_text="REWRITE thing",
        top_k=5, rerank_k=2,
    )
    assert {query for query, _ in retriever.rerank_calls} == {"the question"}

    retriever = FakeRetriever()
    grid = strategies.retrieve_grid(
        "the question", retriever, FakeBM25(), reformulated_text="REWRITE thing",
        top_k=5, rerank_k=2, rerank_with="reform",
    )
    assert {query for query, _ in retriever.rerank_calls} == {"the question", "REWRITE thing"}
    assert grid.rerank_with == "reform"


def test_identical_query_marks_reuse_instead_of_retrieving_twice():
    # This is the 18/20 bug made structurally visible: no second retrieval, and
    # the grid says the column is a copy instead of leaving two silent equals.
    retriever = FakeRetriever()
    grid = strategies.retrieve_grid(
        "same", retriever, FakeBM25(), reformulated_text="same", top_k=5, rerank_k=2
    )
    assert grid.reused_variant == "reform"
    assert retriever.retrieve_calls == [("same", 5)], "must not embed one string twice"
    assert grid.lists["dense__reform"] == grid.lists["dense__orig"]

    no_reform = strategies.retrieve_grid("same", FakeRetriever(), FakeBM25(), top_k=5, rerank_k=2)
    assert len(no_reform.lists) == 5


def test_prerank_map_tracks_the_source_list_of_each_rerank_cell():
    grid = strategies.retrieve_grid(
        "question", FakeRetriever(), FakeBM25(), reformulated_text="REWRITE thing",
        top_k=5, rerank_k=3,
    )
    assert strategies.prerank_map("dense__orig", grid.lists) == {}
    assert strategies.prerank_map("bm25__reform", grid.lists) == {}
    assert strategies.prerank_map("dense_rerank__orig", grid.lists)["co5"] == 5
    assert strategies.prerank_map("hybrid_rerank__orig", grid.lists)["co1"] == 1
    assert strategies.prerank_map("dense_rerank__reform", grid.lists)["cr5"] == 5


# --- reformulation parsing -----------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"information_need": "cause of X"}', "cause of X"),
        ('```json\n{"information_need": "cause of X"}\n```', "cause of X"),
        ('Sure, here you go:\n{"information_need": "cause of X"}\nHope that helps.', "cause of X"),
        ('prefix {"information_need": "cause of X"} suffix', "cause of X"),
        ('{"information_need": "a \\"quoted\\" need"}', 'a "quoted" need'),
        ('{"information_need": "\\n"}', None),
    ],
)
def test_parse_information_need_survives_wrapping(raw, expected):
    # Every non-empty case below was a *fallback* in the first run: `json.loads`
    # over a free-form completion, and the question went back out as its own
    # rewrite while the row still looked successful.
    assert reformulate.parse_information_need(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["I cannot answer that.", '{"information_need": ""}', '{"information_need": "   "}',
     '{"other_field": "cause of X"}', ""],
)
def test_parse_information_need_refuses_thin_content(raw):
    # An empty `information_need` must not count as a rewrite: retrieving the
    # question minus nothing and labelling it `__reform` is the old bug wearing a
    # success badge.
    assert reformulate.parse_information_need(raw) is None


class FakeClient:
    """Minimal `LLMClient` double: scripted completions, counted calls."""

    def __init__(self, *contents, structured=None, raises=None) -> None:
        self.config = type("Cfg", (), {"name": "fake-model"})()
        self._contents = list(contents)
        self._structured = structured
        self._raises = raises
        self.calls: list[str] = []

    async def agenerate(self, prompt, *, system_prompt=None, **kwargs):
        self.calls.append("agenerate")
        if self._raises is not None:
            raise self._raises
        return type("Res", (), {"content": self._contents.pop(0) if self._contents else ""})()

    async def agenerate_structured(self, prompt, output_type, *, system_prompt=None, **kwargs):
        self.calls.append("agenerate_structured")
        if self._structured is None:
            raise reformulate.LLMError("structured output failed: JSON parse error")
        return self._structured


def test_reformulate_query_parses_a_fenced_completion_without_falling_back():
    client = FakeClient('Here:\n```json\n{"information_need": "chronic mesenteric ischemia"}\n```')
    result = asyncio.run(reformulate.reformulate_query(question(), client))
    assert result.query == "chronic mesenteric ischemia"
    assert result.fallback is False
    assert "information_need" in (result.raw_response or "")
    assert client.calls == ["agenerate"], "no second call once the tolerant parse works"


def test_reformulate_query_retries_structured_when_free_form_is_unparsable():
    client = FakeClient(
        "I think the answer is beta.",
        structured=reformulate.Reformulation(information_need="diagnostic test for X"),
    )
    result = asyncio.run(reformulate.reformulate_query(question(), client))
    assert result.query == "diagnostic test for X"
    assert result.fallback is False
    assert client.calls == ["agenerate", "agenerate_structured"]


def test_reformulate_query_records_a_fallback_it_could_not_avoid():
    result = asyncio.run(reformulate.reformulate_query(question(), FakeClient("no json here")))
    assert result.query == question().question, "the query stays usable..."
    assert result.fallback is True
    assert result.error, "...but a fallback with no recorded reason is the old bug"
    assert result.method.startswith("fallback")


def test_reformulate_query_reraises_transport_errors_instead_of_labelling_them():
    import openai

    error = reformulate.LLMError("endpoint down")
    error.__cause__ = openai.APIConnectionError(request=None)
    with pytest.raises(reformulate.LLMError):
        asyncio.run(reformulate.reformulate_query(question(), FakeClient(raises=error)))
    # ...while a *parse* failure of the same shape must fall back, not raise.
    assert not reformulate._is_transport(reformulate.LLMError("bad json"))


def test_manual_result_never_claims_a_prompt_ran():
    result = reformulate.manual_result("  a hand written need  ")
    assert result.query == "a hand written need"
    assert result.fallback is False
    assert "typed by hand" in result.prompt


# --- judge cache ---------------------------------------------------------------


def test_the_harness_keeps_no_private_copy_of_the_rubric():
    # The ~820 paid verdicts were worded by the harness, and this guard existed so
    # that a reworded second copy elsewhere could not silently orphan them. The R1
    # move removed the harness's copy outright -- it stamps the sha and never words
    # a prompt -- so the guard is now structural: exactly one wording, and the two
    # modules that touch it resolve it out of `eval.rubric`.
    from medical_rag.eval import harness, judge

    assert not [n for n in vars(harness) if n.startswith("_JUDGE")], (
        "a second copy of the rubric in the harness would let the wording it "
        "grades with drift from the sha it stamps"
    )
    assert judge.JUDGE_SYSTEM_PROMPT is judge_prompt.JUDGE_SYSTEM_PROMPT
    assert judge.build_judge_prompt is judge_prompt.build_judge_prompt
    assert harness.judge_prompt_sha is judge_prompt.judge_prompt_sha
    built = judge_prompt.build_judge_prompt(question(), [chunk("c1")])
    assert "c1" in built and "NOT told which option is correct" in built
    assert built.count("beta") == 1, "the gold option appears once, as an option -- never as the answer"
    assert len(judge_prompt.judge_prompt_sha()) == 12


def test_cache_round_trip_prompt_invalidation_and_torn_lines(tmp_path):
    path = tmp_path / "judged.jsonl"
    record = {
        "question_id": "42", "chunk_id": "c1", "relevance": "relevant",
        "supports_options": ["B"], "reason": "names it",
        "judge_prompt_sha": judge_prompt.judge_prompt_sha(),
    }
    judge_cache.append_records(path, [record])
    verdicts, counts = judge_cache.load_cache(path)
    assert verdicts[("42", "c1")]["relevance"] == "relevant"
    assert counts == {"loaded": 1, "stale_prompt": 0, "malformed": 0, "missing_file": 0}

    # A rubric edit must invalidate rather than mix: same key, other sha, ignored.
    judge_cache.append_records(path, [dict(record, judge_prompt_sha="0" * 12)])
    _, counts = judge_cache.load_cache(path)
    assert counts["stale_prompt"] == 1 and counts["loaded"] == 1

    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"question_id": "42", "chu')  # what kill -9 leaves behind
    _, counts = judge_cache.load_cache(path)
    assert counts["malformed"] == 1, "a torn tail must cost a warning, not the whole cache"


def test_import_run_seeds_the_cache_from_a_finished_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {"question_id": "1", "correct_answer": "B",
         "judgments": {"c1": {"relevance": "relevant", "supports_options": ["B"], "reason": "r"}}},
        {"question_id": "2", "correct_answer": "A",
         "judgments": {"c2": {"relevance": "irrelevant", "supports_options": [], "reason": "r"}}},
        {"question_id": "3", "error": "judge died",
         "judgments": {"c9": {"relevance": "relevant", "supports_options": ["B"], "reason": "r"}}},
    ]
    (run / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    cache = tmp_path / "cache.jsonl"

    assert judge_cache.import_run(run, cache) == {"questions": 2, "imported": 2, "already_cached": 0}
    assert judge_cache.import_run(run, cache)["imported"] == 0, "re-import must not duplicate"
    verdicts, _ = judge_cache.load_cache(cache)
    assert ("3", "c9") not in verdicts, "an error row's judgments were never earned"


class FakeJudge:
    """Returns a fixed set of chunk_ids per batch, so batching is observable."""

    def __init__(self, returned_per_batch):
        self.config = type("Cfg", (), {"name": "judge-model"})()
        self.returned = list(returned_per_batch)
        self.prompts: list[str] = []

    async def agenerate_structured(self, prompt, output_type, *, system_prompt=None, **kwargs):
        from medical_rag.eval.rubric import ChunkJudgment

        self.prompts.append(prompt)
        return judge_prompt.JudgeVerdict(
            judgments=[
                ChunkJudgment(chunk_id=cid, relevance="relevant", supports_options=["B"], reason="r")
                for cid in self.returned.pop(0)
            ]
        )


def test_ensure_verdicts_judges_only_what_the_cache_does_not_have(tmp_path):
    chunks = [chunk(f"c{i}") for i in range(1, 6)]
    cache_path = tmp_path / "cache.jsonl"
    cache: dict = {}
    judge = FakeJudge([["c1", "c2", "c3"], ["c4"]])  # c5 is deliberately never returned

    first = asyncio.run(
        judge_cache.ensure_verdicts(question(), chunks, cache, client=judge, cache_path=cache_path, batch=3)
    )
    assert len(judge.prompts) == 2, "five chunks at batch=3 is two calls"
    assert first.judged == 4 and first.new_chunk_ids == ["c1", "c2", "c3", "c4"]
    assert first.missing_judgment == 1 and first.problems, "a dropped chunk must not read as irrelevant"

    judge_again = FakeJudge([])
    second = asyncio.run(
        judge_cache.ensure_verdicts(question(), chunks[:4], dict(cache), client=judge_again, cache_path=cache_path, batch=3)
    )
    assert judge_again.prompts == [], "re-inspecting a question must cost zero judge calls"
    assert second.cached == 4 and second.judged == 0

    _, counts = judge_cache.load_cache(cache_path)
    assert counts["loaded"] == 4, "verdicts have to be on disk, not just in memory"


# --- metrics -------------------------------------------------------------------


def test_first_relevant_rank_ignores_partial():
    judgments = {
        "c1": {"relevance": "partial", "supports_options": []},
        "c2": {"relevance": "relevant", "supports_options": []},
    }
    assert metrics.first_relevant_rank(["c1", "c2"], judgments) == 2
    assert metrics.first_relevant_rank(["c1"], judgments) is None


def test_recall_counts_supports_options_not_relevance():
    judgments = {
        "c1": {"relevance": "relevant", "supports_options": ["C"]},
        "c2": {"relevance": "irrelevant", "supports_options": ["B"]},
    }
    assert metrics.gold_hit(["c1"], judgments, "B") is False
    assert metrics.gold_hit(["c1", "c2"], judgments, "B") is True
    assert metrics.gold_hit(["c1"], judgments, "C") is True


def test_aggregate_treats_a_missing_column_as_unmeasured_not_as_zero():
    rows = [
        {"correct_answer": "B", "candidates": {"dense__orig": ["c1"], "dense__reform": ["c2"]},
         "judgments": {"c1": {"relevance": "relevant", "supports_options": ["B"]},
                       "c2": {"relevance": "relevant", "supports_options": ["B"]}}},
        {"correct_answer": "B", "candidates": {"dense__orig": ["c3"]},
         "judgments": {"c3": {"relevance": "irrelevant", "supports_options": []}}},
    ]
    summary = metrics.aggregate(rows, ["dense__orig", "dense__reform"], ks=(1,))
    assert summary["methods"]["dense__orig"]["n"] == 2
    assert summary["methods"]["dense__orig"]["per_k"][1]["semantic_recall"] == 0.5
    reform = summary["methods"]["dense__reform"]
    assert reform["n"] == 1, "a question the column never ran on must lower n, not the average"
    assert reform["per_k"][1]["semantic_recall"] == 1.0
    assert metrics.aggregate([{"error": "boom"}], ["dense__orig"], ks=(1,))["n_questions"] == 0


# --- rendering -----------------------------------------------------------------


def _sample_grid():
    return strategies.retrieve_grid(
        "A patient has a thing. What is it?", FakeRetriever(), FakeBM25(),
        reformulated_text="REWRITE the information need", top_k=4, rerank_k=2,
    )


def test_unjudged_chunk_is_unmeasured_never_irrelevant():
    grid = _sample_grid()
    judgments = {
        c.chunk_id: {"relevance": "relevant", "supports_options": ["B"], "reason": "r"}
        for c in grid.lists["dense__orig"][:2]
    }
    cells = render.build_cells(grid.lists, judgments, "B", ks=(2, 4))
    dense = next(cell for cell in cells if cell.method == "dense__orig")
    assert dense.unjudged == 2, "four retrieved, two judged"
    assert dense.chunks[0].judged and not dense.chunks[3].judged
    assert dense.chunks[0].gold_hit and dense.chunks[3].relevance == "?"
    assert dense.metrics["per_k"][4]["recall"] is True, "the judged part already hits gold"


def test_fallback_and_legacy_ids_render_as_caveats_not_blank_columns():
    grid = _sample_grid()
    judgments = {
        c.chunk_id: {"relevance": "relevant", "supports_options": ["B"], "reason": "r"}
        for c in grid.all_chunks().values()
    }
    view = render.build_question_view(
        question(), grid, judgments, split="train",
        reform={"query": "REWRITE the information need", "method": "manual", "fallback": False},
        ks=(2, 4),
    )
    html = render.render_html([view], meta={"judge_prompt_sha": "abc"})
    assert "abc" in html and "q-42" in html and "dense__reform" in html and "gold B" in html

    legacy = render.render_html([render.view_from_row(
        {"question_id": "42", "correct_answer": "B", "reformulation_fallback": True,
         "candidates": {"dense": ["c1"], "reform_dense": ["c1"]},
         "judgments": {"c1": {"relevance": "relevant", "supports_options": ["B"], "reason": "r"}}},
        question(), {}, ks=(1,),
    )])
    assert "FELL BACK" in legacy
    assert "legacy" in legacy, "`reform_dense` must not render as a clean grid cell"


def test_terminal_and_markdown_agree_with_the_cell_metrics():
    grid = _sample_grid()
    judgments = {
        c.chunk_id: {"relevance": "relevant", "supports_options": ["B"], "reason": "r"}
        for c in grid.all_chunks().values()
    }
    view = render.build_question_view(question(), grid, judgments, split="train", ks=(2, 4))
    dumped = render.render_terminal(view, max_chunks=2)
    assert "HIT" in dumped and "dense:orig" in dumped
    assert "HIT" in render.render_markdown(view).replace("✅", "HIT")


# --- chunk store ---------------------------------------------------------------


class FakeTable:
    """Stand-in for a LanceDB table: `search().where(...).to_list()`."""

    def __init__(self, rows):
        self._rows = rows
        self.queries = 0

    def search(self):
        table = self

        class Query:
            def __init__(self):
                self.clause = ""

            def where(self, clause):
                self.clause = clause
                return self

            def to_list(self):
                table.queries += 1
                inside = self.clause.split("(", 1)[1].rsplit(")", 1)[0]
                wanted = {part.strip().strip("'") for part in inside.split(",")}
                return [row for row in table._rows if row["chunk_id"] in wanted]

        return Query()


def test_chunk_sidecar_round_trip_append_and_damage(tmp_path):
    path = tmp_path / "chunks.jsonl"
    chunk_store.write_chunks(path, [chunk("c1", body="first")])
    chunk_store.append_chunks(path, [chunk("c2", body="second"), chunk("c1", body="revised")])
    found = chunk_store.read_chunks(path)
    assert set(found) == {"c1", "c2"} and found["c1"].content == "revised"

    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n{torn")
    assert set(chunk_store.read_chunks(path)) == {"c1", "c2"}, "one torn line is not an unreadable sidecar"
    assert chunk_store.read_chunks(tmp_path / "absent.jsonl") == {}


def test_resolve_chunks_prefers_the_sidecar_and_batches_the_index(tmp_path):
    sidecar = tmp_path / "run" / "chunks.jsonl"
    sidecar.parent.mkdir()
    chunk_store.write_chunks(sidecar, [chunk("c1", body="from sidecar")])
    table = FakeTable([
        {"chunk_id": "c2", "title": "t2", "content": "from index"},
        {"chunk_id": "c3", "title": "t3", "content": "from index"},
    ])

    found, source = chunk_store.resolve_chunks(["c1", "c2", "c3"], run_dir=sidecar.parent, table=table)
    assert found["c1"].content == "from sidecar" and found["c2"].content == "from index"
    assert table.queries == 1, "one filter query for the batch, not one per chunk"
    assert "sidecar" in source and "LanceDB" in source

    found, source = chunk_store.resolve_chunks(["c1", "c2"], run_dir=sidecar.parent, table=None)
    assert set(found) == {"c1"} and "no index handle" in source

    found, source = chunk_store.resolve_chunks(["c9"], run_dir=None, table=None)
    assert found == {} and "unresolved" in source


# --- the inspector's row shape ---------------------------------------------------


def test_inspection_rows_round_trip_through_the_harness_shape(tmp_path):
    # `inspect_retrieval` writes `results.jsonl` in the harness's shape so
    # `judge_cache --import-run` and grep keep working on an inspection dir; that
    # only holds if question/options survive the trip, which the first run lacked.
    from medical_rag.eval import inspector as inspect_retrieval

    grid = _sample_grid()
    judgments = {
        c.chunk_id: {"relevance": "relevant", "supports_options": ["B"], "reason": "r"}
        for c in grid.all_chunks().values()
    }
    view = render.build_question_view(
        question(), grid, judgments, split="train",
        reform={"query": "REWRITE the information need", "method": "manual", "fallback": False},
        ks=(2, 4),
    )
    row = inspect_retrieval.row_from_view(view)
    assert set(row["candidates"]) == set(strategies.METHOD_GRID)
    assert row["question"] == question().question and row["options"]["B"] == "beta"
    assert row["correct_answer"] == "B" and row["reformulation"]["query"].startswith("REWRITE")
    assert len(row["judgments"]) == len(grid.all_chunks())

    written = inspect_retrieval.write_artifacts(
        tmp_path / "inspect", [view], {"mode": "test"}, html=True
    )
    assert (tmp_path / "inspect" / "report.html").exists()
    assert (tmp_path / "inspect" / "questions" / "q42.md").exists()
    assert len(chunk_store.read_chunks(written["chunks"])) == len(grid.all_chunks())

    restored = json.loads((tmp_path / "inspect" / "results.jsonl").read_text().splitlines()[0])
    again = render.view_from_row(restored, question(), chunk_store.read_chunks(written["chunks"]), ks=(2, 4))
    assert [cell.method for cell in again.cells] == [cell.method for cell in view.cells]
    assert again.gold_letter == "B"

def _grid_row(**over) -> dict:
    """A row in the shape the *grid* harness stores: nested `reformulation` dict."""
    row = {
        "question_id": "42",
        "correct_answer": "B",
        "question": question().question,
        "options": question().options,
        "queries": {"orig": question().question, "reform": "REWRITE the information need"},
        "reformulation": {
            "query": "REWRITE the information need",
            "method": "agenerate + tolerant parse",
            "prompt_source": "production build_reformulation_prompt()",
            "fallback": False,
            "error": None,
        },
        "candidates": {
            "dense__orig": ["c1"],
            "dense__reform": ["c2"],
            "hybrid__orig": ["c1", "c2"],
            "hybrid__reform": ["c1", "c2"],
            "bm25__orig": ["c2"],
            "bm25__reform": ["c1"],
        },
        "judgments": {
            "c1": {"relevance": "relevant", "supports_options": ["B"], "reason": "r"},
            "c2": {"relevance": "irrelevant", "supports_options": [], "reason": "r"},
        },
    }
    row.update(over)
    return row


def test_a_grid_rows_nested_reform_record_is_not_read_as_no_reform():
    # The pre-grid run stored `reformulated_query`/`reformulation_fallback` as flat
    # fields; the grid harness stores the whole record under `reformulation`. Reading
    # only the flat pair made every grid row render as "Reformulation: off
    # (--no-reform)" -- an artifact claiming the lever was never pulled, on a run
    # whose own row says it was pulled and produced a different column.
    chunks = {c.chunk_id: c for c in (chunk("c1"), chunk("c2"))}
    view = render.view_from_row(_grid_row(), None, chunks, ks=(1,))
    assert view.reform["query"] == "REWRITE the information need"
    assert view.reform["fallback"] is False
    assert "hydration" not in view.notes, "the row carries its own question and options"

    markdown = render.render_markdown(view)
    assert "REWRITE the information need" in markdown
    assert "Reformulation off" not in markdown
    assert "Reformulation: off" not in render.render_html([view])


def test_a_fallback_marks_every_identical_reform_column_not_just_dense():
    # `--no-reform` off-states aside: a fallback means the rewrite equalled the
    # question, so an identical `__reform` column measured nothing. Marking only
    # `dense__reform` (the one the legacy run had) would present the other seven as
    # measured-and-matched, which is the opposite finding.
    chunks = {c.chunk_id: c for c in (chunk("c1"), chunk("c2"))}
    row = _grid_row(
        reformulation={"query": question().question, "fallback": True, "error": "unparseable"},
        candidates={
            "dense__orig": ["c1"],
            "dense__reform": ["c1"],
            "hybrid__orig": ["c1", "c2"],
            "hybrid__reform": ["c1", "c2"],
            "bm25__orig": ["c2"],
            "bm25__reform": ["c1"],
        },
    )
    view = render.view_from_row(row, None, chunks, ks=(2,))
    assert view.cell("dense__reform").reused and view.cell("hybrid__reform").reused
    assert not view.cell("bm25__reform").reused, "a differing column was really retrieved"
    assert not view.cell("dense__orig").reused, "orig is never the reused one"


def test_the_legacy_flat_reform_shape_still_renders_as_legacy():
    # `_legacy_row`'s two flat fields are the only record a pre-grid row has; the
    # page must still name the run as pre-grid rather than invent a method for it.
    view = render.view_from_row(
        {
            "question_id": "42",
            "correct_answer": "B",
            "reformulated_query": "rewritten need",
            "reformulation_fallback": True,
            "candidates": {"dense": ["c1"], "reform_dense": ["c1"]},
            "judgments": {
                "c1": {"relevance": "relevant", "supports_options": ["B"], "reason": "r"}
            },
        },
        None,
        {"c1": chunk("c1")},
        ks=(1,),
    )
    assert view.reform["method"] == "legacy run (pre-grid, pre-fix)"
    assert view.cell("dense__reform").reused
    assert "FELL BACK" in render.render_markdown(view)




# --- re-reading a run: legacy ids aggregate, and re-rendering costs nothing ----
#
# `judge_harness` is imported inside these tests on purpose: it pulls in the
# retriever (torch) through `raw_rag`, and the rest of this file avoids that. The
# `--rerender` path itself makes no calls, but importing the module to reach it is
# the price, so it is paid twice rather than at collection.


def _legacy_row(index: int) -> dict:
    """A row in the shape the 2026-09-29 run stored: six flat, peer method ids."""
    letter = "B" if index % 2 else "A"
    return {
        "question_id": str(index),
        "correct_answer": letter,
        "reformulated_query": f"rewritten need {index}",
        "reformulation_fallback": index % 2 == 0,
        "candidates": {
            "dense": [f"c{index}a", f"c{index}b"],
            "dense_rerank": [f"c{index}b"],
            "bm25": [f"b{index}a"],
            "hybrid": [f"c{index}a", f"b{index}a"],
            "hybrid_rerank": [f"c{index}b"],
            "reform_dense": [f"c{index}b"] if index % 2 else [f"c{index}a", f"c{index}b"],
        },
        "judgments": {
            f"c{index}a": {"relevance": "relevant", "supports_options": ["B"], "reason": "r"},
            f"c{index}b": {"relevance": "irrelevant", "supports_options": [], "reason": "r"},
            f"b{index}a": {"relevance": "partial", "supports_options": [], "reason": "r"},
        },
    }


def _rerender_args(run: Path, findings: Path):
    import argparse

    return argparse.Namespace(rerender=str(run), findings_path=str(findings))


def test_canonicalize_rows_maps_legacy_ids_and_keeps_the_original_row(tmp_path: Path) -> None:
    from medical_rag.eval import harness as judge_harness

    row = _legacy_row(1)
    row["candidates"]["paraphrase_rerank"] = ["zz"]  # unrecognised: dropped, never guessed
    [canonical] = judge_harness.canonicalize_rows([row])
    assert canonical["candidates"]["dense__orig"] == ["c1a", "c1b"]
    assert canonical["candidates"]["dense__reform"] == ["c1b"]
    assert canonical["dropped_method_ids"] == ["paraphrase_rerank"]
    assert canonical["legacy_method_ids"] == [
        "bm25->bm25__orig",
        "dense->dense__orig",
        "dense_rerank->dense_rerank__orig",
        "hybrid->hybrid__orig",
        "hybrid_rerank->hybrid_rerank__orig",
        "reform_dense->dense__reform",
    ]
    # Stored artifacts are never mutated by re-reading them.
    assert "reform_dense" in row["candidates"] and "dense__reform" not in row["candidates"]
    # And re-canonicalising grid rows is the identity -- `--rerender` must be idempotent.
    assert judge_harness.canonicalize_rows([canonical])[0] is canonical


def test_legacy_rows_produce_metrics_rather_than_empty_ones() -> None:
    from medical_rag.eval import harness as judge_harness

    summary = judge_harness.compute_metrics([_legacy_row(1), _legacy_row(2)])
    assert summary["methods"]["dense__orig"]["n"] == 2
    assert summary["methods"]["dense__reform"]["n"] == 2
    # Never run in that shape: unmeasured, not zero.
    assert summary["methods"]["bm25__reform"]["n"] == 0
    assert summary["methods"]["bm25__reform"]["per_k"][5]["semantic_recall"] is None


def test_rerender_reads_a_run_offline_and_labels_the_legacy_ids(tmp_path: Path) -> None:
    import asyncio

    from medical_rag.eval import harness as judge_harness

    run = tmp_path / "run"
    run.mkdir()
    with (run / "results.jsonl").open("w", encoding="utf-8") as handle:
        for index in (1, 2):
            handle.write(json.dumps(_legacy_row(index)) + "\n")
        handle.write("{ torn line\n")  # a killed run leaves one of these behind

    findings = tmp_path / "FINDINGS.md"
    assert asyncio.run(judge_harness.amain(_rerender_args(run, findings))) == 0
    text = findings.read_text(encoding="utf-8")
    assert "| dense__orig | 2 |" in text
    assert "Stored method ids" in text and "reform_dense->dense__reform" in text
    # 18/20 of the real run fell back; here, one of two rows did -- the caveat
    # must be the "read it with these excluded" one, not the "all copies" one.
    assert "1/2" in text and "Do not quote it." not in text
    assert "not recorded -- this run predates the verdict" in text


def test_rerender_stops_on_a_run_it_cannot_read(tmp_path: Path) -> None:
    import asyncio

    from medical_rag.eval import harness as judge_harness

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "results.jsonl").write_text("\n{ not json at all\n", encoding="utf-8")
    for run, expect in ((tmp_path / "missing", "no results.jsonl"), (empty, "no readable rows")):
        with pytest.raises(SystemExit) as excinfo:
            asyncio.run(judge_harness.amain(_rerender_args(run, tmp_path / "f.md")))
        assert expect in str(excinfo.value)


# --- the corpus ceiling probe --------------------------------------------------
#
# The judge harness grades chunks it chose itself, so it cannot separate "not
# retrieved" from "not in StatPearls" -- and `FINDINGS.md`'s 65% recall@20 means a
# third of the pinned sample sits in one of those two buckets with no way to say
# which. These tests pin the two things that make the probe trustworthy rather than
# merely impressive: it never lets a reader forget that its query contained the
# gold answer, and a literal miss comes back as "not stated in these words", never
# as "absent from the corpus".


class FakeCorpus:
    """The BM25 index's shape: parallel id/title/content lists, plus `search()`.

    `ceiling` reads `.contents` for the scan and `retrieve_base` calls `.search()`,
    so one small object stands in for the 467 MB pickle and the whole probe is
    testable without loading a model or touching LanceDB.
    """

    def __init__(self, bodies: dict[str, str]) -> None:
        self.chunk_ids = list(bodies)
        self.titles = [f"title of {cid}" for cid in bodies]
        self.contents = [bodies[cid] for cid in bodies]
        self._by_id = dict(bodies)
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return [chunk(cid, body=self._by_id[cid], score=float(k - i)) for i, cid in enumerate(self.chunk_ids[:k])]


def test_containment_scan_counts_every_match_but_shows_only_a_sample() -> None:
    index = FakeCorpus(
        {
            "a": "Chronic pancreatitis causes steatorrhea.",
            "b": "no mention here",
            "c": "chronic PANCREATITIS is the answer",  # case-insensitive on purpose
        }
    )
    scan = ceiling.containment_scan(index, "chronic pancreatitis", limit=10)
    assert scan.matches == 2, "the census is case-insensitive and covers every chunk, not the first page"
    assert len(scan.hits) == 2 and not scan.truncated
    assert scan.ids == {"a", "c"}

    scan = ceiling.containment_scan(index, "chronic pancreatitis", limit=1)
    assert (scan.matches, len(scan.hits), scan.truncated) == (2, 1, True), (
        "a 1-hit list must be labelled a sample -- otherwise 'showing 1' reads as 'only 1 exists'"
    )
    assert scan.ids == {"a", "c"}, "the census survives a truncated display"


def test_a_chunk_matching_twice_counts_once_in_the_census() -> None:
    # 278 chunks say "CT angiography"; two of them saying it five times each is not
    # 288. The number on the page is a count of chunks.
    scan = ceiling.containment_scan(FakeCorpus({"a": "beta ... beta ... beta"}), "beta")
    assert (scan.matches, len(scan.hits)) == (1, 1)


def test_best_rank_scores_against_every_match_not_just_the_displayed_sample() -> None:
    # Found by running the probe against the real corpus: "CT angiography" matches 278
    # chunks and the scan shows 20, so scoring `best_rank` against the sample reported
    # "the corpus states it, retrieval will not find it" for chunks retrieval had
    # surfaced. Here the sample is one hit and the chunk retrieval finds is a different
    # one -- the same bug at a size a test can assert on.
    bodies = {"x1": "alpha", "x2": "alpha", "m1": "beta", "m2": "beta", "m3": "beta"}
    corpus = FakeCorpus(bodies)

    class PointRetriever:
        def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
            return [chunk("m3"), chunk("x1")][:k]

        def rerank(self, query: str, chunks, k: int) -> list[RetrievedChunk]:
            return list(chunks)[:k]

    result = ceiling.probe("beta", PointRetriever(), corpus, top_k=2, rerank_k=1, limit=1)
    assert (result.matches, len(result.hits)) == (3, 1)
    assert result.best_rank["dense"] == 1, "m3 matched and ranked #1; the sample only showed m1"
    assert result.surfaced["dense"] == 1
    assert result.best_rank["bm25"] is None, "bm25 got x1/x2, neither of which states it"
    page = render.render_ceiling(result)
    assert "retrieval will not find it" not in page, (
        "that verdict must not fire when a strategy did surface a matching chunk"
    )


def test_containment_scan_refuses_a_needle_too_short_to_mean_anything() -> None:
    # `be` matches a large fraction of English medical prose; returning a count for
    # it would look like a finding about where an answer is stated.
    with pytest.raises(ValueError, match="at least"):
        ceiling.containment_scan(FakeCorpus({"a": "beta"}), "be")


def test_the_probe_labels_itself_oracle_everywhere_a_reader_might_stop() -> None:
    result = ceiling.probe(
        "beta", FakeRetriever(), FakeCorpus({"c1": "beta disease", "c2": "other"}),
        top_k=2, rerank_k=2,
    )
    assert "NOT a retrieval lever" in result.label
    assert result.to_dict()["oracle"] is True
    page = render.render_ceiling(result)
    assert "NOT a retrieval lever" in page, "the label must be on the page, not only in the data"
    assert result.query == "beta" and result.corpus_chunks == 2


def test_the_probe_names_the_case_where_no_query_side_lever_can_win() -> None:
    # `zz9` states the answer but is never returned by either retriever, which is
    # the finding that should end a reformulation effort rather than start one.
    corpus = FakeCorpus({"c1": "alpha", "c2": "gamma", "zz9": "beta is correct"})

    def never_returns_zz9(query: str, k: int) -> list[RetrievedChunk]:
        return [chunk("c1"), chunk("c2")][:k]

    retriever = FakeRetriever()
    retriever.retrieve = never_returns_zz9  # type: ignore[assignment]
    retriever.rerank = lambda q, chunks, k: list(chunks)[:k]  # type: ignore[assignment]
    result = ceiling.probe("beta", retriever, corpus, top_k=2, rerank_k=2)

    assert result.matches == 1
    assert set(result.best_rank) == set(result.lists)
    assert all(rank is None for rank in result.best_rank.values())
    page = render.render_ceiling(result)
    assert "retrieval will not find it" in page
    assert "reformulation" in page, "the page must say what this rules out, not just the number"


def test_a_literal_miss_is_reported_as_wording_not_absence() -> None:
    # Phase 8: most correct answers chose wording found nowhere in the excerpts, so
    # 0 literal matches cannot mean "StatPearls lacks it" or the probe would
    # over-claim and kill a phase that could still have won.
    result = ceiling.probe(
        "duct obstruction", FakeRetriever(), FakeCorpus({"c1": "beta disease"}), top_k=1, rerank_k=1
    )
    assert result.matches == 0
    assert "paraphras" in result.warning.lower()
    assert "not proof" in result.warning.lower()


def test_best_ranks_reports_absence_for_every_strategy_not_an_empty_row() -> None:
    ranks, counts = ceiling.best_ranks({"dense": [chunk("c1"), chunk("c2")]}, set())
    assert ranks == {"dense": None} and counts == {"dense": 0}
    ranks, counts = ceiling.best_ranks({"dense": [chunk("c1"), chunk("c2")]}, {"c2"})
    assert ranks == {"dense": 2} and counts == {"dense": 1}


# --- the served page and the static report share one renderer -------------------


def test_render_main_is_exactly_what_render_html_puts_inside_main() -> None:
    # The live page swaps this block, so if the two ever diverge the browser shows a
    # layout the tracked report does not have -- and the report is what FINDINGS.md
    # is written from.
    view = render.build_question_view(question(), _sample_grid(), {}, split="train", ks=(2, 4))
    body = render.render_main([view], meta={"top_k": "6"})
    page = render.render_html([view], meta={"top_k": "6"})
    start, end = page.index("<main id='main'>") + len("<main id='main'>"), page.index("</main>")
    assert page[start:end] == body


def test_a_written_report_still_needs_no_javascript_to_read() -> None:
    view = render.build_question_view(question(), _sample_grid(), {}, split="train", ks=(2, 4))
    static = render.render_html([view])
    assert "<script" not in static and "class='bar'" not in static
    live = render.render_html(
        [view],
        live=True,
        toolbar=render.render_toolbar({"retrieval_enabled": True, "judge_enabled": True}),
    )
    assert "<script" in live and "class='bar'" in live


def test_the_toolbar_shows_the_servers_knobs_and_disables_what_cannot_work() -> None:
    state = {
        "question_id": "42", "query": "REWRITE me", "top_k": 30, "rerank_k": 7,
        "bm25_top_k": 400, "rerank_with": "reform", "ks": "5 10 20",
        "retrieval_enabled": False, "judge_enabled": False, "unjudged": 12,
        "questions": ["42", "43"], "in_split": 10178, "probe_default": "beta", "cached": 99,
    }
    bar = render.render_toolbar(state)
    for expected in ("value='30'", "value='7'", "value='400'", "value='5 10 20'", "REWRITE me", "beta"):
        assert expected in bar, f"the toolbar must show the knobs actually in force, not defaults: {expected}"
    assert "2 suggested, 10178 in split" in bar, "the field must say it accepts more than it suggests"
    assert "value='reform' selected" in bar
    assert bar.count("disabled") >= 3, "no retriever means no Retrieve, no Apply and no Judge"
    assert "--no-retrieve" in bar, "a disabled control has to say why, or it reads as a missing feature"
    assert "Judge unseen (12)" in bar


# --- the served session: cost discipline and one view per question --------------


_CHUNK_ID_IN_PROMPT = re.compile(r"chunk_id=(\S+)")


class FakeJudgeClient:
    """Stand-in for `LLMClient` on the judge path; grades every id the prompt names.

    Reading ids out of the prompt rather than being handed them is deliberate:
    `ensure_verdicts` joins verdicts back by `chunk_id`, so a prompt that stopped
    printing them would silently leave every chunk unjudged. This fake makes that a
    test failure instead of a run of question marks.
    """

    def __init__(self, *, relevance: str = "relevant", supports: tuple[str, ...] = ("B",)) -> None:
        self.config = types.SimpleNamespace(name="fake_judge")
        self.calls = 0
        self._relevance = relevance
        self._supports = list(supports)

    async def agenerate_structured(self, prompt, output_type, *, system_prompt=None):
        self.calls += 1
        ids = _CHUNK_ID_IN_PROMPT.findall(prompt)
        if not ids:
            raise AssertionError("judge prompt names no chunk_id -- verdicts cannot be joined back")
        return judge_prompt.JudgeVerdict(
            judgments=[
                judge_prompt.ChunkJudgment(
                    chunk_id=cid,
                    relevance=self._relevance,
                    supports_options=self._supports,
                    reason=f"graded {cid}",
                )
                for cid in ids
            ]
        )


class FakeInspector:
    """Only the surface `serve.Backend` touches, with no models behind any of it.

    Everything the server would cost to test -- embedder, cross-encoder, the BM25
    pickle, a completion from the judge endpoint -- is the same FakeRetriever /
    FakeBM25 pair the grid tests use, so the loop bridge and all nine routes are
    exercisable offline.
    """

    def __init__(self, *, out_dir: Path, cache_path: Path, ks: tuple[int, ...] = (2, 4)) -> None:
        self.args = types.SimpleNamespace(judge_cache=cache_path, judge_batch=24)
        self.out_dir = out_dir
        self.split = "train"
        self.top_k, self.rerank_k, self.bm25_top_k = 6, 3, 6
        self.rerank_with = "question"
        self.ks = ks
        self.cache: dict[tuple[str, str], dict] = {}
        self.cache_counts = {"loaded": 0, "stale_prompt": 0, "malformed": 0}
        self.retriever, self.bm25 = FakeRetriever(), FakeBM25()
        self.judge: FakeJudgeClient | None = None
        self.judge_model = types.SimpleNamespace(name="fake_judge")
        self.breaker = None
        self.table = None
        self.judge_sha = judge_prompt.judge_prompt_sha()
        self.views: list[render.QuestionView] = []
        self.opened: dict[str, bool] = {}
        self.closed = False
        self.inspections = 0

    async def open(self, *, retrieval: bool, judge: bool) -> None:
        self.opened = {"retrieval": retrieval, "judge": judge}

    async def close(self) -> None:
        self.closed = True

    def meta(self, extra: dict | None = None) -> dict:
        return {"mode": "fake", **({"extra": extra} if extra else {})}

    async def inspect(self, question: MedQAQuestion, *, query_override: str | None = None):
        self.inspections += 1
        grid = strategies.retrieve_grid(
            question.question, self.retriever, self.bm25,
            reformulated_text=query_override, top_k=self.top_k, rerank_k=self.rerank_k,
            rerank_with=self.rerank_with,
        )
        judgments = {
            c.chunk_id: self.cache[(question.id, c.chunk_id)]
            for c in grid.all_chunks().values()
            if (question.id, c.chunk_id) in self.cache
        }
        view = render.build_question_view(question, grid, judgments, split=self.split, ks=self.ks)
        self.views.append(view)
        return view


def _backend(tmp_path: Path, *, judged: bool = False) -> serve.Backend:
    insp = FakeInspector(out_dir=tmp_path, cache_path=tmp_path / "cache.jsonl")
    if judged:
        insp.judge = FakeJudgeClient()
    backend = serve.Backend(insp, {question().id: question()})
    backend.start()
    return backend


def test_reinspecting_replaces_the_view_so_a_summary_counts_questions_not_clicks(tmp_path) -> None:
    # `Inspector.inspect` appends -- right for the REPL, wrong for a page you can
    # hit Refresh on. `aggregate_views` averages over the view list, so five clicks
    # would print n=5 and report one question's recall as five.
    backend = _backend(tmp_path)
    q = question()
    for _ in range(3):
        backend.submit(backend.inspect_route({"question_id": q.id}), timeout=10)
    assert len(backend.insp.views) == 1
    assert len(backend.session) == 1
    summary = render.aggregate_views(backend.insp.views)
    assert {entry["n"] for entry in summary.values()} == {1}


def test_judging_costs_no_retrieval_and_no_second_rewrite(tmp_path) -> None:
    # Re-inspecting to pick up fresh verdicts would re-run the reformulator: a paid
    # completion to re-derive a query we already hold, and a different string would
    # silently swap the column the reader is looking at.
    backend = _backend(tmp_path, judged=True)
    q = question()
    backend.submit(backend.inspect_route({"question_id": q.id}), timeout=10)
    retrieves = len(backend.insp.retriever.retrieve_calls)
    inspections = backend.insp.inspections

    html = backend.submit(backend.judge_route({}), timeout=30)
    assert backend.insp.judge.calls >= 1
    assert len(backend.insp.retriever.retrieve_calls) == retrieves, "judging must not re-retrieve"
    assert backend.insp.inspections == inspections, "judging must not re-run the reformulator"
    assert "REL" in html

    view = backend.session[q.id].view
    assert all(chunk.judged for cell in view.cells for chunk in cell.chunks)
    # And the verdicts are on disk, so the next session gets them for free.
    assert (tmp_path / "cache.jsonl").exists()


def test_the_judge_button_refuses_when_every_verdict_is_already_cached(tmp_path) -> None:
    # "Nothing to spend" is a real answer; silently re-grading is how a cached run
    # ends up paying for itself twice.
    backend = _backend(tmp_path, judged=True)
    q = question()
    backend.submit(backend.inspect_route({"question_id": q.id}), timeout=10)
    backend.submit(backend.judge_route({}), timeout=30)
    calls = backend.insp.judge.calls
    with pytest.raises(serve.HttpError) as excinfo:
        backend.submit(backend.judge_route({}), timeout=30)
    assert excinfo.value.status == 409
    assert backend.insp.judge.calls == calls


def test_a_question_id_outside_the_split_is_rejected_not_redrawn(tmp_path) -> None:
    backend = _backend(tmp_path)
    with pytest.raises(serve.HttpError) as excinfo:
        backend.submit(backend.inspect_route({"question_id": "not-a-question"}), timeout=10)
    assert excinfo.value.status == 400 and "not in split" in str(excinfo.value)


def test_the_id_field_suggests_the_sample_but_still_accepts_any_split_id(tmp_path) -> None:
    # The dev split is 10,178 questions. Suggesting all of them would put ~200 KB of
    # <option> markup on every page render; restricting the field to the pinned
    # sample would break the ad-hoc lookup that makes one odd question checkable. A
    # datalist suggests without restricting, so both are satisfiable -- this pins
    # that they stay that way.
    pool = {cid: question().model_copy(update={"id": cid}) for cid in ("1312", "7777")}
    insp = FakeInspector(out_dir=tmp_path, cache_path=tmp_path / "c.jsonl")
    backend = serve.Backend(insp, pool, ["1312", "9999-not-in-split"])
    backend.start()
    assert backend.browse == ["1312"], "a suggested id outside the split is dropped, not rendered"
    backend.submit(backend.inspect_route({"question_id": "7777"}), timeout=10)
    assert backend.current == "7777", "an id nobody suggested must still retrieve"


def test_knob_validation_names_the_constraint_it_hit(tmp_path) -> None:
    backend = _backend(tmp_path)
    backend.submit(backend.inspect_route({"question_id": question().id}), timeout=10)
    cases = {
        "top_k": ("0", "between 1 and 200"),
        "rerank_k": ("nonsense", "must be an integer"),
        "bm25_top_k": ("99999", "between 1 and 1000"),
    }
    for name, (value, expect) in cases.items():
        with pytest.raises(serve.HttpError) as excinfo:
            backend.submit(backend.knobs_route({name: value}), timeout=10)
        assert expect in str(excinfo.value), f"{name}={value}"

    # rerank_k above top_k passes both bounds and is still nonsense: the reranker
    # would be handed nothing to reorder, which looks like a lever that did nothing.
    with pytest.raises(serve.HttpError) as excinfo:
        backend.submit(backend.knobs_route({"top_k": "8", "rerank_k": "9"}), timeout=10)
    assert "nothing to reorder" in str(excinfo.value)
    assert backend.insp.top_k == 6, "a rejected request must not leave its knobs applied"

    # A metric past the retrieval depth is undefined, not zero.
    with pytest.raises(serve.HttpError) as excinfo:
        backend.submit(backend.knobs_route({"ks": "10 20"}), timeout=10)
    assert "exceeds top_k" in str(excinfo.value)

    # Applying a legal pair works and is reported.
    applied = backend.submit(backend.knobs_route({"top_k": "10", "rerank_k": "4"}), timeout=10)
    assert (backend.insp.top_k, backend.insp.rerank_k) == (10, 4)
    assert "top_k=10" in applied


def test_a_probe_only_session_still_persists_its_probes(tmp_path) -> None:
    # Found by clicking Save in the running app: it refused until a question had been
    # inspected, so a session spent probing the corpus -- a result in its own right,
    # and the only place a ceiling measurement is recorded -- could not be written
    # down at all.
    insp = FakeInspector(out_dir=tmp_path, cache_path=tmp_path / "cache.jsonl")
    insp.bm25 = FakeCorpus({"c1": "beta disease", "c2": "alpha"})
    backend = serve.Backend(insp, {})
    backend.start()

    with pytest.raises(serve.HttpError, match="nothing to write"):
        backend.submit(backend.report_route({}), timeout=30)

    backend.submit(backend.probe_route({"text": "beta disease"}), timeout=30)
    written = backend.submit(backend.report_route({}), timeout=30)
    assert "probes.jsonl" in written and (tmp_path / "probes.jsonl").exists()
    record = json.loads((tmp_path / "probes.jsonl").read_text(encoding="utf-8").strip())
    assert record["oracle"] is True and record["matches"] == 1, "the saved row keeps its own warning"


def test_reading_a_session_back_after_save_needs_no_endpoint(tmp_path) -> None:
    backend = _backend(tmp_path, judged=True)
    backend.submit(backend.inspect_route({"question_id": question().id}), timeout=10)
    backend.submit(backend.judge_route({}), timeout=30)
    written = backend.submit(backend.report_route({}), timeout=30)
    for name in ("report.html", "results.jsonl", "chunks.jsonl", "context.md"):
        assert name in written and (tmp_path / name).exists()
    sidecar = chunk_store.read_chunks(tmp_path / "chunks.jsonl")
    assert sidecar and all(record.content for record in sidecar.values())


def _request(port: int, method: str, path: str, body: dict | None = None) -> tuple[int, str]:
    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        payload = json.dumps(body) if body is not None else None
        connection.request(method, path, body=payload, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8")
    finally:
        connection.close()


def test_the_loop_bridge_serves_pages_errors_and_all_over_real_http(tmp_path) -> None:
    """The risky part, end to end: handler threads in, the Inspector's loop out.

    `LLMClient` binds to whichever loop it first runs on, so this whole shape exists
    to keep one loop. A unit test of `submit()` would not catch a route that blocks
    it, a response missing `Content-Length` under HTTP/1.1, or an error path that
    answers 200 with a traceback in the body -- and a 200-with-error is the one
    failure that leaves a stale grid on screen looking current.
    """
    insp = FakeInspector(out_dir=tmp_path, cache_path=tmp_path / "cache.jsonl")
    insp.judge = FakeJudgeClient()
    # The probe scans `.contents`, which the BM25 pickle has and FakeBM25 does not.
    insp.bm25 = FakeCorpus({"co1": "beta is the answer", "bo1": "alpha", "bo2": "gamma"})
    ports: list[int] = []
    thread = threading.Thread(
        target=serve.serve,
        args=(insp, {question().id: question()}),
        kwargs={"port": 0, "on_port": ports.append, "retrieval": True, "judge": True},
        daemon=True,  # dies with the test session; serve_forever has no other exit
    )
    thread.start()
    deadline = time.time() + 15
    while not ports and time.time() < deadline:
        time.sleep(0.05)
    assert ports, "the server never bound a port"
    port = ports[0]

    status, page = _request(port, "GET", "/")
    assert status == 200 and "<main id='main'>" in page
    assert "class='bar'" in page and "rag.inspect()" in page, "the page must actually be the live shell"
    assert insp.opened == {"retrieval": True, "judge": True}, "open() must run on the backend loop"

    status, state = _request(port, "GET", "/api/state")
    assert status == 200 and json.loads(state)["top_k"] == 6

    status, grid = _request(port, "POST", "/api/inspect", {"question_id": "42"})
    assert status == 200 and "q42" in grid

    status, error = _request(port, "POST", "/api/knobs", {"top_k": "abc"})
    assert status == 400 and "must be an integer" in error, "an invalid knob must not answer 200"

    status, error = _request(port, "POST", "/api/inspect", {"question_id": "999999"})
    assert status == 400 and "not in split" in error

    status, probed = _request(port, "POST", "/api/probe", {"text": "beta"})
    assert status == 200 and "NOT a retrieval lever" in probed

    status, error = _request(port, "GET", "/api/nonesuch")
    assert status == 404





