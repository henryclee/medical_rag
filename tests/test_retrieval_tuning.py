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

The experiment modules import each other by bare name (`from strategies import
...`), the way `scripts/exploration/` does, so the directory goes on `sys.path`
rather than each module being loaded by path with `importlib` -- they are not
packages, and they are not meant to be importable from `src/`.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_EXP = _REPO / "experiments" / "retrieval_tuning"
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

import chunk_store  # noqa: E402
import judge_cache  # noqa: E402
import judge_prompt  # noqa: E402
import metrics  # noqa: E402
import reformulate  # noqa: E402
import render  # noqa: E402
import strategies  # noqa: E402
from medical_rag.data.load_medqa import MedQAQuestion  # noqa: E402
from medical_rag.retrieval.retriever import RetrievedChunk  # noqa: E402


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


def test_judge_rubric_matches_the_harness_so_cached_verdicts_stay_valid():
    # The 2026-09-29 run paid ~820 verdicts with the harness's wording. If that
    # wording had moved into `judge_prompt.py` reworded, every cached verdict would
    # silently belong to a rubric nobody chose.
    import judge_harness

    assert judge_harness._JUDGE_SYSTEM_PROMPT == judge_prompt.JUDGE_SYSTEM_PROMPT
    assert judge_harness._JUDGE_JSON_INSTRUCTION == judge_prompt.JUDGE_JSON_INSTRUCTION
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
        from judge_prompt import ChunkJudgment

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
    import inspect_retrieval

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
    import judge_harness

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
    import judge_harness

    summary = judge_harness.compute_metrics([_legacy_row(1), _legacy_row(2)])
    assert summary["methods"]["dense__orig"]["n"] == 2
    assert summary["methods"]["dense__reform"]["n"] == 2
    # Never run in that shape: unmeasured, not zero.
    assert summary["methods"]["bm25__reform"]["n"] == 0
    assert summary["methods"]["bm25__reform"]["per_k"][5]["semantic_recall"] is None


def test_rerender_reads_a_run_offline_and_labels_the_legacy_ids(tmp_path: Path) -> None:
    import asyncio

    import judge_harness

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

    import judge_harness

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "results.jsonl").write_text("\n{ not json at all\n", encoding="utf-8")
    for run, expect in ((tmp_path / "missing", "no results.jsonl"), (empty, "no readable rows")):
        with pytest.raises(SystemExit) as excinfo:
            asyncio.run(judge_harness.amain(_rerender_args(run, tmp_path / "f.md")))
        assert expect in str(excinfo.value)





