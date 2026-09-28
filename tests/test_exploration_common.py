"""Tests for scripts/exploration/_common.py -- the sample pinning and the
artifact writer that Phases 5-9 all log through.

These are the pieces an exploratory finding rests on, and both fail *silently*:
a sample-selection function that drifts across environments breaks the
"same 20 questions" claim that Phases 5 and 6 make of each other, and a writer
that drops rows breaks `--resume` in a way that is only discovered an hour into
a run. The rest of `scripts/` is plain CLI wiring and is exercised by hand.

Loaded by path rather than imported: `scripts/` is deliberately not an importable
package (see `_common.py`'s docstring -- nothing in `medical_rag` may depend on
it), so the test reaches in without teaching the package about it.
"""

import importlib.util
import random
import sys
from pathlib import Path

import pytest

from medical_rag.config import ModelConfig
from medical_rag.data.load_medqa import MedQAQuestion
from medical_rag.generation.llm import GenerationResult

_MODULE_NAME = "exploration_common"
_PATH = Path(__file__).resolve().parents[1] / "scripts" / "exploration" / "_common.py"


@pytest.fixture(scope="module")
def common():
    """Load `_common.py` once per module, by file path."""
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module's own imports resolve against sys.modules.
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _question(index: int, answer_idx: str = "A", letters: str = "ABCD") -> MedQAQuestion:
    options = {letter: f"option {letter} for question {index}" for letter in letters}
    return MedQAQuestion(
        id=str(index),
        question=f"Question {index}?",
        options=options,
        answer_idx=answer_idx,
        answer_text=options[answer_idx],
    )


def _model(**overrides) -> ModelConfig:
    fields = {
        "name": "model_a",
        "base_url": "http://127.0.0.1:8081/v1",
        "api_model_name": "Some-Served-Model",
        "api_key_env": "MODEL_A_API_KEY",
        "max_new_tokens": 1024,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    fields.update(overrides)
    return ModelConfig(**fields)


def _row(common, *, tag="main", question=None, content="ANSWER: A", model=None, **overrides):
    return common.result_row(
        model=model or _model(),
        question=question or _question(1, "A"),
        tag=tag,
        condition_id="phase5_closed_book",
        split="dev",
        prompt="a closed-book prompt",
        result=GenerationResult(
            content=content, finish_reason="stop", completion_tokens=50, elapsed_s=2.0
        ),
        overrides=overrides,
    )


def test_select_sample_is_stable_across_calls(common):
    questions = [_question(i) for i in range(100)]
    assert common.select_sample(questions, 20) == common.select_sample(questions, 20)


def test_select_sample_survives_reloading_in_a_different_order(common):
    """The subset is a function of `(sample_seed, question_id)`, not of load order.

    `random.Random(seed).sample()` fails this: it depends on the order the split
    happened to load in, so a `datasets` version bump that returns rows in a
    different order would silently change "the same 20 questions" that Phases 5
    and 6 claim to share.

    Growth is *not* covered, and deliberately not papered over: adding questions
    can put a new id below the cut, so a grown split can redraw borderline
    members. That is why `context.md` records the ids themselves -- a later phase
    reuses the pinned ids by name rather than re-deriving them.
    """
    questions = [_question(i) for i in range(100)]
    expected = {question.id for question in common.select_sample(questions, 20, sample_seed=7)}

    shuffled = questions[:]
    random.Random(0).shuffle(shuffled)
    assert {q.id for q in common.select_sample(shuffled, 20, sample_seed=7)} == expected

    subset_of_pool = [_question(i) for i in range(50, 100)] + [_question(i) for i in range(50)]
    assert {q.id for q in common.select_sample(subset_of_pool, 20, sample_seed=7)} == expected


def test_sample_seed_redraws_a_different_set(common):
    questions = [_question(i) for i in range(100)]
    first = {q.id for q in common.select_sample(questions, 20, sample_seed=1)}
    second = {q.id for q in common.select_sample(questions, 20, sample_seed=2)}
    assert first != second
    assert len(first & second) < 20


def test_sample_is_ordered_unique_and_sized(common):
    questions = [_question(i) for i in range(100)]
    sample = common.select_sample(questions, 20)
    assert len(sample) == 20
    assert len({q.id for q in sample}) == 20
    # Dataset order, so results.jsonl reads in ascending id rather than hash order.
    assert [q.id for q in sample] == sorted((q.id for q in sample), key=int)


def test_select_sample_rejects_impossible_sizes(common):
    questions = [_question(i) for i in range(5)]
    with pytest.raises(ValueError):
        common.select_sample(questions, 6)
    with pytest.raises(ValueError):
        common.select_sample(questions, 0)


def test_gold_distribution_exposes_a_skewed_draw(common):
    questions = [_question(i, "A") for i in range(3)] + [_question(9, "B")]
    assert dict(common.gold_distribution(questions)) == {"A": 3, "B": 1}


def test_run_writer_resumes_from_rows_it_flushed(common, tmp_path):
    """`--resume` is only honest if completed keys survive a new process."""
    writer = common.RunWriter("phase5", root=tmp_path)
    assert writer.completed_keys() == set()
    writer.write_row({"model": "model_a", "question_id": "3", "tag": "main"})
    writer.write_row({"model": "model_b", "question_id": "3", "tag": "cap4096"})
    writer.close()

    resumed = common.RunWriter("phase5", run_dir=writer.dir)
    assert resumed.dir == writer.dir
    assert resumed.completed_keys() == {
        ("model_a", "3", "main"),
        ("model_b", "3", "cap4096"),
    }
    assert len(resumed.load_rows()) == 2
    resumed.close()


def test_run_writer_tolerates_a_torn_last_line(common, tmp_path):
    """A kill can land mid-write; half a row must not cost the run the others."""
    writer = common.RunWriter("phase5", root=tmp_path)
    writer.write_row({"model": "model_a", "question_id": "1", "tag": "main"})
    with writer.results_path.open("a", encoding="utf-8") as handle:
        handle.write('{"model": "model_a", "question_')
    writer.close()

    reopened = common.RunWriter("phase5", run_dir=writer.dir)
    assert len(reopened.load_rows()) == 1
    reopened.close()


def test_two_runs_in_the_same_second_do_not_share_a_directory(common, tmp_path):
    """A re-run after a failure must not append into the failed run's rows."""
    first = common.RunWriter("phase5", root=tmp_path)
    second = common.RunWriter("phase5", root=tmp_path)
    assert first.dir != second.dir
    first.close()
    second.close()


def test_chains_are_kept_for_every_row_and_skipped_only_when_disabled(common, tmp_path):
    """Chains for correct rows too: failures-only retention hides the base rate."""
    writer = common.RunWriter("phase5", root=tmp_path)
    path = writer.write_chain("model_b", "7", "# gold=B answer=C finish=length", "long chain")
    assert path is not None and path.name == "model_b__7.md"
    text = path.read_text()
    assert text.startswith("# gold=B") and "long chain" in text
    assert writer.write_chain("model_b", "8", "# h", "   ") is None
    writer.close()

    silent = common.RunWriter("phase5", root=tmp_path, save_chains=False)
    assert silent.chains_dir is None
    assert silent.write_chain("model_b", "9", "# h", "body") is None
    assert not (silent.dir / "chains").exists()
    silent.close()


def test_row_rejects_a_letter_outside_the_questions_options(common):
    """An `E` on a four-option item is a format failure, not a wrong answer.

    `GenerationResult.parsed_answer` accepts ABCDE unconditionally, so it returns
    `"E"` here. Phase 12 has to tell `format_failure` from `reasoning_failure`,
    which is why the exploratory row recomputes against the question's own
    options; the gap is a Phase 9 input, and the assertion documents it.
    """
    question = _question(3, answer_idx="B")
    result = GenerationResult(
        content="ANSWER: E", finish_reason="stop", completion_tokens=40, elapsed_s=1.0
    )
    assert result.parsed_answer == "E"

    row = common.result_row(
        model=_model(),
        question=question,
        tag="main",
        condition_id="phase5_closed_book",
        split="dev",
        prompt="a prompt of some length",
        result=result,
    )
    assert row["parsed_answer"] is None
    assert row["unanswered"] is True
    assert row["is_correct"] is False
    assert row["correct_answer"] == "B"


def test_row_carries_its_own_params_and_the_endpoint_it_hit(common):
    row = _row(common, question=_question(4, "B"), content="ANSWER: B", max_new_tokens=512)
    assert row["is_correct"] is True and row["unanswered"] is False
    assert row["split"] == "dev" and row["question_id"] == "4"
    assert row["params"]["max_new_tokens"] == 512  # the override, not models.yaml's
    assert row["params"]["temperature"] == 0.0
    assert row["endpoint"] == "http://127.0.0.1:8081/v1"
    assert row["api_model"] == "Some-Served-Model"
    assert row["tok_s"] == 25.0  # 50 tokens / 2.0 s
    assert row["prompt_chars"] == len("a closed-book prompt")


def test_summarize_excludes_other_tags_by_default(common):
    """A caps-check row repeats a question under a different tag; folding it in would double-count."""
    rows = [
        _row(common),
        _row(common, question=_question(2, "B"), content="ANSWER: B"),
        _row(common, tag="cap4096", question=_question(1, "A"), content="ANSWER: C"),
    ]
    assert "n=2  correct=2 (100%)" in common.summarize(rows)
    assert "n=3" in common.summarize(rows, tag=None)


def test_letter_distribution_separates_unparsed_from_wrong(common):
    rows = [
        _row(common, question=_question(1, "A"), content="ANSWER: C"),
        _row(common, question=_question(2, "A"), content="I cannot determine"),
        _row(common, question=_question(3, "A"), content="ANSWER: A"),  # correct: not an error
    ]
    assert dict(common.letter_distribution(rows)) == {"C": 1, "unparsed": 1}


def test_error_rows_are_counted_out_of_accuracy_but_not_lost(common, tmp_path):
    writer = common.RunWriter("phase5", root=tmp_path)
    writer.write_row(_row(common))
    writer.write_row(
        common.error_row(
            model=_model(),
            question=_question(2, "A"),
            tag="main",
            condition_id="phase5_closed_book",
            split="dev",
            exc=RuntimeError("boom"),
            wall_s=3.0,
        )
    )
    rows = writer.load_rows()
    summary = common.summarize(rows)
    assert "n=1" in summary and "1 errored row(s): 2" in summary and "boom" in summary
    writer.close()


def test_percentile_of_nothing_is_not_zero(common):
    assert common.percentile([], 0.5) is None
    assert common.percentile([3, 1, 2], 0.5) == 2
