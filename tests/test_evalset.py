"""Tests for `medical_rag.eval.evalset` -- the stratified hand-check and snapshot.

Like the rest of the suite, these target the failure modes R2 is built to
prevent, not arithmetic:

* a rebuild with the same seed reproduces the committed review set
  record-for-record, and a different seed produces a different one
  (`build_hand_check` determinism) -- a sample a human graded must never
  silently change under them
* strata cannot be silently short: a cache that cannot fill S3 exits rather
  than shipping a table that reads as more evidence than it is
* the rendered page is blind -- judge verdicts and gold letters stay out of the
  CSV's human eye-line, and the gold letter sits in a spoiler in the markdown
* `score_hand_check` refuses a half-graded CSV and reports disagreement in both
  directions at the stratum level (G1 reads that table)
* the snapshot round-trips: export then check is OK; any cache drift, key or
  field, trips `check_snapshot`
* the CLI subcommands return process codes a shell script can gate on
"""

import csv
import json
from pathlib import Path

import pytest

from medical_rag.data.load_medqa import MedQAQuestion
from medical_rag.eval import evalset
from medical_rag.eval.rubric import judge_prompt_sha


def q(question_id: str, answer: str = "B") -> MedQAQuestion:
    return MedQAQuestion(
        id=question_id,
        question=f"vignette {question_id}: what is it?",
        options={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        answer_idx=answer,
        answer_text="beta",
        meta_info=None,
    )


def verdict(question_id: str, chunk_id: str, relevance: str = "relevant",
            supports: list[str] | None = None) -> dict:
    return {
        "question_id": question_id, "chunk_id": chunk_id,
        "relevance": relevance, "supports_options": supports or [],
        "reason": f"why {question_id} {chunk_id}",
        "judge_prompt_sha": judge_prompt_sha(),
        "source_run": "test",
        "model": "test-judge",
        "judged_at": "2026-10-11T00:00:00+00:00",
    }


class StubChunks:
    """resolve_chunk stub keyed by chunk_id; anything else is a test bug."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, chunk_id: str) -> tuple[str, str]:
        self.seen.append(chunk_id)
        return f"title of {chunk_id}", f"prose of {chunk_id}"


def seed_cache(path: Path, n_questions: int = 12) -> list[str]:
    """Write a full cache: every question carries a relevant, a partial, an
    irrelevant non-control chunk, plus all four control chunks (so S1's 8 and
    the S2/S3/S4 pools all fill), and a letter-discrimination pair."""
    from medical_rag.eval.judge import append_records

    question_ids = [f"q{i:03d}" for i in range(1, n_questions + 1)]
    records = []
    for qid in question_ids:
        records.append(verdict(qid, f"good_{qid}_1", "relevant", ["B"]))
        records.append(verdict(qid, f"topical_{qid}_2", "partial", []))
        records.append(verdict(qid, f"junk_{qid}_3", "irrelevant", []))
        for control in evalset.CONTROL_CHUNK_IDS:
            records.append(verdict(qid, control, "irrelevant", []))
    # one letter-discrimination pair: gold C, supports exclude C
    records.append(verdict(question_ids[0], f"hard_{question_ids[0]}_4", "relevant", ["A", "B"]))
    append_records(path, records)
    return question_ids


def question_pool(question_ids: list[str]) -> list[MedQAQuestion]:
    # gold C on even-numbered ids so the S5 stratum has a discriminating pair
    return [q(qid, "C" if int(qid[1:]) % 2 == 0 else "B") for qid in question_ids]


def build(cache: Path, questions: list[MedQAQuestion], tmp_path: Path,
          resolve_chunk=None, seed: int | None = None) -> list[evalset.HandCheckPair]:
    kwargs = {"seed": seed} if seed is not None else {}
    return evalset.build_hand_check(
        cache, questions=questions,
        resolve_chunk=resolve_chunk or StubChunks(), **kwargs,
    )

# --- build: determinism, quotas, refusal ----------------------------------------


def test_rebuild_is_record_for_record_and_seed_sensitive(tmp_path):
    ids = seed_cache(tmp_path / "cache.jsonl")
    questions = question_pool(ids)

    first = build(tmp_path / "cache.jsonl", questions, tmp_path)
    second = build(tmp_path / "cache.jsonl", questions, tmp_path)
    assert [(evalset._pair_id(p.question_id, p.chunk_id), p.stratum) for p in first] == [(evalset._pair_id(p.question_id, p.chunk_id), p.stratum) for p in second]

    third = build(tmp_path / "cache.jsonl", questions, tmp_path, seed=7)
    assert [(evalset._pair_id(p.question_id, p.chunk_id), p.stratum) for p in first] != [(evalset._pair_id(p.question_id, p.chunk_id), p.stratum) for p in third], \
        "a different seed must produce a different review set, or the sample is not a sample"


def test_strata_quotas_and_no_repeated_question_within_stratum(tmp_path):
    ids = seed_cache(tmp_path / "cache.jsonl")
    pairs = build(tmp_path / "cache.jsonl", question_pool(ids), tmp_path)

    counts = {s: sum(1 for p in pairs if p.stratum == s) for s in evalset.STRATUM_SIZES}
    assert counts == evalset.STRATUM_SIZES, "every stratum must hit its quota exactly"

    for stratum in evalset.STRATUM_SIZES:
        if stratum == "S1_control":
            continue  # controls take 2 records per chunk, so a question may repeat there
        qids = [p.question_id for p in pairs if p.stratum == stratum]
        assert len(qids) == len(set(qids)), f"{stratum} repeats a question"

    control_qids = {p.question_id for p in pairs if p.stratum == "S1_control"}
    assert len(control_qids) >= 5, "controls must spread over >=5 questions"
    assert {p.chunk_id for p in pairs if p.stratum == "S1_control"} == set(evalset.CONTROL_CHUNK_IDS)


def test_s5_is_letter_discrimination_only(tmp_path):
    ids = seed_cache(tmp_path / "cache.jsonl")
    pairs = build(tmp_path / "cache.jsonl", question_pool(ids), tmp_path)
    gold_by_id = {question.id: question.answer_idx for question in question_pool(ids)}
    for pair in pairs:
        if pair.stratum == "S5_letter":
            assert gold_by_id[pair.question_id] not in pair.judge_supports, \
                "S5 must pick pairs where the supports exclude the gold letter"


def test_short_stratum_exits_instead_of_shipping_short(tmp_path):
    ids = seed_cache(tmp_path / "cache.jsonl", n_questions=4)
    with pytest.raises(SystemExit, match="S1 controls"):
        build(tmp_path / "cache.jsonl", question_pool(ids), tmp_path)
    # a cache wide enough for S1 but with an emptied S3 pool
    ids = seed_cache(tmp_path / "cache.jsonl", n_questions=5)
    path = tmp_path / "cache2.jsonl"
    lines = [line for line in (tmp_path / "cache.jsonl").open(encoding="utf-8")
             if json.loads(line)["relevance"] != "partial"]
    path.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(SystemExit):  # either "pool too small" or the quota line fires
        build(path, question_pool(ids), tmp_path)


def test_empty_cache_exits_with_direction(tmp_path):
    with pytest.raises(SystemExit, match="no verdicts usable"):
        build(tmp_path / "cache.jsonl", question_pool(["q001"]), tmp_path)


def test_unknown_chunk_exits_via_resolver(tmp_path):
    ids = seed_cache(tmp_path / "cache.jsonl")

    def failing(chunk_id: str) -> tuple[str, str]:
        raise SystemExit(f"chunk {chunk_id} not found in the store")

    with pytest.raises(SystemExit, match="not found in the store"):
        build(tmp_path / "cache.jsonl", question_pool(ids), tmp_path, resolve_chunk=failing)


def test_control_chunk_missing_from_cache_exits(tmp_path):
    ids = seed_cache(tmp_path / "cache.jsonl", n_questions=12)
    path = tmp_path / "cache.jsonl"
    victim = evalset.CONTROL_CHUNK_IDS[0]
    lines = [line for line in path.open(encoding="utf-8")
             if json.loads(line)["chunk_id"] != victim]
    path.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(SystemExit, match=victim):
        build(path, question_pool(ids), tmp_path)


# --- render: blindness ----------------------------------------------------------


def test_render_is_blind_and_spoilered(tmp_path):
    ids = seed_cache(tmp_path / "cache.jsonl")
    pairs = build(tmp_path / "cache.jsonl", question_pool(ids), tmp_path)
    md_path, csv_path = evalset.render_hand_check(pairs, tmp_path / "out")

    markdown = md_path.read_text(encoding="utf-8")
    assert "||" in markdown, "gold letters are spoilered in the markdown"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == len(pairs)
    assert all(not r["human_relevance"] and not r["human_supports"] and not r["human_note"]
               for r in rows), "human columns ship empty -- that is what grading fills"
    assert all(r["judge_relevance"] and r["judge_reason"] for r in rows)
    by_id = {r["pair_id"]: r for r in rows}
    for pair in pairs:
        row = by_id[evalset._pair_id(pair.question_id, pair.chunk_id)]
        assert row["judge_relevance"] == pair.judge_relevance
        assert row["gold_letter"] == pair.question.answer_idx.upper()


def test_render_round_trip_through_score(tmp_path):
    ids = seed_cache(tmp_path / "cache.jsonl")
    pairs = build(tmp_path / "cache.jsonl", question_pool(ids), tmp_path)
    _md, csv_path = evalset.render_hand_check(pairs, tmp_path / "out")

    # grade it: agree everywhere
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    for row in rows:
        row["human_relevance"] = row["judge_relevance"]
        row["human_supports"] = row["judge_supports"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = evalset.score_hand_check(csv_path)
    assert "G1" in report
    assert "YES -- G1 control condition met" in report

# --- score: validation and the table --------------------------------------------


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_score_refuses_half_graded_and_bad_vocabulary(tmp_path):
    ids = seed_cache(tmp_path / "cache.jsonl")
    pairs = build(tmp_path / "cache.jsonl", question_pool(ids), tmp_path)
    _md, csv_path = evalset.render_hand_check(pairs, tmp_path / "out")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    fields = list(rows[0])

    with pytest.raises(SystemExit, match="not gradeable"):
        evalset.score_hand_check(csv_path), "ungraded CSV must not produce a table"

    rows[3]["human_relevance"] = "maybe"
    _write_csv(csv_path, rows, fields)
    with pytest.raises(SystemExit, match="human_relevance"):
        evalset.score_hand_check(csv_path)

    rows[3]["human_relevance"] = "relevant"
    rows[3]["human_supports"] = "Z"
    _write_csv(csv_path, rows, fields)
    with pytest.raises(SystemExit, match="human_supports"):
        evalset.score_hand_check(csv_path)


def test_score_reports_disagreement_in_both_directions(tmp_path):
    ids = seed_cache(tmp_path / "cache.jsonl")
    pairs = build(tmp_path / "cache.jsonl", question_pool(ids), tmp_path)
    _md, csv_path = evalset.render_hand_check(pairs, tmp_path / "out")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    fields = list(rows[0])

    # 2 lenient pairs: judge irrelevant, human relevant -> judge-over-human
    lenient_ids = {r["pair_id"] for r in rows if r["judge_relevance"] == "irrelevant"}
    lenient_ids = set(sorted(lenient_ids)[:2])
    for row in rows:
        row["human_relevance"] = ("relevant" if row["pair_id"] in lenient_ids
                                  else row["judge_relevance"])
        row["human_supports"] = row["judge_supports"]
    _write_csv(csv_path, rows, fields)
    report = evalset.score_hand_check(csv_path)
    assert "judge over human" in report
    assert "human over judge" in report, "the table must name both failure directions"


def test_score_gate_line_fires_on_control_disagreement(tmp_path):
    ids = seed_cache(tmp_path / "cache.jsonl")
    pairs = build(tmp_path / "cache.jsonl", question_pool(ids), tmp_path)
    _md, csv_path = evalset.render_hand_check(pairs, tmp_path / "out")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    fields = list(rows[0])
    for row in rows:
        row["human_relevance"] = row["judge_relevance"]
        row["human_supports"] = row["judge_supports"]
    control = next(r for r in rows if r["stratum"] == "S1_control")
    control["human_relevance"] = "relevant"  # human disagrees with the known-bad call
    _write_csv(csv_path, rows, fields)
    report = evalset.score_hand_check(csv_path)
    assert "NO -- G1 FAILS on controls" in report
    assert "G1 control condition met" not in report
# --- snapshot -------------------------------------------------------------------


def test_snapshot_round_trip_and_drift(tmp_path, monkeypatch):
    from medical_rag.eval.judge import append_records

    cache = tmp_path / "cache.jsonl"
    seed_cache(cache, n_questions=12)
    out = tmp_path / "snap"
    monkeypatch.setattr(evalset, "DEFAULT_JUDGE_CACHE", cache)
    monkeypatch.setattr(evalset, "SNAPSHOT_SAMPLE_FILE", tmp_path / "no_sample.json")
    monkeypatch.setattr(evalset, "_latest_run_context", lambda: (None, None))
    monkeypatch.setattr(evalset, "load_config", lambda _p: type(
        "Cfg", (), {
            "models": {"judge_model": type("M", (), {
                "name": "judge_model", "api_model_name": "test-model",
                "base_url": "http://x", "temperature": 1.0, "top_p": 0.95,
                "max_new_tokens": 64})()},
            "dev_split": "train",
            "retrieval": type("R", (), {
                "corpus": "statpearls", "embedding_model": "embed",
                "reranker_model": "rerank", "index_dir": str(tmp_path)})(),
        })())
    monkeypatch.setattr(evalset, "resolve_index_dir", lambda config: tmp_path)
    monkeypatch.setattr(evalset, "_index_sha", lambda index_dir: "deadbeef")

    paths = evalset.export_snapshot(out, cache_path=cache)
    assert len(paths) == 2
    sha = judge_prompt_sha()
    lines = [json.loads(line) for line in (out / f"verdicts_{sha}.jsonl").open(encoding="utf-8")]
    assert len(lines) == 12 * 7 + 1
    assert all("prose" not in json.dumps(line) for line in lines), \
        "the snapshot must not embed chunk prose -- the corpus is the prose-of-record"
    metadata = json.loads((out / "snapshot_metadata.json").read_text(encoding="utf-8"))
    assert metadata["judge_model"]["api_model_name"] == "test-model"
    assert metadata["corpus"]["index_sha256"] == "deadbeef"
    assert metadata["sample"]["path"] is None, "a missing sample file is a note, not a crash"

    assert evalset.check_snapshot(out) is True

    # drift: a new verdict lands in the cache -> check must trip
    append_records(cache, [verdict("q001", "fresh_chunk", "relevant", ["B"])])
    assert evalset.check_snapshot(out) is False

    # drift: an exported verdict's field is edited in the cache -> check must trip
    records, _ = evalset.load_cache(cache)
    victim_key = next(k for k in records if k[1].startswith("good_"))
    records[victim_key]["relevance"] = "partial"
    cache.write_text("".join(json.dumps(r) + "\n" for r in records.values()), encoding="utf-8")
    assert evalset.check_snapshot(out) is False, "field-level drift must trip, not just key drift"

    # restore and re-export, then a truncated trailing line (interrupted write)
    # must trip cleanly -- not crash with a traceback
    records[victim_key]["relevance"] = "relevant"
    cache.write_text("".join(json.dumps(r) + "\n" for r in records.values()), encoding="utf-8")
    paths = evalset.export_snapshot(out, cache_path=cache)
    assert evalset.check_snapshot(out) is True
    with (out / f"verdicts_{sha}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"question_id": "q001", "chunk_id": "cut')
    assert evalset.check_snapshot(out) is False, "a truncated snapshot must trip, not crash"


def test_server_sampling_is_pasted_verbatim(tmp_path, monkeypatch):
    settings = {"models": {"test-model": {"temperature": 0.7, "force_sampling": True}}}
    path = tmp_path / "model_settings.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setattr(evalset, "OMLX_MODEL_SETTINGS", path)
    assert evalset._server_sampling("test-model") == {"temperature": 0.7, "force_sampling": True}

    monkeypatch.setattr(evalset, "OMLX_MODEL_SETTINGS", tmp_path / "missing.json")
    assert "_missing" in evalset._server_sampling("test-model")

    monkeypatch.setattr(evalset, "OMLX_MODEL_SETTINGS", path)  # file back, wrong model key
    assert "_note" in evalset._server_sampling("other-model"), \
        "an unknown judge model must be a note, not a silent {}"
# --- CLI ------------------------------------------------------------------------


def test_cli_build_and_score_exit_codes(tmp_path, monkeypatch):
    ids = seed_cache(tmp_path / "cache.jsonl")
    questions = question_pool(ids)
    monkeypatch.setattr(evalset, "load_medqa", lambda _split: questions)
    monkeypatch.setattr(evalset, "HAND_CHECK_DIR", tmp_path / "r2")

    assert evalset.main(["--cache", str(tmp_path / "cache.jsonl"),
                         "--hand-check-dir", str(tmp_path / "r2"),
                         "build"], resolve_chunk=StubChunks()) == 0
    assert (tmp_path / "r2" / "hand_check.md").exists()

    with pytest.raises(SystemExit):
        evalset.main(["score"]), "ungraded csv exits"

    csv_path = tmp_path / "r2" / "hand_check_records.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    for row in rows:
        row["human_relevance"] = row["judge_relevance"]
        row["human_supports"] = row["judge_supports"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert evalset.main(["score"]) == 0

    assert evalset.main([]) == 2, "no subcommand prints help and exits 2"