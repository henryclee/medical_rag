"""Shared sample pinning and artifact writing for the exploratory phases.

Not a module of the package: `scripts/` is not importable, and this is
exploration plumbing that the study's public surface must not depend on. It is
imported by the sibling scripts in this directory (Python puts this directory
on `sys.path` when one of them is run), and it is promoted into
`experiment/runner.py` at Phase 9 if it earns it, so the study does not end up
with two logging systems.

PLAN.md's exploratory-artifact convention is the reason this file exists. An
exploratory phase is not finished when its check passes; it is finished when its
evidence is reviewable months later, at the Phase 10 freeze, by someone asking
"why did `model_b` get this one wrong?" with a `models.yaml` that has already
been edited since. Three things follow, and they live here rather than in each
script so that Phase 6 cannot drift from Phase 5:

1. **A pinned sample.** `select_sample()` ranks by a hash of
   `(sample_seed, question_id)` instead of `random.Random(seed).sample()`, which
   depends on the order the split happened to load in and has changed internals
   across CPython versions. Hash-ranking is unaffected by both, but it is *not*
   immune to the split growing -- a new question can rank inside the cut -- so
   the chosen ids are written to `context.md` and reused by name downstream.
2. **A raw run directory** under `outputs/exploration/<phase>/<stamp>/` --
   `results.jsonl`, `chains/<model>__<question_id>.md`, `context.md`. Rows are
   flushed the moment they are written, so a run interrupted 25 minutes in keeps
   its evidence instead of losing it, and `completed_keys()` makes `--resume`
   mean "do not re-fire those requests".
3. **Self-describing rows.** Each row keeps `QuestionResult`'s field names (see
   `interfaces.md`) so Phase 12's `classify_failure()` can read exploratory data
   without a translation layer, and snapshots the `params` active for it, so the
   log stays honest after `conditions.yaml`/`models.yaml` move on (open
   question 9).

Every row also carries `split`, because `MedQAQuestion.id` is the positional
index *within* a split -- dev-split question `"7"` and test-split question `"7"`
are different questions, and a join that ignores `split` is silently wrong.
"""

import hashlib
import json
import platform
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import openai
from loguru import logger

from medical_rag.config import ExperimentConfig, ModelConfig
from medical_rag.data.load_medqa import DATASET_ID, MedQAQuestion
from medical_rag.generation.llm import GenerationResult
from medical_rag.generation.prompt import parse_answer

# Pinned so a re-run reproduces a previous phase's sample by default. The value
# is arbitrary; that it is recorded in context.md and FINDINGS.md is not.
DEFAULT_SAMPLE_SEED = 1

SELECTION_METHOD = "sha256(seed:question_id) ascending, first k"


def _sort_key(question: MedQAQuestion) -> tuple[int, int, str]:
    """Order questions by dataset position, tolerating non-numeric ids."""
    return (0, int(question.id), "") if question.id.isdigit() else (1, 0, question.id)


def select_sample(
    questions: Sequence[MedQAQuestion], sample_size: int, sample_seed: int = DEFAULT_SAMPLE_SEED
) -> list[MedQAQuestion]:
    """Choose `sample_size` questions deterministically for this `sample_seed`.

    The ranking depends only on `(sample_seed, question_id)`, so the draw is
    unaffected by load order and by interpreter version -- which is what lets
    Phase 6 honestly say "the same 20 questions". A grown pool can still redraw
    borderline members, so callers record the ids they got. Results come back in
    dataset order so rows read naturally.
    """
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    if sample_size > len(questions):
        raise ValueError(
            f"sample_size {sample_size} exceeds the {len(questions)} questions in the split"
        )

    ranked = sorted(
        questions,
        key=lambda q: (hashlib.sha256(f"{sample_seed}:{q.id}".encode()).hexdigest(), q.id),
    )
    return sorted(ranked[:sample_size], key=_sort_key)


def select_by_ids(
    questions: Sequence[MedQAQuestion], ids: Iterable[str]
) -> list[MedQAQuestion]:
    """Fetch specific questions by `question_id`, in the order requested.

    Phase 6 asks "what changes when the same 20 questions get retrieved
    context?", which only means something if they are the *same* questions.
    Re-deriving them with `select_sample()` is not that: its own docstring
    concedes a grown pool redraws borderline members, and Phase 5 §8.4 watched
    an endpoint swap replace 5 of 20 answers with nothing edited in this repo.
    So the ids are the contract, they are looked up by name, and the re-derivation
    is kept only as a check that prints -- see `raw_rag.py --dry-run`.

    Every missing id is reported at once. Raising on the first one, or silently
    returning the ones that did resolve, would shrink every paired table without
    saying so -- the exact quiet-partial failure this function exists to prevent.
    """
    requested = [str(question_id) for question_id in ids]
    by_id = {question.id: question for question in questions}
    missing = [question_id for question_id in requested if question_id not in by_id]
    if missing:
        raise KeyError(
            f"{len(missing)} of {len(requested)} pinned question_id(s) are not in the split: "
            f"{', '.join(missing)}. The split changed since these were pinned; do not "
            f"re-draw silently -- re-pin the sample and record the re-pin in FINDINGS.md."
        )
    return [by_id[question_id] for question_id in requested]


def gold_distribution(questions: Iterable[MedQAQuestion]) -> Counter:
    """Gold-letter counts of a sample, so bias in the *sample* is visible."""
    return Counter(question.answer_idx.upper() for question in questions)


def warn_on_degenerate_sample(questions: Sequence[MedQAQuestion]) -> Counter:
    """Flag a sample whose gold letters are too skewed to support bias analysis.

    Phase 5 tabulates the option-letter distribution of each model's *errors*
    precisely to catch a model that favours one letter. That reading is
    worthless if the sample itself is half one letter, so the skew is reported
    before any request is fired rather than discovered in the findings.
    """
    counts = gold_distribution(questions)
    total = sum(counts.values())
    if not total:
        return counts

    letter, largest = counts.most_common(1)[0]
    if largest / total > 0.5:
        logger.warning(
            "sample gold distribution is skewed: {} is {}/{} of it ({}) -- "
            "error-letter bias will be hard to read; re-draw with a different "
            "--sample-seed if this matters for the phase",
            letter,
            largest,
            total,
            dict(sorted(counts.items())),
        )
    return counts


def git_rev() -> tuple[str, bool]:
    """Short HEAD revision and whether the working tree was dirty.

    Recorded so a finding can be attributed to exactly this code. `dirty=True`
    matters: an exploratory result produced by uncommitted changes is a weaker
    claim, and the alternative -- refusing to run -- would block legitimate
    throwaway experiments.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout
        return head, bool(porcelain.strip())
    except Exception as exc:  # not a git checkout, or git missing
        logger.warning("could not read git revision ({}); recording rev='unknown'", exc)
        return "unknown", False


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _unique_dir(path: Path) -> Path:
    """Disambiguate a run directory that a previous run already claimed.

    The stamp only has second resolution, so a smoke test re-run immediately
    after a failure -- the most likely thing to happen in this directory --
    would otherwise append its rows into the failed run's `results.jsonl` and
    hand `--resume` a directory describing two runs. Cheap to prevent, and the
    alternative (refusing to start) is worse than a `-1` suffix.
    """
    candidate, suffix = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}-{suffix}")
        suffix += 1
    return candidate


DEFAULT_RUN_ROOT = "outputs/exploration"


class RunWriter:
    """One run's raw artifact directory, written incrementally.

    `outputs/exploration/<phase>/<UTC-stamp>/` is gitignored and regenerable;
    `experiments/<phase>/FINDINGS.md` is what a human keeps. The writer only
    ever touches the former, and never `outputs/runs/` -- exploratory rows must
    not be mistaken for pilot or confirmatory results (PLAN.md's guardrail).
    """

    def __init__(
        self,
        phase: str,
        *,
        root: str | Path | None = DEFAULT_RUN_ROOT,
        run_dir: str | Path | None = None,
        save_chains: bool = True,
    ) -> None:
        # run_dir reopens an existing directory, which is what makes --resume
        # re-attach to a half-finished run instead of starting a second one.
        # `root=None` falls back to the exploration root rather than crashing: a
        # script that registers its own `--output-dir` with an unset default
        # means "default", and this is the one place that decides what that is.
        base = DEFAULT_RUN_ROOT if root is None else root
        self.dir = Path(run_dir) if run_dir else _unique_dir(Path(base) / phase / _utc_stamp())
        self.dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.dir / "results.jsonl"
        self.save_chains = save_chains
        self.chains_dir = None if not save_chains else self.dir / "chains"
        if self.chains_dir is not None:
            self.chains_dir.mkdir(parents=True, exist_ok=True)
        self._handle = None

    @property
    def stamp(self) -> str:
        return self.dir.name

    def _append(self, text: str) -> None:
        if self._handle is None:
            self._handle = self.results_path.open("a", encoding="utf-8")
        self._handle.write(text)
        # Flush, not close-and-reopen: the point is that a run killed 25 minutes
        # in has already handed its rows to the OS, so --resume can pick up where
        # it stopped instead of re-billing the same completions.
        self._handle.flush()

    def write_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Append one row and echo it, so stdout and results.jsonl agree."""
        self._append(json.dumps(row, ensure_ascii=False) + "\n")
        print("RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
        return row

    def completed_keys(self) -> set[tuple[str, str, str]]:
        """`(model, question_id, tag)` triples that earned a completion.

        Error rows are deliberately NOT included. A row written by `error_row()`
        records that a call was *attempted and failed* (endpoint down, HTTP 5xx),
        not that the question was answered; counting it as done made `--resume`
        skip the exact questions that never ran. That trap cost the first Phase 5
        run 17 silent no-ops (`experiments/phase5/FINDINGS.md` §7), and Phase 9's
        resumability claim would have inherited it. A row that legitimately came
        back unanswered-but-answered (no parsable ANSWER:) *does* count: it is a
        measured outcome, not a lost request.
        """
        return {(row.get("model", "?"), row.get("question_id", "?"), row.get("tag", "?"))
                for row in self.load_rows() if "error" not in row}

    def load_rows(self) -> list[dict[str, Any]]:
        """Every row on disk, so a resumed run summarizes the whole run.

        Tolerates a torn last line: the writer flushes per row, but a kill can
        still land mid-write, and one half-row must not lose the run's other 39.
        """
        rows: list[dict[str, Any]] = []
        if not self.results_path.exists():
            return rows
        with self.results_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("skipping unparseable line in {}", self.results_path)
        return rows

    def write_chain(self, model: ModelConfig | str, tag: str, header: str, body: str) -> Path | None:
        """Persist a chain of thought -- where the failure mode actually lives.

        Kept for every row, not just wrong ones: failures-only retention hides
        the base rate needed to interpret the failures.
        """
        if self.chains_dir is None or not body.strip():
            return None
        name = model.name if isinstance(model, ModelConfig) else str(model)
        path = self.chains_dir / f"{name}__{tag}.md"
        path.write_text(f"{header.rstrip()}\n\n{body.strip()}\n", encoding="utf-8")
        return path

    def write_context(
        self,
        *,
        config: ExperimentConfig,
        model_keys: Sequence[str],
        questions: Sequence[MedQAQuestion],
        split: str,
        sample_seed: int,
        argv: Sequence[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Write `context.md`: the resolved state that produced these rows.

        The one file that lets a finding be re-derived later. It records the
        resolved per-model parameters rather than a path to `models.yaml`,
        because that file will be edited -- and at the Phase 10 freeze the
        question is what the run actually used, not what the config says now.
        No secrets land here: `api_key_env` is the name of an environment
        variable, and its value is never read into config.
        """
        rev, dirty = git_rev()
        ids = [question.id for question in questions]
        argv_text = " ".join([Path(sys.argv[0]).name, *(argv or sys.argv[1:])])

        lines = [
            f"# Exploration run context -- {self.dir.parent.name}/{self.stamp}",
            "",
            f"- **Started (UTC):** {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"- **git rev:** `{rev}`{' (dirty working tree -- uncommitted changes were live)' if dirty else ''}",
            f"- **Command:** `{argv_text}`",
            f"- **Python:** {platform.python_version()} on {platform.machine()} {platform.system()}",
            f"- **Dataset:** `{DATASET_ID}` split `{split}` ({len(questions)} sampled; "
            "exploratory phases read `config.dev_split` only)",
            f"- **Sample:** size={len(ids)}, sample_seed={sample_seed}, selection={SELECTION_METHOD}",
            f"- **Artifacts:** `{self.dir}` (`results.jsonl`, "
            + ("`chains/`)" if self.chains_dir else "chains disabled)"),
            "",
            "## Sampled question_ids (pinned -- later phases reuse exactly these)",
            "",
            "```",
            ",".join(ids),
            "```",
            "",
            "## Resolved parameters per model arm",
            "",
        ]
        for key in model_keys:
            model = config.models[key]
            lines.append(f"### `{key}` -> `{model.api_model_name}` @ `{model.base_url}`")
            lines.append("")
            for field, value in model.model_dump().items():
                if field in {"name", "api_model_name", "base_url", "api_key_env"}:
                    continue
                lines.append(f"- `{field}`: {value}")
            lines.append(f"- `api_key_env`: `{model.api_key_env}` (name only; value never logged)")
            lines.append("")

        if extra:
            lines += ["## Run-specific notes", ""]
            lines += [f"- **{key}**: {value}" for key, value in extra.items()]
            lines.append("")

        path = self.dir / "context.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def answer_letter(result: GenerationResult, question: MedQAQuestion) -> str | None:
    """The scored letter, restricted to this question's real options.

    Deliberately not `result.parsed_answer`, which defaults to accepting A-E: on
    an A-D question an answer of `E` must count as *unparseable*, not as a wrong
    answer, or the format-failure rate gets miscounted as a reasoning failure
    (Phase 12's `format_failure` vs. `reasoning_failure`). The gap between this
    and `GenerationResult.parsed_answer` is recorded in Phase 5's FINDINGS as an
    input to Phase 9's `QuestionResult` decision; production code is untouched
    by exploratory phases.
    """
    return parse_answer(result.content, valid_letters=set(question.options))


def params_snapshot(model: ModelConfig, **overrides: Any) -> dict[str, Any]:
    """The generation parameters active for one row, inline.

    Open question 9's convention: a log row that has to be joined back against
    `models.yaml` to be interpretable is a row that becomes noise the first time
    that file is edited.
    """
    snapshot = {
        "temperature": model.temperature,
        "top_p": model.top_p,
        "max_new_tokens": model.max_new_tokens,
        "answer_recovery_max_tokens": model.answer_recovery_max_tokens,
    }
    snapshot.update({key: value for key, value in overrides.items() if value is not None})
    return snapshot


def result_row(
    *,
    model: ModelConfig,
    question: MedQAQuestion,
    tag: str,
    condition_id: str,
    split: str,
    prompt: str,
    result: GenerationResult,
    overrides: dict[str, Any] | None = None,
    retrieval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One `QuestionResult`-shaped row plus this phase's cost/format fields.

    Field names come first from `interfaces.md`'s `QuestionResult` so Phase 12
    can read exploratory data without a translation layer; `retrieved_chunk_ids`
    and `verifier_decisions` are present-but-empty because this arm is
    closed-book, and Phase 6 fills them rather than renaming anything.
    `api_model`/`endpoint` are kept from `probe_models.py`'s rationale: a
    `models.yaml` whose two entries point at one endpoint silently degenerates
    the model factor, and the row is the only place that can disprove it.

    `retrieval` is the retrieval pass that produced this row's prompt, shaped by
    the calling script as `{"chunk_ids": [...], "context_chars": int, "chunks":
    [...], "params": {...]}` -- `chunk_ids` in *final prompt order*, because that
    is what Phase 11's context-position analysis and Phase 12's failure classes
    need, and the pre-rerank order cannot be recovered from the row once
    `Retriever.rerank()` truncates. The per-chunk provenance goes in `chunks` and
    the k's/device/stage timings in `retrieval`, so a reader can tell a bad
    retrieval from a bad reranker from a bad generation without re-running any of
    it. Omitted (closed-book) the row is exactly what Phase 5 wrote, which is what
    lets Phases 5 and 6 be joined row-for-row.
    """
    answer = answer_letter(result, question)
    tok_s = round(result.completion_tokens / result.elapsed_s, 1) if result.elapsed_s else None
    row = {
        "condition_id": condition_id,
        "tag": tag,
        "question_id": question.id,
        "split": split,
        "question": question.question,
        "reformulated_query": None,
        "retrieved_chunk_ids": [],
        "verifier_decisions": None,
        "raw_model_output": result.content,
        "parsed_answer": answer,
        "correct_answer": question.answer_idx.upper(),
        "is_correct": answer is not None and answer == question.answer_idx.upper(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model.name,
        "api_model": model.api_model_name,
        "endpoint": model.base_url,
        "unanswered": answer is None,
        "truncated": result.truncated,
        "finish_reason": result.finish_reason,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "wall_s": round(result.elapsed_s, 1),
        "tok_s": tok_s,
        "prompt_chars": len(prompt),
        "content_chars": len(result.content),
        "reasoning_chars": len(result.reasoning),
        "recovery_attempted": result.recovery_attempted,
        "recovered": result.recovered,
        "params": params_snapshot(model, **(overrides or {})),
    }
    if retrieval is not None:
        row["retrieved_chunk_ids"] = list(retrieval["chunk_ids"])
        row["context_chars"] = retrieval.get("context_chars")
        row["chunks"] = retrieval.get("chunks", [])
        row["retrieval"] = retrieval.get("params", {})
    return row


def error_row(
    *,
    model: ModelConfig,
    question: MedQAQuestion,
    tag: str,
    condition_id: str,
    split: str,
    exc: Exception,
    wall_s: float,
    prompt: str = "",
) -> dict[str, Any]:
    """A row for a call that never produced a completion.

    Recorded rather than raised away: one dead question must not cost the other
    nineteen, and "the endpoint was down for this id" is itself evidence.

    The row keeps the same keys as `result_row()` wherever it can, including
    `params`. Open question 9's whole argument is that a row must be readable
    without re-joining `models.yaml`, and an aborted row is the one most likely
    to be read months later against an edited config -- the first Phase 5 run's
    error rows carried no `params` at all, which is the counterexample that
    fixed this. `finish_reason` and `tok_s` stay None and `completion_tokens`
    stays null: there was no completion to have them for. That is different from
    zero, which would claim a measured-empty generation.
    """
    return {
        "condition_id": condition_id,
        "tag": tag,
        "question_id": question.id,
        "split": split,
        "question": question.question,
        "reformulated_query": None,
        "retrieved_chunk_ids": [],
        "verifier_decisions": None,
        "model": model.name,
        "api_model": model.api_model_name,
        "endpoint": model.base_url,
        "correct_answer": question.answer_idx.upper(),
        "raw_model_output": "",
        "parsed_answer": None,
        "is_correct": False,
        "unanswered": True,
        "truncated": False,
        "finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "tok_s": None,
        "prompt_chars": len(prompt),
        "content_chars": 0,
        "reasoning_chars": 0,
        "recovery_attempted": False,
        "recovered": False,
        "params": params_snapshot(model),
        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        "wall_s": round(wall_s, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


class EndpointCircuitBreaker:
    """Give up on an arm whose endpoint has died, instead of draining the queue.

    Phase 5's first run lost minutes to exactly this: the `model_b` server died
    under an in-flight request, and the harness then faithfully fired 17 more
    completions at a dead port, each logging a row that reads like a model
    failure (`experiments/phase5/FINDINGS.md` §7). Transport failure is not data,
    and a dead endpoint does not recover because you asked it nineteen more times.

    Classification is by the underlying `openai` exception, not by message text:
    `LLMClient._complete()` raises `LLMError(...) from exc`, so `__cause__` still
    carries the real one. Connection/timeout and 5xx errors count toward the
    threshold; an auth rejection trips at once, since every later call would fail
    identically. A 4xx other than auth does not count -- that is bad input, which
    one question can plausibly trigger on its own. A completion that came back
    with no parsable `ANSWER:` never reaches here: `agenerate()` returns those,
    and they are measured outcomes rather than failures.
    """

    def __init__(self, model_name: str, threshold: int = 3) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.consecutive = 0
        self.reason: str | None = None

    @property
    def tripped(self) -> bool:
        return self.reason is not None

    def record_success(self) -> None:
        """One live response clears the streak -- this counts *consecutive* refusals."""
        self.consecutive = 0

    def record_failure(self, exc: BaseException) -> None:
        """Count one failed call against the endpoint, tripping if it looks dead."""
        cause = exc.__cause__
        if isinstance(cause, openai.AuthenticationError):
            self._trip(f"endpoint rejected the key ({str(cause)[:150]})")
            return
        if isinstance(cause, openai.APIStatusError) and cause.status_code < 500:
            return
        if isinstance(cause, (openai.APIConnectionError, openai.APIStatusError)):
            self.consecutive += 1
            if self.consecutive >= self.threshold:
                self._trip(
                    f"{self.consecutive} consecutive transport failures "
                    f"({type(cause).__name__}: {str(cause)[:150]})"
                )

    def _trip(self, reason: str) -> None:
        self.reason = reason
        logger.error(
            "circuit breaker tripped for '{}': {} -- abandoning this arm instead of "
            "firing further requests at it",
            self.model_name,
            reason,
        )

    def abort_problem(self, skipped: int) -> str:
        """The problems-list line an abandoned arm must produce, sized by what it cost."""
        return (
            f"{self.model_name}: ABANDONED, endpoint went away ({self.reason}) -- "
            f"{skipped} question(s) never attempted. These are missing rows, not "
            f"model failures, and must not be read as either."
        )


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile, `None` for an empty sample (no zero-on-empty)."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return float(ordered[index])


def letter_distribution(rows: Sequence[dict[str, Any]], *, of_errors_only: bool = True) -> Counter:
    """Which option letters a model's *errors* land on.

    A model that habitually answers one letter confounds every delta in the
    study, and the only way to notice is to count it. `unparsed` is tallied
    alongside the letters so an unanswered row is never read as a wrong letter.
    """
    counts: Counter = Counter()
    for row in rows:
        if "error" in row or (of_errors_only and row.get("is_correct")):
            continue
        counts[row.get("parsed_answer") or "unparsed"] += 1
    return counts


def summarize(rows: Sequence[dict[str, Any]], *, tag: str | None = "main") -> str:
    """Per-model summary of a run's rows, in the shape FINDINGS.md needs.

    Everything printed here is also in `results.jsonl`; the point of printing it
    is that the numbers a phase's decisions rest on get read at the terminal,
    not only discovered later by whoever re-parses JSONL.

    Filtered to one `tag` by default because a caps check deliberately repeats
    questions under a different `max_new_tokens`: folding those repeats in would
    quietly double-count accuracy. Pass `tag=None` for every row.
    """
    if tag is not None:
        rows = [row for row in rows if row.get("tag") == tag]
    lines: list[str] = []
    errors = [row for row in rows if "error" in row]
    models = sorted({row.get("model", "?") for row in rows})

    for name in models:
        mine = [row for row in rows if row.get("model") == name and "error" not in row]
        failed = [row for row in errors if row.get("model") == name]
        lines.append(f"=== {name} ===")
        if not mine:
            lines.append(f"  no successful rows ({len(failed)} errored) -- see the problems list")
            continue

        n = len(mine)
        correct = sum(bool(row["is_correct"]) for row in mine)
        unanswered = sum(bool(row["unanswered"]) for row in mine)
        truncated = sum(bool(row["truncated"]) for row in mine)
        fired = [row for row in mine if row["recovery_attempted"]]
        recovered = [row for row in fired if row["recovered"]]
        tokens = [float(row["completion_tokens"] or 0) for row in mine]
        walls = [float(row["wall_s"] or 0) for row in mine]
        rates = [float(row["tok_s"]) for row in mine if row.get("tok_s")]
        finishes = Counter(str(row.get("finish_reason")) for row in mine)

        lines.append(
            f"  n={n}  correct={correct} ({100 * correct / n:.0f}%)  unanswered={unanswered}  "
            f"truncated={truncated}"
        )
        lines.append(
            f"  finish_reason: {dict(sorted(finishes.items()))}  "
            f"recovery: {len(recovered)}/{len(fired)} recovered of {len(fired)} fired"
        )
        lines.append(
            f"  wall s: median {_fmt(statistics.median(walls))}  max {_fmt(max(walls))}  "
            f"total {_fmt(sum(walls))}"
        )
        if rates:
            lines.append(f"  tok/s: median {_fmt(statistics.median(rates))}")
        lines.append(
            f"  completion tokens: p50 {_fmt(percentile(tokens, 0.5))}  "
            f"p90 {_fmt(percentile(tokens, 0.9))}  max {_fmt(max(tokens))}  "
            f"total {_fmt(sum(tokens))}"
        )
        prompt_chars = [int(row.get("prompt_chars") or 0) for row in mine]
        if prompt_chars:
            lines.append(
                f"  prompt chars: min {min(prompt_chars)} max {max(prompt_chars)}"
            )
        letters = letter_distribution(mine)
        lines.append(
            "  error letters: "
            + (", ".join(f"{k}={v}" for k, v in sorted(letters.items())) if letters else "none")
        )
        if failed:
            ids = ", ".join(str(row["question_id"]) for row in failed)
            lines.append(f"  {len(failed)} errored row(s): {ids}")
        lines.append("")

    if errors:
        lines.append(f"ERRORS ({len(errors)}):")
        for row in errors:
            lines.append(f"  {row['model']}/{row['question_id']} [{row['tag']}]: {row['error']}")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f}"


# --- retrieval-aware helpers (Phase 6 onward) --------------------------------
#
# Everything below is additive: `closed_book.py`'s rows and summaries are
# unchanged, because none of it is reachable without a caller that passes
# `retrieval=`. It lives here rather than in `raw_rag.py` because Phases 7-9 need
# the same three judgements -- did the excerpts reach the prompt, did the
# excerpt contain what was asked, and did reranking actually move anything --
# and a second implementation of those per script is how a study ends up with
# two numbers for one quantity.

def parse_notes(notes: Sequence[str]) -> dict[str, str]:
    """Turn repeated `--note KEY=VALUE` strings into `context.md`'s note rows.

    oMLX applies per-model sampling (top_k, thinking budget, repetition
    penalty) from its own `~/.omlx/model_settings.json`, which lives outside
    this repo and outside git, so `params_snapshot()` understates what produced
    a row. A note is how a run records what it otherwise could not prove -- and
    for Phase 6 it is mandatory, because a phase whose whole point is a *delta*
    against Phase 5 is worthless unless the endpoint that served both is shown
    to be the same one. Phase 5 §8.4 is the precedent: an endpoint swap alone
    moved `model_a` from 50% to 70% on 5 of the same 20 questions.

    `closed_book.py` carries an identical private copy; Phase 9's runner deletes
    it. Duplicated deliberately rather than importing Phase 5's script, which
    the plan freezes.
    """
    parsed: dict[str, str] = {}
    for note in notes:
        key, sep, value = note.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--note expects KEY=VALUE, got {note!r}")
        parsed[key.strip()] = value.strip()
    return parsed


# Typographic characters a StatPearls excerpt and a MedQA option can disagree on
# for reasons that mean nothing. Dropped rather than folded, so the helper below
# stays a substring test and does not grow into a parser.
_STRIP_CHARS = dict.fromkeys(map(ord, "\u2018\u2019\u201c\u201d\u2013\u2014\u2212'\""), None)


def _normalize(text: str) -> str:
    """Casefold, drop quotes/dashes, collapse whitespace -- for containment tests."""
    return " ".join(text.translate(_STRIP_CHARS).casefold().split())


def chunk_match_rank(needle: str, chunks: Sequence[Any]) -> int | None:
    """1-based prompt index of the first chunk containing `needle`, else None.

    The cheapest available answer to "did the context we handed the model
    actually contain the thing it needed?", which is where every RAG failure
    classification eventually reduces. It is a string test, so read it with both
    of its errors in mind: it *under*-counts (an excerpt explaining
    "noncaseating granulomas" without the gold phrase "sarcoidosis" reads as
    absent, though it is the useful excerpt) and it *over*-counts (an excerpt
    that names the gold option in order to rule it out reads as present).
    Under-counting is the commoner error at this corpus size, so treat a low hit
    rate as evidence about the metric before evidence about the retriever, and
    never call this accuracy.

    Accepts anything with a `.content` (a `RetrievedChunk` or a stand-in).
    """
    probe = _normalize(needle)
    if not probe:
        return None
    for index, chunk in enumerate(chunks, start=1):
        if probe in _normalize(chunk.content):
            return index
    return None


def missing_chunk_evidence(prompt: str, chunks: Sequence[Any]) -> list[str]:
    """Chunks that were retrieved but did not survive into the rendered prompt.

    Forty completions' worth of interpretation rests on the assumption that the
    chunks named in a row are the chunks the model saw. `format_context()`'s
    contract is `"[n] title\\ncontent"` blocks; if a later refactor of
    `prompt.py` renumbers, dedupes, or truncates them, every row would silently
    describe a prompt that was never sent, and Phase 11's position analysis would
    be measuring nothing. Checked once per question before the first request, so
    the failure surfaces as a problem line instead of as a finding that is wrong.

    Returns one line per gap; empty means every chunk is verifiably present.
    """
    problems: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        if f"[{index}] {chunk.title}" not in prompt:
            problems.append(
                f"[{index}] chunk_id={chunk.chunk_id}: '[{index}] {chunk.title}' header missing"
            )
        probe = chunk.content[:80]
        if probe and probe not in prompt:
            problems.append(f"[{index}] chunk_id={chunk.chunk_id}: content head missing")
    return problems


def retrieval_stats(rows: Sequence[dict[str, Any]], *, tag: str | None = "main") -> dict[str, Any]:
    """Retrieval/rerank cost and churn for a run, deduplicated per pass.

    `results.jsonl` carries one row per *(arm, question)*, but Phase 6 retrieves
    once per question and hands both arms the same context, so two rows describe
    one retrieval pass. Averaging over rows would double the sample and weight
    each question by however many arms happened to run, so units are deduped on
    `(question_id, chunk_ids)`: that collapses genuinely shared passes and keeps
    different ones apart, which matters as soon as Phase 7's arms retrieve
    different contexts for the same question.

    Churn is deliberately not a correlation coefficient. For each pass, read the
    kept chunks' pre-rerank vector ranks in prompt order: a chunk is *moved* when
    that sequence differs from its sorted form, and *promoted* when its vector
    rank was past the cut (`top_k_rerank`), i.e. it enters the prompt only
    because the reranker exists. `moved=0, promoted=0` is exactly what a
    rerank-off control must produce, which makes this the test that the control
    is a control.

    Scores are reported as two separate means and are *not* comparable: vector
    scores are cosine similarities (these run ~0.6-0.85, so "high similarity"
    barely discriminates) and rerank scores are `CrossEncoder.predict()` logits
    (unbounded, may be negative). Dividing one by the other would be the first
    mistake a reader of this table is likely to make, so they get separate keys
    instead of a ratio.
    """
    if tag is not None:
        rows = [row for row in rows if row.get("tag") == tag]
    # `error_row()` never sets `chunks`, so failed calls drop out here instead of
    # being counted as retrievals that returned nothing.
    with_chunks = [row for row in rows if row.get("chunks")]

    units: dict[tuple[Any, tuple[Any, ...]], dict[str, Any]] = {}
    for row in with_chunks:
        units.setdefault((row.get("question_id"), tuple(row.get("retrieved_chunk_ids") or ())), row)
    passes = list(units.values())

    stats: dict[str, Any] = {
        "rows": len(with_chunks),
        "passes": len(passes),
        "questions": len({row.get("question_id") for row in passes}),
        "chunks_per_pass": None,
        "context_chars": {},
        "stage_s": {},
        "vector_scores": {},
        "rerank_scores": {},
        "churn": {"passes": 0, "moved": 0, "kept": 0, "promoted": 0},
    }
    if not passes:
        return stats

    sizes = [len(row["chunks"]) for row in passes]
    stats["chunks_per_pass"] = round(sum(sizes) / len(sizes), 2)

    chars = [int(row.get("context_chars") or 0) for row in passes]
    stats["context_chars"] = {
        "min": min(chars),
        "median": percentile(chars, 0.5),
        "max": max(chars),
    }
    context_total, prompt_total = sum(chars), sum(int(row.get("prompt_chars") or 0) for row in passes)
    stats["context_share_of_prompt"] = round(context_total / prompt_total, 3) if prompt_total else None

    retrieve = [float(_retrieval_param(row, "retrieve_s") or 0.0) for row in passes]
    rerank = [float(_retrieval_param(row, "rerank_s") or 0.0) for row in passes]
    stats["stage_s"] = {
        "retrieve_median": round(statistics.median(retrieve), 3),
        "retrieve_max": round(max(retrieve), 3),
        "rerank_median": round(statistics.median(rerank), 3),
        "rerank_max": round(max(rerank), 3),
        "total": round(sum(retrieve) + sum(rerank), 2),
    }

    top1 = [_retrieval_param(row, "vector_top1_score") for row in passes]
    top1 = [value for value in top1 if value is not None]
    weakest = [_retrieval_param(row, "vector_last_score") for row in passes]
    weakest = [value for value in weakest if value is not None]
    if top1:
        stats["vector_scores"] = {
            "top1_mean": round(sum(top1) / len(top1), 3),
            "top1_min": round(min(top1), 3),
        }
    if weakest:
        stats["vector_scores"]["weakest_retrieved_mean"] = round(sum(weakest) / len(weakest), 3)

    kept_scores = []
    for row in passes:
        scores = [c["rerank_score"] for c in row["chunks"] if c.get("rerank_score") is not None]
        if scores:
            kept_scores.append(sum(scores) / len(scores))
    if kept_scores:
        stats["rerank_scores"] = {
            "kept_mean": round(sum(kept_scores) / len(kept_scores), 3),
            "kept_min": round(min(kept_scores), 3),
            "kept_max": round(max(kept_scores), 3),
        }

    for row in passes:
        cut = int(_retrieval_param(row, "top_k_rerank") or len(row["chunks"]))
        ranks = [chunk.get("vector_rank") for chunk in row["chunks"]]
        if not ranks or any(not isinstance(rank, int) for rank in ranks):
            continue  # a pass with no recorded vector ranks cannot be churn-scored
        stats["churn"]["kept"] += len(ranks)
        moved = sum(1 for got, want in zip(ranks, sorted(ranks)) if got != want)
        promoted = sum(1 for rank in ranks if rank > cut)
        stats["churn"]["moved"] += moved
        stats["churn"]["promoted"] += promoted
        if moved or promoted:
            stats["churn"]["passes"] += 1
    return stats


def _retrieval_param(row: dict[str, Any], key: str) -> Any:
    return (row.get("retrieval") or {}).get(key)


def summarize_retrieval(rows: Sequence[dict[str, Any]], *, tag: str | None = "main") -> str:
    """`retrieval_stats()` as a printable block, in the shape FINDINGS.md needs.

    Split from `summarize()` because the questions they answer differ: that one
    asks what the model did, this one asks whether the pipeline handed it
    something worth reading and what the retrieval stages cost. Retrieval cost
    also belongs on its own line because it is the only place a reader can see
    that the reranker -- not the LLM -- was or was not the slow stage.
    """
    stats = retrieval_stats(rows, tag=tag)
    if not stats["passes"]:
        return f"RETRIEVAL (tag={tag!r}): no rows with chunks -- closed-book rows?"

    lines = [
        f"RETRIEVAL (tag={tag!r}): {stats['passes']} pass(es) over "
        f"{stats['questions']} question(s), {stats['rows']} row(s)"
    ]
    chars, stage = stats["context_chars"], stats["stage_s"]
    share = stats["context_share_of_prompt"]
    lines.append(
        f"  chunks/pass: {stats['chunks_per_pass']}  context chars: "
        f"min {chars['min']} median {_fmt(chars['median'])} max {_fmt(chars['max'])}"
        + (f"  ({100 * share:.1f}% of the prompt)" if share else "")
    )
    lines.append(
        f"  s/question: retrieve median {stage['retrieve_median']} max {stage['retrieve_max']}  "
        f"rerank median {stage['rerank_median']} max {stage['rerank_max']}  "
        f"total {stage['total']} (both arms share one pass)"
    )
    if stats["vector_scores"]:
        scores = stats["vector_scores"]
        lines.append(
            "  vector cosine (bounded 0-1): top-1 mean "
            f"{scores['top1_mean']} min {scores['top1_min']}"
            + (f"  weakest of the retrieve set mean {scores['weakest_retrieved_mean']}" if "weakest_retrieved_mean" in scores else "")
        )
    if stats["rerank_scores"]:
        scores = stats["rerank_scores"]
        lines.append(
            "  reranker logit (UNCALIBRATED -- not comparable to the cosine row): "
            f"kept mean {scores['kept_mean']} "
            f"range {scores['kept_min']}..{scores['kept_max']}"
        )
    churn = stats["churn"]
    share_moved = f" ({100 * churn['moved'] / churn['kept']:.0f}% of kept chunks)" if churn["kept"] else ""
    lines.append(
        f"  rerank churn: {churn['moved']} chunk(s) moved position{share_moved}; "
        f"{churn['promoted']} entered the prompt from past the top-k_rerank cut; "
        f"{churn['passes']}/{stats['passes']} pass(es) changed at all "
        "(a rerank-off pass must read 0/0/0)"
    )
    return "\n".join(lines)




