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
        root: str | Path = "outputs/exploration",
        run_dir: str | Path | None = None,
        save_chains: bool = True,
    ) -> None:
        # run_dir reopens an existing directory, which is what makes --resume
        # re-attach to a half-finished run instead of starting a second one.
        self.dir = Path(run_dir) if run_dir else _unique_dir(Path(root) / phase / _utc_stamp())
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
        """`(model, question_id, tag)` triples already present in results.jsonl."""
        return {(row.get("model", "?"), row.get("question_id", "?"), row.get("tag", "?"))
                for row in self.load_rows()}

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
) -> dict[str, Any]:
    """One `QuestionResult`-shaped row plus this phase's cost/format fields.

    Field names come first from `interfaces.md`'s `QuestionResult` so Phase 12
    can read exploratory data without a translation layer; `retrieved_chunk_ids`
    and `verifier_decisions` are present-but-empty because this arm is
    closed-book, and Phase 6 fills them rather than renaming anything.
    `api_model`/`endpoint` are kept from `probe_models.py`'s rationale: a
    `models.yaml` whose two entries point at one endpoint silently degenerates
    the model factor, and the row is the only place that can disprove it.
    """
    answer = answer_letter(result, question)
    tok_s = round(result.completion_tokens / result.elapsed_s, 1) if result.elapsed_s else None
    return {
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


def error_row(
    *,
    model: ModelConfig,
    question: MedQAQuestion,
    tag: str,
    condition_id: str,
    split: str,
    exc: Exception,
    wall_s: float,
) -> dict[str, Any]:
    """A row for a call that never produced a completion.

    Recorded rather than raised away: one dead question must not cost the other
    nineteen, and "the endpoint was down for this id" is itself evidence.
    """
    return {
        "condition_id": condition_id,
        "tag": tag,
        "question_id": question.id,
        "split": split,
        "model": model.name,
        "api_model": model.api_model_name,
        "endpoint": model.base_url,
        "correct_answer": question.answer_idx.upper(),
        "parsed_answer": None,
        "is_correct": False,
        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        "wall_s": round(wall_s, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


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
                f"  closed-book prompt chars: min {min(prompt_chars)} max {max(prompt_chars)}"
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
