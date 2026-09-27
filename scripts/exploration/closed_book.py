"""Phase 5: closed-book generation smoke test over a pinned dev-split sample.

Runs `LLMClient` + `build_answer_prompt(context_chunks=None)` over real MedQA
items from `config.dev_split` and writes the exploratory-artifact trio: rows,
every chain of thought, and `context.md`. Tracked and re-runnable on purpose --
this is the evidence base for the Phase 10 design freeze, not a throwaway.

`scripts/probe_models.py` stays the pre-flight for endpoint health, cost and the
recovery path, but its four hand-written questions are not a substitute for real
MedQA items: they cannot show how a *reasoning* model behaves across a sample,
which is what the `max_new_tokens` budget has to be sized against.

Four questions this run has to answer (PLAN.md Phase 5):

1. Does every response parse to a valid option letter via `parse_answer()` on
   real items? Accuracy is reported too, but n=20 is plumbing evidence, never an
   accuracy estimate.
2. How often does `finish_reason="length"` fire, and does the one-shot ANSWER
   recovery close it, at the *configured* budgets (`model_a` 1,024 / `model_b`
   16,384)? Those counts -- not guesswork -- decide whether `model_a`'s budget is
   large enough and whether `model_b`'s can come down (open question 8).
3. Does `seed` actually bite (open question 10)? Phase 4 proved the parameter is
   *accepted*; if repeated seeds produce identical draws, the study's 3-seed
   design contributes zero variance and every CI built on it is fiction.
   `--checks seed` runs the same prompt at the same seed twice and at two
   different seeds, at each model's own temperature > 0.
4. Which option letters does each model's *errors* land on? A model biased
   toward one letter confounds every delta in the study.

Requests are sequential within a model by default. Passing `seed` disables
batched serving in `mlx_lm.server` (experiments/phase4/FINDINGS.md), and both
endpoints share one Apple Silicon GPU, so `--concurrent-models` would trade the
per-model tok/s that Phase 13's wall-clock estimate depends on for a shorter
wall clock here. The concurrency sweep belongs to Phase 9 (open question 12).

Cost warning: `model_b` writes a chain of thought into a non-standard
`reasoning` field and, at its recommended temperature, failed to close within
3,072 tokens on 4 of 6 closed-book questions in Phase 4. At its 16,384-token
ceiling and ~60-78 tok/s, one question can take ~4.5 minutes, so a 20-question
pass is 15-90 minutes. Rows are flushed as they complete; interrupt freely and
resume with `--resume outputs/exploration/phase5/<stamp>`.

    set -a; source .env; set +a
    .venv/bin/python scripts/exploration/closed_book.py --dry-run
    .venv/bin/python scripts/exploration/closed_book.py --sample-size 20
"""

import argparse
import asyncio
import time
from collections import Counter
from typing import Any

from loguru import logger

from medical_rag.config import ExperimentConfig, ModelConfig, load_config
from medical_rag.data.load_medqa import MedQAQuestion, load_medqa
from medical_rag.generation.llm import LLMClient, LLMError
from medical_rag.generation.prompt import build_answer_prompt

from _common import (
    DEFAULT_SAMPLE_SEED,
    RunWriter,
    error_row,
    result_row,
    select_sample,
    summarize,
    warn_on_degenerate_sample,
)

PHASE = "phase5"
CONDITION_ID = "phase5_closed_book"

# The seed check's four draws per question. Same-seed repeats answer "is the
# draw reproducible?"; different-seed draws answer "does the seed contribute
# variance?" -- the 3-seed design needs both, and they are separate questions,
# so they are measured separately.
SEED_LABELS: tuple[tuple[str, int], ...] = (
    ("seed-same-1", 0),
    ("seed-same-2", 0),
    ("seed-diff-1", 100),
    ("seed-diff-2", 101),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/default.yaml", help="path to default.yaml")
    parser.add_argument(
        "--models", nargs="*", default=None, help="model keys (default: every model in models.yaml)"
    )
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="pins which questions are drawn; recorded in context.md",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="generation seed for the main pass (default: each model's own seed)",
    )
    parser.add_argument(
        "--checks",
        nargs="*",
        choices=("main", "seed", "caps"),
        default=["main", "seed"],
        help="which checks to run",
    )
    parser.add_argument(
        "--seed-questions", type=int, default=2, help="questions to repeat for the seed check"
    )
    parser.add_argument(
        "--caps",
        nargs="*",
        type=int,
        default=None,
        help="extra max_new_tokens ceilings to re-measure (contingency pass, off by default)",
    )
    parser.add_argument(
        "--cap-questions", type=int, default=5, help="questions to use for each --caps pass"
    )
    parser.add_argument(
        "--out-root", default="outputs/exploration", help="gitignored root for run directories"
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="RUN_DIR",
        help="re-open an existing run dir and skip (model, question_id, tag) rows it already has",
    )
    parser.add_argument("--no-chains", action="store_true", help="do not save chains of thought")
    parser.add_argument(
        "--concurrent-models",
        action="store_true",
        help="run the two endpoints simultaneously (faster wall clock, depressed tok/s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="pin and print the sample and the planned call count; fire no requests",
    )
    return parser.parse_args()


async def preflight(model: ModelConfig, client: LLMClient) -> str | None:
    """Confirm the endpoint is up and serving the model `models.yaml` names.

    Cheap insurance against losing an hour: one `GET /v1/models` catches a dead
    endpoint and a misnamed `api_model_name` before 20 long completions fail
    one at a time.
    """
    try:
        served = await client.client.models.list()
    except Exception as exc:
        return (
            f"{model.name}: endpoint {model.base_url} is not answering -- "
            f"{type(exc).__name__}: {str(exc)[:150]}"
        )

    ids = {item.id for item in served.data}
    if model.api_model_name not in ids:
        return (
            f"{model.name}: {model.base_url} serves {sorted(ids)} but models.yaml names "
            f"'{model.api_model_name}' -- every completion would be rejected or, worse, "
            "answered by a different model than the arm claims"
        )
    return None


def chain_header(row: dict[str, Any]) -> str:
    """The one-line provenance stamped atop every saved chain of thought."""
    return (
        f"# {row['model']} q{row['question_id']} [{row['tag']}] "
        f"seed={row['seed']} max_new_tokens={row['params']['max_new_tokens']}\n"
        f"gold={row['correct_answer']} answer={row['parsed_answer']} "
        f"finish={row['finish_reason']} completion_tokens={row['completion_tokens']} "
        f"wall_s={row['wall_s']} prompt_chars={row['prompt_chars']} "
        f"recovery={row['recovery_attempted']}/{row['recovered']}"
    )


async def run_main_check(
    model: ModelConfig,
    client: LLMClient,
    questions: list[MedQAQuestion],
    seed: int | None,
    split: str,
    writer: RunWriter,
    done: set[tuple[str, str, str]],
) -> list[str]:
    """One closed-book completion per sampled question, at the model's own budget.

    Returns the problems worth exiting non-zero on. Rows are already flushed to
    `results.jsonl` as they arrive, and the final summary re-reads that file, so
    a resumed run reports the whole run rather than just this session's slice.
    """
    problems: list[str] = []
    overrides = {} if seed is None else {"seed": seed}

    for index, question in enumerate(questions, start=1):
        if (model.name, question.id, "main") in done:
            logger.info("{}/{} already logged -- skipping (resume)", model.name, question.id)
            continue

        prompt = build_answer_prompt(question.question, question.options, None)
        started = time.perf_counter()
        try:
            result = await client.agenerate(prompt, seed=seed)
        except LLMError as exc:
            writer.write_row(
                error_row(
                    model=model,
                    question=question,
                    tag="main",
                    condition_id=CONDITION_ID,
                    split=split,
                    exc=exc,
                    wall_s=time.perf_counter() - started,
                )
            )
            problems.append(f"{model.name}/q{question.id}: {exc}")
            continue

        row = writer.write_row(
            result_row(
                model=model,
                question=question,
                tag="main",
                condition_id=CONDITION_ID,
                split=split,
                prompt=prompt,
                result=result,
                overrides=overrides,
            )
        )
        writer.write_chain(
            model, question.id, chain_header(row), result.reasoning or result.content
        )

        logger.info(
            "[{}/{}] q{} gold={} answer={} finish={} tokens={} {:.1f}s{}",
            index,
            len(questions),
            question.id,
            row["correct_answer"],
            row["parsed_answer"],
            row["finish_reason"],
            row["completion_tokens"],
            row["wall_s"],
            " TRUNCATED" if row["truncated"] else "",
        )
        if row["unanswered"]:
            # A legitimate experimental outcome, not a broken pipeline -- the row
            # is scored incorrect. It is still the single most interesting thing
            # in the run, so it gets a warning and its chain gets saved.
            logger.warning(
                "{} returned no parsable answer for q{} (finish_reason={}, {} reasoning "
                "chars, recovery={}) -- would be scored incorrect",
                model.name,
                question.id,
                row["finish_reason"],
                row["reasoning_chars"],
                row["recovery_attempted"],
            )
    return problems


def _tally(flags: list[bool]) -> str:
    return "n/a" if not flags else f"{sum(flags)}/{len(flags)}"


VERDICT_NOTES = {
    "bites": "seed contributes variance, so the 3-seed design is honest.",
    "inert": "every draw was identical regardless of seed, so repeated seeds buy zero "
    "variance -- report seed variance as unsupported by this endpoint, or drive "
    "repetition from temperature instead (open question 10).",
    "nondeterministic-with-fixed-seed": "the endpoint does not pin the draw at a fixed "
    "seed, so no reproducibility claim survives: rerunning the pilot would not "
    "reproduce the pilot.",
    "partial": "mixed signals -- the sample is too small to settle it; raise "
    "--seed-questions before the Phase 10 freeze.",
    "inconclusive": "no question completed all four draws; nothing was measured.",
}


def seed_verdict(
    model_name: str,
    draws: dict[str, dict[str, str]],
    letters: dict[str, dict[str, str | None]],
    temperature: float,
) -> tuple[str, str]:
    """Interpret the same-seed vs different-seed comparison.

    Compares the graded `content`, not the reasoning text: identical content on a
    reasoning model is strong evidence the whole draw was identical, and content
    is what scoring reads, so this is the determinism claim that matters.
    Returns `(verdict, report)`; the verdict string is what FINDINGS.md cites.
    """
    labels = [label for label, _ in SEED_LABELS]
    complete = [qid for qid in draws if all(label in draws[qid] for label in labels)]

    same = [draws[q][labels[0]] == draws[q][labels[1]] for q in complete]
    diff = [draws[q][labels[2]] == draws[q][labels[3]] for q in complete]
    cross = [draws[q][labels[0]] == draws[q][labels[2]] for q in complete]
    same_letter = [
        letters[q].get(labels[0]) is not None
        and letters[q][labels[0]] == letters[q][labels[1]]
        for q in complete
    ]
    diff_letter = [
        letters[q].get(labels[2]) is not None
        and letters[q][labels[2]] == letters[q][labels[3]]
        for q in complete
    ]

    if not complete:
        verdict = "inconclusive"
    elif not all(same):
        verdict = "nondeterministic-with-fixed-seed"
    elif all(cross):
        verdict = "inert"
    elif all(diff) and not any(cross):
        verdict = "bites"
    else:
        verdict = "partial"

    report = [
        f"seed check ({model_name}, temperature={temperature}, "
        f"{len(complete)} question(s) x {len(labels)} draws)",
        f"  same-seed pair identical:          {_tally(same)}",
        f"  different-seed pair identical:     {_tally(diff)}",
        f"  same- vs different-seed identical: {_tally(cross)}",
        f"  letter agrees, same seed:          {_tally(same_letter)}",
        f"  letter agrees, different seeds:    {_tally(diff_letter)}",
        f"  VERDICT: {verdict} -- {VERDICT_NOTES[verdict]}",
    ]
    if len(complete) < len(draws):
        report.append(
            f"  note: {len(draws) - len(complete)} question(s) lacked a full set of "
            f"{len(labels)} draws (errors or an unfinished resume) and were excluded"
        )
    return verdict, "\n".join(report)


async def run_seed_check(
    model: ModelConfig,
    client: LLMClient,
    questions: list[MedQAQuestion],
    seed: int | None,
    split: str,
    writer: RunWriter,
    done: set[tuple[str, str, str]],
) -> tuple[list[str], list[str]]:
    """Does `seed` bite? Open question 10, which decides the whole variance design.

    Phase 4 only proved the parameter is *accepted* -- a 200 after adding
    `seed=123` to a temperature-0 call. Under a 3-seed design that distinction
    is fatal: if the endpoint ignores the seed, or ignores sampling altogether,
    the repetitions contribute no variance and every interval built on them is
    fiction. Four draws per question separate the two failure modes.

    Returns `(problems, report_lines)`; the report is what lands in
    `seed_check.md` and in FINDINGS.md.
    """
    base_seed = seed if seed is not None else model.seed

    problems: list[str] = []
    draws: dict[str, dict[str, str]] = {}
    letters: dict[str, dict[str, str | None]] = {}
    wanted = dict(SEED_LABELS)
    prior = {
        (row["question_id"], row["tag"]): row
        for row in writer.load_rows()
        if row.get("model") == model.name and row.get("tag") in wanted
    }

    for question in questions:
        prompt = build_answer_prompt(question.question, question.options, None)
        for label, offset in SEED_LABELS:
            draws.setdefault(question.id, {})
            letters.setdefault(question.id, {})
            seed_value = base_seed + offset

            if (model.name, question.id, label) in done:
                row = prior.get((question.id, label))
                if row is not None:
                    draws[question.id][label] = row.get("raw_model_output") or ""
                    letters[question.id][label] = row.get("parsed_answer")
                    logger.info(
                        "{}/{} {} already logged -- reusing (resume)",
                        model.name,
                        question.id,
                        label,
                    )
                    continue

            try:
                result = await client.agenerate(prompt, seed=seed_value)
            except LLMError as exc:
                problems.append(f"{model.name}/q{question.id}[{label}]: {exc}")
                continue
            row = writer.write_row(
                result_row(
                    model=model,
                    question=question,
                    tag=label,
                    condition_id=f"{CONDITION_ID}_seed",
                    split=split,
                    prompt=prompt,
                    result=result,
                    overrides={"seed": seed_value},
                )
            )
            writer.write_chain(
                model,
                f"{question.id}.{label}",
                chain_header(row),
                result.reasoning or result.content,
            )
            draws[question.id][label] = result.content
            letters[question.id][label] = row["parsed_answer"]
            logger.info(
                "seed-check q{} {} seed={} -> {} ({} content / {} reasoning chars, {:.1f}s)",
                question.id,
                label,
                seed_value,
                row["parsed_answer"],
                row["content_chars"],
                row["reasoning_chars"],
                row["wall_s"],
            )

    _, report = seed_verdict(model.name, draws, letters, model.temperature)
    return problems, report.splitlines()


async def run_caps_check(
    model: ModelConfig,
    client: LLMClient,
    questions: list[MedQAQuestion],
    seed: int | None,
    cap: int,
    split: str,
    writer: RunWriter,
    done: set[tuple[str, str, str]],
) -> list[str]:
    """Re-run a few questions under a lower `max_new_tokens` ceiling.

    The contingency pass for open question 8: if `model_b`'s 16,384-token budget
    is only ever consumed by chains of thought that never reach `ANSWER:`, the
    honest fix is a smaller ceiling, not a bigger one. Running the same questions
    at 4,096/8,192 measures the cost of that choice instead of arguing about it.
    """
    problems: list[str] = []

    for question in questions:
        tag = f"cap{cap}"
        if (model.name, question.id, tag) in done:
            logger.info("{}/{} already logged -- skipping (resume)", model.name, question.id)
            continue

        prompt = build_answer_prompt(question.question, question.options, None)
        started = time.perf_counter()
        try:
            result = await client.agenerate(prompt, seed=seed, max_new_tokens=cap)
        except LLMError as exc:
            writer.write_row(
                error_row(
                    model=model,
                    question=question,
                    tag=tag,
                    condition_id=f"{CONDITION_ID}_{tag}",
                    split=split,
                    exc=exc,
                    wall_s=time.perf_counter() - started,
                )
            )
            problems.append(f"{model.name}/q{question.id}[{tag}]: {exc}")
            continue

        row = writer.write_row(
            result_row(
                model=model,
                question=question,
                tag=tag,
                condition_id=f"{CONDITION_ID}_{tag}",
                split=split,
                prompt=prompt,
                result=result,
                overrides=({"seed": seed} if seed is not None else {}) | {"max_new_tokens": cap},
            )
        )
        writer.write_chain(model, f"{question.id}.{tag}", chain_header(row), result.reasoning or result.content)
        logger.info(
            "cap{} q{} -> {} finish={} tokens={}",
            cap,
            question.id,
            row["parsed_answer"],
            row["finish_reason"],
            row["completion_tokens"],
        )
    return problems


# Rough throughput band for these two MLX endpoints (experiments/phase4). Used
# only to print a worst-case cost notice before firing an hour of requests; the
# run's own tok/s is reported from measured rows.
TPS_PESSIMISTIC = 40.0
TPS_OPTIMISTIC = 80.0


def planned_calls(args: argparse.Namespace, n_sample: int, n_repeated: int, n_models: int) -> dict:
    """How many completions each check will fire, so cost is visible up front."""
    plan = {}
    if "main" in args.checks:
        plan["main"] = n_sample * n_models
    if "seed" in args.checks:
        plan["seed"] = n_repeated * len(SEED_LABELS) * n_models
    if args.caps:
        plan["caps"] = len(args.caps) * n_repeated * n_models
    return plan


def print_plan(
    args: argparse.Namespace,
    config: ExperimentConfig,
    keys: list[str],
    sample: list[MedQAQuestion],
    gold: Counter,
    plan: dict,
) -> None:
    """Echo the pinned sample and the cost of proceeding, before any request."""
    print(f"pinned sample (sample_seed={args.sample_seed}, split={config.dev_split}):")
    print("  " + ",".join(question.id for question in sample))
    print(f"  gold letters: {dict(sorted(gold.items()))}")
    print()
    print("arms:")
    for key in keys:
        model = config.models[key]
        print(
            f"  {key}: {model.api_model_name} @ {model.base_url} temperature={model.temperature} "
            f"max_new_tokens={model.max_new_tokens} recovery={model.answer_recovery_max_tokens} "
            f"seed={model.seed}"
        )
    print()
    print(f"planned completions: {plan} (total {sum(plan.values())})")
    for key in keys:
        ceiling = config.models[key].max_new_tokens
        print(
            f"  worst case {key}: {ceiling / TPS_PESSIMISTIC / 60:.1f}-"
            f"{ceiling / TPS_OPTIMISTIC / 60:.1f} min/question if every completion ran "
            "to its ceiling -- a bound, not an estimate"
        )
    print("  (rows flush as they complete; Ctrl-C is safe, resume with --resume <run dir>)")


async def amain(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    keys = args.models or list(config.models)
    unknown = [key for key in keys if key not in config.models]
    if unknown:
        raise SystemExit(
            f"--models {unknown} not found in {args.config}; available: {sorted(config.models)}"
        )
    if not args.checks:
        raise SystemExit("--checks cannot be empty")

    questions = load_medqa(config.dev_split)
    sample = select_sample(questions, args.sample_size, args.sample_seed)
    gold = warn_on_degenerate_sample(sample)
    repeated = sample[: max(0, args.seed_questions)]
    cap_questions = sample[: max(0, args.cap_questions)]

    print_plan(
        args,
        config,
        keys,
        sample,
        gold,
        planned_calls(args, len(sample), len(repeated), len(keys)),
    )
    if args.dry_run:
        print("\ndry run -- no requests sent")
        return 0

    reports: list[str] = []
    with RunWriter(
        PHASE, root=args.out_root, run_dir=args.resume, save_chains=not args.no_chains
    ) as writer:
        done = writer.completed_keys()
        if done:
            print(f"\nresuming {writer.dir}: {len(done)} row(s) already logged")
        writer.write_context(
            config=config,
            model_keys=keys,
            questions=sample,
            split=config.dev_split,
            sample_seed=args.sample_seed,
            extra={
                "checks": ", ".join(args.checks),
                "seed-check questions": (
                    ",".join(question.id for question in repeated)
                    if "seed" in args.checks
                    else "n/a (seed check not requested)"
                ),
                "caps": args.caps or "none",
                "concurrent_models": args.concurrent_models,
                "seed report": "seed_check.md, written after the run",
                "scoring note": "parsed_answer is recomputed against each question's own "
                "option letters, not GenerationResult.parsed_answer's hardcoded ABCDE",
            },
        )
        print(f"run dir: {writer.dir}\n", flush=True)

        async def drive(key: str) -> list[str]:
            """Run every requested check for one arm; return its problems."""
            model = config.models[key]
            problems: list[str] = []
            try:
                client = LLMClient(model)
            except LLMError as exc:
                return [f"{key}: {exc}"]

            async with client:
                blocked = await preflight(model, client)
                if blocked:
                    return [blocked]
                print(f"--- {key}: {model.api_model_name} @ {model.base_url} ---", flush=True)

                if "main" in args.checks:
                    problems += await run_main_check(
                        model, client, sample, args.seed, config.dev_split, writer, done
                    )
                if "seed" in args.checks:
                    seed_problems, report = await run_seed_check(
                        model, client, repeated, args.seed, config.dev_split, writer, done
                    )
                    problems += seed_problems
                    if report:
                        # extend, not +=: rebinding `reports` here would need
                        # nonlocal and would silently drop the other arm's report.
                        reports.extend([f"## {model.name}", *report, ""])
                for cap in args.caps or []:
                    problems += await run_caps_check(
                        model,
                        client,
                        cap_questions,
                        args.seed,
                        cap,
                        config.dev_split,
                        writer,
                        done,
                    )
            return problems

        if args.concurrent_models:
            per_model = await asyncio.gather(*(drive(key) for key in keys))
        else:
            per_model = [await drive(key) for key in keys]

        problems = [problem for model_problems in per_model for problem in model_problems]
        rows = writer.load_rows()
        for key in keys:
            if not any(row.get("model") == config.models[key].name for row in rows):
                problems.append(f"{key}: produced no rows at all")
        if reports:
            (writer.dir / "seed_check.md").write_text(
                "# Seed check -- open question 10\n\n" + "\n".join(reports) + "\n", encoding="utf-8"
            )

    print("\n" + summarize(rows))
    if reports:
        print("\n" + "\n".join(reports))
    print(f"\nartifacts: {writer.dir}")

    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for problem in problems:
            print(f"  - {problem}")
        print("\nexit 1: not every planned call produced a row. Truncated and unanswered "
              "rows are findings, not problems.")
        return 1
    print(
        "\nEvery planned call produced a row. Next: copy the measured numbers and "
        "representative chains into experiments/phase5/FINDINGS.md"
    )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(amain(parse_args())))


if __name__ == "__main__":
    main()
