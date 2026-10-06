"""Phase 5: closed-book generation smoke test over a pinned dev-split sample.

Runs `LLMClient` + `build_answer_prompt(context_chunks=None)` over real MedQA
items from `config.dev_split` and writes the exploratory-artifact trio: rows,
every chain of thought, and `context.md`. Tracked and re-runnable on purpose --
this is the evidence base for the Phase 10 design freeze, not a throwaway.

`scripts/probe_models.py` stays the pre-flight for endpoint health, cost and the
recovery path, but its four hand-written questions are not a substitute for real
MedQA items: they cannot show how a *reasoning* model behaves across a sample,
which is what the `max_new_tokens` budget has to be sized against.

Three questions this run has to answer (PLAN.md Phase 5):

1. Does every response parse to a valid option letter via `parse_answer()` on
   real items? Accuracy is reported too, but n=20 is plumbing evidence, never an
   accuracy estimate.
2. How often does `finish_reason="length"` fire, and does the one-shot ANSWER
   recovery close it, at the *configured* budgets (`model_a` 1,024 / `model_b`
   16,384)? Those counts -- not guesswork -- decide whether `model_a`'s budget is
   large enough and whether `model_b`'s can come down (open question 8).
3. Which option letters does each model's *errors* land on? A model biased
   toward one letter confounds every delta in the study.

(Open question 10 -- whether `seed` actually bites -- was settled by an earlier
run of this script and is now resolved by removing `seed` from the pipeline
entirely: PLAN.md item 10. There is no longer a seed check here.)

Requests are sequential within a model by default. Both endpoints share one
Apple Silicon GPU, so `--concurrent-models` would trade the per-model tok/s
that Phase 13's wall-clock estimate depends on for a shorter wall clock here.
The concurrency sweep belongs to Phase 9 (open question 12).

Run one arm over the pinned Phase 5 sample (all 20 questions x both models = 40
calls; `--limit` shrinks it for a live check). Results land in a new timestamped
folder under `--output-dir`, which defaults to `outputs/exploration/<script-stem>/`
-- e.g. `outputs/exploration/closed_book/20260927T162758Z/`.

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

from medical_rag.eval.runlog import (
    DEFAULT_SAMPLE_SEED,
    EndpointCircuitBreaker,
    RunWriter,
    error_row,
    result_row,
    select_sample,
    summarize,
    warn_on_degenerate_sample,
)

PHASE = "phase5"
CONDITION_ID = "phase5_closed_book"


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
        "--checks",
        nargs="*",
        choices=("main", "caps"),
        default=["main"],
        help="which checks to run",
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
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="record a fact about this run in context.md; repeatable. Needed because "
        "the endpoint now applies per-model sampling (top_k, thinking budget, "
        "repetition penalty) from its OWN config, which lives outside this repo "
        "and outside git -- without a note, a row's `params` understates what "
        "actually produced it (open question 9).",
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
        f"max_new_tokens={row['params']['max_new_tokens']}\n"
        f"gold={row['correct_answer']} answer={row['parsed_answer']} "
        f"finish={row['finish_reason']} completion_tokens={row['completion_tokens']} "
        f"wall_s={row['wall_s']} prompt_chars={row['prompt_chars']} "
        f"recovery={row['recovery_attempted']}/{row['recovered']}"
    )


async def run_main_check(
    model: ModelConfig,
    client: LLMClient,
    questions: list[MedQAQuestion],
    split: str,
    writer: RunWriter,
    done: set[tuple[str, str, str]],
    breaker: EndpointCircuitBreaker,
) -> list[str]:
    """One closed-book completion per sampled question, at the model's own budget.

    Returns the problems worth exiting non-zero on. Rows are already flushed to
    `results.jsonl` as they arrive, and the final summary re-reads that file, so
    a resumed run reports the whole run rather than just this session's slice.
    """
    problems: list[str] = []

    for index, question in enumerate(questions, start=1):
        if (model.name, question.id, "main") in done:
            logger.info("{}/{} already logged -- skipping (resume)", model.name, question.id)
            continue

        if breaker.tripped:
            problems.append(breaker.abort_problem(len(questions) - index + 1))
            break

        prompt = build_answer_prompt(question.question, question.options, None)
        started = time.perf_counter()
        try:
            result = await client.agenerate(prompt)
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
                    prompt=prompt,
                )
            )
            problems.append(f"{model.name}/q{question.id}: {exc}")
            breaker.record_failure(exc)
            continue

        breaker.record_success()

        row = writer.write_row(
            result_row(
                model=model,
                question=question,
                tag="main",
                condition_id=CONDITION_ID,
                split=split,
                prompt=prompt,
                result=result,
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


async def run_caps_check(
    model: ModelConfig,
    client: LLMClient,
    questions: list[MedQAQuestion],
    cap: int,
    split: str,
    writer: RunWriter,
    done: set[tuple[str, str, str]],
    breaker: EndpointCircuitBreaker,
) -> list[str]:
    """Re-run a few questions under a lower `max_new_tokens` ceiling.

    The contingency pass for open question 8: if `model_b`'s 16,384-token budget
    is only ever consumed by chains of thought that never reach `ANSWER:`, the
    honest fix is a smaller ceiling, not a bigger one. Running the same questions
    at 4,096/8,192 measures the cost of that choice instead of arguing about it.
    """
    problems: list[str] = []

    for index, question in enumerate(questions, start=1):
        tag = f"cap{cap}"
        if (model.name, question.id, tag) in done:
            logger.info("{}/{} already logged -- skipping (resume)", model.name, question.id)
            continue

        if breaker.tripped:
            problems.append(breaker.abort_problem(len(questions) - index + 1))
            break

        prompt = build_answer_prompt(question.question, question.options, None)
        started = time.perf_counter()
        try:
            result = await client.agenerate(prompt, max_new_tokens=cap)
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
                    prompt=prompt,
                )
            )
            problems.append(f"{model.name}/q{question.id}[{tag}]: {exc}")
            breaker.record_failure(exc)
            continue

        breaker.record_success()

        row = writer.write_row(
            result_row(
                model=model,
                question=question,
                tag=tag,
                condition_id=f"{CONDITION_ID}_{tag}",
                split=split,
                prompt=prompt,
                result=result,
                overrides={"max_new_tokens": cap},
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


def planned_calls(args: argparse.Namespace, n_sample: int, n_cap_questions: int, n_models: int) -> dict:
    """How many completions each check will fire, so cost is visible up front."""
    plan = {}
    if "main" in args.checks:
        plan["main"] = n_sample * n_models
    if args.caps:
        plan["caps"] = len(args.caps) * n_cap_questions * n_models
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
            f"max_new_tokens={model.max_new_tokens} recovery={model.answer_recovery_max_tokens}"
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


def parse_notes(notes: list[str]) -> dict[str, str]:
    """Turn repeated `--note KEY=VALUE` strings into `context.md`'s note rows.

    The endpoint now supplies sampling parameters this repo never sees or sends
    (oMLX keeps them in `~/.omlx/model_settings.json`, outside git), so
    `params_snapshot()` understates what actually produced a row. A note is how a
    run records what it otherwise could not prove -- e.g. the 4,096-token thinking
    budget that is the reason `model_b` terminates at all (open question 9).
    """
    parsed: dict[str, str] = {}
    for note in notes:
        key, sep, value = note.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--note expects KEY=VALUE, got {note!r}")
        parsed[key.strip()] = value.strip()
    return parsed


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
    cap_questions = sample[: max(0, args.cap_questions)]

    print_plan(
        args,
        config,
        keys,
        sample,
        gold,
        planned_calls(args, len(sample), len(cap_questions), len(keys)),
    )
    if args.dry_run:
        print("\ndry run -- no requests sent")
        return 0

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
                "caps": args.caps or "none",
                "concurrent_models": args.concurrent_models,
                "scoring note": "parsed_answer is recomputed against each question's own "
                "option letters, not GenerationResult.parsed_answer's hardcoded ABCDE",
                **parse_notes(args.note),
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

            # One breaker per arm: both arms now share one oMLX process, so a
            # failure in `model_a`'s calls says nothing about `model_b`'s queue,
            # and abandoning the wrong arm would cost rows the other could still
            # have answered.
            breaker = EndpointCircuitBreaker(model.name)

            async with client:
                blocked = await preflight(model, client)
                if blocked:
                    return [blocked]
                print(f"--- {key}: {model.api_model_name} @ {model.base_url} ---", flush=True)

                if "main" in args.checks:
                    problems += await run_main_check(
                        model, client, sample, config.dev_split, writer, done, breaker
                    )
                for cap in args.caps or []:
                    problems += await run_caps_check(
                        model,
                        client,
                        cap_questions,
                        cap,
                        config.dev_split,
                        writer,
                        done,
                        breaker,
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

    print("\n" + summarize(rows))
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
