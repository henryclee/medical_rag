"""CLI live check of the served models: cost, latency, and the recovery path.

Goes through the real `LLMClient` with the real `config/models.yaml`, so it
exercises the `GenerationResult` path (chain-of-thought extraction plus the
one-shot ANSWER recovery) exactly as the experiment runner will. Run it after
any change to `generation/llm.py`, `generation/prompt.py`, or `models.yaml`,
and before a pilot run, to confirm every endpoint is up, that the configured
models are actually distinct, and what one answer costs.

Two checks:
  latency    closed-book probe questions, per model: wall time, completion
             tokens, tok/s, finish_reason, and correctness against known golds.
  recovery   forces the chain of thought to exhaust `max_new_tokens`, then
             confirms the one-shot recovery call closes with a parsable answer.

Requests are deliberately sequential: `mlx_lm.server` disables batched serving
when a seed is sent, and concurrency would distort the timing measured here.

Requires the endpoints named in config/models.yaml to be running and their API
keys exported (`set -a && source .env && set +a`).

Rows print as JSON and are also appended to `<out-dir>/probe-<stamp>.jsonl`;
every chain of thought is written to `<out-dir>/chains/<stamp>/`, because the
answer line alone rarely shows why a reasoning model went wrong. Exits non-zero
if an endpoint failed or a recovery attempt did not recover.
"""

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from medical_rag.config import ModelConfig, load_config
from medical_rag.generation.llm import GenerationResult, LLMClient, LLMError
from medical_rag.generation.prompt import build_answer_prompt

# Medically unambiguous items whose gold letter varies, so no single option
# position is favoured. These are a plumbing check, not an accuracy estimate:
# four items cannot rank models, they only show the path works end to end.
PROBE_QUESTIONS: list[tuple[str, str, dict[str, str], str]] = [
    (
        "q1",
        "A 45-year-old woman presents with bilateral facial nerve palsy, uveitis, and "
        "elevated serum angiotensin-converting enzyme. Chest imaging shows bilateral hilar "
        "lymphadenopathy. Which of the following is the most likely diagnosis?",
        {
            "A": "Sarcoidosis",
            "B": "Lyme disease",
            "C": "Guillain-Barre syndrome",
            "D": "Multiple sclerosis",
        },
        "A",
    ),
    (
        "q2",
        "A 19-year-old basketball player collapses and dies during a game. Autopsy shows "
        "asymmetric septal hypertrophy with myocyte disarray. A harsh systolic murmur was "
        "heard at the left sternal border that increased with the Valsalva strain. Which of "
        "the following is the most likely diagnosis?",
        {
            "A": "Dilated cardiomyopathy",
            "B": "Hypertrophic obstructive cardiomyopathy",
            "C": "Bicuspid aortic valve stenosis",
            "D": "Atrial septal defect",
        },
        "B",
    ),
    (
        "q3",
        "A 62-year-old man with a 40 pack-year smoking history presents with weight loss, "
        "dry mouth, and proximal muscle weakness. Deep tendon reflexes are absent but "
        "reappear after 10 seconds of maximal voluntary contraction. CT shows a right hilar "
        "mass. Which of the following antibodies is most likely present?",
        {
            "A": "Anti-acetylcholine receptor antibody",
            "B": "Anti-MuSK antibody",
            "C": "Anti-voltage-gated P/Q-type calcium channel antibody",
            "D": "Anti-striated muscle antibody",
        },
        "C",
    ),
    (
        "q4",
        "A 58-year-old man develops rapidly progressive dementia over six weeks with startle "
        "myoclonus and cerebellar ataxia. MRI diffusion-weighted imaging shows cortical "
        "ribboning, and CSF is positive for 14-3-3 protein. Which of the following is the "
        "most likely diagnosis?",
        {
            "A": "Alzheimer disease",
            "B": "Hashimoto encephalopathy",
            "C": "Herpes simplex encephalitis",
            "D": "Creutzfeldt-Jakob disease",
        },
        "D",
    ),
]

# Needs a chain long enough to outrun --caps, which is the point: the first
# completion must end on finish_reason='length' with no ANSWER: line.
RECOVERY_QUESTION_ID = "q3"
RECOVERY_CAPS: tuple[int, ...] = (200, 400)


def _base_row(model: ModelConfig, tag: str) -> dict:
    """Identity fields on every row.

    `api_model` and `endpoint` are recorded so a run's log can prove the two
    arms really are different served models -- a `models.yaml` where both
    entries point at one endpoint silently degenerates the model factor.
    """
    return {
        "model": model.name,
        "api_model": model.api_model_name,
        "endpoint": model.base_url,
        "tag": tag,
    }


def _result_row(model: ModelConfig, tag: str, gold: str, result: GenerationResult) -> dict:
    answer = result.parsed_answer
    return {
        **_base_row(model, tag),
        "gold": gold,
        "answer": answer,
        "correct": answer == gold,
        "unanswered": answer is None,
        "truncated": result.truncated,
        "finish": result.finish_reason,
        # GenerationResult.elapsed_s and completion_tokens already include the
        # recovery call when one fired, so these are true per-question costs.
        "wall_s": round(result.elapsed_s, 1),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "reasoning_chars": len(result.reasoning),
        "content_chars": len(result.content),
        "recovery_attempted": result.recovery_attempted,
        "recovered": result.recovered,
    }


def _error_row(model: ModelConfig, tag: str, exc: Exception, wall_s: float) -> dict:
    return {**_base_row(model, tag), "error": str(exc)[:200], "wall_s": round(wall_s, 1)}


def _dump_chain(
    chains_dir: Path | None,
    model: ModelConfig,
    tag: str,
    gold: str,
    result: GenerationResult,
) -> None:
    """Save the chain of thought, which is where the failure mode actually lives."""
    if chains_dir is None:
        return
    body = result.reasoning or result.content
    if not body.strip():
        return
    chains_dir.mkdir(parents=True, exist_ok=True)
    header = (
        f"# {model.name} {tag}\n"
        f"gold={gold} answer={result.parsed_answer} finish={result.finish_reason} "
        f"completion_tokens={result.completion_tokens} wall_s={result.elapsed_s:.1f} "
        f"recovery={result.recovery_attempted}/{result.recovered}\n\n"
    )
    (chains_dir / f"{model.name}__{tag}.md").write_text(header + body + "\n")


def _emit(row: dict) -> dict:
    print("RESULT " + json.dumps(row), flush=True)
    return row


async def run_latency_check(
    model: ModelConfig,
    client: LLMClient,
    seed: int | None,
    chains_dir: Path | None,
) -> tuple[list[dict], list[str]]:
    """Closed-book cost/latency over the probe set, at the model's own budget."""
    rows: list[dict] = []
    problems: list[str] = []
    for qid, question, options, gold in PROBE_QUESTIONS:
        prompt = build_answer_prompt(question, options, None)
        started = time.perf_counter()
        try:
            result = await client.agenerate(prompt, seed=seed)
        except LLMError as exc:
            rows.append(_emit(_error_row(model, qid, exc, time.perf_counter() - started)))
            problems.append(f"{model.name}/{qid}: {exc}")
            continue
        row = _emit(_result_row(model, qid, gold, result))
        rows.append(row)
        _dump_chain(chains_dir, model, qid, gold, result)
        if row["unanswered"]:
            # A legitimate experimental outcome, not a broken pipeline: the row
            # would be scored incorrect. Warn, but do not fail the check.
            logger.warning(
                "{} returned no parsable answer for {} (finish_reason={}) -- "
                "would be scored incorrect",
                model.name,
                qid,
                result.finish_reason,
            )
    return rows, problems


async def run_recovery_check(
    model: ModelConfig,
    client: LLMClient,
    seed: int | None,
    caps: list[int],
    chains_dir: Path | None,
) -> tuple[list[dict], list[str]]:
    """Starve the chain of thought and confirm the rescue call closes the answer."""
    rows: list[dict] = []
    problems: list[str] = []
    question, options, gold = next(
        (q, o, g) for tag, q, o, g in PROBE_QUESTIONS if tag == RECOVERY_QUESTION_ID
    )
    prompt = build_answer_prompt(question, options, None)
    for cap in caps:
        tag = f"recovery-cap{cap}"
        started = time.perf_counter()
        try:
            result = await client.agenerate(prompt, seed=seed, max_new_tokens=cap)
        except LLMError as exc:
            rows.append(_emit(_error_row(model, tag, exc, time.perf_counter() - started)))
            problems.append(f"{model.name}/{tag}: {exc}")
            continue
        row = _emit({**_result_row(model, tag, gold, result), "cap": cap})
        rows.append(row)
        _dump_chain(chains_dir, model, tag, gold, result)
        if row["recovery_attempted"] and not row["recovered"]:
            problems.append(
                f"{model.name}/{tag}: recovery call fired but still produced no parsable "
                f"ANSWER (answer_recovery_max_tokens={model.answer_recovery_max_tokens})"
            )
    return rows, problems


def summarize(model: ModelConfig, rows: list[dict]) -> str:
    answered = [row for row in rows if "error" not in row]
    closed_book = [row for row in answered if not row["tag"].startswith("recovery")]
    recovery = [row for row in answered if row["tag"].startswith("recovery")]

    lines = [f"=== {model.name} ({model.api_model_name} @ {model.base_url}) ==="]
    if closed_book:
        n = len(closed_book)
        wall = sum(row["wall_s"] for row in closed_book)
        tokens = sum(row["completion_tokens"] or 0 for row in closed_book)
        correct = sum(row["correct"] for row in closed_book)
        unanswered = sum(row["unanswered"] for row in closed_book)
        truncated = sum(row["truncated"] for row in closed_book)
        lines.append(
            f"  closed-book: {correct}/{n} correct, {unanswered} unanswered, "
            f"{truncated} truncated"
        )
        if wall > 0:
            lines.append(
                f"  cost: {wall / n:.1f}s and {tokens / n:.0f} completion tokens per "
                f"question, {tokens / wall:.0f} tok/s"
            )
    if recovery:
        attempted = [row for row in recovery if row["recovery_attempted"]]
        recovered = [row for row in attempted if row["recovered"]]
        lines.append(
            f"  recovery: {len(attempted)}/{len(recovery)} caps truncated, "
            f"{len(recovered)}/{len(attempted)} recovered a parsable answer"
        )
    if all(not line.startswith("  ") for line in lines):
        lines.append("  every call errored -- see the problems list")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/default.yaml", help="path to default.yaml")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="model keys from models.yaml (default: all of them)",
    )
    parser.add_argument(
        "--checks",
        nargs="*",
        choices=("latency", "recovery"),
        default=("latency", "recovery"),
        help="which checks to run",
    )
    parser.add_argument("--seed", type=int, default=None, help="default: each model's own seed")
    parser.add_argument(
        "--caps",
        nargs="*",
        type=int,
        default=list(RECOVERY_CAPS),
        help="max_new_tokens ceilings for the recovery check",
    )
    parser.add_argument(
        "--out-dir", default="outputs/probes", help="where to write rows and chains"
    )
    parser.add_argument(
        "--no-chains", action="store_true", help="do not save chains of thought to disk"
    )
    return parser.parse_args()


async def amain(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    keys = args.models or list(config.models)
    unknown = [key for key in keys if key not in config.models]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; models.yaml defines {sorted(config.models)}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"probe-{stamp}.jsonl"
    chains_dir = None if args.no_chains else out_dir / "chains" / stamp

    all_rows: list[dict] = []
    problems: list[str] = []
    for key in keys:
        model = config.models[key]
        logger.info("probing '{}' -> {} ({})", model.name, model.base_url, model.api_model_name)
        try:
            client = LLMClient(model)
        except LLMError as exc:  # e.g. its API key env var is unset
            problems.append(f"{key}: {exc}")
            continue
        async with client:
            rows, row_problems = ([], [])
            if "latency" in args.checks:
                rows, row_problems = await run_latency_check(model, client, args.seed, chains_dir)
            if "recovery" in args.checks:
                more_rows, more_problems = await run_recovery_check(
                    model, client, args.seed, args.caps, chains_dir
                )
                rows += more_rows
                row_problems += more_problems
        print(summarize(model, rows), flush=True)
        all_rows += rows
        problems += row_problems

    with jsonl_path.open("a", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")
    logger.info("Wrote {} rows to {}", len(all_rows), jsonl_path)
    if chains_dir:
        logger.info("Saved chains of thought under {}", chains_dir)

    if problems:
        logger.error(
            "Check FAILED ({} problem(s)):\n  {}", len(problems), "\n  ".join(problems)
        )
        return 1
    logger.info(
        "Check OK: {} probes across {} model(s), no endpoint or recovery failures",
        len(all_rows),
        len(keys),
    )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(amain(parse_args())))


if __name__ == "__main__":
    main()


