"""Phase 6: raw RAG end-to-end, over the questions Phase 5 answered closed-book.

Runs `Retriever.retrieve()` -> `Retriever.rerank()` -> `build_answer_prompt()` with
context -> `LLMClient` over the *same 20 dev-split questions* Phase 5 answered
closed-book, and pairs the two runs, so the RAG-vs-closed-book delta is a set of
per-question transitions instead of a difference between two accuracy numbers
computed over samples that would not have been the same sample.

The baseline is `outputs/exploration/phase5/20260928T020841Z` (40 rows, every call
HTTP 200, zero truncations, 100% parse rate: `model_a` 8/20, `model_b` 11/20).
`sample_seed=1`, and the ids it recorded are hardcoded here as
`SAMPLE_QUESTION_IDS`. `--dry-run` re-derives the draw with `select_sample()` and
exits non-zero if the pin has drifted: `select_sample()`'s own docstring concedes
a grown pool redraws borderline members, and Phase 5 §8.4 watched an endpoint
swap alone replace 5 of 20 answers with nothing edited in this repo. A pin nobody
checks is a comment, so this one is checked in code, on every dry run.

What this run has to show (PLAN.md Phase 6):

1. The full chain runs over all 20 questions x both arms.
2. `retrieved_chunk_ids` is populated on every row and rerank reorders relative
   to the vector-only order -- recorded per chunk as `vector_rank`, so the churn
   claim is auditable rather than asserted (`summarize_retrieval()`).
3. `rescued` / `distracted` / `both_wrong` transitions vs Phase 5, in
   `pairing.md`, keyed by `question_id`.
4. The context actually reached the prompt: `missing_chunk_evidence()` compares
   `format_context()`'s rendering against the string that was sent, so an empty
   context or a silently truncated one cannot pass.

For `distracted` rows -- the interesting failure, where context moved a correct
answer to a wrong one -- the row also records `answer_chunk_rank`: the prompt
position of the chunk whose text contains the *switched-to* option's wording, or
null if no excerpt names it. That distinguishes "the excerpt talked the model out
of it" from "the model invented the alternative", which is the split Phase 12's
failure classes need and nothing in the pipeline currently measures.

Retrieval is in-process (embedder + cross-encoder on MPS) while the LLM is an HTTP
call to oMLX, so with `--concurrent-questions 1` (default) the embedding pass runs
while nothing is in flight. Raising that overlaps the embedder with served
generation on one GPU: measure before trusting any tok/s number from it, and note
which value produced the rows. `--concurrent-models` additionally puts both arms'
calls for one question in flight together -- and both arms are served by the *same*
oMLX process (config/models.yaml's "ONE SERVER SERVES BOTH" note: `:8080` pools 12
models, they are not two endpoints), so that concurrency buys nothing and makes
per-arm tok/s meaningless. Sequential is the default for both reasons.

    set -a; source .env; set +a
    .venv/bin/python scripts/exploration/raw_rag.py --dry-run
    .venv/bin/python scripts/exploration/raw_rag.py --limit 2      # live smoke
    .venv/bin/python scripts/exploration/raw_rag.py               # full 20 x 2
"""

import argparse
import asyncio
import json
import time
from collections import Counter, namedtuple
from contextlib import AsyncExitStack
from enum import Enum
from pathlib import Path
from typing import Any, Sequence, TextIO

from loguru import logger

from medical_rag.config import ExperimentConfig, ModelConfig, load_config
from medical_rag.data.load_medqa import MedQAQuestion, load_medqa
from medical_rag.generation.llm import LLMClient, LLMError
from medical_rag.generation.prompt import build_answer_prompt, format_context
from medical_rag.retrieval.embedder import Embedder
from medical_rag.retrieval.index import load_index
from medical_rag.retrieval.retriever import RetrievedChunk, Retriever

from _common import (
    DEFAULT_RUN_ROOT,
    DEFAULT_SAMPLE_SEED,
    EndpointCircuitBreaker,
    RunWriter,
    chunk_match_rank,
    error_row,
    missing_chunk_evidence,
    parse_notes,
    result_row,
    select_by_ids,
    select_sample,
    summarize,
    summarize_retrieval,
    warn_on_degenerate_sample,
)

PHASE = "phase6"
CONDITION_ID = "phase6_raw_rag"
# The Phase 5 run these ids were pinned from and whose rows this run pairs
# against. Named here because a delta is meaningless without naming what it is a
# delta against; `--baseline-dir` can point elsewhere, but the default and this
# constant have to agree or `pairing.md` lies.
BASELINE_RUN = "phase5/20260928T020841Z"
DEFAULT_BASELINE_DIR = f"outputs/exploration/{BASELINE_RUN}"

# Exactly the ids `select_sample(dev_pool, 20, sample_seed=1)` returns -- i.e. the
# block `outputs/exploration/phase5/20260928T020841Z/context.md` records. This
# list, not the draw, is what Phase 6 runs; `--dry-run` re-derives it.
SAMPLE_QUESTION_IDS: list[str] = [
    "1312", "2391", "3998", "4002", "5209", "5949", "6205", "6264", "6727",
    "6753", "6771", "7159", "7502", "7553", "8966", "9293", "9473", "9572",
    "9597", "10064",
]

# Rough throughput band measured on these endpoints (Phase 4/5). Used only to
# print a cost bound before firing ~40 long completions; the run's own tok/s is
# reported from the rows it wrote.
TPS_PESSIMISTIC = 40.0
TPS_OPTIMISTIC = 80.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/default.yaml", help="path to default.yaml")
    parser.add_argument(
        "--models", nargs="*", default=None, help="model keys (default: every model in config)"
    )
    parser.add_argument(
        "--split", default=None,
        help="MedQA split to resolve ids against (default: config.dev_split). The pinned "
        "ids came from dev_split; the full split is a superset, so either resolves them.",
    )
    parser.add_argument(
        "--question-ids", nargs="*", default=None,
        help=f"override the pinned sample (default: the {len(SAMPLE_QUESTION_IDS)} ids in "
        "SAMPLE_QUESTION_IDS). An id that does not resolve is a hard error listing every "
        "missing id, never a silent skip that shrinks the paired table.",
    )
    parser.add_argument(
        "--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED,
        help="seed the --dry-run re-derivation checks the pin against (default: the seed "
        "Phase 5 used; passing anything else is how you watch the pin fail)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="run only the first N pinned questions (live smoke: --limit 2)",
    )
    parser.add_argument(
        "--index-dir", default=None,
        help="LanceDB index directory (default: data/index/<config.retrieval.corpus>)",
    )
    parser.add_argument(
        "--top-k-retrieve", type=int, default=None,
        help="vector candidates to pull (default: config.retrieval.top_k_retrieve)",
    )
    parser.add_argument(
        "--top-k-rerank", type=int, default=None,
        help="chunks to keep after rerank (default: config.retrieval.top_k_rerank)",
    )
    parser.add_argument(
        "--baseline-dir", default=DEFAULT_BASELINE_DIR,
        help="run dir to pair against, containing results.jsonl (default: "
        f"{DEFAULT_BASELINE_DIR})",
    )
    parser.add_argument(
        "--concurrent-questions", type=int, default=1,
        help="questions in flight at once (default: 1 = fully sequential). >1 overlaps the "
        "in-process embedder/reranker with the served models' generation on one GPU",
    )
    parser.add_argument(
        "--concurrent-models", action="store_true",
        help="run both arms' calls for one question in flight together (default: sequential)",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_RUN_ROOT,
        help="run-artifact root (default: %(default)s, per _common.RunWriter)",
    )
    parser.add_argument(
        "--no-chains", action="store_true", help="do not save chains of thought"
    )
    parser.add_argument(
        "--resume", default=None, metavar="RUN_DIR",
        help="re-open an existing run dir and skip the (model, question_id, tag) rows it "
        "already has (error rows do not count as done)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="check the pin against a fresh re-derivation, print the retrieval/prompt plan "
        "and the baseline coverage, and make no calls of any kind -- the index is not "
        "loaded and no endpoint is contacted",
    )
    parser.add_argument(
        "--note", action="append", default=[], metavar="KEY=VALUE",
        help="record a fact about this run in context.md; repeatable. Mandatory in spirit "
        "for this phase: oMLX applies per-model sampling from its own config outside git, "
        "and a delta against Phase 5 only means anything if the endpoint that served both "
        "is shown to be the same one (Phase 5 section 8.4).",
    )
    return parser.parse_args()


# --- the closed-book baseline this run is a delta against -------------------


def load_baseline(run_dir: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    """`(model, question_id)` -> the closed-book row that answered that question.

    Pairing is only honest if the pair exists. Phase 5's first run
    (`20260927T162758Z`) carries 48 rows, two of which are `model_b` questions
    that never closed, and pointing `--baseline-dir` at a run with a different
    sample would silently yield a table full of `unpaired`. So the rows are read
    here, `tag != "main"` and error rows are dropped (an error row records an
    attempt, not an outcome), and the plan print reports the coverage actually
    found rather than the coverage hoped for.
    """
    path = Path(run_dir)
    if not path.is_file():
        path = path / "results.jsonl"
    if not path.exists():
        raise SystemExit(
            f"no results.jsonl under {run_dir!r} -- pass --baseline-dir a run directory "
            f"that answered these questions (default: {DEFAULT_BASELINE_DIR})"
        )

    baseline: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping unparseable line in {}", path)
                continue
            if row.get("tag") != "main" or "error" in row:
                continue
            baseline[(row.get("model"), row.get("question_id"))] = row

    if not baseline:
        raise SystemExit(f"{path} contains no usable main-tag rows -- nothing to pair against")
    return baseline


def endpoint_drift(
    baseline: dict[tuple[str, str], dict[str, Any]],
    config: ExperimentConfig,
    keys: list[str],
) -> list[str]:
    """Complain if an arm would now be served somewhere else than in the baseline.

    Phase 5 §8.4: swapping the serving stack for the *same* model and *same* 20
    questions moved `model_a` from 50% to 70%. A Phase 6 delta computed across two
    different endpoints therefore measures the endpoint, not the retrieval, and
    the rows would not show it without joining two run directories by hand.
    Checking at plan time means the run says so before it costs 40 completions.
    """
    problems: list[str] = []
    for key in keys:
        model = config.models[key]
        prior = {
            (row.get("endpoint"), row.get("api_model"))
            for (name, _question_id), row in baseline.items()
            if name == model.name
        }
        if not prior:
            problems.append(
                f"{key}: the baseline has no rows for model '{model.name}' -- every paired "
                f"cell would read 'unpaired' (baseline dir: {DEFAULT_BASELINE_DIR})"
            )
            continue
        if len(prior) > 1:
            problems.append(
                f"{key}: the baseline itself spans {len(prior)} endpoints "
                f"{sorted(prior, key=str)} -- it is not one run, so it cannot anchor a delta"
            )
        for endpoint, api_model in sorted(prior, key=str):
            if (endpoint, api_model) != (model.base_url, model.api_model_name):
                problems.append(
                    f"{key}: baseline answered at {endpoint} as '{api_model}', this run would "
                    f"send to {model.base_url} as '{model.api_model_name}' -- the delta would "
                    "measure the endpoint, not the context (Phase 5 §8.4)"
                )
    return problems


OUTCOMES = (
    "both_right",
    "rescued",
    "distracted",
    "both_wrong",
    "newly_answered",
    "newly_unanswered",
    "both_unanswered",
    "unpaired",
)


def row_letter(row: dict[str, Any] | None) -> str:
    """The letter a row answered, or `?` when it never got to one."""
    return "?" if not row else (row.get("parsed_answer") or "?")


def outcome_of(rag: dict[str, Any], base: dict[str, Any] | None) -> str:
    """The closed-book -> RAG transition for one (model, question) pair.

    `rescued` and `distracted` are the two the phase exists to count, so the
    answerability transitions are kept separate from them rather than folded into
    `both_wrong`: a question `model_b` refused closed-book and then answered with
    context is not the same event as one it answered right both ways, and lumping
    them together would let a formatting change masquerade as a retrieval effect.
    `unpaired` (no baseline row) is reported rather than skipped -- a table that
    silently has 17 rows where it claims 20 is how a study loses an afternoon.
    """
    if base is None:
        return "unpaired"
    if rag.get("unanswered"):
        return "both_unanswered" if base.get("unanswered") else "newly_unanswered"
    if base.get("unanswered"):
        return "newly_answered"
    if base.get("is_correct"):
        return "both_right" if rag.get("is_correct") else "distracted"
    return "rescued" if rag.get("is_correct") else "both_wrong"


def _question_sort_key(question_id: Any) -> tuple[int, int, str]:
    """Dataset order where ids are numeric, so the table matches the pool order."""
    text = str(question_id or "")
    return (0, int(text), "") if text.isdigit() else (1, 0, text)


def pairing_markdown(rows: list[dict[str, Any]], model_names: list[str]) -> str:
    """The per-question transition table, as `pairing.md`.

    Written out rather than left on stdout because these are the rows a human has
    to read to write the findings: the `distracted` cells in particular are only
    interpretable next to `answer_chunk_rank` (was the wrong option even named in
    the excerpts?), and nobody should have to re-run the script -- or re-pay for
    the completions -- to see them.
    """
    lines = [
        f"# Phase 6 pairing -- RAG vs closed-book (baseline {BASELINE_RUN})",
        "",
        "`outcome` is the closed-book -> RAG transition for the same (model, question).",
        "`cb`/`rag` are the letters each arm answered (`?` = no parsable answer). `rank`/",
        "`grank` are the prompt positions of the excerpts containing the *answered* and the",
        "*gold* option's wording (`-` = no excerpt names it) -- a `distracted` row with",
        "`rank=-` is a model that invented the alternative, not one the excerpts argued",
        "into it, and a `both_wrong` row with `grank=-` is a retrieval miss, not a model miss.",
        "",
    ]
    for name in model_names:
        mine = [row for row in rows if row.get("model") == name and row.get("tag") == "main"]
        if not mine:
            continue
        tally = Counter(row.get("outcome", "unpaired") for row in mine)
        base_correct = sum(1 for row in mine if row.get("base_correct"))
        rag_correct = sum(
            1 for row in mine if not row.get("unanswered") and row.get("is_correct")
        )
        net = tally["rescued"] - tally["distracted"]
        lines += [
            f"## `{name}` ({len(mine)} question(s))",
            "",
            "| outcome | n |",
            "| --- | --- |",
        ]
        lines += [f"| {outcome} | {tally[outcome]} |" for outcome in OUTCOMES if tally[outcome]]
        lines += [
            "",
            f"closed-book correct {base_correct} -> RAG correct {rag_correct} of {len(mine)} "
            f"(net rescued−distracted {net:+d}). n={len(mine)} is plumbing evidence, not an "
            "accuracy estimate.",
            "",
            "| question | gold | cb | rag | outcome | rank | grank | chunks |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in sorted(mine, key=lambda row: _question_sort_key(row.get("question_id"))):
            lines.append(
                f"| {row.get('question_id')} | {row.get('correct_answer') or '?'} "
                f"| {row.get('base_answer') or '-'} | {row_letter(row)} "
                f"| {row.get('outcome')} | {dash(row.get('answer_chunk_rank'))} "
                f"| {dash(row.get('gold_chunk_rank'))} "
                f"| {len(row.get('retrieved_chunk_ids') or [])} |"
            )
        lines.append("")
    return "\n".join(lines)


def dash(value: Any) -> str:
    """An explicit `-` for a missing rank, so a column never reads as a blank cell."""
    return "-" if value is None else str(value)



def chain_header(row: dict[str, Any]) -> str:
    """The provenance stamped atop a saved chain -- here, which chunks were in the prompt.

    A closed-book chain is legible on its own; a RAG chain is not, because its
    answer depends on text the file does not otherwise contain. The chunk ids make
    the answer reconstructible from `prompts/`, which is what makes a `distracted`
    case reviewable a week later.
    """
    chunks = ", ".join(row.get("retrieved_chunk_ids") or []) or "none"
    return (
        f"# {row['model']} q{row['question_id']} [{row['tag']}] "
        f"max_new_tokens={row['params']['max_new_tokens']}\n"
        f"gold={row['correct_answer']} answer={row['parsed_answer']} "
        f"finish={row['finish_reason']} completion_tokens={row['completion_tokens']} "
        f"wall_s={row['wall_s']} context_chars={row.get('context_chars')} "
        f"outcome={row.get('outcome')} recovery={row['recovery_attempted']}/{row['recovered']}\n"
        f"chunks={chunks}"
    )


class Preflight(Enum):
    """Whether an arm may spend its completions on the endpoint it was given."""

    PROCEED = "proceed"
    BLOCK = "block"


async def preflight(model: ModelConfig, client: LLMClient) -> tuple[Preflight, list[str]]:
    """Confirm the endpoint answers and serves the model `models.yaml` names.

    Returns a decision and the lines to print, rather than logging or raising: this
    runs once per arm before ~20 completions each, and the two failure modes want
    different handling. A dead endpoint blocks the whole run, but a misnamed
    `api_model_name` only invalidates *that* arm -- dropping the arm and pairing
    the other is a smaller loss than a run whose rows came from some other model.
    Phase 5 §8.4 is the reason `BLOCK` refuses rather than warns: an endpoint
    serving a different model produces rows that look identical and measure
    something else.
    """
    try:
        served = await client.client.models.list()
    except Exception as exc:  # noqa: BLE001 - any failure here means "do not send"
        return Preflight.BLOCK, [
            f"{model.name}: endpoint {model.base_url} is not answering -- "
            f"{type(exc).__name__}: {str(exc)[:150]}. Start the server that serves this model "
            "(environment.md) or repoint config/models.yaml."
        ]

    ids = sorted({item.id for item in served.data if item.id})
    if model.api_model_name not in ids:
        return Preflight.BLOCK, [
            f"{model.name}: {model.base_url} serves {ids} but models.yaml names "
            f"'{model.api_model_name}' -- every completion would be rejected or, worse, "
            "answered by a different model than this arm claims"
        ]
    return Preflight.PROCEED, [
        f"{model.name}: {model.base_url} serves {model.api_model_name} (offering {', '.join(ids)})"
    ]


# --- retrieval --------------------------------------------------------------


def resolve_index_dir(
    config: ExperimentConfig, index_dir: str | Path | None = None
) -> Path:
    """Where the LanceDB table lives, given `--index-dir` or the config's corpus.

    `RetrievalConfig` names a `corpus`, not a directory, so the mapping is here
    rather than in the config -- but it is one function used by both the plan print
    and the loader so the number in `context.md` cannot disagree with the table
    actually queried.
    """
    if index_dir:
        return Path(index_dir)
    return Path("data/index") / config.retrieval.corpus


def build_retriever(
    config: ExperimentConfig, index_dir: str | Path | None = None
) -> tuple[Retriever, dict[str, Any]]:
    """Load the Phase 2 index and wire the Phase 3 retriever onto it.

    Returns the retriever and the *static* retrieval fields every row of this run
    shares -- index path, row count, the two encoder names, the device. Taken once
    here rather than re-read per question, because `Embedder` exposes no
    `model_name` attribute and a row that silently recorded `None` for the
    embedding model would look like a config change rather than a missing getter.

    Nothing here rebuilds anything: `build_index.py` records a corpus hash, so a
    silently rebuilt index would be a second, invisible variable under a run whose
    whole claim is a delta against Phase 5.
    """
    path = resolve_index_dir(config, index_dir)
    if not path.exists():
        raise SystemExit(
            f"no index at {path} -- run .venv/bin/python scripts/build_index.py first "
            "(environment.md §5.1: the index is built once and reused by every phase)"
        )
    table = load_index(path)
    retriever = Retriever(
        Embedder(model_name=config.retrieval.embedding_model),
        table,
        config.retrieval.reranker_model,
    )
    static = {
        "index_dir": str(path),
        "index_rows": table.count_rows(),
        "embedding_model": config.retrieval.embedding_model,
        "reranker_model": config.retrieval.reranker_model,
        "device": retriever.embedder.device,
    }
    return retriever, static



def retrieve_pass(
    retriever: Retriever,
    question: MedQAQuestion,
    top_k_retrieve: int,
    top_k_rerank: int,
    static: dict[str, Any],
) -> tuple[list[RetrievedChunk], str, dict[str, Any]]:
    """One retrieval for one question, plus the measurements Phase 6 claims.

    Returns `(kept, context, meta)`. Called once per question, before any arm is
    asked, because both arms must see the *same* context for a paired comparison
    to mean anything: retrieving per arm would let `model_b`'s prompt differ from
    `model_a`'s on the same question, and the delta would then contain a different
    retrieval rather than a different model.

    Two traps here are worth stating, because both were measured rather than
    assumed from the signatures:

    * `Retriever.rerank()` returns copies with `score` *overwritten* by the
      cross-encoder logit, so the cosine similarity only exists on the pre-rerank
      list. It is captured there, per chunk, into `vector_score`; a row that
      reported the post-rerank `score` twice would make reranking look like a
      rescaling of the vector scores, and Phase 10's "reranking adds no
      information" argument would be made out of an artifact.
    * `vector_rank` is likewise captured pre-rerank, since the final list alone
      cannot distinguish a reorder from a pass-through -- and `moved`/`promoted`
      in `summarize_retrieval()` are read straight off it.

    `top_k_retrieve` is passed explicitly even though it equals
    `config.retrieval.top_k_retrieve` today, so the recorded number is the number
    used rather than one inferred from a default.
    """
    started = time.perf_counter()
    candidates = retriever.retrieve(question.question, top_k=top_k_retrieve)
    retrieve_s = round(time.perf_counter() - started, 3)
    vector = {chunk.chunk_id: (rank, chunk.score) for rank, chunk in enumerate(candidates, 1)}

    started = time.perf_counter()
    kept = retriever.rerank(question.question, candidates, top_k=top_k_rerank)
    rerank_s = round(time.perf_counter() - started, 3)

    chunks = []
    for rank, chunk in enumerate(kept, start=1):
        vector_rank, vector_score = vector.get(chunk.chunk_id, (None, None))
        chunks.append(
            {
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "prompt_rank": rank,
                "vector_rank": vector_rank,
                "vector_score": vector_score,
                "rerank_score": chunk.score,
                "content_chars": len(chunk.content),
            }
        )

    context = format_context(kept)
    meta = {
        "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
        "chunks": chunks,
        "context_chars": len(context),
        "params": {
            **static,
            "top_k_retrieve": top_k_retrieve,
            "top_k_rerank": top_k_rerank,
            "retrieve_s": retrieve_s,
            "rerank_s": rerank_s,
            "candidates": len(candidates),
            "kept": len(kept),
            "vector_top1_score": candidates[0].score if candidates else None,
            "vector_last_score": candidates[-1].score if candidates else None,
            "rerank_top_score": kept[0].score if kept else None,
        },
    }
    return kept, context, meta



def prompt_artifact(question: MedQAQuestion, prompt: str, meta: dict[str, Any]) -> str:
    """The exact string handed to the model, with the ranking that produced it.

    PLAN.md asks for a prompt artifact and no reviewer trusts a printed
    description of a prompt. `vector_score` and `rerank_score` are printed side by
    side because they are different quantities -- cosine similarity vs a
    cross-encoder logit -- and a row that shows only the second makes reranking
    look like it rescaled the scores.
    """
    lines = [
        f"### q{question.id} -- CONDITION {CONDITION_ID} -- the EXACT string sent",
        "",
        "```",
        prompt,
        "```",
        "",
        f"- question: {question.question}",
        f"- gold: {question.answer_idx} | options: "
        f"{json.dumps(question.options, ensure_ascii=False)}",
        f"- context_chars={meta['context_chars']} retrieve_s={meta['params']['retrieve_s']} "
        f"rerank_s={meta['params']['rerank_s']} "
        f"(index `{meta['params']['index_dir']}`, {meta['params']['index_rows']} rows, "
        f"top_k {meta['params']['top_k_retrieve']}->{meta['params']['top_k_rerank']}, "
        f"device {meta['params']['device']})",
    ]
    for chunk in meta["chunks"]:
        lines.append(
            f"  - [{chunk['prompt_rank']}] {chunk['chunk_id']} \"{chunk['title']}\" "
            f"vector_rank={chunk['vector_rank']} "
            f"vector={_fmt_score(chunk['vector_score'])} "
            f"rerank={_fmt_score(chunk['rerank_score'])}"
        )
    return "\n".join(lines)


def _fmt_score(value: float | None) -> str:
    """Four decimals, or an explicit dash -- never a silent 0.0000 for a gap."""
    return "-" if value is None else f"{value:.4f}"


# --- the plan: what a reviewer needs to see before 40 completions are spent --


def pin_check(
    questions: Sequence[MedQAQuestion], ids: Sequence[str], sample_seed: int
) -> tuple[bool, list[str]]:
    """Re-derive Phase 5's draw and say whether the hardcoded pin still matches it.

    The run itself uses the pinned ids regardless of what this says -- `--limit`
    and `--question-ids` exist, and a check that overrode the sample would defeat
    the point of pinning. What the check buys is visibility: `select_sample()`'s
    docstring concedes a grown pool redraws borderline members, so "same 20
    questions" is a claim about a list, not about a seed, and the only way to keep
    it honest is to recompute the seed's draw every time and print the diff.
    """
    drawn = [question.id for question in select_sample(questions, len(ids), sample_seed)]
    if drawn == list(ids):
        return True, [
            f"pin OK: select_sample(pool={len(questions)}, size={len(ids)}, "
            f"seed={sample_seed}) still returns exactly these ids"
        ]

    shared = set(drawn) & set(ids)
    return False, [
        f"PIN DRIFT: select_sample(pool={len(questions)}, size={len(ids)}, seed={sample_seed}) "
        f"now returns {len(shared)}/{len(ids)} of the pinned ids",
        f"  added:   {', '.join(sorted(set(drawn) - set(ids))) or '-'}",
        f"  dropped: {', '.join(sorted(set(ids) - set(drawn))) or '-'}",
        "  this run still answers the pinned ids (that is the point), but a fresh draw is "
        "no longer Phase 5's sample -- do not swap the seed to make the check pass. "
        "Re-pin deliberately and record the re-pin in experiments/phase6/FINDINGS.md.",
    ]


def answer_chunk_rank(
    question: MedQAQuestion, letter: str | None, chunks: Sequence[RetrievedChunk]
) -> int | None:
    """Prompt position of the excerpt whose text names the option `letter`.

    Read with `chunk_match_rank`'s own caveat, which is why it is a rank and not a
    boolean: the test under-counts an excerpt that supports an option without
    quoting its wording, and over-counts one that names it in order to rule it
    out. Its value is the *contrast* -- `distracted` with `rank=-` versus
    `distracted` with `rank=2` is the difference between a model that invented an
    answer and one the excerpts argued into it.
    """
    if not letter:
        return None
    option = question.options.get(letter)
    return chunk_match_rank(option or "", chunks)


Arm = namedtuple("Arm", "model client breaker")


def print_plan(
    args: argparse.Namespace,
    config: ExperimentConfig,
    keys: list[str],
    sample: list[MedQAQuestion],
    pin: tuple[bool, list[str]],
    gold: Counter,
    baseline: dict[tuple[str, str], dict[str, Any]],
    retrieval: dict[str, Any],
    drift: list[str],
    calls: int,
) -> None:
    """Echo the pin, the arms, the retrieval settings, and the cost of proceeding.

    `--dry-run` prints this and stops, and a live run prints it before firing its
    first completion, because everything a later finding would have to reconstruct
    -- which endpoint, which k, which 20 ids, whether the pin still holds -- is
    cheaper to read here than to recover from two run directories afterwards.
    """
    split = args.split or config.dev_split
    print(f"phase={PHASE} condition_id={CONDITION_ID} split={split}")
    print(f"pinned sample ({len(sample)} of {len(SAMPLE_QUESTION_IDS)} question(s), "
          f"from {BASELINE_RUN}):")
    print("  " + ",".join(question.id for question in sample))
    print(f"  gold letters: {dict(sorted(gold.items()))}")
    for message in pin[1]:
        print(f"  {message}")
    print()

    print("arms:")
    for key in keys:
        model = config.models[key]
        print(
            f"  {key}: {model.api_model_name} @ {model.base_url} temperature={model.temperature} "
            f"max_new_tokens={model.max_new_tokens} "
            f"recovery={model.answer_recovery_max_tokens}"
        )
    print()

    print("retrieval (one shared pass per question; both arms see the same prompt):")
    # The plan prints before the index loads (a --dry-run must not pay for it), so
    # rows/device are still None here. Say so: a bare "-" next to `rows=` reads as
    # "empty index", which is the one thing this line must not imply.
    print(
        f"  index: {retrieval['index_dir']} rows={dash(retrieval['index_rows'])} "
        f"device={dash(retrieval['device'])}"
        + ("" if retrieval["index_rows"] else "   (resolves when the index loads)")
    )
    print(
        f"  embedder: {retrieval['embedding_model']}  reranker: {retrieval['reranker_model']}"
    )
    print(
        f"  top_k: retrieve {retrieval['top_k_retrieve']} -> keep {retrieval['top_k_rerank']}"
        f" (config default {config.retrieval.top_k_retrieve}->"
        f"{config.retrieval.top_k_rerank})"
    )
    print()

    print(f"baseline: {args.baseline_dir} ({len(baseline)} main row(s))")
    for key in keys:
        name = config.models[key].name
        paired = sum(1 for question in sample if (name, question.id) in baseline)
        unpaired = [
            question.id for question in sample if (name, question.id) not in baseline
        ]
        print(
            f"  {name}: {paired}/{len(sample)} of this run's questions paired"
            + (f"; unpaired: {', '.join(unpaired)}" if unpaired else "")
        )
    for line in drift:
        print(f"  DRIFT: {line}")
    print()

    print(f"planned completions: {calls} ({len(keys)} arm(s) x {len(sample)} question(s))")
    for key in keys:
        ceiling = config.models[key].max_new_tokens
        print(
            f"  worst case {key}: {ceiling / TPS_PESSIMISTIC / 60:.1f}-"
            f"{ceiling / TPS_OPTIMISTIC / 60:.1f} min/question if every completion ran to "
            "its ceiling -- a bound, not an estimate"
        )
    if args.concurrent_questions > 1 or args.concurrent_models:
        print(
            f"  concurrency: questions={args.concurrent_questions} "
            f"models_together={args.concurrent_models} -- the in-process embedder/reranker "
            "will overlap the served models' generation on one GPU, so read this run's "
            "tok/s as interference, not as throughput"
        )


async def run_question(
    question: MedQAQuestion,
    arms: dict[str, Arm],
    retriever: Retriever,
    retrieval: dict[str, Any],
    baseline: dict[tuple[str, str], dict[str, Any]],
    writer: RunWriter,
    done: set[tuple[str, str, str]],
    prompts: TextIO,
    args: argparse.Namespace,
    split: str,
) -> list[str]:
    """Retrieve once for `question`, then have every pending arm answer that prompt.

    Returns problem strings. Retrieving once and handing both arms the identical
    string is the point of the function: two calls on two different rankings would
    make a `rescued` row ambiguous about whether the context or the ranking caused
    the flip. A question retrieval cannot furnish evidence for is a problem, not an
    unanswered row -- scoring a context-free prompt as RAG would read back as a
    model failure.
    """
    pending = [
        key for key, arm in arms.items() if (arm.model.name, question.id, "main") not in done
    ]
    if not pending:
        logger.info("q{} already logged for every arm -- skipping", question.id)
        return []

    try:
        chunks, _context, meta = await asyncio.to_thread(
            retrieve_pass,
            retriever,
            question,
            retrieval["top_k_retrieve"],
            retrieval["top_k_rerank"],
            retrieval,
        )
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return [f"q{question.id}: retrieval failed ({type(exc).__name__}): {exc}"]
    if not chunks:
        return [
            f"q{question.id}: retrieval returned 0 chunks from {retrieval['index_dir']} "
            f"({dash(retrieval['index_rows'])} rows) -- refusing to score a context-free "
            "prompt as RAG"
        ]

    prompt = build_answer_prompt(question.question, question.options, chunks)
    gaps = missing_chunk_evidence(prompt, chunks)
    if gaps:
        return [
            f"q{question.id}: retrieved evidence did not reach the prompt: {gap}" for gap in gaps
        ]
    prompts.write(prompt_artifact(question, prompt, meta) + "\n\n")
    prompts.flush()

    gold_rank = answer_chunk_rank(question, question.answer_idx, chunks)

    async def ask(key: str) -> list[str]:
        arm = arms[key]
        base_row = baseline.get((arm.model.name, question.id))
        started = time.perf_counter()
        try:
            result = await arm.client.agenerate(prompt)
        except LLMError as exc:
            writer.write_row(
                error_row(
                    model=arm.model,
                    question=question,
                    tag="main",
                    condition_id=CONDITION_ID,
                    split=split,
                    exc=exc,
                    wall_s=time.perf_counter() - started,
                    prompt=prompt,
                )
            )
            arm.breaker.record_failure(exc)
            return [f"{arm.model.name}/q{question.id}: {exc}"]
        arm.breaker.record_success()

        row = result_row(
            model=arm.model,
            question=question,
            tag="main",
            condition_id=CONDITION_ID,
            split=split,
            prompt=prompt,
            result=result,
            retrieval=meta,
        )
        row["base_answer"] = "-" if base_row is None else row_letter(base_row)
        row["base_correct"] = None if base_row is None else bool(base_row.get("is_correct"))
        row["outcome"] = outcome_of(row, base_row)
        row["answer_changed"] = bool(
            base_row
            and not base_row.get("unanswered")
            and not row["unanswered"]
            and base_row.get("parsed_answer") != row["parsed_answer"]
        )
        row["answer_chunk_rank"] = answer_chunk_rank(question, row["parsed_answer"], chunks)
        row["gold_chunk_rank"] = gold_rank
        writer.write_row(row)
        if not args.no_chains:
            writer.write_chain(
                arm.model, question.id, chain_header(row), result.reasoning or result.content
            )
        logger.info(
            "{} q{} -> {} (context={} chars, retrieve={:.2f}s, rerank={:.2f}s)",
            arm.model.name,
            question.id,
            row_letter(row),
            meta["context_chars"],
            meta["params"]["retrieve_s"],
            meta["params"]["rerank_s"],
        )
        return []

    if args.concurrent_models and len(pending) > 1:
        groups = await asyncio.gather(*(ask(key) for key in pending))
        return [problem for group in groups for problem in group]
    problems: list[str] = []
    for key in pending:
        problems += await ask(key)
    return problems


async def amain(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    keys = args.models or list(config.models)
    unknown = sorted(set(keys) - set(config.models))
    if unknown:
        raise SystemExit(
            f"--models {','.join(unknown)} not in models.yaml "
            f"(known: {', '.join(config.models)})"
        )

    split = args.split or config.dev_split
    ids = list(args.question_ids or SAMPLE_QUESTION_IDS)
    pool = load_medqa(split)
    try:
        sample = select_by_ids(pool, ids)
    except KeyError as exc:
        raise SystemExit(str(exc.args[0]) if exc.args else exc) from exc
    if args.limit is not None:
        sample = sample[: args.limit]
    if not sample:
        raise SystemExit("no questions left to answer after --limit")
    gold = warn_on_degenerate_sample(sample)

    baseline = load_baseline(args.baseline_dir)
    drift = endpoint_drift(baseline, config, keys)
    pin = pin_check(pool, ids, args.sample_seed)

    top_k_retrieve = args.top_k_retrieve or config.retrieval.top_k_retrieve
    top_k_rerank = args.top_k_rerank or config.retrieval.top_k_rerank
    if top_k_rerank > top_k_retrieve:
        raise SystemExit(
            f"--top-k-rerank {top_k_rerank} exceeds --top-k-retrieve {top_k_retrieve} "
            "(rerank can only narrow, never invent candidates)"
        )
    retrieval = {
        "index_dir": str(resolve_index_dir(config, args.index_dir)),
        "index_rows": None,
        "device": None,
        "embedding_model": config.retrieval.embedding_model,
        "reranker_model": config.retrieval.reranker_model,
        "top_k_retrieve": top_k_retrieve,
        "top_k_rerank": top_k_rerank,
    }

    print_plan(
        args, config, keys, sample, pin, gold, baseline, retrieval, drift,
        len(keys) * len(sample),
    )

    if args.dry_run:
        print(
            "\ndry run -- no index load, no preflight, no requests. Would write to "
            f"{Path(args.output_dir) / PHASE}/<UTC-stamp>/ (results.jsonl, chains/, prompts.md, "
            "pairing.md, context.md)"
        )
        if drift or not pin[0]:
            print("exit 1: the run would compare itself against a baseline it cannot pair with.")
            return 1
        return 0

    try:
        retriever, static = await asyncio.to_thread(
            build_retriever, config, retrieval["index_dir"]
        )
    except Exception as exc:  # noqa: BLE001 - reported as the reason nothing ran
        raise SystemExit(
            f"retriever unavailable for {retrieval['index_dir']}: {type(exc).__name__}: {exc}"
        ) from exc
    retrieval |= static
    print(
        f"\nretrieval ready: {retrieval['index_rows']} rows on {retrieval['device']} (the "
        "first question's retrieve_s carries the cold-start cost)\n",
        flush=True,
    )

    problems: list[str] = []
    async with AsyncExitStack() as stack:
        arms: dict[str, Arm] = {}
        for key in keys:
            model = config.models[key]
            client = await stack.enter_async_context(LLMClient(model))
            decision, lines = await preflight(model, client)
            for line in lines:
                print(f"  {line}")
            if decision == Preflight.BLOCK:
                problems.append(
                    f"{model.name}: refused to send -- the endpoint is not the one models.yaml "
                    "names, so any delta against Phase 5 would be unattributable"
                )
                continue
            arms[key] = Arm(model=model, client=client, breaker=EndpointCircuitBreaker(model.name))
        print()

        if not arms:
            problems.append("no arm passed preflight -- nothing to answer with")
            for problem in problems:
                print(f"PROBLEM: {problem}")
            return 1

        with RunWriter(
            PHASE, root=args.output_dir, run_dir=args.resume, save_chains=not args.no_chains
        ) as writer:
            done = writer.completed_keys()
            if done:
                print(f"resume: {len(done)} row(s) already in {writer.dir}, skipping them\n")
            paired = sum(
                1
                for arm in arms.values()
                for question in sample
                if (arm.model.name, question.id) in baseline
            )
            writer.write_context(
                config=config,
                model_keys=keys,
                questions=sample,
                split=split,
                sample_seed=args.sample_seed,
                extra={
                    "condition_id": CONDITION_ID,
                    "baseline run": f"{args.baseline_dir} -- Phase 5 closed-book, "
                    f"{len(baseline)} main row(s) loaded, {paired}/{len(arms) * len(sample)} "
                    "of this run's (model, question) pairs paired",
                    "sample selection": f"PINNED ids copied from {BASELINE_RUN}'s context.md, "
                    "not a fresh draw; this run re-derives the seed's draw and prints the diff",
                    "index": f"{retrieval['index_dir']} ({retrieval['index_rows']} rows, "
                    f"{retrieval['embedding_model']}, {retrieval['device']})",
                    "retrieval": f"top_k_retrieve={retrieval['top_k_retrieve']} -> "
                    f"top_k_rerank={retrieval['top_k_rerank']}, reranker="
                    f"{retrieval['reranker_model']}; one shared pass per question, identical "
                    "prompt for both arms",
                    "concurrency": f"questions={args.concurrent_questions} "
                    f"models_together={args.concurrent_models}",
                    "artifacts": "prompts.md (the exact string sent, per question), pairing.md "
                    "(per-question transition against the baseline)",
                    **parse_notes(args.note),
                },
            )
            print(f"run dir: {writer.dir}\n", flush=True)

            with (writer.dir / "prompts.md").open("a", encoding="utf-8") as prompts:
                if args.concurrent_questions > 1:
                    # The circuit-breaker stop is only evaluated between questions,
                    # so a batched run drains its whole queue after an endpoint
                    # dies. The sequential default is what bounds that blast radius,
                    # which is why nothing here is concurrent unless asked.
                    groups = await asyncio.gather(
                        *(
                            run_question(
                                question, arms, retriever, retrieval, baseline, writer, done,
                                prompts, args, split,
                            )
                            for question in sample
                        )
                    )
                    problems += [problem for group in groups for problem in group]
                else:
                    for index, question in enumerate(sample, start=1):
                        if all(arm.breaker.tripped for arm in arms.values()):
                            problems.append(
                                "every arm's endpoint circuit-broke -- abandoning the "
                                f"remaining {len(sample) - index + 1} question(s); the rows "
                                "above are the whole of this run"
                            )
                            break
                        print(f"--- q{question.id} ({index}/{len(sample)}) ---")
                        problems += await run_question(
                            question, arms, retriever, retrieval, baseline, writer, done,
                            prompts, args, split,
                        )

            rows = writer.load_rows()
            for arm in arms.values():
                if not any(row.get("model") == arm.model.name for row in rows):
                    problems.append(f"{arm.model.name}: produced no rows at all")

    print("\n" + summarize(rows))
    print("\n" + summarize_retrieval(rows))
    pairing = pairing_markdown(rows, [arm.model.name for arm in arms.values()])
    pairing_path = writer.dir / "pairing.md"
    pairing_path.write_text(pairing + "\n", encoding="utf-8")
    print("\n" + pairing)
    print(f"\nartifacts: {writer.dir}")

    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for problem in problems:
            print(f"  - {problem}")
        print("\nexit 1: not every planned call produced a row. Unanswered, distracted and "
              "unpaired rows are findings, not problems.")
        return 1
    print(
        "\nEvery planned call produced a row. Next: copy the transition table, the retrieval "
        "churn numbers and the rescued/distracted examples into "
        "experiments/phase6/FINDINGS.md, naming the endpoint and concurrency that produced "
        "them. Remember `pairing.md` lists only what this run measured -- an id the baseline "
        "never answered shows up as `unpaired`, not as a zero."
    )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(amain(parse_args())))


if __name__ == "__main__":
    main()
