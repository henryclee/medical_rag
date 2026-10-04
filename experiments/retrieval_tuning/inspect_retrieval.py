"""Manual retrieval inspection -- look at what each strategy actually retrieved.

    set -a; source .env; set +a

    # read an already-judged run: no embedder, no BM25 pickle, no requests
    .venv/bin/python experiments/retrieval_tuning/inspect_retrieval.py \\
        --from-run outputs/exploration/retrieval_tuning/20260929T120215Z --open

    # one question, live: retrieve the 5x2 grid, judge only unseen chunks
    .venv/bin/python experiments/retrieval_tuning/inspect_retrieval.py --question-id 1312

    # tweak the reformulation lever without paying for the reformulation
    .venv/bin/python experiments/retrieval_tuning/inspect_retrieval.py \\
        --question-id 1312 --query 'chronic pancreatitis with steatorrhea from duct obstruction'

    # keep the embedder and BM25 loaded, poke at several questions
    .venv/bin/python experiments/retrieval_tuning/inspect_retrieval.py --repl

Why: `FINDINGS.md` claims `dense` reached 65% and `bm25` 10%. Those numbers are
not actionable until you can read the chunk `dense_rerank` put at #1 and the one
it left at #17, so this tool is built for reading rather than for scoring -- and
for reading *both* query variants side by side, because "did reformulation help"
is the question this directory exists to answer.

Two modes, deliberately different costs:

* **read** (`--from-run`) -- zero LLM calls, zero models. Chunk text comes from
  the run's `chunks.jsonl` sidecar if it has one, else a LanceDB filter query.
  The 2026-09-29 run has neither a sidecar nor question text in its rows, so
  those come from the dataset and the index, and anything unresolvable is
  labelled rather than hidden.
* **live** (`--question-id`, `--all`, `--repl`) -- loads the retriever, the BM25
  pickle and the judge client, retrieves the whole grid, and asks the judge only
  for `(question, chunk)` pairs missing from `judge_cache/judged_chunks.jsonl`.
  Re-inspecting a question you looked at yesterday is free; the first look costs
  one judge batch per ~24 new chunks.

Free-text probing (`--question ... --option A=... --answer A`) runs the same grid
on a question that is not in MedQA at all -- the cheapest way to find out why
BM25 keeps missing a paraphrase.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import webbrowser
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from loguru import logger

from medical_rag.config import load_config
from medical_rag.data.load_medqa import MedQAQuestion, load_medqa
from medical_rag.generation.llm import LLMClient, LLMError
from medical_rag.retrieval.index import load_index
from medical_rag.retrieval.retriever import RetrievedChunk

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _path in (_HERE, _REPO_ROOT / "scripts" / "exploration"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _common import (  # noqa: E402
    DEFAULT_RUN_ROOT,
    EndpointCircuitBreaker,
    _unique_dir,
    parse_notes,
    select_by_ids,
)
from bm25_index import DEFAULT_BM25_PATH, load_or_build_bm25_index  # noqa: E402
from chunk_store import resolve_chunks, write_chunks  # noqa: E402
from judge_cache import DEFAULT_JUDGE_CACHE, ensure_verdicts, load_cache  # noqa: E402
from judge_prompt import judge_prompt_sha  # noqa: E402
from raw_rag import Preflight, build_retriever, preflight, resolve_index_dir  # noqa: E402
from reformulate import (  # noqa: E402
    ReformulationResult,
    load_template,
    manual_result,
    reformulate_query,
)
from render import (  # noqa: E402
    QuestionView,
    build_question_view,
    render_html,
    render_markdown,
    render_terminal,
    view_from_row,
)
from strategies import aretrieve_grid  # noqa: E402

PHASE = "retrieval_tuning"
CONDITION_ID = "retrieval_inspection"
K_DEFAULT = (5, 10, 20)



def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config/default.yaml", help="path to default.yaml")
    parser.add_argument("--split", default=None, help="MedQA split (default: config.dev_split)")

    modes = parser.add_argument_group("what to inspect")
    modes.add_argument(
        "--from-run", default=None, metavar="RUN_DIR",
        help="read an existing run's results.jsonl -- no models loaded, no calls made",
    )
    modes.add_argument(
        "--question-id", action="append", default=[], metavar="ID",
        help="inspect this question (repeatable); live unless combined with --from-run",
    )
    modes.add_argument(
        "--all", action="store_true",
        help="the pinned 20 questions (judge_harness.SAMPLE_QUESTION_IDS)",
    )
    modes.add_argument("--question", default=None, help="free-text question (live mode)")
    modes.add_argument(
        "--option", action="append", default=[], metavar="LETTER=TEXT",
        help="an option for --question, e.g. A=Chronic pancreatitis (repeatable)",
    )
    modes.add_argument("--answer", default=None, help="gold letter for --question")
    modes.add_argument(
        "--repl", action="store_true",
        help="interactive loop: keep the embedder and BM25 loaded between questions",
    )

    grid = parser.add_argument_group("grid knobs")
    grid.add_argument("--top-k", type=int, default=None, help="retrieve k (default: config)")
    grid.add_argument("--rerank-k", type=int, default=None, help="keep k after rerank")
    grid.add_argument(
        "--bm25-top-k", type=int, default=None,
        help="BM25 depth, separate from --top-k (default: --top-k). The lever to pull if "
        "you suspect BM25's 10% recall is a depth artifact and not a vocabulary one",
    )
    grid.add_argument(
        "--rerank-with", choices=("question", "reform"), default="question",
        help="which string the cross-encoder sees; 'question' keeps the reform column a "
        "single-variable change (see strategies.py)",
    )
    grid.add_argument("--k", nargs="*", type=int, default=list(K_DEFAULT), help="k values")
    grid.add_argument("--index-dir", default=None, help="LanceDB index directory")
    grid.add_argument("--bm25-index", default=str(DEFAULT_BM25_PATH), help="pickled BM25 path")
    grid.add_argument("--rebuild-bm25", action="store_true", help="rebuild the BM25 pickle")

    reform = parser.add_argument_group("reformulation knobs")
    reform.add_argument("--no-reform", action="store_true", help="skip the reform column entirely")
    reform.add_argument(
        "--query", default=None, metavar="TEXT",
        help="hand-written information need; skips the reformulation call, so no fallback "
        "can silently flatten the reform column",
    )
    reform.add_argument(
        "--reform-prompt", default=None, metavar="FILE",
        help="prompt template file with a {question} placeholder, replacing production's",
    )

    judge = parser.add_argument_group("judging knobs")
    judge.add_argument(
        "--no-judge", action="store_true",
        help="render from the verdict cache only; unjudged chunks show as '?'",
    )
    judge.add_argument("--judge-cache", default=str(DEFAULT_JUDGE_CACHE), help="verdict cache path")
    judge.add_argument("--judge-batch", type=int, default=24, help="chunks per judge call")

    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", default=DEFAULT_RUN_ROOT, help="artifact root")
    out.add_argument("--out-name", default=None, help="name the dir (else inspect_<UTC>)")
    out.add_argument("--no-html", action="store_true", help="terminal dump only, write no files")
    out.add_argument("--open", dest="open_browser", action="store_true", help="open the report")
    out.add_argument(
        "--max-chunks", type=int, default=10,
        help="chunks per cell in the terminal dump (HTML always shows all); 0 = all",
    )
    out.add_argument("--dry-run", action="store_true", help="print the plan; make no calls")
    out.add_argument("--note", action="append", default=[], metavar="KEY=VALUE", help="context.md row")
    return parser.parse_args(argv)


# --- inputs ------------------------------------------------------------------


def read_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Every row of a run's `results.jsonl`, with a torn tail line skipped.

    The last line of a run killed mid-write is a partial JSON object. Skipping it
    with a warning is right for a viewer; the same line in a metrics pipeline
    would have to be a hard error, which is why this does not live in `_common`.
    """
    path = Path(run_dir) / "results.jsonl"
    if not path.exists():
        raise SystemExit(f"--from-run {run_dir}: no results.jsonl to read")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.open(encoding="utf-8"), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("{}:{}: torn line skipped (interrupted mid-write?)", path, number)
    return rows


def custom_question(args: argparse.Namespace) -> MedQAQuestion | None:
    """Build a `MedQAQuestion` from `--question/--option/--answer`.

    Options and a gold letter are required: without them the grid's only number
    is `relevant_fraction`, a different measurement than the one the page is laid
    out to answer. Refusing beats rendering a page that looks like a recall
    inspection and is not one.
    """
    if args.question is None:
        return None
    options: dict[str, str] = {}
    for raw in args.option:
        letter, sep, text = raw.partition("=")
        if not sep or not letter.strip() or not text.strip():
            raise SystemExit(f"--option must look like A=Some text, got {raw!r}")
        options[letter.strip().upper()] = text.strip()
    if len(options) < 2:
        raise SystemExit("--question needs at least two --option LETTER=TEXT values")
    gold = (args.answer or "").strip().upper()
    if gold not in options:
        raise SystemExit(f"--answer {gold!r} is not among the given options {sorted(options)}")
    digest = hashlib.sha256(args.question.encode("utf-8")).hexdigest()[:8]
    return MedQAQuestion(
        id=args.question_id[0] if args.question_id else f"custom-{digest}",
        question=args.question,
        options=options,
        answer_idx=gold,
        answer_text=options[gold],
        meta_info="ad-hoc question typed into inspect_retrieval.py",
    )


def resolve_target_questions(
    args: argparse.Namespace, pool: list[MedQAQuestion] | None
) -> list[MedQAQuestion]:
    """The questions to inspect, from ids / --all / the free-text question.

    Errors name every missing id (via `select_by_ids`) instead of inspecting the
    subset that resolved -- a page for 18 of 20 pinned questions that looks
    complete is the failure mode this repo has already paid for twice.
    """
    from judge_harness import SAMPLE_QUESTION_IDS  # noqa: PLC0415 - import stays local

    targets: list[MedQAQuestion] = []
    ids = list(args.question_id)
    if args.all:
        ids = ids + [qid for qid in SAMPLE_QUESTION_IDS if qid not in ids]
    if ids:
        if pool is None:
            raise SystemExit("--question-id/--all need the MedQA split loaded")
        try:
            targets.extend(select_by_ids(pool, ids))
        except KeyError as exc:
            raise SystemExit(str(exc.args[0]) if exc.args else exc) from exc
    custom = custom_question(args)
    if custom is not None:
        targets.append(custom)
    return targets


# --- artifacts ----------------------------------------------------------------


def write_context(path: Path, meta: dict[str, Any], notes: dict[str, str]) -> None:
    """`context.md` for an inspection -- same idea as a run's, smaller contract.

    Records the knobs that produced the page, because an inspection you cannot
    re-run is a screenshot. `judge_prompt_sha` and `chunk_source` are here for the
    same reason: they decide whether the verdicts on the page belong to this
    rubric, and whether the prose came from the index or a sidecar.
    """
    lines = ["# Retrieval inspection context", ""]
    lines += [f"- **{key}**: {value}" for key, value in meta.items()]
    if notes:
        lines += ["", "## Notes", ""] + [f"- **{key}**: {value}" for key, value in notes.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def row_from_view(view: QuestionView) -> dict[str, Any]:
    """A harness-shaped row, so an inspection is greppable like a run.

    Method keys are grid ids and `judgments` holds the verdicts actually shown, so
    `judge_cache.py --import-run` can re-import an inspection dir. `question` and
    `options` are included on purpose -- their absence is what made the first
    run's rows unreadable by anything but the dataset that produced them.
    """
    return {
        "question_id": view.question_id,
        "split": view.split,
        "condition_id": CONDITION_ID,
        "question": view.question,
        "options": view.options,
        "correct_answer": view.gold_letter,
        "queries": view.queries,
        "reformulation": {
            key: value for key, value in (view.reform or {}).items() if key != "prompt"
        },
        "candidates": {cell.method: [c.chunk_id for c in cell.chunks] for cell in view.cells},
        "judgments": {
            chunk.chunk_id: {
                "relevance": chunk.relevance,
                "supports_options": chunk.supports_options,
                "reason": chunk.reason,
            }
            for cell in view.cells
            for chunk in cell.chunks
            if chunk.judged
        },
        "provenance": view.provenance,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def dedupe_chunks(views: Sequence[QuestionView]) -> list[RetrievedChunk]:
    """Every distinct chunk across the inspected questions, first body winning.

    A chunk whose body never resolved (an old run, no sidecar, no index handle) is
    still written, with empty text: the sidecar is then a complete record of what
    the report *displayed*, including the parts it displayed as unresolved.
    """
    seen: dict[str, RetrievedChunk] = {}
    for view in views:
        for cell in view.cells:
            for chunk in cell.chunks:
                seen.setdefault(
                    chunk.chunk_id,
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        title=chunk.title,
                        content=chunk.content,
                        score=chunk.score if isinstance(chunk.score, float) else 0.0,
                    ),
                )
    return list(seen.values())


def write_artifacts(
    out_dir: Path, views: Sequence[QuestionView], meta: dict[str, Any], *, html: bool = True
) -> dict[str, Path]:
    """`report.html` + one markdown per question + rows + chunk sidecar + context."""
    written: dict[str, Path] = {}
    if not html:
        return written
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.html"
    report.write_text(
        render_html(views, title=f"Retrieval inspection -- {out_dir.name}", meta=meta),
        encoding="utf-8",
    )
    written["report"] = report

    questions = out_dir / "questions"
    questions.mkdir(exist_ok=True)
    for view in views:
        target = questions / f"q{view.question_id}.md"
        target.write_text(render_markdown(view), encoding="utf-8")
        written[f"q{view.question_id}"] = target

    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for view in views:
            handle.write(json.dumps(row_from_view(view), ensure_ascii=False) + "\n")
    write_chunks(out_dir / "chunks.jsonl", dedupe_chunks(views))
    notes = {f"q{view.question_id}/{key}": value for view in views for key, value in view.notes.items()}
    write_context(out_dir / "context.md", meta, notes)
    written["results"] = out_dir / "results.jsonl"
    written["chunks"] = out_dir / "chunks.jsonl"
    written["context"] = out_dir / "context.md"
    return written
# --- the inspector -------------------------------------------------------------


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class Inspector:
    """The loaded world: index, BM25, judge client, verdict cache, out dir.

    Constructed once and reused -- by a one-shot run and by every turn of the
    REPL. The REPL exists because `Embedder`, the cross-encoder and the 467 MB
    BM25 pickle load in ~40 s while one grid takes ~2 s: re-running the script per
    guess pays the expensive part to exercise the cheap one.
    """

    def __init__(self, args: argparse.Namespace, config: Any) -> None:
        self.args = args
        self.config = config
        self.split = args.split or config.dev_split
        self.top_k = args.top_k or config.retrieval.top_k_retrieve
        self.rerank_k = args.rerank_k or config.retrieval.top_k_rerank
        self.bm25_top_k = args.bm25_top_k or self.top_k
        self.ks = tuple(sorted(args.k or K_DEFAULT))
        self.rerank_with = args.rerank_with
        self.template = load_template(args.reform_prompt)
        self.notes: dict[str, str] = dict(parse_notes(args.note))
        self.index_dir = resolve_index_dir(config, args.index_dir)
        self.out_dir = _unique_dir(
            Path(args.output_dir) / PHASE / (args.out_name or f"inspect_{_utc_stamp()}")
        )
        self.cache: dict[tuple[str, str], dict[str, Any]] = {}
        self.cache_counts: dict[str, int] = {}
        self.judge: LLMClient | None = None
        self.judge_model: Any = None
        self.reformulator: LLMClient | None = None
        self.breaker: EndpointCircuitBreaker | None = None
        self.retriever: Any = None
        self.bm25: Any = None
        self.table: Any = None
        self.judge_sha = judge_prompt_sha()
        self.views: list[QuestionView] = []
        self._stack: AsyncExitStack | None = None

    async def open(self, *, retrieval: bool, judge: bool) -> None:
        """Load only what this mode needs -- read mode loads no models and no BM25.

        Skipping the BM25 pickle in read mode is not a micro-optimisation: it is
        467 MB and two model loads that a "just show me yesterday's verdicts"
        request should not pay, and paying them is what makes people stop looking.
        """
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        self.cache, self.cache_counts = load_cache(self.args.judge_cache, judge_sha=self.judge_sha)
        print(
            f"judge cache: {self.cache_counts['loaded']} verdict(s) usable"
            + (f", {self.cache_counts['stale_prompt']} stale (other rubric)" if self.cache_counts["stale_prompt"] else "")
            + (f", {self.cache_counts['malformed']} malformed" if self.cache_counts["malformed"] else "")
        )
        self.table = load_index(self.index_dir)
        if not retrieval:
            return

        self.retriever, static = await asyncio.to_thread(build_retriever, self.config, str(self.index_dir))
        print(f"retriever ready: {static['index_rows']} rows on {static['device']}")
        self.bm25 = await asyncio.to_thread(
            load_or_build_bm25_index, self.index_dir, self.args.bm25_index,
            rebuild=self.args.rebuild_bm25,
        )
        print(f"BM25 ready: {len(self.bm25.chunk_ids)} chunks ({self.args.bm25_index})")
        if not judge:
            return

        self.judge_model = self.config.models["judge_model"]
        self.judge = await self._stack.enter_async_context(LLMClient(self.judge_model))
        if not self.args.no_reform:
            reform_model = self.config.models["model_a"]
            self.reformulator = await self._stack.enter_async_context(LLMClient(reform_model))
            decision, lines = await preflight(reform_model, self.reformulator)
            print("  " + " ".join(lines))
            if decision is Preflight.BLOCK:
                raise SystemExit("reformulator failed preflight -- aborting before any calls")
        decision, lines = await preflight(self.judge_model, self.judge)
        print("  " + " ".join(lines))
        if decision is Preflight.BLOCK:
            raise SystemExit("judge failed preflight -- aborting before any calls")
        self.breaker = EndpointCircuitBreaker(self.judge_model.name)

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(None, None, None)
            self._stack = None

    def meta(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        meta = {
            "generated": _utc_stamp(),
            "mode": "live retrieval" if self.retriever else "read-only replay",
            "source": self.args.from_run or "fresh retrieval",
            "split": self.split,
            "top_k": f"{self.top_k} -> rerank {self.rerank_k}",
            "bm25_top_k": self.bm25_top_k,
            "rerank_with": self.rerank_with,
            "columns": "orig + reform" if not self.args.no_reform else "orig only (--no-reform)",
            "judge_prompt_sha": self.judge_sha,
            "judge_cache": f"{self.args.judge_cache} ({self.cache_counts.get('loaded', 0)} cached)",
            **self.notes,
        }
        if extra:
            meta.update(extra)
        return meta

    async def inspect(
        self, question: MedQAQuestion, *, query_override: str | None = None
    ) -> QuestionView:
        """Retrieve the grid for one question, judge only what is unjudged, render it.

        `query_override` is the REPL's per-turn hand-written information need; it
        takes precedence over `--query` and over the reformulator, so a tweak costs
        a retrieval pass (~2 s) instead of a generation (~20 s) -- and it can never
        silently fall back, which is the whole point of the manual path.
        """
        assert self.retriever is not None and self.bm25 is not None, "open(retrieval=True) first"
        reform: ReformulationResult | None = None
        if query_override:
            reform = manual_result(query_override)
        elif self.args.query:
            reform = manual_result(self.args.query)
        elif not self.args.no_reform and self.reformulator is not None:
            reform = await reformulate_query(question, self.reformulator, template=self.template)

        grid = await aretrieve_grid(
            question.question,
            self.retriever,
            self.bm25,
            reformulated_text=reform.query if reform else None,
            top_k=self.top_k,
            rerank_k=self.rerank_k,
            bm25_top_k=self.bm25_top_k,
            rerank_with=self.rerank_with,
        )
        union = list(grid.all_chunks().values())
        judgments: dict[str, Any] = {}
        cost = {"cached": 0, "judged": 0, "batches": 0, "missing": 0}
        problems: list[str] = []

        if self.judge is None:
            for chunk in union:
                record = self.cache.get((question.id, chunk.chunk_id))
                if record is None:
                    cost["missing"] += 1
                else:
                    judgments[chunk.chunk_id] = record
            if cost["missing"]:
                problems.append(
                    f"{cost['missing']} chunk(s) have no cached verdict and --no-judge was given"
                )
        else:
            assert self.breaker is not None
            outcome = await ensure_verdicts(
                question,
                union,
                self.cache,
                client=self.judge,
                cache_path=self.args.judge_cache,
                breaker=self.breaker,
                batch=self.args.judge_batch,
                judge_sha=self.judge_sha,
                source_run=self.out_dir.name,
            )
            judgments = outcome.verdicts
            cost = {
                "cached": outcome.cached,
                "judged": outcome.judged,
                "batches": outcome.batches,
                "missing": outcome.missing_judgment,
            }
            problems = list(outcome.problems)

        provenance = {
            "mode": "live retrieval",
            "top_k": f"{self.top_k} -> rerank {self.rerank_k}",
            "bm25_top_k": self.bm25_top_k,
            "rerank_with": self.rerank_with,
            "index_dir": str(self.index_dir),
            "judge": getattr(self.judge_model, "api_model_name", "cache only (--no-judge)"),
            "judge_prompt_sha": self.judge_sha,
            "chunks": (
                f"{len(union)} unique -- {cost['cached']} from cache, "
                f"{cost['judged']} newly judged in {cost['batches']} call(s)"
            ),
        }
        notes = dict(self.notes)
        notes.update({f"problem_{i + 1}": problem for i, problem in enumerate(problems)})
        view = build_question_view(
            question,
            grid,
            judgments,
            split=self.split,
            reform=reform.model_dump() if reform else None,
            ks=self.ks,
            provenance=provenance,
            notes=notes,
        )
        self.views.append(view)
        print(render_terminal(view, max_chunks=(self.args.max_chunks or None)), flush=True)
        return view
# --- modes ---------------------------------------------------------------------


def views_from_run(
    insp: Inspector, rows: Sequence[dict[str, Any]], pool_by_id: dict[str, MedQAQuestion]
) -> list[QuestionView]:
    """Read mode: rows -> views, resolving chunk text once for the whole run.

    One `resolve_chunks` call for the union of ids, not one per question: each
    LanceDB filter query is a full scan of the table, so per-question resolution
    would turn a free replay into a multi-minute one for 20 questions.
    """
    args = insp.args
    usable = [row for row in rows if "error" not in row]
    skipped = len(rows) - len(usable)
    wanted = {str(question_id) for question_id in args.question_id}
    if wanted:
        absent = wanted - {str(row.get("question_id")) for row in usable}
        if absent:
            print(
                f"note: {len(absent)} requested id(s) are not in {args.from_run}: "
                f"{', '.join(sorted(absent))} -- inspect them live without --from-run"
            )
        usable = [row for row in usable if str(row.get("question_id")) in wanted]

    ids = [
        chunk_id
        for row in usable
        for ids in (row.get("candidates") or {}).values()
        for chunk_id in ids
    ]
    chunk_lookup, source = resolve_chunks(ids, run_dir=args.from_run, table=insp.table)
    print(f"resolved {len(chunk_lookup)} chunk bodies via {source}")
    if skipped:
        print(f"skipped {skipped} error row(s) -- see {Path(args.from_run) / 'results.jsonl'}")

    views: list[QuestionView] = []
    for row in usable:
        view = view_from_row(
            row,
            pool_by_id.get(str(row.get("question_id"))),
            chunk_lookup,
            ks=insp.ks,
            provenance=insp.meta({"mode": "read-only replay", "chunk_source": source}),
            notes=dict(insp.notes),
            chunk_source=source,
        )
        views.append(view)
        print(render_terminal(view, max_chunks=(args.max_chunks or None)), flush=True)
    insp.views.extend(views)
    return views


async def run_live(insp: Inspector, targets: Sequence[MedQAQuestion]) -> int:
    """Live mode: inspect each target, stop early if the judge endpoint dies."""
    problems: list[str] = []
    for index, question in enumerate(targets, start=1):
        if insp.breaker is not None and insp.breaker.tripped:
            problems.append(
                f"judge endpoint circuit-broke ({insp.breaker.reason}); skipped the remaining "
                f"{len(targets) - index + 1} question(s) instead of firing at a dead port"
            )
            break
        print(f"\n=== q{question.id} ({index}/{len(targets)}) ===", flush=True)
        try:
            await insp.inspect(question)
        except LLMError as exc:
            if insp.breaker is not None:
                insp.breaker.record_failure(exc)
            problems.append(f"q{question.id}: judge/reformulation call failed: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - index/BM25 failure is not an LLM failure
            problems.append(f"q{question.id}: retrieval failed: {type(exc).__name__}: {exc}")
            continue
        # Rewrite the report after every question: an inspection abandoned halfway
        # should still leave a readable page for what it did get through.
        write_artifacts(insp.out_dir, insp.views, insp.meta(), html=not insp.args.no_html)
    for problem in problems:
        print(f"PROBLEM: {problem}")
    return 1 if problems else 0


HELP = """commands:
  i <question_id>   inspect a question from the split (fresh grid, judges only unseen chunks)
  u <text>          set the hand-written information need, then re-inspect ('u' alone clears it)
  r                 drop the override and let the reformulator answer again
  t <k>             top_k                  bm <k>   bm25_top_k
  rk <k>            rerank_k               rw question|reform   which string reranks
  k <k1> <k2> ...   k values to report
  n <key>=<value>   attach a note to every page in this session
  w                 rewrite report.html / questions/*.md / results.jsonl
  h                 this help              x       quit (artifacts are written on exit)"""


async def run_repl(insp: Inspector, pool: Sequence[MedQAQuestion]) -> None:
    """The loop -- kept alive so the embedder and the BM25 pickle stay loaded.

    `input()` runs in a thread: the same event loop owns the judge's HTTP calls,
    and blocking it on a keystroke would hold an in-flight completion open.
    """
    by_id = {question.id: question for question in pool}
    current: MedQAQuestion | None = None
    override: str | None = None
    print(HELP)
    print(f"{len(by_id)} question(s) in split {insp.split}; artifacts -> {insp.out_dir}")
    while True:
        try:
            line = await asyncio.to_thread(
                input, f"[t={insp.top_k} bm={insp.bm25_top_k} rk={insp.rerank_k}]> "
            )
        except (EOFError, KeyboardInterrupt):
            print()
            return
        line = line.strip()
        if not line:
            continue
        command, _, rest = line.partition(" ")
        rest = rest.strip()
        try:
            if command in ("x", "exit", "quit"):
                return
            elif command == "h":
                print(HELP)
            elif command == "i":
                question = by_id.get(rest)
                if question is None:
                    print(f"no {rest!r} in split {insp.split} -- check --split")
                    continue
                current = question
                await insp.inspect(question, query_override=override)
            elif command == "u":
                override = rest or None
                print("override = " + (repr(override) if override else "<cleared: reformulator decides>"))
                if current is None:
                    print("(no current question yet -- remembered for the next 'i <id>')")
                else:
                    await insp.inspect(current, query_override=override)
            elif command == "r":
                override = None
                if current is None:
                    print("no current question")
                else:
                    await insp.inspect(current, query_override=None)
            elif command == "k":
                ks = sorted({int(value) for value in rest.split() if value.isdigit()})
                if not ks:
                    print("k takes numbers, e.g. 'k 5 10 20'")
                else:
                    insp.ks = tuple(ks)
                    print(f"ks = {insp.ks}")
            elif command == "t":
                insp.top_k = int(rest)
                print(f"top_k = {insp.top_k}")
            elif command == "bm":
                insp.bm25_top_k = int(rest)
                print(
                    f"bm25_top_k = {insp.bm25_top_k} -- deeper BM25 does not fix vocabulary; "
                    "compare bm25__orig's '1st rel' before believing a gain"
                )
            elif command == "rk":
                insp.rerank_k = int(rest)
                print(f"rerank_k = {insp.rerank_k}")
            elif command == "rw":
                if rest not in ("question", "reform"):
                    print("rw takes 'question' or 'reform'")
                else:
                    insp.rerank_with = rest
                    print(f"rerank_with = {rest}")
            elif command == "n":
                key, sep, value = rest.partition("=")
                if not sep:
                    print("n key=value")
                else:
                    insp.notes[key.strip()] = value.strip()
                    print(f"noted {key.strip()}")
            elif command == "w":
                written = write_artifacts(
                    insp.out_dir, insp.views, insp.meta(), html=not insp.args.no_html
                )
                print("wrote " + (", ".join(sorted(str(path) for path in written.values())) or "nothing"))
            else:
                print(f"unknown command {command!r} -- h for help")
        except ValueError as exc:
            print(f"bad argument: {exc}")
        except LLMError as exc:
            print(f"endpoint error: {exc}")


async def amain(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    read_mode = bool(args.from_run)
    rows = read_rows(Path(args.from_run)) if read_mode else []

    # The split is loaded only when question text is actually needed. The
    # 2026-09-29 rows store no question text, so a replay of *that* run still pays
    # the (local, cached) dataset load; a run whose rows do carry their text, and a
    # `--question`-only probe, do not. If the load fails in read mode the report
    # still renders -- ids and verdicts are the part that cost money.
    need_pool = bool(
        args.question_id or args.all or args.question or args.repl
        or (read_mode and any("question" not in row for row in rows))
    )
    pool: list[MedQAQuestion] = []
    if need_pool:
        try:
            pool = load_medqa(args.split or config.dev_split)
        except Exception as exc:  # noqa: BLE001 - dataset/HF failure, mode-dependent severity
            if not read_mode:
                raise SystemExit(f"could not load the MedQA split: {exc}") from exc
            print(f"WARNING: could not load the split ({exc}) -- question text will be missing")

    insp = Inspector(args, config)
    targets = [] if read_mode else resolve_target_questions(args, pool)
    print(
        f"mode={'read-only replay' if read_mode else 'live retrieval'} split={insp.split} "
        f"top_k={insp.top_k}->rerank {insp.rerank_k} bm25_top_k={insp.bm25_top_k} "
        f"rerank_with={insp.rerank_with} columns={'orig only' if args.no_reform else 'orig+reform'} "
        f"ks={insp.ks}"
    )
    if read_mode:
        print(f"rows: {len(rows)} in {args.from_run}")
    else:
        print(f"targets: {', '.join('q' + question.id for question in targets) or '(none)'}")

    if args.dry_run:
        print(
            "\ndry run -- no index, no BM25, no models, no requests. Would write to "
            f"{insp.out_dir}/ (report.html, questions/*.md, results.jsonl, chunks.jsonl, context.md)"
        )
        return 0
    if not read_mode and not targets and not args.repl:
        raise SystemExit(
            "nothing to inspect: pass --question-id / --all / --question ..., or --repl, or --from-run"
        )

    await insp.open(retrieval=not read_mode, judge=not (read_mode or args.no_judge))
    status = 0
    try:
        if args.repl:
            await run_repl(insp, pool)
        elif read_mode:
            views_from_run(insp, rows, {question.id: question for question in pool})
        else:
            status = await run_live(insp, targets)
    finally:
        written = write_artifacts(insp.out_dir, insp.views, insp.meta(), html=not args.no_html)
        await insp.close()

    if written:
        print("\nartifacts: " + ", ".join(sorted(str(path) for path in written.values())))
        report = written.get("report")
        if report is not None and args.open_browser:
            webbrowser.open(report.resolve().as_uri())
    else:
        print("\n(--no-html: nothing written)")
    return status


def main() -> None:
    raise SystemExit(asyncio.run(amain(parse_args())))


if __name__ == "__main__":
    main()







