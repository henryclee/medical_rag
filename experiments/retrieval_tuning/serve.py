"""`--serve`: the inspection grid as a localhost web app, driving the loaded process.

Why a server instead of the REPL: the two surfaces were in different places. The
knobs lived in a terminal loop that prints 110-character chunk bodies
(`render.render_terminal` says so itself: "the HTML is where you read prose"), and
the prose lived in a static `report.html` you switched to a browser to read and
had to reload by hand. Every iteration was *type in terminal → squint → switch
tabs → reload*, and outside the REPL it was *re-run the script*, which pays ~40 s
of embedder + cross-encoder + 467 MB BM25 pickle to exercise one grid -- ~12 s
measured on this machine for the five `__orig` cells at k=20 (two embedding passes,
two exact searches over 380k rows, two BM25 scans, two rerank passes), so the load
is the same order as the work and gets paid every time.

This keeps that loaded state and puts a form in front of it. `Inspector` is
untouched: it is constructed and `open()`ed exactly as `--repl` would, and every
route calls the same `aretrieve_grid` / `ensure_verdicts` / `build_cells` /
`render_main` the CLI calls, so the page and `FINDINGS.md` cannot drift apart.

Two constraints shape the plumbing:

* **One asyncio loop, on its own thread.** `Inspector.inspect()` is async and
  `LLMClient` wraps an `AsyncOpenAI`, which binds to the loop it first ran on --
  so a handler cannot call `asyncio.run()` per request. Handler threads post
  coroutines to this loop with `run_coroutine_threadsafe` and block on the result.
* **One grid at a time.** An `asyncio.Lock` serialises retrieval and judging: the
  cross-encoder is MPS work and `model_a`/`model_b`/`judge_model` all share one
  oMLX process on `:8080` (PLAN.md, ADR-0005), so two tabs firing at once buys
  contention and a misread rather than a faster answer.

Judging stays behind a button. Pages render from `judge_cache/judged_chunks.jsonl`
with unjudged chunks as `?` (already `render.build_cells`' behaviour: unmeasured,
never silently `irrelevant`), and the button spends completions that land in the
append-only cache -- so a question costs once, ever.

Localhost only, no auth: this is a single-user dev tool that renders full corpus
text and can spend endpoint time. It binds `127.0.0.1` by default for exactly that
reason -- do not repoint it at `0.0.0.0` on a shared host.
"""

from __future__ import annotations

import asyncio
import html
import json
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence
from urllib.parse import parse_qs, urlparse

from loguru import logger

from medical_rag.data.load_medqa import MedQAQuestion
from medical_rag.retrieval.retriever import RetrievedChunk

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ceiling import CeilingResult, probe  # noqa: E402
from judge_cache import ensure_verdicts  # noqa: E402
from render import (  # noqa: E402
    QuestionView,
    build_cells,
    render_ceiling,
    render_html,
    render_main,
    render_toolbar,
)

# A handler that waits forever is worse than one that times out: the page would
# show a spinner over a stale grid, which is the "looks current, isn't" failure
# this whole directory is paranoid about. Generous, because a judge batch against
# a shared endpoint is slow by design.
TIMEOUTS = {"inspect": 180.0, "knobs": 180.0, "judge": 1_800.0, "probe": 600.0, "chunk": 300.0}

# Guards against typing `99999` into top_k and waiting on a full-corpus sort. Deep
# is a legitimate lever -- `bm25_top_k` exists precisely because depth is suspect --
# an accidental six-digit value is not.
K_LIMITS = {"top_k": (1, 200), "rerank_k": (1, 200), "bm25_top_k": (1, 1000)}


class HttpError(Exception):
    """A client error to render as a plain-text body with this status code."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class SessionItem:
    """One inspected question: what it was, and what the grid showed for it.

    `question` is nullable because a row seeded from an old run may have stored
    `question_id` only -- no text, no options. Such a view stays on the page (its
    ids and verdicts are the part that cost money) but cannot be judged or
    re-retrieved, because the judge rubric needs all four options and there are
    none. Deleting it to keep the types tidy would hide the run's gap instead of
    naming it, which is what `view_from_row`'s hydration note already does.
    """

    question: MedQAQuestion | None
    view: QuestionView

    def require_question(self) -> MedQAQuestion:
        """The question, or a rejection explaining why this row cannot be graded.

        The judge rubric needs all four options and `build_answer_prompt` needs the
        text, so a row that stored `question_id` alone cannot be judged or
        re-retrieved. Refusing names the gap; proceeding would spend a completion to
        grade an empty option list.
        """
        if self.question is None:
            raise HttpError(
                409,
                f"q{self.view.question_id} stored no question text/options, so there is nothing "
                "to judge against -- inspect it live (drop --from-run) to grade it",
            )
        return self.question


class Backend:
    """The Inspector plus the loop it lives on, with a sync facade for handlers.

    Handler threads call the plain `*_route` helpers via `submit()`; each coroutine
    runs on `loop` and holds `lock`, so `session`, `probes` and `insp.views` have
    exactly one owner and a `Save` landing mid-inspect cannot write a half-updated
    view list.
    """

    def __init__(self, insp: Any, pool: dict[str, MedQAQuestion], browse: Sequence[str] = ()) -> None:
        self.insp = insp
        self.pool = pool
        # What the id field autocompletes. The dev split is 10,178 questions, and
        # emitting one <option> per question would put ~200 KB of markup on every
        # page render to autocomplete a field nobody uses by browsing -- whereas the
        # pinned sample is the set PLAN.md requires this project revisit. Any other
        # id still works: a datalist suggests, it does not restrict.
        self.browse = [cid for cid in browse if cid in pool] or list(pool)
        self.session: dict[str, SessionItem] = {}
        self.probes: list[dict[str, Any]] = []
        self.current: str | None = None
        self.override: str | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.lock: asyncio.Lock | None = None

    def ingest(self, question: MedQAQuestion | None, view: QuestionView) -> None:
        """Add or replace one question's grid in the session, keeping browse order.

        Replace-not-append matters: `render.aggregate_views` averages over the view
        list, so a re-render that appended would report n=2 for one question.
        """
        self.session[view.question_id] = SessionItem(question=question, view=view)
        self.current = view.question_id
        self._sync_views()

    # --- loop lifecycle ------------------------------------------------------

    def start(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.lock = asyncio.Lock()
        self.thread = threading.Thread(target=self._run, name="inspector-loop", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        assert self.loop is not None
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro: Any, *, timeout: float | None = None) -> Any:
        assert self.loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def shutdown(self) -> None:
        """Close the endpoint clients on their own loop, then stop that loop."""
        if self.loop is None or not self.loop.is_running():
            return

        async def _close() -> None:
            try:
                await self.insp.close()
            except Exception as exc:  # noqa: BLE001 - shutting down; log and move on
                logger.warning("closing inspector during shutdown: {}", exc)

        try:
            asyncio.run_coroutine_threadsafe(_close(), self.loop).result(30)
        except Exception as exc:  # noqa: BLE001
            logger.warning("shutdown: {}", exc)
        self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread is not None:
            self.thread.join(timeout=5)

    # --- shared state --------------------------------------------------------

    def _sync_views(self) -> None:
        """Make the Inspector's view list match the session, re-renders in place.

        `Inspector.inspect()` appends, which is right for the REPL (every turn is a
        fresh look) and wrong here: clicking Retrieve five times would put q1312 in
        the report five times, and `render.aggregate_views` averages over the view
        *list* -- so its summary row would read n=5 for one question, and every
        recall figure would be that same question counted five times.
        """
        self.insp.views = [item.view for item in self.session.values()]

    def _item(self) -> SessionItem:
        item = self.session.get(self.current or "")
        if item is None:
            raise HttpError(400, "no question on screen -- type a question id first")
        return item

    def _require_retrieval(self) -> None:
        if self.insp.retriever is None or self.insp.bm25 is None:
            raise HttpError(
                409, "this session loaded no retriever (--no-retrieve) -- restart without it to retrieve"
            )

    # --- state / page --------------------------------------------------------

    async def state(self) -> dict[str, Any]:
        item = self.session.get(self.current or "")
        unjudged = (
            len(
                {
                    chunk.chunk_id
                    for cell in item.view.cells
                    for chunk in cell.chunks
                    if not chunk.judged
                }
            )
            if item
            else 0
        )
        # The toolbar shows what the server is actually running with, not the
        # config's defaults: knobs drift during a session and a control bar that
        # lies about its own state is worse than no control bar.
        return {
            "question_id": self.current or "",
            "query": self.override or (item.view.queries.get("reform", "") if item else ""),
            "top_k": self.insp.top_k,
            "rerank_k": self.insp.rerank_k,
            "bm25_top_k": self.insp.bm25_top_k,
            "rerank_with": self.insp.rerank_with,
            "ks": " ".join(str(k) for k in self.insp.ks),
            "retrieval_enabled": self.insp.retriever is not None,
            "judge_enabled": self.insp.judge is not None,
            "unjudged": unjudged,
            "questions": self.browse,
            "in_split": len(self.pool),
            "probe_default": item.view.gold_text if item else "",
            "cached": self.insp.cache_counts.get("loaded", 0),
        }

    async def page(self) -> str:
        toolbar = render_toolbar(await self.state())
        self._sync_views()
        return render_html(
            self.insp.views,
            title=f"Retrieval inspection -- {self.insp.out_dir.name}",
            meta=self.insp.meta(),
            live=True,
            toolbar=toolbar,
        )

    # --- routes ------------------------------------------------------------

    async def inspect_route(self, body: dict[str, Any]) -> str:
        async with self._lock():
            question, override = self._resolve(body)
            await self._retrieve(question, override)
            return render_main(self.insp.views, meta=self.insp.meta())

    async def judge_route(self, body: dict[str, Any]) -> str:
        async with self._lock():
            if self.insp.judge is None:
                raise HttpError(409, "no judge endpoint loaded -- restart without --no-judge to judge")
            item = self._item()
            question = item.require_question()
            union = self._union(item)
            # Cache keys go through the view's id, which equals `question.id` by
            # construction -- one spelling of the key, so a verdict written by the
            # live path is found by the read path and never silently re-bought.
            pending = [
                chunk
                for chunk in union
                if (item.view.question_id, chunk.chunk_id) not in self.insp.cache
            ]
            if not pending:
                raise HttpError(409, "every chunk here already has a cached verdict -- nothing to spend")
            outcome = await ensure_verdicts(
                question,
                pending,
                self.insp.cache,
                client=self.insp.judge,
                cache_path=self.insp.args.judge_cache,
                breaker=self.insp.breaker,
                batch=self.insp.args.judge_batch,
                judge_sha=self.insp.judge_sha,
                source_run=self.insp.out_dir.name,
            )
            logger.info(
                "served judge: {} new verdict(s) in {} call(s), {} unanswered",
                outcome.judged,
                outcome.batches,
                outcome.missing_judgment,
            )
            self._reattach_verdicts(item)
            return render_main(self.insp.views, meta=self.insp.meta())

    async def knobs_route(self, body: dict[str, Any]) -> str:
        async with self._lock():
            applied = self._apply_knobs(body)
            logger.info("knobs: {}", " ".join(applied))
            meta = self.insp.meta({"applied": " ".join(applied)}) if applied else self.insp.meta()
            if not self.session:
                return render_main([], meta=meta)
            item = self._item()
            await self._retrieve(item.question, self.override)
            return render_main(self.insp.views, meta=meta)

    async def probe_route(self, body: dict[str, Any]) -> str:
        async with self._lock():
            text = str(body.get("text", "")).strip()
            if not text:
                raise HttpError(400, "probe needs text to search the corpus with")
            self._require_retrieval()
            # Off the loop: the containment scan walks 380k chunk bodies and the
            # oracle grid is seconds of MPS work -- either would stall the judge's
            # in-flight HTTP calls, which live on this same loop.
            result: CeilingResult = await asyncio.to_thread(
                probe,
                text,
                self.insp.retriever,
                self.insp.bm25,
                top_k=self.insp.top_k,
                rerank_k=self.insp.rerank_k,
                bm25_top_k=self.insp.bm25_top_k,
            )
            self.probes.append(result.to_dict())
            return render_ceiling(result)

    async def chunk_route(self, chunk_id: str) -> str:
        from chunk_store import resolve_chunks

        if self.insp.table is None:
            raise HttpError(409, "no index handle in this session")
        lookup, source = await asyncio.to_thread(
            resolve_chunks, [chunk_id], run_dir=self.insp.out_dir, table=self.insp.table
        )
        found = lookup.get(chunk_id)
        if found is None:
            raise HttpError(404, f"{chunk_id!r} is not in the index ({source})")
        return (
            f"<div><b>{html.escape(found.title or '(no title)')}</b> "
            f"<span class='idc'>{html.escape(chunk_id)} · via {html.escape(source)}</span></div>"
            f"<div class='body'>{html.escape(found.content or '(empty body)')}</div>"
        )

    async def report_route(self, body: dict[str, Any]) -> str:
        async with self._lock():
            self._sync_views()
            if not self.insp.views and not self.probes:
                raise HttpError(409, "nothing inspected and nothing probed -- nothing to write")
            written = await asyncio.to_thread(self._write_report)
            return ", ".join(str(path) for path in written)

    # --- route internals ----------------------------------------------------

    def _lock(self) -> asyncio.Lock:
        assert self.lock is not None
        return self.lock

    def _resolve(self, body: dict[str, Any]) -> tuple[MedQAQuestion, str | None]:
        """Which question to act on, and with what information-need override.

        A missing `question_id` means "the one on screen", which is what makes the
        knob Apply button one click. An id outside the split is a 400 rather than a
        no-op: a button that silently redraws the previous grid looks like it worked.
        """
        raw_id = body.get("question_id")
        if raw_id in (None, ""):
            question = self._item().require_question()
        else:
            question = self.pool.get(str(raw_id))
            if question is None:
                raise HttpError(400, f"question {str(raw_id)!r} is not in split {self.insp.split}")
        if "query" in body:
            raw = body.get("query")
            self.override = None if raw in (None, "") else (str(raw).strip() or None)
        return question, self.override

    async def _retrieve(self, question: MedQAQuestion, override: str | None) -> QuestionView:
        self._require_retrieval()
        view = await self.insp.inspect(question, query_override=override)
        # `inspect()` appends to a list the session owns; undo its append so a
        # re-render replaces the question instead of duplicating it (see _sync_views).
        if self.insp.views and self.insp.views[-1] is view:
            self.insp.views.pop()
        self.session[question.id] = SessionItem(question=question, view=view)
        self.current = question.id
        self._sync_views()
        return view

    def _union(self, item: SessionItem) -> list[RetrievedChunk]:
        """Every distinct chunk the grid surfaced, as `ensure_verdicts` wants them."""
        union: dict[str, RetrievedChunk] = {}
        for cell in item.view.cells:
            for chunk in cell.chunks:
                union.setdefault(
                    chunk.chunk_id,
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        title=chunk.title,
                        content=chunk.content,
                        score=chunk.score if isinstance(chunk.score, float) else 0.0,
                    ),
                )
        return list(union.values())

    def _reattach_verdicts(self, item: SessionItem) -> None:
        """Re-grade the grid already in memory -- no re-retrieval, no re-rewrite.

        The alternative is to inspect again, which re-runs the reformulator and
        spends a completion to re-derive a query we already have -- and might get a
        *different* string, silently changing the column the reader is looking at.
        """
        view = item.view
        lists = {
            cell.method: [
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    content=chunk.content,
                    score=chunk.score if isinstance(chunk.score, float) else 0.0,
                )
                for chunk in cell.chunks
            ]
            for cell in view.cells
        }
        judgments = {
            chunk.chunk_id: self.insp.cache[(view.question_id, chunk.chunk_id)]
            for cell in view.cells
            for chunk in cell.chunks
            if (view.question_id, chunk.chunk_id) in self.insp.cache
        }
        view.cells = build_cells(
            lists,
            judgments,
            view.gold_letter,
            ks=view.ks,
            legacy=[cell.method for cell in view.cells if cell.legacy],
        )

    def _apply_knobs(self, body: dict[str, Any]) -> list[str]:
        """Validate the whole request, then set it -- never half-apply.

        A blank field means "leave it", not "set it to zero": clearing a box to
        retype it should not fire a retrieval with `top_k=0`. Validation is a full
        pass before any assignment because the knobs interact, and applying
        `top_k=8` before rejecting `rerank_k=9` would leave the running server
        holding a combination it just refused -- which the next unrelated request
        would then retrieve with. Each message names the constraint it hit, since
        the message is where you learn that `rerank_k` above `top_k` leaves the
        reranker nothing to reorder.
        """
        pending: dict[str, int] = {}
        for name, (low, high) in K_LIMITS.items():
            raw = str(body.get(name, "")).strip()
            if not raw:
                continue
            try:
                value = int(raw)
            except ValueError as exc:
                raise HttpError(400, f"{name} must be an integer, got {raw!r}") from exc
            if not low <= value <= high:
                raise HttpError(400, f"{name} must be between {low} and {high}, got {value}")
            pending[name] = value

        # The pair has to be checked against the *resulting* state, not the request:
        # `rerank_k=9` is fine on its own and broken next to `top_k=8`.
        rerank_k = pending.get("rerank_k", self.insp.rerank_k)
        top_k = pending.get("top_k", self.insp.top_k)
        if rerank_k > top_k:
            raise HttpError(
                400,
                f"rerank_k ({rerank_k}) > top_k ({top_k}) -- the reranker would have nothing "
                "to reorder",
            )

        ks: tuple[int, ...] | None = None
        raw_ks = str(body.get("ks", "")).strip()
        if raw_ks:
            ks = sorted({int(tok) for tok in raw_ks.replace(",", " ").split() if tok.isdigit()})
            if not ks:
                raise HttpError(400, f"ks needs positive integers, got {raw_ks!r}")
            if any(k > top_k for k in ks):
                raise HttpError(
                    400,
                    f"ks {ks} exceeds top_k ({top_k}) -- a metric past the retrieval depth is "
                    "undefined, not zero",
                )

        variant = str(body.get("rerank_with", "")).strip()
        if variant and variant not in ("question", "reform"):
            raise HttpError(400, "rerank_with is 'question' or 'reform'")
        if not pending and ks is None and not variant:
            raise HttpError(400, "no knobs in the request")

        applied: list[str] = []
        for name, value in pending.items():
            setattr(self.insp, name, value)
            applied.append(f"{name}={value}")
        if ks is not None:
            self.insp.ks = ks
            applied.append(f"ks={ks}")
        if variant:
            self.insp.rerank_with = variant
            applied.append(f"rerank_with={variant}")
        return applied

    def _write_report(self) -> list[Path]:
        from inspect_retrieval import write_artifacts

        meta = self.insp.meta({"served": "yes -- the header's knobs are the ones that produced this"})
        written = list(write_artifacts(self.insp.out_dir, self.insp.views, meta).values())
        if self.probes:
            path = self.insp.out_dir / "probes.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for record in self.probes:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written.append(path)
        return written


class Handler(BaseHTTPRequestHandler):
    """Nine routes, HTML fragments out, errors as plain text with a real status.

    Errors go back as `text/plain` bodies because the client drops them straight
    into the status line -- the one place a human using this tool looks. Returning
    200 with an error string in it is the specific thing to avoid: it would leave a
    stale grid on screen looking current.
    """

    server_version = "RetrievalInspect/1"
    protocol_version = "HTTP/1.1"

    @property
    def backend(self) -> Backend:
        return self.server.backend  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 - name fixed by stdlib
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send(200, self.backend.submit(self.backend.page(), timeout=180.0), "text/html")
            elif parsed.path == "/api/state":
                state = self.backend.submit(self.backend.state(), timeout=30.0)
                self._send(200, json.dumps(state, indent=2), "application/json")
            elif parsed.path == "/api/chunk":
                chunk_id = (parse_qs(parsed.query).get("id") or [""])[0].strip()
                if not chunk_id:
                    raise HttpError(400, "usage: /api/chunk?id=<chunk_id>")
                self._send(
                    200,
                    self.backend.submit(self.backend.chunk_route(chunk_id), timeout=TIMEOUTS["chunk"]),
                    "text/html",
                )
            else:
                raise HttpError(404, f"no route {parsed.path}")
        except Exception as exc:  # noqa: BLE001 - one response path per request
            self._fail(exc)

    def do_POST(self) -> None:  # noqa: N802 - name fixed by stdlib
        parsed = urlparse(self.path)
        name = parsed.path.rsplit("/", 1)[-1]
        route = {
            "inspect": self.backend.inspect_route,
            "judge": self.backend.judge_route,
            "knobs": self.backend.knobs_route,
            "probe": self.backend.probe_route,
            "report": self.backend.report_route,
        }.get(name)
        if route is None:
            self._fail(HttpError(404, f"no route {parsed.path}"))
            return
        try:
            body = self._json_body()
            self._send(
                200,
                self.backend.submit(route(body), timeout=TIMEOUTS.get(name, 180.0)),
                "text/html",
            )
        except Exception as exc:  # noqa: BLE001 - the client surfaces the message
            self._fail(exc)

    # --- plumbing ----------------------------------------------------------

    def _json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            parsed = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HttpError(400, f"body is not JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise HttpError(400, "body must be a JSON object")
        return parsed

    def _fail(self, exc: Exception) -> None:
        # A 500 is our bug, so the traceback goes to the log and the one-line message
        # goes to the client (it renders into the status line). Never answer 200 with
        # an error in the body -- that leaves a stale grid on screen reading as current.
        status = getattr(exc, "status", None)
        if status is None:
            status = 500
            logger.exception("serve: unhandled {}", type(exc).__name__)
        self._send(status, f"{type(exc).__name__}: {exc}", "text/plain")

    def _send(self, status: int, body: str, kind: str) -> None:
        payload = body.encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", f"{kind}; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if payload:
                self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            # The browser gave up on a slow judge call. Normal, and printing a
            # traceback every time it happens would bury the log lines that matter.
            pass

    def log_message(self, fmt: str, *args: Any) -> None:
        line = fmt % args
        if " 200 " in line or "favicon" in line:
            return
        print(f"[serve] {line}", flush=True)


def serve(
    insp: Any,
    pool: dict[str, MedQAQuestion],
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    retrieval: bool = True,
    judge: bool = True,
    browse: Sequence[str] = (),
    on_open: Callable[[], Awaitable[Sequence[tuple[MedQAQuestion | None, QuestionView]]]] | None = None,
    on_port: Callable[[int], None] | None = None,
) -> int:
    """Block serving `insp` until Ctrl-C. Returns the port actually bound.

    `--port 0` picks its own, which is what lets a second `--serve` run alongside
    the first (two hypotheses worth comparing side by side, each with its own
    loaded index) -- the chosen port gets printed.

    `insp.open()` runs *here, on the backend loop*, not in the caller: `LLMClient`
    builds an `AsyncOpenAI` bound to whichever loop it first runs on, so a client
    opened on the CLI's loop would raise "attached to a different loop" the first
    time a handler thread used it. `on_open` is the seam for anything else that
    needs that handle (seeding a page from `--from-run`'s rows) -- it runs after
    `open()` and before the port is listening, so the first request already has a
    grid to show and there is no handler that could race the seed. `on_port` reports
    the bound port, which with `--port 0` is unknowable beforehand.
    """
    backend = Backend(insp, pool, browse)
    backend.start()
    backend.submit(insp.open(retrieval=retrieval, judge=judge), timeout=900.0)
    if on_open is not None:
        for question, view in backend.submit(on_open(), timeout=900.0):
            backend.ingest(question, view)
    server = ThreadingHTTPServer((host, port), Handler)
    server.backend = backend  # type: ignore[attr-defined]
    actual_port = server.server_address[1]
    if on_port is not None:
        on_port(actual_port)
    loaded = "retriever + BM25 loaded" if retrieval else "read-only (--no-retrieve)"
    graded = "judge endpoint live" if judge else "no endpoint: verdicts from cache only"
    print(
        f"\n  http://{host}:{actual_port}/   (Ctrl-C to stop)\n"
        f"  {len(backend.browse)} id(s) offered, {len(pool)} in split · artifacts -> {insp.out_dir}\n"
        f"  {loaded} · {graded}\n"
        "  probes use the gold answer -- logged to probes.jsonl, never to FINDINGS.md\n",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.shutdown()
        server.server_close()
        backend.shutdown()
    return actual_port
