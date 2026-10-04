"""Rendering the grid so a human can read it: HTML, markdown, terminal.

Why this exists at all: after the first tuning run, the only way to ask "what
did retrieval actually bring back for question 1312?" was to read
`results.jsonl` -- 40 `chunk_id`s and a verdict, with the text they name living
in a 467 MB pickle and a LanceDB table. A number you cannot open is a number you
end up trusting, which is how a 10-point recall delta got written into
`FINDINGS.md` from a lever that never fired.

Layout follows how the question is actually asked -- "did the reformulation
help?" -- so the two query variants sit side by side per strategy rather than as
ten flat columns. Gold-supporting chunks get a star and stay open; `irrelevant`
ones collapse, because the useful manual read is "why did the good one rank 7th
under dense and 1st under dense_rerank", not 40 paragraphs of StatPearls.

Self-contained HTML (inline CSS, no JS needed to read it, no CDN) so a report
opens from `file://` on a laptop with the network off.
"""

from __future__ import annotations

import html
import textwrap
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from medical_rag.data.load_medqa import MedQAQuestion
from medical_rag.retrieval.retriever import RetrievedChunk

from metrics import K_VALUES, cell_metrics
from strategies import (
    BASE_STRATEGIES,
    METHOD_GRID,
    QUERY_VARIANTS,
    GridResult,
    canonical_method,
    prerank_map,
    split_method,
)

RELEVANCE_BADGE = {"relevant": "REL", "partial": "PAR", "irrelevant": "IRR"}
MISSING = "?"


@dataclass
class ChunkView:
    """One chunk as rendered inside one cell, with its verdict attached."""

    rank: int
    chunk_id: str
    title: str
    content: str
    score: float | None = None
    relevance: str = MISSING
    supports_options: list[str] = field(default_factory=list)
    reason: str = ""
    gold_hit: bool = False
    prerank: int | None = None
    judged: bool = True
    body_available: bool = True


@dataclass
class CellView:
    """One grid cell: a strategy under one query variant."""

    method: str
    base: str
    variant: str
    chunks: list[ChunkView] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    reused: bool = False
    legacy: bool = False
    unjudged: int = 0

    @property
    def label(self) -> str:
        return self.method if not self.legacy else f"{self.method} (legacy)"


@dataclass
class QuestionView:
    """Everything one question's inspection page needs, from either mode."""

    question_id: str
    question: str
    options: dict[str, str]
    gold_letter: str
    split: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    queries: dict[str, str] = field(default_factory=dict)
    reform: dict[str, Any] = field(default_factory=dict)
    cells: list[CellView] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    ks: tuple[int, ...] = K_VALUES

    @property
    def gold_text(self) -> str:
        return self.options.get(self.gold_letter, "")

    def cell(self, method: str) -> CellView | None:
        return next((cell for cell in self.cells if cell.method == method), None)

    def variants(self) -> list[str]:
        present = {cell.variant for cell in self.cells}
        return [variant for variant in QUERY_VARIANTS if variant in present]

    def methods(self) -> list[str]:
        present = {cell.method for cell in self.cells}
        return [method for method in METHOD_GRID if method in present]


def _chunk_view(
    rank: int,
    chunk: RetrievedChunk,
    judgment: Mapping[str, Any] | None,
    gold: str,
    prerank: int | None,
) -> ChunkView:
    judgment = judgment or {}
    supports = sorted({str(o).upper() for o in judgment.get("supports_options", [])})
    return ChunkView(
        rank=rank,
        chunk_id=chunk.chunk_id,
        title=chunk.title,
        content=chunk.content,
        score=chunk.score,
        relevance=judgment.get("relevance", MISSING),
        supports_options=supports,
        reason=judgment.get("reason", ""),
        gold_hit=gold in supports,
        prerank=prerank,
        judged=bool(judgment),
        body_available=bool(chunk.content),
    )


def build_cells(
    lists: Mapping[str, Sequence[RetrievedChunk]],
    judgments: Mapping[str, Mapping[str, Any]],
    gold: str,
    *,
    ks: Sequence[int] = K_VALUES,
    legacy: Sequence[str] = (),
) -> list[CellView]:
    """Attach verdicts and per-cell metrics to every ranked list, grid order.

    `judgments` is the union the judge was asked about; a chunk missing from it
    renders as `?` and is counted in `unjudged` rather than defaulted to
    `irrelevant` -- the difference is whether the cell's recall looks low or
    looks unmeasured, and only the second one is honest.
    """
    cells: list[CellView] = []
    legacy_ids = set(legacy)
    for method in METHOD_GRID:
        if method not in lists:
            continue
        chunks = list(lists[method])
        source = prerank_map(method, {m: lists[m] for m in lists})
        views = [
            _chunk_view(rank, chunk, judgments.get(chunk.chunk_id), gold, source.get(chunk.chunk_id))
            for rank, chunk in enumerate(chunks, start=1)
        ]
        ids = [chunk.chunk_id for chunk in chunks]
        cells.append(
            CellView(
                method=method,
                base=split_method(method)[0],
                variant=split_method(method)[1],
                chunks=views,
                metrics=cell_metrics(ids, judgments, gold, ks=ks),
                reused=False,
                legacy=method in legacy_ids,
                unjudged=sum(1 for view in views if not view.judged),
            )
        )
    return cells


def build_question_view(
    question: MedQAQuestion,
    grid: GridResult,
    judgments: Mapping[str, Mapping[str, Any]],
    *,
    split: str = "",
    reform: Mapping[str, Any] | None = None,
    ks: Sequence[int] = K_VALUES,
    chunk_lookup: Mapping[str, RetrievedChunk] | None = None,
    provenance: Mapping[str, Any] | None = None,
    notes: Mapping[str, str] | None = None,
) -> QuestionView:
    """A live-mode view: the grid this process just retrieved.

    `chunk_lookup` overrides the chunks inside `grid` -- in live mode they
    already carry their text, so it is only consulted when a stored id has to be
    resolved from a sidecar or the index instead. `split` is a parameter because
    `MedQAQuestion` does not carry one: the loader takes it, the record does not
    return it.
    """
    lists: dict[str, list[RetrievedChunk]] = {}
    for method, chunks in grid.lists.items():
        lists[method] = [chunk_lookup.get(c.chunk_id, c) if chunk_lookup else c for c in chunks]
    view = QuestionView(
        question_id=question.id,
        question=question.question,
        options=dict(question.options),
        gold_letter=question.answer_idx.upper(),
        split=split,
        meta={"answer_text": question.answer_text, "meta_info": question.meta_info},
        queries=dict(grid.variant_queries),
        reform=dict(reform or {}),
        cells=build_cells(lists, judgments, question.answer_idx.upper(), ks=ks),
        provenance=dict(provenance or {}),
        notes=dict(notes or {}),
        ks=tuple(ks),
    )
    if grid.reused_variant:
        for cell in view.cells:
            if cell.variant == grid.reused_variant:
                cell.reused = True
    return view
    if grid.reused_variant:
        for cell in view.cells:
            if cell.variant == grid.reused_variant:
                cell.reused = True
    return view


def _stored_reform(row: Mapping[str, Any]) -> dict[str, Any]:
    """The reformulation record a row carries, in either shape it was written in.

    The grid harness stores a nested `reformulation` dict (the whole
    `ReformulationResult`); the pre-grid run stored two flat fields. Reading only
    the flat pair made a grid row look like it had been run with `--no-reform`,
    which is the opposite of what the row says.
    """
    record = row.get("reformulation")
    if isinstance(record, Mapping):
        return dict(record)
    return {
        "query": row.get("reformulated_query"),
        "fallback": row.get("reformulation_fallback"),
        "error": None,
        "method": "legacy run (pre-grid, pre-fix)",
    }


def view_from_row(
    row: Mapping[str, Any],
    question: MedQAQuestion | None,
    chunk_lookup: Mapping[str, RetrievedChunk],
    *,
    ks: Sequence[int] = K_VALUES,
    provenance: Mapping[str, Any] | None = None,
    notes: Mapping[str, str] | None = None,
    chunk_source: str = "",
) -> QuestionView:
    """A read-mode view: rebuild the grid from a stored row, at zero cost.

    Legacy method ids (`reform_dense`) are aliased onto the grid but flagged, so
    the page says "this column is the old broken one" instead of presenting it as
    a measured reform result. Missing question text (the first run's rows store
    only `question_id`) degrades to ids + verdicts with a visible warning rather
    than a crash -- but then again that run's rows never had options either, so
    the dataset load in `inspect_retrieval.py` is what makes this readable.
    """
    gold = str(row.get("correct_answer", "")).upper()
    judgments = row.get("judgments") or {}
    lists: dict[str, list[RetrievedChunk]] = {}
    legacy: list[str] = []
    for stored, ids in (row.get("candidates") or {}).items():
        try:
            method, was_aliased = canonical_method(stored)
        except ValueError:
            continue
        lists[method] = [
            chunk_lookup.get(chunk_id)
            or RetrievedChunk(chunk_id=chunk_id, title="", content="", score=0.0)
            for chunk_id in ids
        ]
        if was_aliased:
            legacy.append(method)

    view = QuestionView(
        question_id=str(row.get("question_id")),
        question=row.get("question") or (question.question if question else ""),
        options=dict(row.get("options") or (question.options if question else {})),
        gold_letter=gold,
        split=str(row.get("split", "")),
        meta=(
            {"answer_text": question.answer_text, "meta_info": question.meta_info}
            if question
            else dict(row.get("meta") or {})
        ),
        queries={
            # Prefer what the row measured: a harness row carries its own
            # `queries` dict (orig *and* reform), and overwriting it with the
            # dataset text would silently drop the reform query from the report.
            "orig": (row.get("queries") or {}).get("orig")
            or (question.question if question else "")
            or "(question text not in row)",
            **{
                key: value
                for key, value in (row.get("queries") or {}).items()
                if key != "orig" and isinstance(value, str)
            },
        },
        reform=_stored_reform(row),
        cells=build_cells(lists, judgments, gold, ks=ks, legacy=legacy),
        provenance=dict(provenance or {"chunk_source": chunk_source}),
        notes=dict(notes or {}),
        ks=tuple(ks),
    )
    if view.reform.get("fallback"):
        # The "reformulation" was the raw question (JSON-parse fallback), so every
        # `__reform` column is its `__orig` column re-run. Mark the ones that are
        # literally identical, which covers both shapes: the legacy run's aliased
        # `reform_dense` -> `dense__reform`, and a grid run where all eight reform
        # cells came back identical. Without this a reader assumes an identical
        # column was *measured* and matched, which is the opposite finding.
        for base in BASE_STRATEGIES:
            cell, orig = view.cell(f"{base}__reform"), view.cell(f"{base}__orig")
            if cell is None or orig is None:
                continue
            if [c.chunk_id for c in cell.chunks] == [c.chunk_id for c in orig.chunks]:
                cell.reused = True
    if question is None and not (view.question and view.options):
        # Only say the run is unreadable when it really is: harness rows carry
        # their own question text and options, so `--from-run` on them needs no
        # dataset load and must not be stamped with a caveat that is not true.
        view.notes["hydration"] = (
            "question text/options unavailable -- this run's rows stored question_id only"
        )
    return view


# --------------------------------------------------------------------------
# terminal
# --------------------------------------------------------------------------


def _flag(cell: CellView) -> str:
    if cell.reused:
        return "="
    if cell.unjudged:
        return f"~{cell.unjudged}"
    return ""


def _recall_at(cell: CellView | None, k: int) -> str:
    if cell is None:
        return "  -  "
    entry = cell.metrics.get("per_k", {}).get(k) or {}
    return "  HIT" if entry.get("recall") else "  -- "


def render_terminal(view: QuestionView, *, max_chunks: int | None = 10) -> str:
    """The dump `--repl` prints after every turn -- no browser required.

    Metrics first, chunks second: the common question at a terminal is "did the
    rewrite change anything", which the table answers in one glance. Chunk bodies
    are truncated hard here -- the HTML is where you read prose.
    """
    out: list[str] = []
    ks = view.ks
    width = 78
    out.append("=" * width)
    out.append(
        f"q{view.question_id} [{view.split or '?'}]  gold {view.gold_letter}: "
        f"{view.gold_text[:44]}"
    )
    out.append("=" * width)
    out.append(_wrap(view.question, width, "  "))
    for letter in sorted(view.options):
        marker = " <== GOLD" if letter == view.gold_letter else ""
        out.append(f"  {letter}) {_wrap(view.options[letter], width - 6, '      ', first_only=True)}{marker}")

    out.append("")
    reform = view.reform or {}
    if reform.get("query") or reform.get("fallback"):
        out.append(f"reformulation [{reform.get('method', '?')}]  prompt: {reform.get('prompt_source', '?')}")
        if reform.get("query"):
            out.append(_wrap(str(reform.get("query")), width, "  reform: "))
        else:
            out.append("  reform: (no reform query stored in this row)")
        if reform.get("fallback"):
            out.append(
                f"  !! FALLBACK -> reform column IS the orig column: {reform.get('error') or 'not recorded'}"
            )
    elif view.variants() == ["orig"]:
        out.append("reformulation: off (--no-reform) -- orig column only")
    if reform.get("error") and not reform.get("fallback"):
        out.append(f"  (noted) reform error: {reform['error']}")

    out.append("")
    header = f"{'strategy':<22}" + "".join(f"{'r@' + str(k):>6}" for k in ks) + f"{'rel':>6}{'1st':>5}  {'':<20}"
    out.append(header)
    out.append("-" * width)
    for base in BASE_STRATEGIES:
        orig = view.cell(f"{base}__orig")
        reform_cell = view.cell(f"{base}__reform")
        if orig is None and reform_cell is None:
            continue
        for variant, cell in (("orig", orig), ("refm", reform_cell)):
            if cell is None:
                continue
            row = f"{base + ':' + variant:<22}"
            row += "".join(f"{_recall_at(cell, k):>6}" for k in ks)
            fraction = (cell.metrics.get("per_k", {}).get(ks[-1], {}) or {}).get("relevant_fraction")
            row += f"{fraction:>6.2f}" if isinstance(fraction, float) else f"{'-':>6}"
            row += f"{cell.metrics.get('first_relevant_rank') or '-':>5}"
            row += "  " + _flag(cell)
            out.append(row)
    out.append("  HIT = gold-supporting chunk inside k;  = : reform column identical to orig; ~n: n unjudged")

    for base in BASE_STRATEGIES:
        for variant in view.variants():
            cell = view.cell(f"{base}__{variant}")
            if cell is None:
                continue
            shown = cell.chunks if max_chunks is None else cell.chunks[:max_chunks]
            out.append("")
            out.append(f"--- {cell.label} " + "-" * max(0, width - len(cell.label) - 5))
            for chunk in shown:
                star = "*" if chunk.gold_hit else " "
                badge = RELEVANCE_BADGE.get(chunk.relevance, "?")
                moved = "" if chunk.prerank is None else f"(#{chunk.prerank}->)"
                body = chunk.content[:110].replace("\n", " ") if chunk.content else "(body not resolved)"
                out.append(
                    f" {star}{chunk.rank:>2} {badge} {moved:<8} {chunk.title[:28]:<28} "
                    f"{'/'.join(chunk.supports_options) or '-':<4} {body}"
                )
                if chunk.reason:
                    out.append(f"      why: {chunk.reason[:100]}")
    if view.notes:
        out.append("")
        for key, value in view.notes.items():
            out.append(f"note[{key}]: {value}")
    return "\n".join(out)


def _wrap(text: str, width: int, indent: str = "", *, first_only: bool = False) -> str:
    if first_only:
        return text if len(text) <= width else text[: width - 1] + "..."
    return textwrap.fill(text or "", width=width, initial_indent=indent, subsequent_indent=indent)
# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------


def _cell_line(chunk: ChunkView | None) -> str:
    if chunk is None:
        return ""
    star = "⭐ " if chunk.gold_hit else ""
    badge = RELEVANCE_BADGE.get(chunk.relevance, "?")
    moved = "" if chunk.prerank is None else f" (was #{chunk.prerank})"
    opts = "/".join(chunk.supports_options) or "-"
    title = (chunk.title or "(no title)").replace("|", "/")
    return f"{star}{badge} `{opts}` {title}{moved}"


def render_markdown(view: QuestionView) -> str:
    """One question as markdown, for `questions/q<id>.md` and for grep/diff.

    Side-by-side rather than stacked: the manual question is "what did the
    rewrite move", and a reader comparing two columns of the same rank rows it in
    one pass. Chunk prose goes at the bottom, gold-supporting chunks first, so the
    table stays scannable instead of buried under 40 StatPearls paragraphs.
    """
    out: list[str] = []
    ks = view.ks
    out.append(f"# q{view.question_id} — gold `{view.gold_letter}`")
    out.append("")
    out.append(
        f"*split `{view.split or '?'}` · {len(view.cells)} grid cells · chunks from "
        f"`{view.provenance.get('chunk_source', 'live retrieval')}`*"
    )
    out.append("")
    out.append(f"> {view.question}")
    out.append("")
    for letter in sorted(view.options):
        mark = " **← GOLD**" if letter == view.gold_letter else ""
        out.append(f"- `{letter}` {view.options[letter]}{mark}")
    out.append("")

    reform = view.reform or {}
    out.append("## Reformulation")
    out.append("")
    if reform.get("query") or reform.get("fallback"):
        out.append(
            f"- method: `{reform.get('method', '?')}` · prompt: `{reform.get('prompt_source', '?')}`"
        )
        out.append(
            f"- **information need**: {reform['query']}"
            if reform.get("query")
            else "- _no reform query stored in this row_"
        )
        if reform.get("fallback"):
            out.append(
                "- **⚠ FELL BACK to the raw question** — the reform column IS the orig "
                f"column. Reason: `{reform.get('error')}`"
            )
        if reform.get("error") and not reform.get("fallback"):
            out.append(f"- noted error (non-fatal): `{reform['error']}`")
        out.append("")
        out.append("<details><summary>prompt + raw response</summary>")
        out.append("")
        out.append("```\n" + str(reform.get("prompt", "")) + "\n```")
        out.append("```\n" + str(reform.get("raw_response", "")) + "\n```")
        out.append("</details>")
    else:
        out.append("_Reformulation off — orig column only._")
    out.append("")

    out.append("## Metrics")
    out.append("")
    variants = view.variants()
    head = "| strategy | variant | " + " | ".join(f"recall@{k}" for k in ks) + " | rel-frac | 1st rel | unjudged |"
    out.append(head)
    out.append("|" + "---|" * (2 + len(ks) + 3))
    for base in BASE_STRATEGIES:
        for variant in variants:
            cell = view.cell(f"{base}__{variant}")
            if cell is None:
                continue
            per_k = cell.metrics.get("per_k", {})
            row = f"| `{cell.label}` | {variant} | "
            row += " | ".join("✅" if (per_k.get(k) or {}).get("recall") else "—" for k in ks)
            fraction = (per_k.get(ks[-1]) or {}).get("relevant_fraction")
            row += f" | {fraction:.2f} | {cell.metrics.get('first_relevant_rank') or '—'} | {cell.unjudged} |"
            out.append(row)
    out.append("")

    out.append("## Retrieved chunks (orig ▏reform)")
    out.append("")
    for base in BASE_STRATEGIES:
        orig = view.cell(f"{base}__orig")
        other = view.cell(f"{base}__reform")
        if orig is None and other is None:
            continue
        out.append(f"### `{base}`")
        out.append("")
        out.append("| # | orig | reform |")
        out.append("|---|---|---|")
        depth = max(len(orig.chunks) if orig else 0, len(other.chunks) if other else 0)
        for index in range(depth):
            left = orig.chunks[index] if orig and index < len(orig.chunks) else None
            right = other.chunks[index] if other and index < len(other.chunks) else None
            out.append(f"| {index + 1} | {_cell_line(left)} | {_cell_line(right)} |")
        out.append("")
        if other is not None:
            orig_ids = [chunk.chunk_id for chunk in (orig.chunks if orig else [])]
            other_ids = [chunk.chunk_id for chunk in other.chunks]
            only_orig = [c for c in orig_ids if c not in set(other_ids)]
            only_reform = [c for c in other_ids if c not in set(orig_ids)]
            out.append(f"- only in orig ({len(only_orig)}): {_ids(only_orig)}")
            out.append(f"- only in reform ({len(only_reform)}): {_ids(only_reform)}")
            out.append("")

    out.append("## Chunk bodies (gold-supporting first)")
    out.append("")
    ordered = sorted(
        {chunk.chunk_id: chunk for cell in view.cells for chunk in cell.chunks}.values(),
        key=lambda chunk: (not chunk.gold_hit, chunk.rank),
    )
    for chunk in ordered:
        badge = RELEVANCE_BADGE.get(chunk.relevance, "?")
        star = " ⭐ gold-supporting" if chunk.gold_hit else ""
        summary = f"<code>{chunk.chunk_id}</code> — {chunk.title or '(no title)'} · {badge}{star}"
        out.append(f"<details{' open' if chunk.gold_hit else ''}><summary>{summary}</summary>")
        out.append("")
        out.append(chunk.content or "_body not resolved — see `chunk_source` above_")
        if chunk.reason:
            out.append("")
            out.append(f"*judge: {chunk.reason}*")
        out.append("")
        out.append("</details>")
        out.append("")
    if view.notes:
        out.append("## Notes")
        out.append("")
        for key, value in view.notes.items():
            out.append(f"- `{key}`: {value}")
        out.append("")
    return "\n".join(out)


def _ids(chunk_ids: Sequence[str], limit: int = 12) -> str:
    if not chunk_ids:
        return "—"
    shown = ", ".join(f"`{chunk_id}`" for chunk_id in chunk_ids[:limit])
    return shown + (f" …(+{len(chunk_ids) - limit})" if len(chunk_ids) > limit else "")





# --------------------------------------------------------------------------
# html
# --------------------------------------------------------------------------

CSS = """
:root { --rel:#0a7d3c; --par:#b26a00; --irr:#8b8b8b; --gold:#c026a3; --line:#d8d8d8; }
* { box-sizing: border-box; }
body { font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
       margin: 0; color: #17181a; background: #fbfbfa; }
header.page { position: sticky; top: 0; z-index: 5; background: #fff;
       border-bottom: 2px solid #17181a; padding: 10px 18px; }
header.page h1 { font-size: 16px; margin: 0 0 4px; }
nav.qs { font-size: 12px; }
nav.qs a { display: inline-block; padding: 1px 5px; margin: 1px; border: 1px solid var(--line);
       border-radius: 3px; text-decoration: none; color: #17181a; background: #f2f2ef; }
nav.qs a.hit { border-color: var(--gold); }
main { padding: 14px 18px 60px; max-width: 1500px; }
section.q { border-top: 3px solid #17181a; margin: 26px 0; padding-top: 10px; }
h2 { font-size: 15px; margin: 0 0 6px; }
.gold { color: var(--gold); font-weight: 700; }
.qtext { margin: 6px 0; }
ol.opts { margin: 6px 0 10px; padding-left: 22px; }
ol.opts li.gold { color: var(--gold); }
table.meta { border-collapse: collapse; font-size: 12px; margin: 8px 0; }
table.meta td { border: 1px solid var(--line); padding: 2px 7px; vertical-align: top; }
table.meta td.k { color: #666; white-space: nowrap; }
.panel { border: 1px solid var(--line); border-left: 4px solid #2b6cb0; background: #fff;
       padding: 8px 10px; margin: 10px 0; border-radius: 3px; }
.panel.warn { border-left-color: #c53030; background: #fff5f5; }
.panel .need { font-style: italic; }
.badge { display: inline-block; font: 11px/1.4 ui-monospace, monospace; padding: 0 4px;
       border-radius: 3px; border: 1px solid var(--line); background: #eee; }
.badge.REL { color: #fff; background: var(--rel); border-color: var(--rel); }
.badge.PAR { color: #fff; background: var(--par); border-color: var(--par); }
.badge.IRR { color: #fff; background: var(--irr); border-color: var(--irr); }
.badge.miss { color: #fff; background: #444; border-style: dashed; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: start; }
.grid.one { grid-template-columns: 1fr; }
.col { border: 1px solid var(--line); border-radius: 4px; background: #fff; }
.col > h4 { margin: 0; padding: 5px 8px; border-bottom: 1px solid var(--line);
       font-size: 12px; background: #f4f4f1; }
.col.same { opacity: .7; }
details.chunk { border-bottom: 1px dashed #e6e6e2; padding: 4px 8px; }
details.chunk summary { cursor: pointer; list-style: none; }
details.chunk summary::-webkit-details-marker { display: none; }
.rank { display: inline-block; width: 28px; color: #666; font-family: ui-monospace, monospace; }
.moved { color: #2b6cb0; font-family: ui-monospace, monospace; font-size: 11px; }
.title { font-weight: 600; }
.idc { color: #8a8a8a; font-family: ui-monospace, monospace; font-size: 11px; }
.body { margin: 4px 0 4px 28px; white-space: pre-wrap; color: #2a2b2d; }
.why { margin: 2px 0 2px 28px; color: #555; font-size: 12px; }
.star { color: var(--gold); font-weight: 700; }
.opts-chip { font-family: ui-monospace, monospace; font-size: 11px; color: #444; }
table.metrics { border-collapse: collapse; font-size: 12px; margin: 8px 0 12px; }
table.metrics th, table.metrics td { border: 1px solid var(--line); padding: 2px 8px;
       text-align: center; }
table.metrics th { background: #f4f4f1; }
td.hit { background: #e2f5e9; font-weight: 700; }
td.miss { color: #b0b0b0; }
td.delta-up { background: #dff3e6; }
td.delta-down { background: #fde8e8; }
.diff { font-size: 12px; color: #555; margin: 4px 0 8px; padding: 0 8px; }
.flag { font-size: 11px; color: #c53030; }
"""


def _esc(text: Any) -> str:
    return html.escape(str(text if text is not None else ""), quote=False)


def _verdict(chunk: ChunkView) -> str:
    if not chunk.judged:
        return '<span class="badge miss" title="no judge verdict">?</span>'
    label = RELEVANCE_BADGE.get(chunk.relevance, "?")
    return f'<span class="badge {label}" title="{_esc(chunk.relevance)}">{label}</span>'


def _chunk_card(chunk: ChunkView) -> str:
    star = (
        ' <span class="star" title="judge: this chunk supports the gold option">★</span>'
        if chunk.gold_hit
        else ""
    )
    moved = (
        f'<span class="moved">#{chunk.prerank}&rarr;{chunk.rank}</span>'
        if chunk.prerank is not None and chunk.prerank != chunk.rank
        else ""
    )
    chips = " ".join(f'<span class="opts-chip">{_esc(o)}</span>' for o in chunk.supports_options)
    why = f'<div class="why">judge: {_esc(chunk.reason)}</div>' if chunk.reason else ""
    body = chunk.content or "(body not resolved -- see chunk_source in the header)"
    score = f' <span class="idc">{chunk.score:.4f}</span>' if isinstance(chunk.score, float) else ""
    return (
        f'<details class="chunk"{" open" if chunk.gold_hit else ""}>'
        f'<summary><span class="rank">{chunk.rank}</span>{_verdict(chunk)}{star} '
        f'<span class="title">{_esc(chunk.title or "(no title)")}</span> {chips} '
        f'<span class="idc">{_esc(chunk.chunk_id)}</span>{score} {moved}</summary>'
        f'<div class="body">{_esc(body)}</div>{why}</details>'
    )


def _column(cell: CellView | None, ks: Sequence[int]) -> str:
    if cell is None:
        return '<div class="col"><h4>(no such cell in this run)</h4></div>'
    per_k = cell.metrics.get("per_k", {})
    bits = " ".join(f'{"✓" if (per_k.get(k) or {}).get("recall") else "·"}@{k}' for k in ks)
    first = cell.metrics.get("first_relevant_rank")
    bits += f' · 1st rel {"#" + str(first) if first else "—"}'
    if cell.unjudged:
        bits += f' · <span class="flag">{cell.unjudged} unjudged</span>'
    head = f'<h4>{_esc(cell.label)} · {bits}</h4>'
    cards = "".join(_chunk_card(chunk) for chunk in cell.chunks) or '<div class="diff">(empty)</div>'
    if cell.reused:
        cards = (
            '<div class="diff"><span class="flag">identical to the orig column</span> — the '
            "reformulated query equalled the question, so nothing was re-retrieved.</div>" + cards
        )
    if cell.legacy:
        cards = (
            '<div class="diff"><span class="flag">legacy method id</span> — stored as '
            "<code>reform_dense</code> by a pre-grid run; see the run's FINDINGS.md "
            "annotation before quoting this cell.</div>" + cards
        )
    return f'<div class="col{" same" if cell.reused else ""}">{head}{cards}</div>'


def _reform_panel(view: QuestionView) -> str:
    reform = view.reform or {}
    if not reform.get("query") and not reform.get("fallback"):
        return (
            '<div class="panel"><b>Reformulation: off.</b> Only the <code>orig</code> '
            "column exists (<code>--no-reform</code>).</div>"
        )
    fallback = ""
    if reform.get("fallback"):
        fallback = (
            "<br><b>⚠ FELL BACK to the raw question</b> — the reform column <i>is</i> the "
            "orig column, so it measures nothing. Reason: "
            f'<code>{_esc(reform.get("error") or "not recorded by this run")}</code>'
        )
    elif reform.get("error"):
        fallback = f'<br><span class="flag">noted: {_esc(reform["error"])}</span>'
    query = (
        f'<div class="need">{_esc(reform.get("query"))}</div>'
        if reform.get("query")
        else '<div class="idc">(no reform query stored in this row)</div>'
    )
    return (
        f'<div class="panel{" warn" if reform.get("fallback") else ""}">'
        f'<b>Reformulation</b> <span class="badge">{_esc(reform.get("method", "?"))}</span> '
        f'<span class="idc">prompt: {_esc(reform.get("prompt_source", "?"))}</span>'
        f"{query}{fallback}"
        '<details><summary class="idc">prompt + raw response</summary>'
        f'<pre class="body">{_esc(reform.get("prompt", ""))}</pre>'
        f'<pre class="body">{_esc(reform.get("raw_response", ""))}</pre></details></div>'
    )


def _metrics_table(view: QuestionView) -> str:
    ks = list(view.ks)
    variants = view.variants()
    last = ks[-1]
    head = ['<tr><th rowspan="2">strategy</th>']
    for variant in variants:
        head.append(f'<th colspan="{len(ks) + 2}">{variant}</th>')
    if len(variants) > 1:
        head.append(f'<th rowspan="2">Δ@{last}</th>')
    head.append("</tr><tr>")
    for _variant in variants:
        head.extend(f"<th>@{k}</th>" for k in ks)
        head.append("<th>rel</th><th>1st</th>")
    head.append("</tr>")

    rows: list[str] = []
    for base in BASE_STRATEGIES:
        cells = {variant: view.cell(f"{base}__{variant}") for variant in variants}
        if not any(cells.values()):
            continue
        row = [f'<tr><td style="text-align:left">{_esc(base)}</td>']
        for variant in variants:
            cell = cells[variant]
            if cell is None:
                row.append(f'<td colspan="{len(ks) + 2}">—</td>')
                continue
            per_k = cell.metrics.get("per_k", {})
            for k in ks:
                hit = (per_k.get(k) or {}).get("recall")
                row.append(f'<td class="{"hit" if hit else "miss"}">{"✓" if hit else "·"}</td>')
            fraction = (per_k.get(last) or {}).get("relevant_fraction")
            row.append(f'<td>{fraction:.2f}</td>' if isinstance(fraction, float) else "<td>—</td>")
            first = cell.metrics.get("first_relevant_rank")
            row.append(f'<td>{"#" + str(first) if first else "—"}</td>')
        if len(variants) > 1:
            orig = (cells.get("orig").metrics.get("per_k", {}).get(last) or {}).get("recall") if cells.get("orig") else None
            reform_hit = (cells.get("reform").metrics.get("per_k", {}).get(last) or {}).get("recall") if cells.get("reform") else None
            if orig is None or reform_hit is None:
                row.append("<td>—</td>")
            else:
                mark = "same" if orig == reform_hit else ("+1" if reform_hit else "−1")
                css = "delta-up" if mark == "+1" else "delta-down" if mark == "−1" else ""
                row.append(f'<td class="{css}">{mark}</td>')
        row.append("</tr>")
        rows.append("".join(row))
    return f'<table class="metrics">{"".join(head)}{"".join(rows)}</table>'


def _diff_line(orig: CellView | None, other: CellView | None) -> str:
    if orig is None or other is None:
        return ""
    orig_ids = [chunk.chunk_id for chunk in orig.chunks]
    other_ids = [chunk.chunk_id for chunk in other.chunks]
    only_orig = [chunk_id for chunk_id in orig_ids if chunk_id not in set(other_ids)]
    only_other = [chunk_id for chunk_id in other_ids if chunk_id not in set(orig_ids)]
    if not only_orig and not only_other:
        return '<div class="diff">identical sets (order may still differ)</div>'
    return (
        f'<div class="diff">only in <b>orig</b> ({len(only_orig)}): {_esc(", ".join(only_orig[:10]) or "—")}'
        f'<br>only in <b>reform</b> ({len(only_other)}): {_esc(", ".join(only_other[:10]) or "—")}</div>'
    )


def _question_section(view: QuestionView) -> str:
    options = "".join(
        f'<li class="{"gold" if letter == view.gold_letter else ""}"><b>{_esc(letter)}</b> '
        f"{_esc(view.options[letter])}{' ⭐ gold' if letter == view.gold_letter else ''}</li>"
        for letter in sorted(view.options)
    )
    meta_rows = "".join(
        f'<tr><td class="k">{_esc(key)}</td><td>{_esc(value)}</td></tr>'
        for key, value in view.provenance.items()
    )
    note_rows = "".join(
        f'<tr><td class="k">note: {_esc(key)}</td><td class="flag">{_esc(value)}</td></tr>'
        for key, value in view.notes.items()
    )
    variants = view.variants()
    blocks: list[str] = []
    for base in BASE_STRATEGIES:
        orig = view.cell(f"{base}__orig")
        other = view.cell(f"{base}__reform")
        if orig is None and other is None:
            continue
        block = [f'<h4 style="margin:14px 0 4px"><code>{_esc(base)}</code></h4>', '<div class="grid">']
        if "orig" in variants:
            block.append(_column(orig, view.ks))
        if "reform" in variants:
            block.append(_column(other, view.ks))
        block.append("</div>")
        block.append(_diff_line(orig, other))
        blocks.append("".join(block))
    single = ' one' if len(variants) < 2 else ""
    return (
        f'<section class="q" id="q-{_esc(view.question_id)}">'
        f'<h2>q{_esc(view.question_id)} · <span class="gold">gold {_esc(view.gold_letter)}: '
        f"{_esc(view.gold_text)}</span> <span class='idc'>{_esc(view.split)}</span></h2>"
        f'<div class="qtext">{_esc(view.question)}</div><ol class="opts">{options}</ol>'
        f'{_reform_panel(view)}{_metrics_table(view)}'
        f'<table class="meta">{meta_rows}{note_rows}</table>'
        + "".join(blocks).replace('class="grid"', f'class="grid{single}"')
        + "</section>"
    )


def aggregate_views(views: Sequence[QuestionView]) -> dict[str, dict[str, Any]]:
    """Mean per-cell metrics across the questions in a report.

    Recomputed here instead of imported from a run's table because a report can
    mix sources (one old run, two live questions), and a summary that quietly
    averages a different set than the sections below it is worse than no summary.
    """
    totals: dict[str, dict[str, Any]] = {}
    for view in views:
        for cell in view.cells:
            bucket = totals.setdefault(
                cell.method,
                {"n": 0, "per_k": {k: [0, 0.0] for k in view.ks}, "ranks": []},
            )
            bucket["n"] += 1
            for k in view.ks:
                entry = cell.metrics.get("per_k", {}).get(k) or {}
                bucket["per_k"][k][0] += 1 if entry.get("recall") else 0
                bucket["per_k"][k][1] += entry.get("relevant_fraction") or 0.0
            rank = cell.metrics.get("first_relevant_rank")
            if rank:
                bucket["ranks"].append(rank)
    summary: dict[str, dict[str, Any]] = {}
    for method, bucket in totals.items():
        n = bucket["n"]
        summary[method] = {
            "n": n,
            "per_k": {
                k: {
                    "recall": round(hits / n, 3),
                    "rel": round(sum_value / n, 3),
                }
                for k, (hits, sum_value) in bucket["per_k"].items()
            },
            "mean_first_relevant": round(sum(bucket["ranks"]) / len(bucket["ranks"]), 2)
            if bucket["ranks"]
            else None,
        }
    return summary


def render_summary_table(views: Sequence[QuestionView]) -> str:
    summary = aggregate_views(views)
    if not summary:
        return ""
    ks = list(views[0].ks)
    head = ["<tr><th rowspan='2'>method</th>"]
    for k in ks:
        head.append(f"<th colspan='2'>@{k}</th>")
    head.append("<th rowspan='2'>n</th><th rowspan='2'>mean 1st rel</th></tr><tr>")
    for _k in ks:
        head.append("<th>recall</th><th>rel</th>")
    head.append("</tr>")
    rows = []
    for method in METHOD_GRID:
        if method not in summary:
            continue
        entry = summary[method]
        row = f"<tr><td style='text-align:left'><code>{_esc(method)}</code></td>"
        for k in ks:
            per_k = entry["per_k"][k]
            row += f"<td>{per_k['recall']:.2f}</td><td>{per_k['rel']:.2f}</td>"
        row += f"<td>{entry['n']}</td><td>{entry['mean_first_relevant'] or '—'}</td></tr>"
        rows.append(row)
    return f"<h3>Across these {len(views)} question(s)</h3><table class='metrics'>{''.join(head)}{''.join(rows)}</table>"


def render_html(
    views: Sequence[QuestionView],
    *,
    title: str = "Retrieval inspection",
    meta: Mapping[str, Any] | None = None,
) -> str:
    """One self-contained HTML file: header, summary, then one section per question.

    The header carries the provenance that makes the numbers re-runnable (index
    hash, judge-prompt sha, k values, chunk source), and the banner counts
    questions whose reformulation fell back -- the single fact that would have
    stopped the first run's `reform_dense` column from being read as a result.
    """
    meta = dict(meta or {})
    fallbacks = [view.question_id for view in views if (view.reform or {}).get("fallback")]
    last_k = views[0].ks[-1] if views else 0
    links = []
    for view in views:
        hit = any(
            (cell.metrics.get("per_k", {}).get(last_k) or {}).get("recall") for cell in view.cells
        )
        links.append(f'<a href="#q-{_esc(view.question_id)}" class="{"hit" if hit else ""}">q{_esc(view.question_id)}</a>')
    banner = ""
    if fallbacks:
        banner = (
            f'<div class="panel warn"><b>{len(fallbacks)} of {len(views)}</b> question(s) had a '
            "reformulation that fell back to the raw question, so their <code>*__reform</code> "
            "columns are copies of <code>*__orig</code> and must not be read as measurements: "
            f"{_esc(', '.join(fallbacks))}</div>"
        )
    meta_table = "".join(
        f'<tr><td class="k">{_esc(key)}</td><td>{_esc(value)}</td></tr>' for key, value in meta.items()
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{_esc(title)}</title><style>{CSS}</style></head><body>"
        f"<header class='page'><h1>{_esc(title)}</h1>"
        f"<div class='idc'>{_esc(meta.get('generated', ''))}</div>"
        f"<nav class='qs'>{''.join(links)}</nav></header><main>"
        f"<table class='meta'>{meta_table}</table>{banner}"
        f"{render_summary_table(views)}"
        + "".join(_question_section(view) for view in views)
        + "</main></body></html>"
    )





