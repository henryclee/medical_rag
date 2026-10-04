"""Where a chunk's title and body come from, in three escalating costs.

A `results.jsonl` row stores `chunk_id` only -- right for an aggregate run,
wrong for a viewer: to *read* a retrieved chunk the inspector needs its text,
and the two places that have it are expensive. LanceDB needs a table handle (and
a filter query is a 380k-row scan), and `data/index/statpearls_bm25.pkl` is 467
MB to load for four strings.

So the harness writes a `chunks.jsonl` sidecar next to `results.jsonl` for
exactly the chunks it surfaced -- a few hundred rows, kilobytes -- and reading a
judged run back costs nothing. Older runs (the 2026-09-29 one) predate the
sidecar and fall through to LanceDB; a run with neither renders ids and verdicts
with empty bodies rather than pretending to have text it does not have.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from loguru import logger

from medical_rag.retrieval.retriever import RetrievedChunk

SIDECAR_NAME = "chunks.jsonl"


def write_chunks(path: str | Path, chunks: Iterable[RetrievedChunk]) -> Path:
    """Persist `(chunk_id, title, content)` for the chunks a question surfaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {"chunk_id": chunk.chunk_id, "title": chunk.title, "content": chunk.content},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def append_chunks(path: str | Path, chunks: Iterable[RetrievedChunk]) -> Path:
    """Append to a run's sidecar instead of rewriting it.

    Append-only is what makes the sidecar survive `--resume`: a run that died on
    question 12 keeps the 11 questions' text, and re-running appends the rest.
    Duplicate `chunk_id`s are fine -- `read_chunks` keys by id, last write wins,
    and the text of a `chunk_id` does not change inside one frozen index.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {"chunk_id": chunk.chunk_id, "title": chunk.title, "content": chunk.content},
                    ensure_ascii=False,
                )
                + "\n"
            )
        handle.flush()
    return path


def read_chunks(path: str | Path) -> dict[str, RetrievedChunk]:
    """`chunk_id -> RetrievedChunk` from a sidecar; `{}` if it does not exist.

    Score is 0.0: a sidecar chunk is a text lookup, not a ranking, and a score
    read back from a *different* method's list would be a lie about this cell.
    """
    path = Path(path)
    if not path.exists():
        return {}
    found: dict[str, RetrievedChunk] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("chunk sidecar: skipping torn line in {}", path)
                continue
            found[record["chunk_id"]] = RetrievedChunk(
                chunk_id=record["chunk_id"],
                title=record.get("title", ""),
                content=record.get("content", ""),
                score=0.0,
            )
    return found


def fetch_from_table(table: Any, chunk_ids: Sequence[str], *, batch: int = 200) -> dict[str, RetrievedChunk]:
    """Resolve ids through a filter-only LanceDB query -- no vector, no BM25 pickle.

    Batched because each query is a full scan of 380k rows: one query per chunk
    would make opening an old run take minutes for 40 strings.
    """
    wanted = [chunk_id for chunk_id in dict.fromkeys(chunk_ids) if chunk_id]
    found: dict[str, RetrievedChunk] = {}
    for start in range(0, len(wanted), batch):
        group = wanted[start : start + batch]
        quoted = ", ".join("'" + c.replace("'", "''") + "'" for c in group)
        rows = table.search().where(f"chunk_id IN ({quoted})").to_list()
        for row in rows:
            found[row["chunk_id"]] = RetrievedChunk(
                chunk_id=row["chunk_id"],
                title=row.get("title", ""),
                content=row.get("content", ""),
                score=0.0,
            )
    missing = [chunk_id for chunk_id in wanted if chunk_id not in found]
    if missing:
        logger.warning(
            "{} chunk_id(s) not in the index (stale ids from a rebuilt corpus?): {}",
            len(missing),
            ", ".join(missing[:5]),
        )
    return found


def resolve_chunks(
    chunk_ids: Sequence[str],
    *,
    run_dir: str | Path | None = None,
    table: Any = None,
) -> tuple[dict[str, RetrievedChunk], str]:
    """Sidecar first, then LanceDB, then nothing -- with a note saying which.

    Returns `(chunks, source)`; `source` is rendered into the report so a body
    that is missing because the sidecar does not exist for a 5-day-old run is
    visibly an artifact gap, not a retrieval bug.
    """
    if run_dir is not None:
        sidecar = Path(run_dir) / SIDECAR_NAME
        from_sidecar = read_chunks(sidecar)
        missing = [c for c in chunk_ids if c not in from_sidecar]
        if not missing:
            return from_sidecar, f"sidecar ({sidecar})"
        if table is None:
            return from_sidecar, (
                f"{'sidecar' if from_sidecar else 'no sidecar'} at {sidecar}; "
                f"{len(missing)} id(s) absent and no index handle to resolve them"
            )
        from_index = fetch_from_table(table, missing)
        label = "sidecar + LanceDB filter query" if from_sidecar else "LanceDB filter query (no sidecar in this run)"
        return {**from_index, **from_sidecar}, label

    if table is not None:
        return fetch_from_table(table, chunk_ids), "LanceDB filter query"
    return {}, "unresolved -- no sidecar and no index handle"
