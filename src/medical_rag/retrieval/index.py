"""Vector index construction and persistence (LanceDB-backed).

Builds an exact-search vector table from embedded chunks and provides
save/load helpers so the index can be built once (scripts/build_index.py)
and reused across experiment runs. LanceDB tables default to brute-force
exact kNN search as long as no ANN index is explicitly created on them, so
this stays equivalent to a flat, exact-search index (no approximate
retrieval semantics sneak in).
"""

from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.table import Table

TABLE_NAME = "chunks"


def build_index(chunks: list[dict], embeddings: np.ndarray, path: str | Path) -> Table:
    """Create a LanceDB table at `path` with one row per chunk plus its vector."""
    dim = embeddings.shape[1]
    vector_array = pa.FixedSizeListArray.from_arrays(pa.array(embeddings.reshape(-1)), dim)

    data = pa.table(
        {
            "chunk_id": [chunk["chunk_id"] for chunk in chunks],
            "title": [chunk["title"] for chunk in chunks],
            "content": [chunk["content"] for chunk in chunks],
            "vector": vector_array,
        }
    )

    db = lancedb.connect(str(path))
    return db.create_table(TABLE_NAME, data, mode="overwrite")


def load_index(path: str | Path) -> Table:
    db = lancedb.connect(str(path))
    return db.open_table(TABLE_NAME)
