"""Lightweight lexical (BM25) index over the StatPearls corpus.

No lexical index exists elsewhere in the repo -- retrieval so far is dense
only (`medical_rag.retrieval.Retriever`). This module builds one in-memory
with `rank_bm25.BM25Okapi` and pickles it to disk, purely to let
`judge_harness.py` compare a lexical and a hybrid (dense+lexical) retrieval
lever against the dense baseline. It lives here, not in
`src/medical_rag/retrieval/`, because it is unproven: promote it into
production code only if a lever built on it wins at this tuning stage
(`experiments/retrieval_tuning/TUNING.md`).

The tokenizer is a plain lowercase `[a-z0-9]+` splitter -- no stemming, no
stopword removal. That is good enough to compare directionally against dense
retrieval; it is not a production-grade lexical index.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from medical_rag.retrieval.index import load_index
from medical_rag.retrieval.retriever import RetrievedChunk

DEFAULT_BM25_PATH = Path("data/index/statpearls_bm25.pkl")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    """A built BM25 index: parallel `chunk_id`/`title`/`content` lists plus the model."""

    chunk_ids: list[str]
    titles: list[str]
    contents: list[str]
    bm25: BM25Okapi

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [
            RetrievedChunk(
                chunk_id=self.chunk_ids[i],
                content=self.contents[i],
                title=self.titles[i],
                score=float(scores[i]),
            )
            for i in ranked
        ]


def build_bm25_index(index_dir: str | Path, out_path: str | Path = DEFAULT_BM25_PATH) -> BM25Index:
    """Tokenize every chunk in the LanceDB table at `index_dir` and pickle the result.

    One-time cost over 380,454 chunks (a couple of minutes); callers should
    prefer `load_or_build_bm25_index()`, which skips this when the pickle
    already exists.
    """
    table = load_index(index_dir)
    frame = table.to_pandas()
    chunk_ids = frame["chunk_id"].tolist()
    titles = frame["title"].tolist()
    contents = frame["content"].tolist()
    tokenized = [tokenize(content) for content in contents]
    bm25 = BM25Okapi(tokenized)

    index = BM25Index(chunk_ids=chunk_ids, titles=titles, contents=contents, bm25=bm25)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(index, handle)
    return index


def load_bm25_index(path: str | Path = DEFAULT_BM25_PATH) -> BM25Index:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_or_build_bm25_index(
    index_dir: str | Path, path: str | Path = DEFAULT_BM25_PATH, *, rebuild: bool = False
) -> BM25Index:
    """Reuse the pickled index unless `rebuild` is set or none exists yet."""
    if not rebuild and Path(path).exists():
        return load_bm25_index(path)
    return build_bm25_index(index_dir, path)
