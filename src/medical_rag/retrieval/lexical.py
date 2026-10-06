"""Lightweight lexical (BM25) index over the StatPearls corpus.

Nothing like it existed in the package before: retrieval was dense only
(`medical_rag.retrieval.Retriever`). This module builds one in-memory with
`rank_bm25.BM25Okapi` and pickles it to disk so `eval/harness.py` can score a
lexical and a hybrid (dense+lexical) lever against the dense baseline. It was
promoted out of `experiments/retrieval_tuning/` in R1 -- the first grid made it
load-bearing, and `bm25__orig`'s 0/0/10 made it the thing R4 has to fix.

The tokenizer is a plain lowercase `[a-z0-9]+` splitter -- no stemming, no
stopword removal, no field weighting. That is the shape of the degeneracy the
first grid measured (59 distinct chunks filling 100 top-5 slots), and it is a
scheduled lever, not an accident: R4 changes this tokenizer, which is also why the
pickled artifact carries no version stamp yet.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from rank_bm25 import BM25Okapi

from medical_rag.retrieval.index import load_index
from medical_rag.retrieval.retriever import RetrievedChunk

DEFAULT_BM25_PATH = Path("data/index/statpearls_bm25.pkl")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A pickle embeds the *module path* of the class it holds, so the artifact built
# before R1 -- when this file was `experiments/retrieval_tuning/bm25_index.py` --
# names `bm25_index.BM25Index`, a module that no longer exists. Without the remap
# the first load dies with `ModuleNotFoundError: No module named 'bm25_index'`,
# which reads like data corruption and is really a rename. Remap rather than
# demand a rebuild: the artifact is 445 MB of tokenized corpus whose class has not
# changed, so the mapping is exact. Retire this when a rebuilt pickle is on disk
# (R4 changes the tokenizer and rebuilds anyway) -- the load below says so loudly
# instead of letting the shim go unnoticed.
_LEGACY_MODULE_PATHS = {"bm25_index": __name__}


class _LegacyUnpickler(pickle.Unpickler):
    """`pickle.Unpickler` that resolves pre-rename class paths and records it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.remapped: set[str] = set()

    def find_class(self, module: str, name: str) -> Any:
        if module in _LEGACY_MODULE_PATHS:
            self.remapped.add(module)
            module = _LEGACY_MODULE_PATHS[module]
        return super().find_class(module, name)



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
    """Load a pickled index, resolving pre-R1 class paths and saying so.

    A remap is not an error -- the class is the same one -- but it is a fact about
    the artifact worth printing, because it means the pickle predates the tokenizer
    it is being used with.
    """
    with Path(path).open("rb") as handle:
        unpickler = _LegacyUnpickler(handle)
        index = unpickler.load()

    for module in sorted(unpickler.remapped):
        logger.warning(
            "{} was written as `{}.BM25Index`, before this module moved; loaded "
            "through the compat remap. Rebuild (`--rebuild-bm25`) to retire the "
            "shim -- and rebuild anyway whenever the tokenizer below changes",
            path,
            module,
        )
    return index


def load_or_build_bm25_index(
    index_dir: str | Path, path: str | Path = DEFAULT_BM25_PATH, *, rebuild: bool = False
) -> BM25Index:
    """Reuse the pickled index unless `rebuild` is set or none exists yet."""
    if not rebuild and Path(path).exists():
        return load_bm25_index(path)
    return build_bm25_index(index_dir, path)
