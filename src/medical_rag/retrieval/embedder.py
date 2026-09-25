"""Embedding model wrapper for retrieval.

Wraps a sentence-transformers model to embed StatPearls chunks and queries
into a shared vector space for FAISS indexing and search. Uses BGE's
asymmetric retrieval convention: queries get an instruction prefix, indexed
passages do not.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _default_device() -> str:
    # On this project's dev machine, torch==2.14.0's CPU backend produces
    # NaN output for the cross-encoder reranker's BERT architecture
    # (verified: MPS gives correct, non-NaN scores for the same model).
    # Prefer cuda, then mps, and only fall back to cpu if neither exists.
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    def __init__(self, model_name: str, device: str | None = None):
        self.device = device or _default_device()
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_texts(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """Embed a batch of passages (no query instruction prefix)."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > batch_size,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query, with the BGE retrieval instruction prefix."""
        return self.embed_texts([BGE_QUERY_INSTRUCTION + query], batch_size=1)[0]
