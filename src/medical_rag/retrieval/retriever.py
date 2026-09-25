"""Query-time retrieval interface.

Combines the embedder and LanceDB vector table to serve top-k relevant
chunks for a query, and reranks candidates with a cross-encoder, for use
by the generation and experiment modules.
"""

from lancedb.table import Table
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from medical_rag.retrieval.embedder import Embedder


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    title: str
    score: float


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        table: Table,
        reranker_model: str,
    ):
        self.embedder = embedder
        self.table = table
        self.reranker_model = reranker_model
        self._cross_encoder: CrossEncoder | None = None

    @property
    def cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(self.reranker_model, device=self.embedder.device)
        return self._cross_encoder

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_query(query)
        rows = self.table.search(query_embedding).metric("cosine").limit(top_k).to_list()

        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                content=row["content"],
                title=row["title"],
                # LanceDB's cosine metric returns a distance (1 - similarity);
                # convert back to similarity so higher score = more relevant.
                score=1.0 - row["_distance"],
            )
            for row in rows
        ]

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self.cross_encoder.predict(pairs)

        reranked = sorted(zip(chunks, scores), key=lambda pair: -pair[1])[:top_k]
        return [chunk.model_copy(update={"score": float(score)}) for chunk, score in reranked]
