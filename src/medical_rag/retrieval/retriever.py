"""Query-time retrieval interface.

Will combine the embedder and FAISS index to serve top-k relevant chunks
for a given query, for use by the generation and experiment modules.

TODO: class Retriever: load index + embedder + chunk metadata.
TODO: Retriever.retrieve(query, k) -> list[dict] of retrieved chunks.
"""
