"""Embedding model wrapper for retrieval.

Will wrap a sentence-transformers model to embed chunks and queries into a
shared vector space for FAISS indexing and search.

TODO: class Embedder: load a sentence-transformers model.
TODO: Embedder.embed_texts(texts) -> np.ndarray of embeddings.
TODO: Embedder.embed_query(query) -> np.ndarray embedding.
"""
