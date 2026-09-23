"""FAISS index construction and persistence.

Will build a FAISS index from embedded chunks and provide save/load helpers
so the index can be built once (scripts/build_index.py) and reused across
experiment runs.

TODO: build_index(embeddings) -> faiss.Index.
TODO: save_index(index, path) / load_index(path).
TODO: save_chunk_metadata(chunks, path) / load_chunk_metadata(path).
"""
