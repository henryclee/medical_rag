"""CLI script to build the FAISS retrieval index.

Will load StatPearls documents, chunk them, embed the chunks, build a FAISS
index, and persist the index and chunk metadata to disk for later use by
the experiment runner.

TODO: parse CLI args (data dir, output dir, chunk size, embedding model).
TODO: main() -> orchestrates load -> chunk -> embed -> build_index -> save.
"""
