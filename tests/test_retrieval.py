"""Tests for medical_rag.retrieval (embedder, index, retriever + reranker).

Uses the real (small) bge-small-en-v1.5 and ms-marco-MiniLM-L-6-v2 models
against a tiny in-memory fixture corpus -- no StatPearls/NCBI network
dependency. Models are downloaded once and cached by Hugging Face.
"""

from pathlib import Path

import numpy as np

from medical_rag.retrieval.embedder import Embedder
from medical_rag.retrieval.index import build_index, load_index
from medical_rag.retrieval.retriever import RetrievedChunk, Retriever

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FIXTURE_CHUNKS = [
    {"chunk_id": "0", "title": "Diabetes", "content": "Metformin is a first-line treatment for type 2 diabetes mellitus."},
    {"chunk_id": "1", "title": "Hypertension", "content": "ACE inhibitors are commonly used to treat high blood pressure."},
    {"chunk_id": "2", "title": "Asthma", "content": "Inhaled corticosteroids reduce airway inflammation in asthma."},
    {"chunk_id": "3", "title": "Cooking", "content": "Preheat the oven to 350 degrees before baking the bread."},
]


def _build_test_retriever(tmp_path: Path) -> Retriever:
    embedder = Embedder(EMBEDDING_MODEL)
    embeddings = embedder.embed_texts([c["content"] for c in FIXTURE_CHUNKS])
    table = build_index(FIXTURE_CHUNKS, embeddings, tmp_path / "index")
    return Retriever(embedder, table, RERANKER_MODEL)


def test_embed_texts_shape():
    embedder = Embedder(EMBEDDING_MODEL)
    embeddings = embedder.embed_texts([c["content"] for c in FIXTURE_CHUNKS])
    assert embeddings.shape[0] == len(FIXTURE_CHUNKS)
    assert embeddings.dtype == np.float32


def test_index_save_load_roundtrip(tmp_path: Path):
    embedder = Embedder(EMBEDDING_MODEL)
    embeddings = embedder.embed_texts([c["content"] for c in FIXTURE_CHUNKS])
    index_path = tmp_path / "index"
    build_index(FIXTURE_CHUNKS, embeddings, index_path)

    query_embedding = embedder.embed_query("What treats diabetes?")

    reloaded = load_index(index_path)
    results = reloaded.search(query_embedding).metric("cosine").limit(2).to_list()

    assert len(results) == 2
    assert results[0]["chunk_id"] == "0"


def test_retrieve_surfaces_relevant_chunk(tmp_path: Path):
    retriever = _build_test_retriever(tmp_path)
    results = retriever.retrieve("What medication treats diabetes?", top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert results[0].chunk_id == "0"


def test_rerank_reorders_results(tmp_path: Path):
    retriever = _build_test_retriever(tmp_path)
    # Deliberately give the reranker a candidate list where raw embedding
    # similarity and cross-encoder relevance are expected to disagree.
    candidates = [
        RetrievedChunk(chunk_id="3", content=FIXTURE_CHUNKS[3]["content"], title="Cooking", score=0.9),
        RetrievedChunk(chunk_id="0", content=FIXTURE_CHUNKS[0]["content"], title="Diabetes", score=0.1),
    ]

    reranked = retriever.rerank("What medication treats diabetes?", candidates, top_k=2)

    assert reranked[0].chunk_id == "0"
