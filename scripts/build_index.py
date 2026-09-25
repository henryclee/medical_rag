"""CLI script to build the LanceDB retrieval index.

Loads a corpus, embeds every chunk's `contents` field, builds a LanceDB
vector table, and persists it to disk for later use by the retriever and
experiment runner. Idempotent: re-running with an already-built index
directory does not re-embed unless --force is passed.
"""

import argparse
import shutil
import time
from pathlib import Path

from loguru import logger

from medical_rag.data.load_statpearls import load_statpearls
from medical_rag.retrieval.embedder import Embedder
from medical_rag.retrieval.index import build_index

CORPUS_LOADERS = {
    "statpearls": load_statpearls,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=sorted(CORPUS_LOADERS), default="statpearls")
    parser.add_argument("--output-dir", default=None, help="Defaults to data/index/<corpus>")
    parser.add_argument("--data-cache-dir", default=None, help="Defaults to data/<corpus>")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true", help="Rebuild even if the index already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir or f"data/index/{args.corpus}")

    if output_dir.exists() and not args.force:
        logger.info("Index already exists at {} (use --force to rebuild)", output_dir)
        return
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)

    data_cache_dir = args.data_cache_dir or f"data/{args.corpus}"
    load_corpus = CORPUS_LOADERS[args.corpus]

    logger.info("Loading corpus '{}'", args.corpus)
    chunks = load_corpus(cache_dir=data_cache_dir)

    logger.info("Embedding {} chunks with '{}'", len(chunks), args.embedding_model)
    embedder = Embedder(args.embedding_model)
    start = time.monotonic()
    embeddings = embedder.embed_texts([chunk.contents for chunk in chunks], batch_size=args.batch_size)
    elapsed = time.monotonic() - start
    logger.info("Embedded {} chunks in {:.1f}s -> shape {}", len(chunks), elapsed, embeddings.shape)

    logger.info("Building LanceDB index at {}", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_index([chunk.model_dump() for chunk in chunks], embeddings, output_dir)

    size_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    logger.info(
        "Saved index ({} rows, {:.1f}MB) to {}",
        table.count_rows(),
        size_bytes / (1 << 20),
        output_dir,
    )


if __name__ == "__main__":
    main()
