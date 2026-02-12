"""Index documents into the vector store for RAG."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import (
    create_vectorstore,
    load_documents_from_directory,
    load_documents_from_urls,
    split_documents,
)
from src.config import DATA_DIR, VECTORSTORE_DIR

# Default URLs - Lilian Weng's blog posts (from LangChain tutorial)
DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]


def main(use_urls: bool = True, use_local: bool = False):
    """Index documents from URLs and/or local directory."""
    documents = []

    if use_urls:
        print("Loading documents from URLs...")
        docs = load_documents_from_urls(DEFAULT_URLS)
        documents.extend(docs)
        print(f"  Loaded {len(docs)} documents from web")

    if use_local and DATA_DIR.exists():
        print("Loading documents from local directory...")
        docs = load_documents_from_directory(DATA_DIR)
        documents.extend(docs)
        print(f"  Loaded {len(docs)} documents from {DATA_DIR}")

    if not documents:
        print(
            "No documents to index. Add files to the 'data/' folder "
            "or use default URLs (use_urls=True)."
        )
        return

    print("Splitting documents into chunks...")
    splits = split_documents(documents)
    print(f"  Created {len(splits)} chunks")

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    print("Creating vector store...")
    create_vectorstore(splits, persist_directory=VECTORSTORE_DIR)
    print(f"  Vector store saved to {VECTORSTORE_DIR}")
    print("Indexing complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-urls",
        action="store_true",
        help="Skip loading from default URLs",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Also load from local data/ directory",
    )
    args = parser.parse_args()

    main(use_urls=not args.no_urls, use_local=args.local)
