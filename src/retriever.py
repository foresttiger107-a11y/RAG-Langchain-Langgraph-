"""Document loading, splitting, and retriever setup."""

from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  # fallback

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    RETRIEVAL_TOP_K,
    VECTORSTORE_DIR,
)


def get_embeddings() -> OpenAIEmbeddings:
    """Create OpenAI embeddings instance."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required. Set it in .env file.")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def load_documents_from_urls(urls: List[str]) -> List[Document]:
    """Load documents from web URLs."""
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
    return docs


def load_documents_from_directory(
    directory: Optional[Path] = None, glob: str = "**/*.txt"
) -> List[Document]:
    """Load documents from a local directory."""
    path = directory or DATA_DIR
    if not path.exists():
        return []
    loader = DirectoryLoader(
        str(path),
        glob=glob,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split documents into smaller chunks for retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return text_splitter.split_documents(documents)


def create_vectorstore(
    documents: List[Document],
    persist_directory: Optional[Path] = None,
    collection_name: str = "rag_docs",
) -> Chroma:
    """Create and persist a Chroma vector store from documents."""
    embeddings = get_embeddings()
    persist_path = str(persist_directory or VECTORSTORE_DIR)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_path,
        collection_name=collection_name,
    )
    return vectorstore


def load_existing_vectorstore(
    persist_directory: Optional[Path] = None,
    collection_name: str = "rag_docs",
) -> Chroma:
    """Load an existing Chroma vector store."""
    embeddings = get_embeddings()
    persist_path = str(persist_directory or VECTORSTORE_DIR)

    return Chroma(
        persist_directory=persist_path,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def get_retriever(
    vectorstore: Chroma,
    k: int = RETRIEVAL_TOP_K,
    search_type: str = "similarity",
) -> VectorStoreRetriever:
    """Create a retriever from the vector store."""
    return vectorstore.as_retriever(search_kwargs={"k": k}, search_type=search_type)
