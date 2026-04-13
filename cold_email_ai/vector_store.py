"""
Vector store utilities for the AI Cold Email Generator.

This module is responsible for:
  - building embeddings from portfolio documents
  - storing them in a ChromaDB vector store
  - exposing a retriever interface used by the RAG pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from services.model_provider import DEFAULT_EMBEDDING_MODEL, get_embeddings_model

DEFAULT_PERSIST_DIR = Path("chroma_db")


def build_vector_store(
    documents: List[Document],
    persist_dir: Path | str = DEFAULT_PERSIST_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Chroma:
    """
    Create (or overwrite) a Chroma vector store from the given documents.

    Parameters
    ----------
    documents:
        List of LangChain `Document` instances describing the candidate portfolio.
    persist_dir:
        Directory where the ChromaDB index will be stored.
    embedding_model:
        Google embedding model name (e.g. ``models/text-embedding-004``).
    """
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings_model(embedding_model)
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_path),
    )


def load_vector_store(
    persist_dir: Path | str = DEFAULT_PERSIST_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Chroma:
    """
    Load an existing Chroma vector store.

    If the directory does not yet contain a valid Chroma index, this function
    will still construct an empty store that can be populated later.
    """
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings_model(embedding_model)
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_path),
    )
    return vector_store


__all__ = ["build_vector_store", "load_vector_store", "DEFAULT_PERSIST_DIR"]

