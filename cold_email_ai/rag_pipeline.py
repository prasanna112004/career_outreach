"""
RAG (Retrieval‑Augmented Generation) pipeline for the AI Cold Email Generator.

This module wires together:
  - text chunking (RecursiveCharacterTextSplitter)
  - the portfolio vector store
  - a semantic retriever to fetch the most relevant portfolio context
    for a given job description.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import List

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from vector_store import build_vector_store, load_vector_store, DEFAULT_PERSIST_DIR

# Separate Chroma directory so user resume indexes do not overwrite the demo portfolio.
USER_RESUME_PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db_resume"


class LexicalRetriever:
    """Lightweight retriever fallback that does not require embeddings."""

    def __init__(self, chunks: List[Document], k: int = 6):
        self._chunks = chunks
        self._k = max(1, int(k))

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            t
            for t in re.findall(r"[A-Za-z][A-Za-z0-9\+\#\.\-]{1,32}", (text or "").lower())
            if len(t) >= 3
        }

    def _rank(self, query: str) -> List[Document]:
        q = self._tokens(query)
        if not q:
            return self._chunks[: self._k]
        scored: list[tuple[int, Document]] = []
        for doc in self._chunks:
            d = self._tokens(doc.page_content)
            score = len(q & d)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        ranked = [d for _, d in scored]
        if len(ranked) < self._k:
            seen = {id(d) for d in ranked}
            ranked.extend([d for d in self._chunks if id(d) not in seen])
        return ranked[: self._k]

    def invoke(self, query: str) -> List[Document]:
        return self._rank(query)

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self._rank(query)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:
    """
    Split portfolio documents into overlapping chunks.

    RecursiveCharacterTextSplitter works well for long free‑form text and is
    widely used in production‑grade RAG systems.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(documents)


def build_retriever_from_documents(
    documents: List[Document],
    persist_dir: Path | str,
    k: int = 6,
):
    """
    Chunk documents, embed, persist to Chroma, and return a similarity retriever.

    Use a dedicated ``persist_dir`` for user resumes so indexes stay isolated.
    """
    if not documents:
        raise ValueError("No documents to index.")
    chunks = chunk_documents(documents)
    try:
        vector_store = build_vector_store(chunks, persist_dir=persist_dir)
        return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    except Exception:
        # Fallback for environments where embedding models are unavailable.
        return LexicalRetriever(chunks, k=k)


def build_portfolio_retriever(
    documents: List[Document],
    k: int = 4,
):
    """
    Build a semantic similarity retriever from portfolio documents.

    This will:
      1. Chunk the documents
      2. Index them in Chroma
      3. Return a retriever interface
    """
    return build_retriever_from_documents(
        documents, persist_dir=DEFAULT_PERSIST_DIR, k=k
    )


def load_portfolio_retriever(k: int = 4):
    """
    Load an existing portfolio retriever from disk.

    This assumes that the portfolio vector store has already been built.
    """
    vector_store = load_vector_store(persist_dir=DEFAULT_PERSIST_DIR)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever


def retrieve_relevant_portfolio_context(
    job_description_text: str,
    retriever,
) -> List[Document]:
    """
    Run a semantic similarity query over the portfolio retriever.

    Returns a ranked list of portfolio chunks that are most relevant to the
    provided job description.
    """
    # Prefer invoke() (LangChain 0.2+); fallback for older retriever APIs.
    if hasattr(retriever, "invoke"):
        return list(retriever.invoke(job_description_text))
    return list(retriever.get_relevant_documents(job_description_text))


__all__ = [
    "chunk_documents",
    "build_retriever_from_documents",
    "build_portfolio_retriever",
    "USER_RESUME_PERSIST_DIR",
    "load_portfolio_retriever",
    "retrieve_relevant_portfolio_context",
]

