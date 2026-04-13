"""
Resume / demo retriever construction without Streamlit session state.

Used by FastAPI and shared generation paths.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Tuple

from langchain.docstore.document import Document

from portfolio_data import load_portfolio_documents
from rag_pipeline import (
    USER_RESUME_PERSIST_DIR,
    build_portfolio_retriever,
    build_retriever_from_documents,
    retrieve_relevant_portfolio_context,
)
from resume_parser import build_candidate_profile_summary, resume_text_to_documents
from scraper import JobDescription


# Small in-process cache to avoid rebuilding Chroma for identical resume text.
_RETRIEVER_CACHE: Dict[str, Any] = {}
_CACHE_ORDER: list[str] = []
_MAX_CACHE = 8
_DEMO_RETRIEVER: Any | None = None


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_put(fp: str, retriever: Any) -> None:
    if fp in _RETRIEVER_CACHE:
        return
    _RETRIEVER_CACHE[fp] = retriever
    _CACHE_ORDER.append(fp)
    while len(_CACHE_ORDER) > _MAX_CACHE:
        old = _CACHE_ORDER.pop(0)
        _RETRIEVER_CACHE.pop(old, None)


def get_retriever_and_profile(
    use_resume: bool,
    resume_text: str | None,
) -> Tuple[Any, str | None, bool]:
    """
    Returns (retriever, candidate_profile, resume_mode).
    """
    if use_resume:
        if not resume_text or not resume_text.strip():
            raise ValueError("Resume text is required when use_resume is true.")
        fp = _fingerprint(resume_text)
        if fp not in _RETRIEVER_CACHE:
            docs = resume_text_to_documents(resume_text)
            if not docs:
                raise ValueError("Could not parse resume text.")
            retriever = build_retriever_from_documents(
                docs, persist_dir=USER_RESUME_PERSIST_DIR, k=6
            )
            _cache_put(fp, retriever)
        retriever = _RETRIEVER_CACHE[fp]
        profile = build_candidate_profile_summary(resume_text)
        return retriever, profile, True

    global _DEMO_RETRIEVER
    if _DEMO_RETRIEVER is None:
        portfolio_docs = load_portfolio_documents()
        _DEMO_RETRIEVER = build_portfolio_retriever(portfolio_docs, k=4)
    return _DEMO_RETRIEVER, None, False


def retrieve_context_for_job(
    job: JobDescription,
    use_resume: bool,
    resume_text: str | None,
) -> Tuple[list[Document], str | None, bool]:
    """Run RAG retrieval for the given job."""
    retriever, candidate_profile, resume_mode = get_retriever_and_profile(
        use_resume, resume_text
    )
    docs = retrieve_relevant_portfolio_context(job.as_plain_text(), retriever)
    return docs, candidate_profile, resume_mode
