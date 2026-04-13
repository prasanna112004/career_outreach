"""Centralized Groq + optional embeddings model providers."""

from __future__ import annotations

import os

from langchain_groq import ChatGroq

DEFAULT_CHAT_MODEL = "llama-3.1-8b-instant"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def _resolve_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st  # type: ignore

        secret_key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
        if secret_key:
            return secret_key
    except Exception:
        pass
    raise ValueError(
        "GROQ_API_KEY is not set. Add it to your environment or Streamlit secrets."
    )


def get_chat_model(model: str | None = None, temperature: float = 0.4):
    return ChatGroq(
        model=(model or DEFAULT_CHAT_MODEL),
        temperature=temperature,
        api_key=_resolve_api_key(),
    )


def get_embeddings_model(model: str | None = None):
    # Lazy import so Streamlit Cloud can boot even if sentence-transformers/torchvision
    # are unavailable. RAG pipeline already has lexical fallback retrieval.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=(model or DEFAULT_EMBEDDING_MODEL),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
