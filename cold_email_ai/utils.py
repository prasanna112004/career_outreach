"""
Utility helpers for the AI Cold Email Generator.

Includes:
  - simple token counting using `tiktoken`
  - an email quality scoring heuristic
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import tiktoken
from langchain.docstore.document import Document

from scraper import JobDescription


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Estimate the number of tokens in a text string for a given model.

    This is useful to keep prompts within model limits and to debug
    prompt construction for a production‑style system.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to a reasonable default if the model is unknown.
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))


def score_email_quality(
    email_text: str,
    job: JobDescription,
    portfolio_docs: Sequence[Document],
    resume_text: str | None = None,
) -> float:
    """
    Heuristic scoring of generated email quality on a 1–10 scale.

    We approximate three aspects:
      1. Relevance to the job (mentions title/company and job skills)
      2. Personalization (mentions portfolio projects or candidate name)
      3. Clarity (length within reasonable bounds; no extremely short/long text)
    """
    if not email_text:
        return 1.0

    text_lower = email_text.lower()
    score = 0.0
    max_score = 10.0

    # 1) Relevance to job: title, company, skills overlap
    relevance_points = 0.0
    if job.title and job.title.lower().split()[0] in text_lower:
        relevance_points += 1.5
    if job.company and job.company.lower().split()[0] in text_lower:
        relevance_points += 1.5

    skills_overlap = 0
    for s in job.skills:
        if s.lower() in text_lower:
            skills_overlap += 1
    relevance_points += min(skills_overlap * 0.5, 3.0)
    relevance_points = min(relevance_points, 6.0)

    # 2) Personalization: overlap with resume / retrieved context
    personalization_points = 0.0
    portfolio_text = " ".join(doc.page_content for doc in portfolio_docs).lower()
    ref_text = (resume_text or "").lower() + " " + portfolio_text

    if resume_text:
        # Reward if email echoes non-trivial words from the user's resume (length > 4)
        resume_words = {
            w
            for w in ref_text.split()
            if len(w) > 4 and w.isalnum()
        }
        echo = sum(1 for w in list(resume_words)[:200] if w in text_lower)
        personalization_points += min(echo * 0.15, 2.5)
    else:
        if "alex doe" in text_lower or "alex" in text_lower:
            personalization_points += 1.0
        distinctive_terms = ["rag", "langchain", "streamlit", "mlops", "recommendation"]
        matches = sum(
            1
            for term in distinctive_terms
            if term in text_lower and term in portfolio_text
        )
        personalization_points += min(matches * 0.7, 3.0)

    personalization_points = min(personalization_points, 4.0)

    # 3) Clarity: reward emails within an approximate token / length band
    length = len(email_text.split())
    if 120 <= length <= 320:
        clarity_points = 3.0
    elif 80 <= length < 120 or 320 < length <= 450:
        clarity_points = 2.0
    else:
        clarity_points = 1.0

    score = relevance_points + personalization_points + clarity_points
    # Clamp and round to one decimal place, but return a float (UI can cast/format)
    score = max(1.0, min(score, max_score))
    return round(score, 1)


__all__ = ["count_tokens", "score_email_quality"]

