"""
Unified outreach generation (email, LinkedIn, cover letter) for API and Streamlit.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from langchain.docstore.document import Document

from prompt_templates import build_outreach_prompt
from scraper import JobDescription
from .retriever_helpers import retrieve_context_for_job
from .model_provider import DEFAULT_CHAT_MODEL, get_chat_model
from utils import score_email_quality


def _ensure_email_shape(text: str, job: JobDescription) -> str:
    t = (text or "").strip()
    if not t:
        return t

    company = (job.company or "").strip()
    role = (job.title or "the role").strip()
    if company:
        t = t.replace("[Company]", company)

    # Ensure subject line exists and is separated.
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines or not lines[0].lower().startswith("subject:"):
        subj_company = company if company and "unknown" not in company.lower() else "your company"
        lines.insert(0, f"Subject: Application for {role} at {subj_company}")
    t = "\n".join(lines)

    # Ensure greeting starts on its own line.
    t = re.sub(r"(Subject:[^\n]+)\s+(Hi|Hello|Dear)\b", r"\1\n\n\2", t, flags=re.IGNORECASE)
    if not re.search(r"\b(hi|hello|dear)\b", t[:200], flags=re.IGNORECASE):
        t = t + "\n\nHi Hiring Manager,"

    # Force paragraph readability if model returns one giant block.
    body_lines = t.splitlines()
    rebuilt: list[str] = []
    for ln in body_lines:
        if len(ln) > 260 and "." in ln:
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ln) if s.strip()]
            chunk: list[str] = []
            for s in sents:
                chunk.append(s)
                if len(" ".join(chunk)) > 220:
                    rebuilt.append(" ".join(chunk))
                    chunk = []
            if chunk:
                rebuilt.append(" ".join(chunk))
        else:
            rebuilt.append(ln)
    t = "\n\n".join([ln for ln in rebuilt if ln.strip()])

    # Ensure closing exists.
    if not re.search(r"(best regards|regards|sincerely|thank you)", t.lower()):
        t = t.rstrip() + "\n\nThank you for your time and consideration.\n\nBest regards,"
    return t.strip()


def _ensure_linkedin_shape(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    # No subject in LinkedIn.
    t = re.sub(r"^subject:\s*[^\n]+\n*", "", t, flags=re.IGNORECASE).strip()
    paras = [p.strip() for p in re.split(r"\n{2,}", t) if p.strip()]
    if len(paras) == 1 and len(paras[0]) > 300:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paras[0]) if s.strip()]
        paras = []
        chunk: list[str] = []
        for s in sents:
            chunk.append(s)
            if len(" ".join(chunk)) > 180:
                paras.append(" ".join(chunk))
                chunk = []
        if chunk:
            paras.append(" ".join(chunk))
    return "\n\n".join(paras).strip()


def _normalize_generated_text(text: str, output_type: str) -> str:
    """
    Enforce practical length bounds so output respects selected format.
    """
    t = (text or "").strip()
    if not t:
        return t

    if output_type == "linkedin":
        # Keep LinkedIn messages concise and mobile-friendly.
        max_chars = 900
        return t[:max_chars].rstrip() if len(t) > max_chars else t

    words = t.split()
    if output_type == "email":
        max_words = 260
    else:  # cover letter
        max_words = 460

    if len(words) <= max_words:
        return t
    clipped = " ".join(words[:max_words]).strip()
    if not re.search(r"[.!?]$", clipped):
        clipped += "."
    return clipped


def generate_outreach_content(
    job: JobDescription,
    tone: str,
    chat_model: str = DEFAULT_CHAT_MODEL,
    *,
    use_resume: bool,
    resume_text: str | None,
    output_type: str = "email",
) -> Tuple[str, float, List[Document], str | None]:
    """
    Returns generated text, heuristic quality score, retrieved docs, resume text if any.
    """
    context_docs, candidate_profile, resume_mode = retrieve_context_for_job(
        job, use_resume, resume_text
    )

    prompt = build_outreach_prompt(
        job=job,
        portfolio_docs=context_docs,
        tone=tone,
        candidate_profile=candidate_profile,
        resume_mode=resume_mode,
        output_type=output_type,
    )

    llm = get_chat_model(chat_model, temperature=0.6)
    chain = prompt | llm
    response = chain.invoke({})
    text: str = response.content if hasattr(response, "content") else str(response)
    text = _normalize_generated_text(text, output_type)
    if output_type == "email":
        text = _ensure_email_shape(text, job)
    elif output_type == "linkedin":
        text = _ensure_linkedin_shape(text)

    quality = score_email_quality(
        text,
        job,
        context_docs,
        resume_text=resume_text if use_resume else None,
    )
    return text, quality, context_docs, resume_text if use_resume else None
