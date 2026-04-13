"""
Prompt templates for the AI Cold Email Generator.

The main entry point is `build_cold_email_prompt`, which formats a
structured prompt including:
  - candidate introduction
  - relevant portfolio context
  - alignment between skills and job requirements
  - a requested tone for the email
"""

from __future__ import annotations

from typing import Iterable, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from scraper import JobDescription


ALLOWED_TONES = {"professional", "friendly", "direct"}

# Multi-platform outreach formats
OUTPUT_TYPES = {"email", "linkedin", "cover_letter"}


def _normalize_output_type(output_type: str) -> str:
    if not output_type:
        return "email"
    o = output_type.lower().strip()
    if o not in OUTPUT_TYPES:
        return "email"
    return o


def _normalize_tone(tone: str) -> str:
    """Normalize and validate tone input."""
    if not tone:
        return "professional"
    tone_lower = tone.lower().strip()
    if tone_lower not in ALLOWED_TONES:
        return "professional"
    return tone_lower


def format_portfolio_context(docs: Iterable[Document]) -> str:
    """
    Convert retrieved portfolio documents into a compact, human‑readable text
    block that the LLM can condition on.
    """
    lines: List[str] = []
    for i, doc in enumerate(docs, start=1):
        section = doc.metadata.get("section", "portfolio")
        project_name = doc.metadata.get("project_name")
        header_parts = [f"{i}. [{section}]"]
        if project_name:
            header_parts.append(project_name)
        header = " ".join(header_parts)
        lines.append(header + ":\n" + doc.page_content.strip() + "\n")
    return "\n".join(lines) if lines else "No portfolio context available."


def build_outreach_prompt(
    job: JobDescription,
    portfolio_docs: Iterable[Document],
    tone: str = "professional",
    candidate_profile: str | None = None,
    resume_mode: bool = False,
    output_type: str = "email",
) -> ChatPromptTemplate:
    """
    Create a `ChatPromptTemplate` for generating a personalized cold email.

    The system and human messages are carefully structured so the model
    understands the candidate, the job, and the retrieved portfolio context.

    When ``resume_mode`` is True, the candidate profile and retrieved chunks
    come from the user's uploaded resume; the model must not invent employers,
    dates, or metrics not supported by that material.

    ``output_type`` selects email vs LinkedIn DM vs cover letter formatting.
    """
    normalized_tone = _normalize_tone(tone)
    fmt = _normalize_output_type(output_type)
    portfolio_context = format_portfolio_context(portfolio_docs)

    system_template = (
        "You are an expert AI assistant for career outreach: emails, LinkedIn messages, "
        "and cover letters for job seekers.\n"
        "Always write from the candidate's first-person perspective (I/my), never as a recruiter "
        "or as a company representative (we/our team).\n"
        "Follow the requested format and tone. Be concrete; avoid empty buzzwords.\n"
    )
    if fmt == "email":
        system_template += (
            "For email: use a clear subject line and structured body suitable for email.\n"
        )
    elif fmt == "linkedin":
        system_template += (
            "For LinkedIn: write a short, conversational message suitable for a connection "
            "request or InMail — no letterhead, minimal formality.\n"
        )
    else:
        system_template += (
            "For cover letter: use a formal letter layout with greeting, body paragraphs, "
            "and professional closing.\n"
        )

    if resume_mode:
        system_template += (
            "\n**Resume mode:** Only use facts that appear in the Candidate Profile "
            "or Retrieved Context below. Do not fabricate employers, degrees, "
            "certifications, or metrics. If something is unclear, stay vague rather than invent.\n"
        )

    context_label = (
        "### Retrieved resume excerpts (matched to this job)"
        if resume_mode
        else "### Retrieved portfolio context"
    )

    if fmt == "email":
        highlight_line = (
            "- Highlight 1–3 of the most relevant projects, roles, or achievements "
            "**from the resume/context** and tie them to the job needs.\n"
            if resume_mode
            else "- Highlight 1–3 of the most relevant projects, roles, or achievements "
            "and tie them to the job needs.\n"
        )
        impressive_line = (
            "- Make the email **impressive but honest**: strong hook, specific proof points "
            "from the resume, tight structure.\n"
            if resume_mode
            else ""
        )
        human_template = (
            "Write a cold **email** from the candidate (job seeker) to the hiring manager or recruiter.\n\n"
            "### Candidate Profile\n"
            "{candidate_profile}\n\n"
            "### Job Description\n"
            "{job_description}\n\n"
            f"{context_label}\n"
            "{portfolio_context}\n\n"
            "### Requirements\n"
            "- Start with a clear **Subject:** line, then the body.\n"
            "- Greeting rule: address the recruiter/hiring team (e.g., 'Hi Hiring Manager,' or "
            "'Hi Recruiter,'). Never greet the candidate's own name.\n"
            "- Formatting rule: use readable paragraph breaks; do not output one giant paragraph.\n"
            "- Do not use placeholders like [Company] or [Role]; use concrete values from the job details.\n"
            "- Briefly introduce the candidate and mention the target role and company.\n"
            "- Voice rule: write as the applicant seeking this role, not as someone hiring others.\n"
            f"{highlight_line}"
            "- Align skills and experience with the job description.\n"
            "- Close with a confident call to action (e.g., proposing a brief call).\n"
            "- Use a **{tone}** tone.\n"
            "- Keep the body roughly 250–350 words unless the content requires less.\n"
            f"{impressive_line}"
        )
    elif fmt == "linkedin":
        human_template = (
            "Write a **short LinkedIn message** (connection note or InMail style) from the candidate/job seeker.\n\n"
            "### Candidate Profile\n"
            "{candidate_profile}\n\n"
            "### Job Description\n"
            "{job_description}\n\n"
            f"{context_label}\n"
            "{portfolio_context}\n\n"
            "### Requirements\n"
            "- Maximum ~1200 characters total (LinkedIn limits); prefer under 900.\n"
            "- No subject line; start with a brief greeting + hook to the recruiter or hiring team.\n"
            "- Never greet using the candidate's own name.\n"
            "- Use short paragraphs with blank lines between them for readability.\n"
            "- 2–4 short paragraphs or tight bullet-style lines; mobile-friendly.\n"
            "- Mention role/company interest explicitly; 1–2 concrete proof points from context.\n"
            "- Voice rule: first-person applicant voice only; never recruiter-style outreach.\n"
            "- End with a light ask (e.g., open to chat).\n"
            "- Use a **{tone}** tone — professional but human, not stiff.\n"
        )
    else:
        human_template = (
            "Write a **cover letter** for this application from the candidate/job seeker.\n\n"
            "### Candidate Profile\n"
            "{candidate_profile}\n\n"
            "### Job Description\n"
            "{job_description}\n\n"
            f"{context_label}\n"
            "{portfolio_context}\n\n"
            "### Requirements\n"
            "- Formal letter: greeting (e.g., Dear Hiring Manager), 3–5 substantive paragraphs, closing + signature line.\n"
            "- Paragraph 1: role interest and company fit.\n"
            "- Middle paragraphs: relevant experience and achievements tied to JD; use specifics from context.\n"
            "- Final paragraph: enthusiasm and next steps.\n"
            "- Voice rule: first-person applicant voice only; never recruiter-style outreach.\n"
            "- Use a **{tone}** tone (still professional for a cover letter).\n"
            "- Length roughly 350–500 words unless the role is very senior.\n"
        )

    if candidate_profile is None:
        candidate_profile = (
            "Name: Alex Doe\n"
            "Title: Senior Machine Learning Engineer\n"
            "Location: San Francisco, CA\n"
            "Focus: Building production‑grade ML and RAG systems with Python, LangChain, and OpenAI."
        )

    job_text = job.as_plain_text()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            (
                "human",
                human_template,
            ),
        ]
    ).partial(
        candidate_profile=candidate_profile,
        job_description=job_text,
        portfolio_context=portfolio_context,
        tone=normalized_tone,
    )

    return prompt


def build_cold_email_prompt(
    job: JobDescription,
    portfolio_docs: Iterable[Document],
    tone: str = "professional",
    candidate_profile: str | None = None,
    resume_mode: bool = False,
) -> ChatPromptTemplate:
    """Backward-compatible alias: always ``output_type=email``."""
    return build_outreach_prompt(
        job=job,
        portfolio_docs=portfolio_docs,
        tone=tone,
        candidate_profile=candidate_profile,
        resume_mode=resume_mode,
        output_type="email",
    )


__all__ = [
    "build_outreach_prompt",
    "build_cold_email_prompt",
    "format_portfolio_context",
    "ALLOWED_TONES",
    "OUTPUT_TYPES",
]


