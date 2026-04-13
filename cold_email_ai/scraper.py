"""
Web scraper for extracting job descriptions from public job posting URLs.

This module uses `requests` and `BeautifulSoup` to fetch and parse the page,
then applies a few heuristics to extract:
  - job title
  - company name
  - responsibilities / description
  - required skills

The scraper intentionally stays lightweight and best‑effort; for production
use you would typically add per‑site scrapers or use an external service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import requests
from bs4 import BeautifulSoup


@dataclass
class JobDescription:
    """Structured representation of a scraped job posting."""

    url: str
    title: str
    company: str
    responsibilities: str
    skills: List[str]

    @staticmethod
    def from_manual_input(
        title: str,
        company: str,
        responsibilities: str,
        skills: Optional[List[str]] = None,
        url: str = "manual://input",
    ) -> "JobDescription":
        """
        Build a job description when the user (or a browser extension) supplies
        text directly instead of a scrapeable URL.
        """
        return JobDescription(
            url=url,
            title=title.strip() or "Job",
            company=company.strip() or "Company",
            responsibilities=responsibilities.strip() or "",
            skills=list(skills) if skills else [],
        )

    def as_plain_text(self) -> str:
        """
        Flatten the job description into a single text block suitable for
        feeding into an LLM or retriever.
        """
        skills_str = ", ".join(self.skills) if self.skills else "N/A"
        return (
            f"Job Title: {self.title}\n"
            f"Company: {self.company}\n\n"
            f"Responsibilities / Description:\n{self.responsibilities}\n\n"
            f"Required skills: {skills_str}"
        )


def _fetch_html(url: str, timeout: int = 15) -> str:
    """Fetch raw HTML for the given URL."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _extract_title(soup: BeautifulSoup) -> str:
    """Try a few common patterns to find the job title."""
    # <h1> is often the job title
    if soup.h1 and soup.h1.get_text(strip=True):
        return soup.h1.get_text(strip=True)

    # Fallback to <title>
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)

    return "Unknown Job Title"


def _extract_company(soup: BeautifulSoup) -> str:
    """Best‑effort extraction of a company name."""
    # Look for meta tags
    metas = soup.find_all("meta")
    for m in metas:
        prop = m.get("property") or m.get("name") or ""
        if "og:site_name" in prop.lower() or "twitter:title" in prop.lower():
            content = m.get("content")
            if content:
                return content.strip()

    # Simple heuristics over text
    possible = soup.find_all(["span", "div"], string=True, limit=50)
    for tag in possible:
        text = tag.get_text(strip=True)
        if "Inc" in text or "LLC" in text or "Ltd" in text or "GmbH" in text:
            return text

    return "Unknown Company"


def _extract_body_text(soup: BeautifulSoup) -> str:
    """
    Extract the main textual content for responsibilities / description.

    For simplicity we join relevant <p>, <li> and some <div> content,
    excluding headers and nav/footers.
    """
    texts: List[str] = []
    for tag in soup.find_all(["p", "li"]):
        text = tag.get_text(" ", strip=True)
        if len(text) < 40:
            # Skip very short fragments such as menu items.
            continue
        texts.append(text)

    # Fallback if nothing reasonable was found
    if not texts:
        body = soup.body.get_text(" ", strip=True) if soup.body else ""
        return body[:4000]

    combined = "\n".join(texts)
    return combined[:6000]


def _extract_linkedin_fallback_text(soup: BeautifulSoup) -> str:
    """
    LinkedIn often hides full JD body to non-authenticated scrapers.
    Use meta tags / JSON-LD as fallback so we still get useful content.
    """
    candidates: List[str] = []
    for sel, attr in (
        ('meta[property="og:description"]', "content"),
        ('meta[name="description"]', "content"),
    ):
        el = soup.select_one(sel)
        if el and el.get(attr):
            candidates.append(el.get(attr, "").strip())

    for script in soup.find_all("script", {"type": "application/ld+json"}):
        txt = script.get_text(" ", strip=True)
        if txt:
            candidates.append(txt)

    merged = "\n".join([c for c in candidates if len(c) > 20]).strip()
    return merged[:6000]


def _extract_skills_from_text(text: str) -> List[str]:
    """
    Naive skill extraction based on a fixed keyword list.

    In a production system you might use an LLM or an ontology‑based
    matcher. Here we keep it simple and transparent.
    """
    known_skills = [
        "python",
        "java",
        "javascript",
        "typescript",
        "sql",
        "aws",
        "gcp",
        "azure",
        "docker",
        "kubernetes",
        "machine learning",
        "deep learning",
        "nlp",
        "llm",
        "prompt engineering",
        "langchain",
        "pytorch",
        "tensorflow",
        "data engineering",
        "mlops",
        "streamlit",
    ]

    text_lower = text.lower()
    found: List[str] = []
    for skill in known_skills:
        if skill in text_lower:
            found.append(skill)
    return sorted(set(found))


def scrape_job_posting(url: str) -> JobDescription:
    """
    Scrape a job posting URL into a structured `JobDescription`.

    Parameters
    ----------
    url:
        Public job posting URL.

    Returns
    -------
    JobDescription
        Parsed job description data ready for downstream RAG components.
    """
    html = _fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    title = _extract_title(soup)
    company = _extract_company(soup)
    responsibilities = _extract_body_text(soup)
    if "linkedin.com/jobs" in url.lower() and len(responsibilities or "") < 120:
        fallback = _extract_linkedin_fallback_text(soup)
        if len(fallback) > len(responsibilities or ""):
            responsibilities = fallback
    skills = _extract_skills_from_text(responsibilities)

    return JobDescription(
        url=url,
        title=title,
        company=company,
        responsibilities=responsibilities,
        skills=skills,
    )


__all__ = ["JobDescription", "scrape_job_posting"]

