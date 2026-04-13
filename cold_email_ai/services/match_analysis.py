"""
Resume vs job description skill overlap and match score.

Uses Groq to extract normalized skill lists, with a heuristic fallback.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Set, Tuple

from langchain_core.prompts import ChatPromptTemplate

from .model_provider import DEFAULT_CHAT_MODEL, get_chat_model

# Known tech / role keywords for fallback overlap
_FALLBACK_KEYWORDS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "go",
    "rust",
    "sql",
    "aws",
    "gcp",
    "azure",
    "kubernetes",
    "docker",
    "terraform",
    "kafka",
    "redis",
    "postgres",
    "mongodb",
    "react",
    "node",
    "django",
    "fastapi",
    "flask",
    "langchain",
    "pytorch",
    "tensorflow",
    "numpy",
    "pandas",
    "scikit-learn",
    "sklearn",
    "machine learning",
    "deep learning",
    "supervised learning",
    "unsupervised learning",
    "nlp",
    "llm",
    "rag",
    "system design",
    "ci/cd",
    "git",
    "agile",
    "api",
    "rest",
    "graphql",
    "microservices",
]

_GENERIC_NON_SKILLS = {
    "experience",
    "knowledge",
    "responsibilities",
    "requirements",
    "required",
    "preferred",
    "ability",
    "strong",
    "good",
    "excellent",
    "team",
    "work",
    "years",
    "year",
    "role",
    "candidate",
    "company",
}

_ALIASES = {
    "nodejs": "node.js",
    "node": "node.js",
    "postgresql": "postgres",
    "rest api": "rest",
    "rest apis": "rest",
}


def _normalize_skill(s: str) -> str:
    cleaned = re.sub(r"\s+", " ", s.strip().lower())
    cleaned = cleaned.strip(".,;:()[]{}")
    return _ALIASES.get(cleaned, cleaned)


def _extract_json_object(text: str) -> Optional[dict]:
    """Parse first JSON object from model output."""
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _fallback_skills(text: str) -> Set[str]:
    t = text.lower()
    found: Set[str] = set()
    for kw in _FALLBACK_KEYWORDS:
        if kw in t:
            found.add(kw)
    return found


def _token_skills(text: str) -> Set[str]:
    """
    Generic token fallback so JD skill coverage does not collapse to empty.
    """
    out: Set[str] = set()
    for w in re.findall(r"[A-Za-z][A-Za-z0-9\+\#\.\-\/]{1,32}", text or ""):
        wl = w.lower()
        if len(wl) < 3:
            continue
        if wl in _GENERIC_NON_SKILLS:
            continue
        # Keep tech-shaped tokens, acronyms, and fallback skills.
        if (
            ("+" in wl or "#" in wl or "." in wl)
            or (w.isupper() and len(w) <= 6)
            or (wl in _FALLBACK_KEYWORDS)
        ):
            out.add(_normalize_skill(wl))
    return out


def _clean_skill_set(skills: Set[str], max_items: int = 40) -> Set[str]:
    cleaned = []
    for s in skills:
        ns = _normalize_skill(s)
        if not ns or ns in _GENERIC_NON_SKILLS:
            continue
        if len(ns) < 2 or len(ns) > 40:
            continue
        cleaned.append(ns)
    # Stable deterministic order then cap to avoid denominator blowup.
    return set(sorted(dict.fromkeys(cleaned))[:max_items])


def _llm_extract_skill_lists(
    resume_text: str,
    job_text: str,
    chat_model: str,
) -> Optional[Tuple[List[str], List[str]]]:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You extract professional skills and technologies only. "
                "Output valid JSON with keys resume_skills and jd_skills (arrays of short strings). "
                "No prose, no markdown fences.",
            ),
            (
                "human",
                "RESUME:\n{resume}\n\nJOB DESCRIPTION:\n{job}\n\n"
                'Respond with JSON only: {{"resume_skills": [...], "jd_skills": [...]}}',
            ),
        ]
    )
    llm = get_chat_model(chat_model, temperature=0.1)
    chain = prompt | llm
    r = chain.invoke(
        {
            "resume": resume_text[:12000],
            "job": job_text[:12000],
        }
    )
    raw = r.content if hasattr(r, "content") else str(r)
    data = _extract_json_object(raw)
    if not data:
        return None
    rs = data.get("resume_skills") or []
    jd = data.get("jd_skills") or []
    if not isinstance(rs, list) or not isinstance(jd, list):
        return None
    rs_clean = [_normalize_skill(str(x)) for x in rs if str(x).strip()]
    jd_clean = [_normalize_skill(str(x)) for x in jd if str(x).strip()]
    return list(dict.fromkeys(rs_clean)), list(dict.fromkeys(jd_clean))


def compute_match_insights(
    resume_skills: Set[str],
    jd_skills: Set[str],
) -> Tuple[int, List[str], List[str]]:
    """Derive match score, missing, and strong skills."""
    if not jd_skills:
        return 0, [], sorted(resume_skills)

    strong = sorted(resume_skills & jd_skills)
    missing = sorted(jd_skills - resume_skills)
    # Coverage of JD skill expectations by the resume
    score = round(100.0 * len(strong) / max(len(jd_skills), 1))
    score = max(0, min(100, int(score)))
    return score, missing, strong


def analyze_resume_job_match(
    resume_text: str,
    job_description_text: str,
    chat_model: str = DEFAULT_CHAT_MODEL,
) -> dict:
    """
    Return match_score (0-100), missing_skills, strong_skills.

    Response shape matches API contract:
    {"match_score": int, "missing_skills": [...], "strong_skills": [...]}
    """
    try:
        extracted = _llm_extract_skill_lists(resume_text, job_description_text, chat_model)
    except Exception:
        extracted = None
    resume_kw = _fallback_skills(resume_text)
    jd_kw = _fallback_skills(job_description_text)

    if extracted:
        rs_list, jd_list = extracted
        resume_set = _clean_skill_set(set(rs_list))
        jd_set = _clean_skill_set(set(jd_list))
        # Guardrail: model can occasionally return resume_skills but empty jd_skills.
        if not jd_set:
            jd_set = _clean_skill_set(
                _fallback_skills(job_description_text) | _token_skills(job_description_text)
            )
        if not resume_set:
            resume_set = _clean_skill_set(
                _fallback_skills(resume_text) | _token_skills(resume_text)
            )
    else:
        resume_set = _clean_skill_set(_fallback_skills(resume_text) | _token_skills(resume_text))
        jd_set = _clean_skill_set(
            _fallback_skills(job_description_text) | _token_skills(job_description_text)
        )

    # Always augment with deterministic keyword hits to stabilize score/missing behavior.
    resume_set = _clean_skill_set(resume_set | resume_kw)
    jd_set = _clean_skill_set(jd_set | jd_kw)

    score, missing, strong = compute_match_insights(resume_set, jd_set)
    if strong and score == 0:
        score = 1
    return {
        "match_score": score,
        "missing_skills": missing[:25],
        "strong_skills": strong[:25],
    }
