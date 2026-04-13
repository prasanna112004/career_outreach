"""
Static portfolio dataset for the AI Cold Email Generator.

This module defines a small, structured knowledge base describing the
candidate's projects, skills, experience, and achievements. The data
is converted to LangChain `Document` objects so it can be indexed
in a vector store.
"""

from typing import List

from langchain_core.documents import Document


def _base_metadata() -> dict:
    """Return metadata shared by all portfolio documents."""
    return {
        "candidate_name": "Alex Doe",
        "candidate_title": "Senior Machine Learning Engineer",
        "location": "San Francisco, CA",
    }


def load_portfolio_documents() -> List[Document]:
    """
    Build a list of `Document` objects describing the candidate portfolio.

    The documents are intentionally somewhat overlapping so that the
    retriever can surface the most relevant slices given an arbitrary
    job description.
    """
    meta = _base_metadata()
    docs: List[Document] = []

    # High-level profile
    profile_text = (
        "Candidate: Alex Doe, Senior Machine Learning Engineer based in San Francisco.\n"
        "Summary: 6+ years of experience building production ML and generative AI systems, "
        "with a focus on retrieval‑augmented generation (RAG), NLP, and recommendation "
        "systems. Strong background in Python, LangChain, OpenAI models, and MLOps on AWS."
    )
    docs.append(Document(page_content=profile_text, metadata={**meta, "section": "profile"}))

    # Project 1 – RAG search for SaaS product
    project_rag = (
        "Project: RAG Knowledge Search for B2B SaaS Platform.\n"
        "Built an end‑to‑end retrieval‑augmented generation (RAG) system to answer "
        "customer questions over product documentation, support tickets, and release notes.\n"
        "Stack: Python, LangChain, OpenAI embeddings, ChromaDB, FastAPI, Docker, AWS ECS.\n"
        "Highlights: Designed chunking and metadata strategy, implemented semantic search, "
        "and deployed a scalable API used by >500 support agents daily."
    )
    docs.append(
        Document(
            page_content=project_rag,
            metadata={**meta, "section": "project", "project_name": "RAG Knowledge Search"},
        )
    )

    # Project 2 – Cold email and sales automation
    project_sales = (
        "Project: Intelligent Sales Outreach Assistant.\n"
        "Developed a cold‑email generation tool that tailored outreach based on "
        "prospect LinkedIn profiles, company news, and product fit.\n"
        "Stack: Python, LangChain, OpenAI GPT models, PostgreSQL, Streamlit, Celery.\n"
        "Highlights: Increased reply rates by 35% by personalizing messaging and "
        "optimizing prompts for different tones (professional, friendly, direct)."
    )
    docs.append(
        Document(
            page_content=project_sales,
            metadata={**meta, "section": "project", "project_name": "Sales Outreach Assistant"},
        )
    )

    # Project 3 – MLOps & experimentation
    project_mlops = (
        "Project: ML Experimentation & MLOps Platform.\n"
        "Implemented experiment tracking, model registry, and CI/CD for ML models.\n"
        "Stack: Python, MLflow, Kubernetes, Docker, GitHub Actions, AWS S3, SageMaker.\n"
        "Highlights: Reduced model deployment time from weeks to days and increased "
        "reproducibility of experiments across the data science team."
    )
    docs.append(
        Document(
            page_content=project_mlops,
            metadata={**meta, "section": "project", "project_name": "MLOps Platform"},
        )
    )

    # Skills
    skills_text = (
        "Core skills: Python, LangChain, OpenAI API, ChromaDB, Vector Databases, "
        "Retrieval‑Augmented Generation (RAG), NLP, LLM Prompt Engineering.\n"
        "Additional skills: Streamlit, FastAPI, SQL, NoSQL, data modeling, "
        "AWS (ECS, S3, Lambda), Docker, Kubernetes, CI/CD, monitoring, logging.\n"
        "Soft skills: cross‑functional collaboration, product thinking, mentoring, "
        "writing technical documentation, communicating trade‑offs to stakeholders."
    )
    docs.append(Document(page_content=skills_text, metadata={**meta, "section": "skills"}))

    # Experience
    experience_text = (
        "Experience: Senior ML Engineer at GrowthStack (3 years).\n"
        "Led a small team building recommendation systems and marketing automation tools.\n"
        "Previously: ML Engineer at DataBridge (3 years) focusing on NLP pipelines, "
        "entity extraction, and text classification at scale."
    )
    docs.append(Document(page_content=experience_text, metadata={**meta, "section": "experience"}))

    # Achievements
    achievements_text = (
        "Achievements: Drove double‑digit increase in product engagement through "
        "personalized recommendations; shipped 3 production RAG systems; "
        "mentored 5+ junior engineers; co‑authored internal best practices for "
        "LLM safety, evaluation, and prompt design."
    )
    docs.append(Document(page_content=achievements_text, metadata={**meta, "section": "achievements"}))

    return docs


__all__ = ["load_portfolio_documents"]

