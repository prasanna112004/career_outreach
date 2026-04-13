"""
FastAPI backend for the Career Outreach Assistant.

Run locally:
  cd cold_email_ai && uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Integrates with the existing LangChain/Groq stack without changing Streamlit behavior.
"""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    AnalyzeMatchRequest,
    AnalyzeMatchResponse,
    GenerateContentRequest,
    GenerateContentResponse,
    RefineContentRequest,
    RefineContentResponse,
)
from prompt_templates import format_portfolio_context
from scraper import JobDescription, scrape_job_posting
from resume_parser import extract_text_from_upload
from services.content_generation import generate_outreach_content
from services.match_analysis import analyze_resume_job_match
from services.refinement import refine_generated_text

app = FastAPI(
    title="Career Outreach Assistant API",
    version="1.1.0",
    description="Match analysis, multi-format outreach generation, and refinement.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract-resume")
async def extract_resume(resume_file: UploadFile = File(...)):
    try:
        raw = await resume_file.read()
        text = extract_text_from_upload(resume_file.filename or "resume.txt", raw)
        return {"resume_text": text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Resume parse failed: {e}") from e


@app.post("/analyze-match", response_model=AnalyzeMatchResponse)
def analyze_match(body: AnalyzeMatchRequest):
    try:
        result = analyze_resume_job_match(
            body.resume_text,
            body.job_description_text,
            chat_model=body.chat_model,
        )
        return AnalyzeMatchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _resolve_job(body: GenerateContentRequest) -> JobDescription:
    if body.job_url and body.job_url.strip():
        try:
            return scrape_job_posting(body.job_url.strip())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to scrape job URL: {e}") from e
    if body.job_description_text and body.job_description_text.strip():
        return JobDescription.from_manual_input(
            title=body.job_title or "Role",
            company=body.company or "Company",
            responsibilities=body.job_description_text.strip(),
            skills=[],
        )
    raise HTTPException(
        status_code=400,
        detail="Provide either job_url or job_description_text (e.g. from Chrome extension).",
    )


@app.post("/generate-content", response_model=GenerateContentResponse)
def generate_content(body: GenerateContentRequest):
    if body.use_resume and not (body.resume_text and body.resume_text.strip()):
        raise HTTPException(
            status_code=400,
            detail="resume_text is required when use_resume is true.",
        )
    try:
        job = _resolve_job(body)
        text, quality, docs, _ = generate_outreach_content(
            job,
            tone=body.tone,
            chat_model=body.chat_model,
            use_resume=body.use_resume,
            resume_text=body.resume_text if body.use_resume else None,
            output_type=body.output_type,
        )
        preview = format_portfolio_context(docs)
        return GenerateContentResponse(
            generated_text=text,
            quality_score=quality,
            retrieved_context_preview=preview,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/refine-content", response_model=RefineContentResponse)
def refine_content(body: RefineContentRequest):
    try:
        refined = refine_generated_text(
            body.original_generated_text,
            body.instruction,
            chat_model=body.chat_model,
        )
        return RefineContentResponse(refined_text=refined)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
