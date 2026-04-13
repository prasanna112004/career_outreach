"""Pydantic request/response models for the Career Outreach API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from services.model_provider import DEFAULT_CHAT_MODEL


class AnalyzeMatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=20)
    job_description_text: str = Field(..., min_length=20)
    chat_model: str = DEFAULT_CHAT_MODEL


class AnalyzeMatchResponse(BaseModel):
    match_score: int = Field(..., ge=0, le=100)
    missing_skills: list[str]
    strong_skills: list[str]


class GenerateContentRequest(BaseModel):
    """Either provide ``job_url`` (scraped) OR manual job fields + description."""

    job_url: Optional[str] = None
    job_title: Optional[str] = None
    company: Optional[str] = None
    job_description_text: Optional[str] = None

    resume_text: Optional[str] = None
    use_resume: bool = True
    tone: Literal["professional", "friendly", "direct"] = "professional"
    output_type: Literal["email", "linkedin", "cover_letter"] = "email"
    chat_model: str = DEFAULT_CHAT_MODEL

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_url": "https://example.com/job/123",
                "resume_text": "...",
                "use_resume": True,
                "tone": "professional",
                "output_type": "email",
            }
        }
    )


class GenerateContentResponse(BaseModel):
    generated_text: str
    quality_score: float
    retrieved_context_preview: str


class RefineContentRequest(BaseModel):
    original_generated_text: str = Field(..., min_length=10)
    instruction: str = Field(
        ...,
        description="e.g. 'Make shorter', 'Improve tone', or a custom instruction.",
    )
    chat_model: str = DEFAULT_CHAT_MODEL


class RefineContentResponse(BaseModel):
    refined_text: str
