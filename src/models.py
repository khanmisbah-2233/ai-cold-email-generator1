"""Typed data models used across the application."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class JobSummary(BaseModel):
    """Normalized representation of a job description."""

    role: str = Field(default="Unknown role", description="The job title or target role.")
    company: str = Field(default="the company", description="The hiring company name.")
    location: str = Field(default="Not specified", description="Job location or remote policy.")
    experience_level: str = Field(default="Not specified", description="Seniority or years of experience.")
    required_skills: list[str] = Field(default_factory=list, description="Core required skills.")
    preferred_skills: list[str] = Field(default_factory=list, description="Nice-to-have skills.")
    responsibilities: list[str] = Field(default_factory=list, description="Primary responsibilities.")
    description_summary: str = Field(default="", description="Short summary of the role.")
    source_url: str | None = Field(default=None, description="Original job post URL, if supplied.")
    parsing_strategy: Literal["llm", "heuristic"] = "heuristic"

    @field_validator("required_skills", "preferred_skills", "responsibilities", mode="before")
    @classmethod
    def normalize_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def query_text(self) -> str:
        """Build a retrieval query for the portfolio vector store."""
        parts = [
            self.role,
            self.company,
            self.experience_level,
            " ".join(self.required_skills),
            " ".join(self.preferred_skills),
            " ".join(self.responsibilities),
            self.description_summary,
        ]
        return "\n".join(part for part in parts if part)


class RetrievedPortfolioItem(BaseModel):
    """Portfolio item returned by ChromaDB."""

    title: str
    category: str = ""
    skills: str = ""
    description: str = ""
    outcome: str = ""
    url: str = ""
    score: float | None = None
    content: str = ""


class CandidateProfile(BaseModel):
    """Candidate details used in the final email."""

    name: str = "Your Name"
    target_title: str = "Python AI Developer"
    email: str = ""
    phone: str = ""
    portfolio_url: str = ""
    linkedin_url: str = ""
