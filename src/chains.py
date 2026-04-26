"""LangChain chains for job parsing and cold email generation."""

from __future__ import annotations

import json
import re

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .models import CandidateProfile, JobSummary, RetrievedPortfolioItem
from .text import clean_text, truncate_text


TECH_SKILLS = [
    "python",
    "langchain",
    "chromadb",
    "streamlit",
    "rag",
    "retrieval augmented generation",
    "vector search",
    "embeddings",
    "openai",
    "groq",
    "ollama",
    "fastapi",
    "django",
    "flask",
    "sql",
    "postgresql",
    "pandas",
    "numpy",
    "pydantic",
    "api",
    "docker",
    "ci/cd",
    "github actions",
    "data visualization",
    "nlp",
    "machine learning",
    "prompt engineering",
]


def parse_job_description(raw_job_text: str, llm=None, *, source_url: str | None = None) -> JobSummary:
    """Parse a job description with LangChain, falling back to heuristics."""
    raw_job_text = truncate_text(raw_job_text)
    if not raw_job_text:
        raise ValueError("A job description is required.")

    if llm is None:
        return heuristic_job_summary(raw_job_text, source_url=source_url)

    parser = PydanticOutputParser(pydantic_object=JobSummary)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You extract structured hiring requirements from job posts. "
                "Be precise, avoid guessing, and keep every list short and relevant.\n"
                "{format_instructions}",
            ),
            (
                "human",
                "Parse this job post into the requested structure:\n\n{job_text}",
            ),
        ]
    )
    chain = prompt | llm | parser

    try:
        result = chain.invoke(
            {
                "job_text": raw_job_text,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        result.source_url = source_url
        result.parsing_strategy = "llm"
        return result
    except Exception:
        return heuristic_job_summary(raw_job_text, source_url=source_url)


def generate_cold_email(
    *,
    job: JobSummary,
    portfolio_matches: list[RetrievedPortfolioItem],
    candidate: CandidateProfile,
    tone: str,
    llm=None,
) -> str:
    """Generate a tailored cold email grounded in portfolio evidence."""
    if llm is None:
        return template_email(
            job=job,
            portfolio_matches=portfolio_matches,
            candidate=candidate,
            tone=tone,
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You write concise, professional cold emails for job applications. "
                "Use only the supplied job summary, candidate profile, and portfolio evidence. "
                "Do not invent employers, degrees, metrics, certifications, or project results. "
                "Write in a human voice, avoid hype, and keep the email between 140 and 220 words. "
                "Return a subject line followed by the email body.",
            ),
            (
                "human",
                "Tone: {tone}\n\n"
                "Candidate profile:\n{candidate_json}\n\n"
                "Job summary:\n{job_json}\n\n"
                "Relevant portfolio evidence:\n{portfolio_json}\n\n"
                "Write the email now. Include a clear call to discuss the role.",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return clean_text(
        chain.invoke(
            {
                "tone": tone,
                "candidate_json": json.dumps(candidate.model_dump(), indent=2),
                "job_json": json.dumps(job.model_dump(), indent=2),
                "portfolio_json": json.dumps(
                    [item.model_dump() for item in portfolio_matches],
                    indent=2,
                ),
            }
        )
    )


def heuristic_job_summary(raw_job_text: str, *, source_url: str | None = None) -> JobSummary:
    """Best-effort parser used when no LLM is configured."""
    text = clean_text(raw_job_text)
    lower_text = text.lower()
    lines = [line.strip(" -:\t") for line in text.splitlines() if line.strip()]

    role = _extract_by_label(text, ["job title", "role", "position"])
    if not role and lines:
        role = lines[0][:90]

    company = _extract_by_label(text, ["company", "organization", "employer"])
    if not company:
        company_match = re.search(r"\bat\s+([A-Z][A-Za-z0-9&.,' -]{2,60})", text)
        company = company_match.group(1).strip(" .,-") if company_match else "the company"

    location = _extract_by_label(text, ["location"])
    experience_level = _extract_experience(text)
    required_skills = _extract_skills(lower_text)
    preferred_skills = _extract_preferred_skills(lower_text)
    responsibilities = _extract_responsibilities(lines)

    summary = " ".join(lines[:4])
    return JobSummary(
        role=role or "Unknown role",
        company=company or "the company",
        location=location or "Not specified",
        experience_level=experience_level,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        responsibilities=responsibilities,
        description_summary=summary[:600],
        source_url=source_url,
        parsing_strategy="heuristic",
    )


def template_email(
    *,
    job: JobSummary,
    portfolio_matches: list[RetrievedPortfolioItem],
    candidate: CandidateProfile,
    tone: str,
) -> str:
    """Deterministic fallback email for demo mode."""
    strongest = portfolio_matches[:2]
    proof_lines = []
    for item in strongest:
        proof = f"{item.title}: {item.description}"
        if item.outcome:
            proof = f"{proof} {item.outcome}"
        proof_lines.append(proof)

    skill_phrase = ", ".join(job.required_skills[:5]) or "the role's core requirements"
    proof_paragraph = " ".join(proof_lines) or "My portfolio includes relevant Python and AI application work."
    contact_line = _contact_line(candidate)

    return clean_text(
        f"""
Subject: Application for {job.role}

Hi {job.company} team,

I am reaching out about the {job.role} role. Your need for {skill_phrase} stood out because my recent work has focused on building practical AI tools that connect unstructured inputs, retrieval, and clean user-facing workflows.

{proof_paragraph}

I would be glad to bring the same product-minded engineering approach to {job.company}, especially around reliable AI workflows, clear automation, and maintainable Python systems.

Would you be open to a short conversation about how my background could support this role?

Best,
{candidate.name}
{contact_line}
""".strip()
    )


def _extract_by_label(text: str, labels: list[str]) -> str:
    for label in labels:
        pattern = rf"(?im)^\s*{re.escape(label)}\s*[:\-]\s*(.+)$"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def _extract_experience(text: str) -> str:
    match = re.search(r"(\d+\+?\s*(?:-|to)?\s*\d*\+?\s*years?[^.\n,;]*)", text, flags=re.I)
    if match:
        return match.group(1).strip()
    lowered = text.lower()
    for level in ["intern", "junior", "mid-level", "senior", "lead", "principal"]:
        if level in lowered:
            return level.title()
    return "Not specified"


def _extract_skills(lower_text: str) -> list[str]:
    found = []
    for skill in TECH_SKILLS:
        if skill in lower_text and skill.title() not in found:
            found.append(skill.title())
    return found[:12]


def _extract_preferred_skills(lower_text: str) -> list[str]:
    preferred_section = ""
    for marker in ["nice to have", "preferred", "bonus"]:
        if marker in lower_text:
            preferred_section = lower_text.split(marker, 1)[1]
            break
    if not preferred_section:
        return []
    return _extract_skills(preferred_section)[:8]


def _extract_responsibilities(lines: list[str]) -> list[str]:
    responsibilities = []
    capture = False
    for line in lines:
        lowered = line.lower()
        if any(marker in lowered for marker in ["responsibilities", "what you'll do", "you will"]):
            capture = True
            continue
        if capture and any(marker in lowered for marker in ["requirements", "qualifications", "nice to have"]):
            break
        if capture and len(line) > 20:
            responsibilities.append(line[:180])
    return responsibilities[:6]


def _contact_line(candidate: CandidateProfile) -> str:
    parts = [
        candidate.email,
        candidate.phone,
        candidate.portfolio_url,
        candidate.linkedin_url,
    ]
    return "\n".join(part for part in parts if part)
