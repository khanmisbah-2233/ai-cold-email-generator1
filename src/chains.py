"""LangChain chains for job parsing and cold email generation."""

from __future__ import annotations

import json
import re
from json import JSONDecodeError

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
    "computer vision",
    "image recognition",
    "object detection",
    "video analysis",
    "augmented reality",
    "opencv",
    "tensorflow",
    "pytorch",
    "deep learning",
    "visual data",
]

WEAK_EMAIL_PHRASES = [
    "although my background",
    "i am eager to leverage my transferable skills",
    "transferable skills",
    "my skills can be adapted",
    "learn and apply new concepts quickly",
    "i may not have",
    "while i do not have",
    "strong foundation",
    "passion for developing innovative solutions",
    "confident in my ability",
    "my ability to learn",
    "skills can support your team's goals",
    "contribute to the development of solutions for applications such as",
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
                "You are a precise job-post parser for a cold-email generator. "
                "Extract the hiring company, role title, location, seniority, required skills, "
                "preferred skills, and responsibilities from any job description. "
                "Do not guess missing facts. If a company or location is not present, use "
                "'the company' or 'Not specified'. Keep lists short and relevant.\n"
                "{format_instructions}",
            ),
            (
                "human",
                "Parse this job post into the requested structure. Return only the requested structured output.\n\n{job_text}",
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
        return parse_job_description_json(raw_job_text, llm, source_url=source_url)


def parse_job_description_json(raw_job_text: str, llm, *, source_url: str | None = None) -> JobSummary:
    """Fallback LLM parser that is tolerant of non-Pydantic JSON responses."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract a job post into strict JSON. Return only valid JSON with these keys: "
                "role, company, location, experience_level, required_skills, preferred_skills, "
                "responsibilities, description_summary. Use arrays for skill and responsibility fields. "
                "Use 'the company' or 'Not specified' when values are missing.",
            ),
            ("human", "{job_text}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({"job_text": raw_job_text})
        payload = _extract_json_object(response)
        result = JobSummary.model_validate(payload)
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
    portfolio_for_prompt = _relevant_portfolio_items(job, portfolio_matches)
    if llm is None:
        return template_email(
            job=job,
            portfolio_matches=portfolio_for_prompt,
            candidate=candidate,
            tone=tone,
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior career outreach writer for professional job applications. "
                "Write confident, polished cold emails across any industry or role. "
                "Ground the email in the supplied job summary, candidate profile, and relevant portfolio evidence. "
                "If portfolio evidence is adjacent rather than exact, frame it positively as related experience; "
                "do not apologize for gaps or sound underqualified. "
                "Never use weak phrases such as 'although my background', 'my skills can be adapted', "
                "'I may not have', 'strong foundation', 'passion for developing innovative solutions', "
                "'confident in my ability', or 'learn and apply new concepts quickly'. "
                "Do not invent employers, degrees, metrics, certifications, project results, or personal history. "
                "Never mention that you were given JSON, a job summary, or portfolio evidence. "
                "Use specific language from the job post, but avoid copying long sentences from it. "
                "Include one or two strongest proof points only when they are relevant. "
                "Use a human, confident, concise voice. Keep the email between 150 and 230 words. "
                "Return exactly this format:\n\n"
                "Subject: Application for <role>\n\n"
                "Hi <company team or hiring team>,\n\n"
                "<3 concise paragraphs: fit for role, relevant evidence, call to discuss>\n\n"
                "Best,\n"
                "<candidate name>\n"
                "<available contact details>",
            ),
            (
                "human",
                "Tone: {tone}\n\n"
                "Candidate profile:\n{candidate_json}\n\n"
                "Job summary:\n{job_json}\n\n"
                "Portfolio evidence that may be relevant:\n{portfolio_json}\n\n"
                "Write a complete professional email for this exact job. "
                "If the company name is missing, address it to the hiring team. "
                "If the candidate's target title differs from the job title, prioritize the job title. "
                "Make the candidate sound qualified, direct, and professional. "
                "Include a clear call to discuss the role.",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    email = clean_text(
        chain.invoke(
            {
                "tone": tone,
                "candidate_json": json.dumps(candidate.model_dump(), indent=2),
                "job_json": json.dumps(job.model_dump(), indent=2),
                "portfolio_json": json.dumps(
                    [item.model_dump() for item in portfolio_for_prompt],
                    indent=2,
                ),
            }
        )
    )
    try:
        email = polish_professional_email(
            email=email,
            job=job,
            portfolio_matches=portfolio_for_prompt,
            candidate=candidate,
            tone=tone,
            llm=llm,
        )
    except Exception:
        pass
    if _needs_email_revision(email):
        email = revise_email(
            email=email,
            job=job,
            portfolio_matches=portfolio_for_prompt,
            candidate=candidate,
            tone=tone,
            llm=llm,
        )
    if _needs_email_revision(email):
        email = write_final_groq_email(
            job=job,
            portfolio_matches=portfolio_for_prompt,
            candidate=candidate,
            tone=tone,
            llm=llm,
        )
    return _ensure_email_structure(email, job=job, candidate=candidate)


def polish_professional_email(
    *,
    email: str,
    job: JobSummary,
    portfolio_matches: list[RetrievedPortfolioItem],
    candidate: CandidateProfile,
    tone: str,
    llm,
) -> str:
    """Always polish Groq drafts into a professional application email."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional job-application editor. Rewrite the draft into a polished, "
                "formal cold email that is ready to send to a hiring team. Keep it concise, confident, "
                "specific to the role, and grounded in the candidate profile and portfolio evidence. "
                "Do not use casual language, exaggerated claims, or weak phrases such as strong foundation, "
                "passion for developing innovative solutions, confident in my ability, transferable skills, "
                "or learn and apply new concepts quickly. Do not invent facts. "
                "Return only the final email with this structure: Subject, greeting, three concise paragraphs, "
                "Best, candidate name, contact details.",
            ),
            (
                "human",
                "Tone: {tone}\n\n"
                "Candidate profile:\n{candidate_json}\n\n"
                "Job summary:\n{job_json}\n\n"
                "Portfolio evidence:\n{portfolio_json}\n\n"
                "Draft email:\n{email}\n\n"
                "Rewrite it as a professional final email.",
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
                "email": email,
            }
        )
    )


def revise_email(
    *,
    email: str,
    job: JobSummary,
    portfolio_matches: list[RetrievedPortfolioItem],
    candidate: CandidateProfile,
    tone: str,
    llm,
) -> str:
    """Ask the LLM to polish weak or hesitant generated drafts."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You rewrite job-application cold emails into a confident professional version. "
                "Remove hesitant or apologetic language. Do not invent facts. "
                "Keep the same exact format: Subject, greeting, 3 concise paragraphs, signoff, contact details. "
                "Never use these phrases: although my background, my skills can be adapted, "
                "I may not have, while I do not have, transferable skills, strong foundation, "
                "passion for developing innovative solutions, confident in my ability, "
                "learn and apply new concepts quickly.",
            ),
            (
                "human",
                "Tone: {tone}\n\n"
                "Job summary:\n{job_json}\n\n"
                "Candidate profile:\n{candidate_json}\n\n"
                "Relevant portfolio evidence:\n{portfolio_json}\n\n"
                "Draft to improve:\n{email}\n\n"
                "Return only the improved email.",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return clean_text(
        chain.invoke(
            {
                "tone": tone,
                "job_json": json.dumps(job.model_dump(), indent=2),
                "candidate_json": json.dumps(candidate.model_dump(), indent=2),
                "portfolio_json": json.dumps(
                    [item.model_dump() for item in portfolio_matches],
                    indent=2,
                ),
                "email": email,
            }
        )
    )


def write_final_groq_email(
    *,
    job: JobSummary,
    portfolio_matches: list[RetrievedPortfolioItem],
    candidate: CandidateProfile,
    tone: str,
    llm,
) -> str:
    """Create a fresh Groq email when a previous Groq draft was too generic."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Write a fresh, polished job-application email. This is the final version shown to a user, "
                "so it must be professional, specific, and confident. Do not apologize for gaps. "
                "Do not use generic motivational phrases. Do not use these exact phrases: "
                "strong foundation, passion for developing innovative solutions, confident in my ability, "
                "transferable skills, learn and apply new concepts quickly. "
                "Use the candidate's actual project evidence when relevant. "
                "Return only the email in this structure: Subject line, greeting, three short paragraphs, "
                "Best, candidate name, contact details.",
            ),
            (
                "human",
                "Tone: {tone}\n\n"
                "Candidate profile:\n{candidate_json}\n\n"
                "Job summary:\n{job_json}\n\n"
                "Portfolio evidence:\n{portfolio_json}\n\n"
                "Write the final email now.",
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
    if not role:
        role = _extract_role_from_text(text)
    if not role and lines:
        role = _clean_role(lines[0][:90])

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
    strongest = _relevant_portfolio_items(job, portfolio_matches)[:2]
    proof_lines = []
    for item in strongest:
        proof = f"{item.title}: {item.description}"
        if item.outcome:
            proof = f"{proof} {item.outcome}"
        proof_lines.append(proof)

    skill_phrase = ", ".join(job.required_skills[:5]) or "the role's core requirements"
    proof_paragraph = " ".join(proof_lines) or (
        f"My portfolio shows practical Python and AI application work relevant to {job.role} responsibilities, "
        "including structured data workflows, reliable automation, and user-focused product delivery."
    )
    contact_line = _contact_line(candidate)
    company_name = job.company.strip()
    has_company = company_name and company_name.lower() not in {"the company", "unknown", "not specified"}
    greeting = f"Hi {company_name} team," if has_company else "Hi hiring team,"
    company_reference = company_name if has_company else "your team"

    return clean_text(
        f"""
Subject: Application for {job.role}

{greeting}

I am writing to apply for the {job.role} role. The focus on {skill_phrase} aligns with my work building practical AI tools, structured data workflows, and clean user-facing applications.

{proof_paragraph}

I would be glad to bring the same product-minded engineering approach to {company_reference}, especially around reliable systems, clear automation, and maintainable Python development.

Would you be open to a short conversation about how my experience can support this role?

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


def _extract_json_object(value: str) -> dict:
    value = (value or "").strip()
    try:
        return json.loads(value)
    except JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _relevant_portfolio_items(
    job: JobSummary,
    portfolio_matches: list[RetrievedPortfolioItem],
) -> list[RetrievedPortfolioItem]:
    """Keep fallback emails from citing projects with no skill overlap."""
    skill_terms = [skill.lower() for skill in job.required_skills if len(skill) > 2]
    if not skill_terms:
        return portfolio_matches

    relevant = []
    for item in portfolio_matches:
        searchable = " ".join(
            [
                item.title,
                item.category,
                item.skills,
                item.description,
                item.outcome,
                item.content,
            ]
        ).lower()
        if any(term in searchable for term in skill_terms):
            relevant.append(item)
    return relevant


def _needs_email_revision(email: str) -> bool:
    lowered = email.lower()
    if any(phrase in lowered for phrase in WEAK_EMAIL_PHRASES):
        return True
    required_markers = ["subject:", "hi ", "best,"]
    return not all(marker in lowered for marker in required_markers)


def _ensure_email_structure(email: str, *, job: JobSummary, candidate: CandidateProfile) -> str:
    """Add missing email wrapper pieces without replacing the Groq-written body."""
    text = clean_text(email)
    lowered = text.lower()

    if "subject:" not in lowered:
        text = f"Subject: Application for {job.role}\n\n{text}"

    lowered = text.lower()
    if "\nhi " not in lowered and not lowered.startswith("hi "):
        subject_match = re.match(r"(?is)^(Subject:[^\n]+)\n*", text)
        if subject_match:
            text = f"{subject_match.group(1)}\n\nHi hiring team,\n\n{text[subject_match.end():].lstrip()}"
        else:
            text = f"Hi hiring team,\n\n{text}"

    lowered = text.lower()
    if "best," not in lowered:
        contact_line = _contact_line(candidate)
        signoff = f"Best,\n{candidate.name}"
        if contact_line:
            signoff = f"{signoff}\n{contact_line}"
        text = f"{text.rstrip()}\n\n{signoff}"

    return clean_text(text)


def _extract_role_from_text(text: str) -> str:
    patterns = [
        r"\b(?:looking|hiring|seeking)\s+for\s+(?:a|an|the)?\s*([A-Za-z0-9 /+#.\-]+?\s+(?:Engineer|Developer|Scientist|Analyst|Specialist|Manager|Designer|Architect|Intern))\b",
        r"\bAs\s+(?:a|an|the)?\s*([A-Za-z0-9 /+#.\-]+?\s+(?:Engineer|Developer|Scientist|Analyst|Specialist|Manager|Designer|Architect|Intern))\b",
        r"\bjoin\s+(?:our|the)\s+team\s+as\s+(?:a|an|the)?\s*([A-Za-z0-9 /+#.\-]+?\s+(?:Engineer|Developer|Scientist|Analyst|Specialist|Manager|Designer|Architect|Intern))\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            return _clean_role(match.group(1))
    return ""


def _clean_role(role: str) -> str:
    role = re.sub(r"\s+", " ", role or "").strip(" .,-:")
    role = re.sub(
        r"^(?:talented|experienced|skilled|motivated|strong|great|excellent)\s+",
        "",
        role,
        flags=re.I,
    )
    role = re.sub(r"\s+to\s+join.*$", "", role, flags=re.I).strip(" .,-:")
    if len(role) > 80:
        return role[:80].rsplit(" ", 1)[0].strip(" .,-:")
    return role


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
