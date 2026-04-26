"""Central configuration for paths, provider options, and defaults."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PORTFOLIO_CSV = DATA_DIR / "portfolio.csv"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "portfolio_projects"

LLM_PROVIDERS = ["Groq", "Demo mode"]
EMBEDDING_PROVIDERS = ["Local hashing", "OpenAI"]

DEFAULT_MODELS = {
    "Groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "Demo mode": "deterministic-template",
}

DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "Groq")
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

SAMPLE_JOB_POST = """
Senior Python AI Engineer
Company: Northstar Talent Systems
Location: Remote

We are looking for a Senior Python AI Engineer to build production-ready AI tools for recruiting and sales teams. The role involves designing LangChain workflows, integrating vector databases such as ChromaDB, building Streamlit prototypes, and creating reliable extraction pipelines from unstructured job and candidate data.

Requirements:
- 4+ years of Python experience
- Strong experience with LangChain, retrieval augmented generation, vector search, and prompt engineering
- Experience building Streamlit or FastAPI applications
- Ability to work with APIs, CSV data, and structured output validation
- Clear communication and strong product sense

Nice to have:
- Groq API experience
- CI/CD and deployment automation
- Data visualization experience
""".strip()
