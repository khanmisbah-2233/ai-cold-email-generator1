from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import streamlit as st
from dotenv import load_dotenv

from src.chains import generate_cold_email, parse_job_description
from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MODELS,
    PORTFOLIO_CSV,
    SAMPLE_JOB_POST,
)
from src.embeddings import create_embedding_function
from src.job_loader import fetch_job_posting
from src.llm import LLMConfigurationError, create_chat_model
from src.models import CandidateProfile


load_dotenv()


SECRET_SECTIONS = ("general", "default", "secrets", "groq", "GROQ", "llm", "LLM")
SECRET_ALIASES = {
    "GROQ_API_KEY": (
        "GROQ_API_KEY",
        "groq_api_key",
        "GROQ_KEY",
        "groq_key",
        "GROQ_TOKEN",
        "groq_token",
    ),
    "GROQ_MODEL": ("GROQ_MODEL", "groq_model", "MODEL", "model"),
    "OPENAI_API_KEY": ("OPENAI_API_KEY", "openai_api_key"),
    "OPENAI_EMBEDDING_MODEL": ("OPENAI_EMBEDDING_MODEL", "openai_embedding_model"),
    "EMBEDDING_PROVIDER": ("EMBEDDING_PROVIDER", "embedding_provider"),
    "PORTFOLIO_CSV": ("PORTFOLIO_CSV", "portfolio_csv"),
    "PORTFOLIO_MATCHES": ("PORTFOLIO_MATCHES", "portfolio_matches"),
    "EMAIL_TONE": ("EMAIL_TONE", "email_tone"),
    "REBUILD_PORTFOLIO_INDEX": ("REBUILD_PORTFOLIO_INDEX", "rebuild_portfolio_index"),
}


def main() -> None:
    st.set_page_config(
        page_title="AI Cold Email Generator",
        page_icon=":email:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    st.title("AI Cold Email Generator")

    settings = build_runtime_settings()
    candidate = render_candidate_profile()
    raw_job_text, source_url, submitted = render_job_input()

    if not submitted:
        return

    if not raw_job_text.strip():
        st.warning("Add a job description before generating an email.")
        return

    with st.status("Preparing generation workflow...", expanded=True) as status:
        st.write("Creating LangChain chat model")
        llm = resolve_llm(settings)
        groq_active = llm is not None
        if groq_active:
            st.write("Groq LLM connected from hidden settings")
        else:
            st.write("Groq LLM is not connected")

        st.write("Creating ChromaDB embedding function")
        embedding_function = resolve_embeddings(settings)

        st.write("Indexing portfolio data")
        from src.portfolio import PortfolioStore

        portfolio_store = PortfolioStore(
            csv_path=Path(settings["portfolio_csv"]),
            persist_directory=CHROMA_DIR,
            collection_name=collection_name_for(
                str(settings["embedding_provider"]),
                DEFAULT_EMBEDDING_MODEL,
            ),
            embedding_function=embedding_function,
        )
        indexed_count = portfolio_store.ensure_index(rebuild=settings["rebuild_index"])

        st.write("Parsing job description")
        job = parse_job_description(raw_job_text, llm=llm, source_url=source_url)

        st.write("Retrieving portfolio matches from ChromaDB")
        portfolio_matches = portfolio_store.search(job, k=settings["top_k"])

        st.write("Generating tailored email")
        email_generated_with_groq = False
        try:
            if llm is None and settings["provider"] == "Groq":
                status.update(label="Groq API key is not connected", state="error", expanded=True)
                render_groq_setup_error()
                return

            email = generate_cold_email(
                job=job,
                portfolio_matches=portfolio_matches,
                candidate=candidate,
                tone=settings["tone"],
                llm=llm,
            )
            email_generated_with_groq = llm is not None
        except Exception as error:
            if llm is not None:
                status.update(label="Groq generation failed", state="error", expanded=True)
                render_groq_runtime_error(error)
                return

            st.warning(f"LLM generation failed: {error}. Using demo email fallback.")
            email = generate_cold_email(
                job=job,
                portfolio_matches=portfolio_matches,
                candidate=candidate,
                tone=settings["tone"],
                llm=None,
            )
            email_generated_with_groq = False
        status.update(label="Email generated", state="complete", expanded=False)

    render_results(
        job=job,
        portfolio_matches=portfolio_matches,
        email=email,
        indexed_count=indexed_count,
        groq_active=email_generated_with_groq,
    )


def build_runtime_settings() -> dict[str, object]:
    """Load hidden runtime settings from environment variables."""
    provider = DEFAULT_LLM_PROVIDER if DEFAULT_LLM_PROVIDER in DEFAULT_MODELS else "Groq"
    embedding_provider = get_setting("EMBEDDING_PROVIDER", "Local hashing")
    if embedding_provider not in {"Local hashing", "OpenAI"}:
        embedding_provider = "Local hashing"

    return {
        "provider": provider,
        "model_name": get_setting(
            "GROQ_MODEL",
            DEFAULT_MODELS.get(provider, "llama-3.3-70b-versatile"),
        ),
        "api_key": get_setting("GROQ_API_KEY", ""),
        "base_url": "",
        "embedding_provider": embedding_provider,
        "embedding_api_key": get_setting("OPENAI_API_KEY", ""),
        "portfolio_csv": get_setting("PORTFOLIO_CSV", str(PORTFOLIO_CSV)),
        "top_k": int(get_setting("PORTFOLIO_MATCHES", "3")),
        "tone": get_setting("EMAIL_TONE", "Professional"),
        "rebuild_index": env_flag("REBUILD_PORTFOLIO_INDEX"),
    }


def render_candidate_profile() -> CandidateProfile:
    with st.expander("Candidate profile", expanded=True):
        first, second, third = st.columns(3)
        name = first.text_input("Name", value="Your Name")
        target_title = second.text_input("Target title", value="Python AI Developer")
        email = third.text_input("Email", value="")

        fourth, fifth, sixth = st.columns(3)
        phone = fourth.text_input("Phone", value="")
        portfolio_url = fifth.text_input("Portfolio URL", value="")
        linkedin_url = sixth.text_input("LinkedIn URL", value="")

    return CandidateProfile(
        name=name,
        target_title=target_title,
        email=email,
        phone=phone,
        portfolio_url=portfolio_url,
        linkedin_url=linkedin_url,
    )


def render_job_input() -> tuple[str, str | None, bool]:
    st.subheader("Job description")
    source = st.radio(
        "Source",
        ["Paste text", "Fetch URL", "Use sample"],
        horizontal=True,
        label_visibility="collapsed",
    )

    with st.form("job_form"):
        source_url = None
        raw_job_text = ""

        if source == "Paste text":
            raw_job_text = st.text_area(
                "Paste job description",
                height=280,
                placeholder="Paste the job post here...",
            )
        elif source == "Fetch URL":
            source_url = st.text_input("Job post URL", placeholder="https://example.com/job-post")
        else:
            raw_job_text = st.text_area(
                "Sample job description",
                value=SAMPLE_JOB_POST,
                height=280,
            )

        submitted = st.form_submit_button("Generate tailored email", type="primary", use_container_width=True)

    if submitted and source == "Fetch URL":
        with st.spinner("Fetching job post..."):
            try:
                raw_job_text = fetch_job_posting(source_url or "")
            except Exception as error:
                st.error(f"Could not fetch the job post: {error}")
                return "", source_url, False

    return raw_job_text, source_url, submitted


def render_results(*, job, portfolio_matches, email: str, indexed_count: int, groq_active: bool) -> None:
    st.divider()

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Role", job.role)
    metric_b.metric("Company", job.company)
    metric_c.metric("Skills found", len(job.required_skills))
    metric_d.metric("Portfolio indexed", indexed_count)

    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        st.subheader("Parsed job")
        st.write(f"**Location:** {job.location}")
        st.write(f"**Experience:** {job.experience_level}")
        if job.required_skills:
            st.write("**Required skills:** " + ", ".join(job.required_skills))
        if job.preferred_skills:
            st.write("**Preferred skills:** " + ", ".join(job.preferred_skills))
        if job.description_summary:
            st.write(job.description_summary)
        st.subheader("Portfolio matches")
        for item in portfolio_matches:
            label = item.title
            if item.score is not None:
                label = f"{item.title} - distance {item.score:.3f}"
            with st.expander(label, expanded=False):
                st.write(item.description)
                st.write(f"**Skills:** {item.skills}")
                if item.outcome:
                    st.write(f"**Outcome:** {item.outcome}")
                if item.url:
                    st.link_button("Open project", item.url)

    with right:
        st.subheader("Generated email")
        if groq_active:
            st.success("Generated with Groq")
        st.text_area("Email draft", value=email, height=520)
        st.download_button(
            "Download email",
            data=email,
            file_name="tailored_cold_email.txt",
            mime="text/plain",
            use_container_width=True,
        )


def render_groq_setup_error() -> None:
    st.error("Groq is not connected on Streamlit Cloud, so the app cannot generate a Groq email yet.")
    st.info(
        "Open Manage app -> Settings -> Secrets and add GROQ_API_KEY and GROQ_MODEL, "
        "then save and reboot the app. The key stays hidden and is not shown in the UI."
    )
    st.code(
        'GROQ_API_KEY = "your_real_groq_key_here"\n'
        'GROQ_MODEL = "llama-3.3-70b-versatile"',
        language="toml",
    )


def render_groq_runtime_error(error: Exception) -> None:
    st.error("Groq is configured, but the Groq request failed before a professional email could be generated.")
    st.info("Check that the Streamlit Secret key is valid, the model name is correct, and then reboot the app.")
    st.warning(f"Reason from Groq/LangChain: {sanitize_error_message(error)}")
    with st.expander("Technical detail", expanded=True):
        st.code(sanitize_error_message(error))


def sanitize_error_message(error: Exception) -> str:
    message = str(error) or error.__class__.__name__
    message = re.sub(r"gsk_[A-Za-z0-9_\-]+", "gsk_***hidden***", message)
    message = re.sub(r"Bearer\s+[A-Za-z0-9._\-]+", "Bearer ***hidden***", message, flags=re.I)
    return message[:1500]


def resolve_llm(settings: dict[str, object]):
    provider = str(settings["provider"])
    if provider == "Demo mode":
        return None

    api_key = str(settings.get("api_key") or "")
    if provider == "Groq":
        api_key = api_key or get_setting("GROQ_API_KEY", "")
    if is_placeholder_secret(api_key):
        api_key = ""

    if provider == "Groq" and not api_key:
        return None

    try:
        return create_chat_model(
            provider,
            model_name=str(settings["model_name"]),
            api_key=api_key,
            base_url=str(settings.get("base_url") or ""),
        )
    except LLMConfigurationError as error:
        st.warning(f"{error} Continuing in demo mode.")
        return None
    except Exception as error:
        st.warning(f"Could not initialize {provider}: {error}. Continuing in demo mode.")
        return None


def resolve_embeddings(settings: dict[str, object]):
    provider = str(settings["embedding_provider"])
    api_key = str(settings.get("embedding_api_key") or "")
    api_key = api_key or get_secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")

    try:
        return create_embedding_function(
            provider,
            openai_api_key=api_key,
            openai_model=DEFAULT_EMBEDDING_MODEL,
        )
    except Exception as error:
        st.warning(f"{error} Using local hashing embeddings instead.")
        return create_embedding_function("Local hashing")


def get_secret(name: str) -> str:
    """Read secrets from Streamlit Cloud or local .streamlit/secrets.toml."""
    try:
        for alias in secret_aliases(name):
            value = normalize_secret_value(st.secrets.get(alias, ""), name)
            if value:
                return value

        for section_name in SECRET_SECTIONS:
            section = st.secrets.get(section_name, {})
            if hasattr(section, "get"):
                for alias in secret_aliases(name, include_generic=True):
                    value = normalize_secret_value(section.get(alias, ""), name)
                    if value:
                        return value
            else:
                value = normalize_secret_value(section, name)
                if value:
                    return value
    except Exception:
        return ""
    return ""


def get_setting(name: str, default: str = "") -> str:
    value = get_secret(name)
    if value:
        return value

    for alias in secret_aliases(name):
        value = normalize_secret_value(os.getenv(alias, ""), name)
        if value:
            return value

    return str(default).strip()


def secret_aliases(name: str, *, include_generic: bool = False) -> tuple[str, ...]:
    aliases = SECRET_ALIASES.get(name, (name,))
    if include_generic and name == "GROQ_API_KEY":
        aliases = (*aliases, "api_key", "key", "token")
    return aliases


def normalize_secret_value(value: object, name: str) -> str:
    if value is None or hasattr(value, "get"):
        return ""

    text = str(value).strip().strip('"').strip("'")
    if not text:
        return ""

    for alias in secret_aliases(name, include_generic=True):
        match = re.search(rf"(?im)^\s*{re.escape(alias)}\s*=\s*[\"']?([^\"'\n#]+)", text)
        if match:
            return match.group(1).strip()

    return text


def is_placeholder_secret(value: str) -> bool:
    normalized = (value or "").strip().lower()
    return normalized in {
        "",
        "your_groq_api_key_here",
        "your_real_groq_api_key",
        "your_real_groq_api_key_here",
    }


def env_flag(name: str) -> bool:
    return get_setting(name, "").strip().lower() in {"1", "true", "yes", "on"}


def collection_name_for(embedding_provider: str, embedding_model: str) -> str:
    suffix = embedding_provider
    if embedding_provider == "OpenAI":
        suffix = f"{embedding_provider}_{embedding_model}"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", suffix).strip("_").lower()
    return f"{COLLECTION_NAME}_{slug}"[:63].rstrip("_-")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.75rem 1rem;
        }
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none;
        }
        textarea {
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
