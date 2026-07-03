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
        page_title="Career Email Copilot",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    settings = build_runtime_settings()
    candidate = render_candidate_profile()
    render_header(settings)
    render_chat_intro(candidate)
    raw_job_text, source_url, submitted = render_job_input()

    if not submitted:
        render_idle_panel()
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
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="brand-mark">AI</div>
                <div>
                    <div class="brand-title">Career Email Copilot</div>
                    <div class="brand-subtitle">Profile and runtime settings</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Candidate Profile")
        name = st.text_input("Name", value="Your Name", key="profile_name")
        target_title = st.text_input(
            "Target title",
            value="Python AI Developer",
            key="profile_target_title",
        )
        email = st.text_input("Email", value="", key="profile_email")
        phone = st.text_input("Phone", value="", key="profile_phone")
        portfolio_url = st.text_input("Portfolio URL", value="", key="profile_portfolio_url")
        linkedin_url = st.text_input("LinkedIn URL", value="", key="profile_linkedin_url")

        st.divider()
        st.caption("The assistant uses this profile to personalize the email signature and positioning.")

    return CandidateProfile(
        name=name,
        target_title=target_title,
        email=email,
        phone=phone,
        portfolio_url=portfolio_url,
        linkedin_url=linkedin_url,
    )


def render_job_input() -> tuple[str, str | None, bool]:
    st.markdown('<div class="section-label">Start A New Draft</div>', unsafe_allow_html=True)
    source = st.radio(
        "Job input source",
        ["Paste text", "Fetch URL", "Use sample"],
        horizontal=True,
        key="job_source",
    )

    with st.form("job_form", border=False):
        source_url = None
        raw_job_text = ""

        if source == "Paste text":
            raw_job_text = st.text_area(
                "Message to assistant",
                height=260,
                key="job_text",
                placeholder=(
                    "Paste the job description here. The assistant will extract requirements, "
                    "match your portfolio, and write a tailored cold email."
                ),
            )
        elif source == "Fetch URL":
            source_url = st.text_input(
                "Public job post URL",
                key="job_url",
                placeholder="https://company.com/careers/software-engineer",
            )
            st.info(
                "Best for public company career pages. LinkedIn often blocks automated fetching; "
                "for LinkedIn jobs, paste the description text instead."
            )
        else:
            raw_job_text = st.text_area(
                "Sample job description",
                value=SAMPLE_JOB_POST,
                height=280,
                key="sample_job_text",
            )

        submitted = st.form_submit_button(
            "Generate with AI assistant",
            type="primary",
            use_container_width=True,
        )

    if submitted and source == "Fetch URL":
        with st.spinner("Fetching job post..."):
            try:
                raw_job_text = fetch_job_posting(source_url or "")
            except Exception as error:
                st.error(f"Could not fetch the job post: {error}")
                return "", source_url, False

    return raw_job_text, source_url, submitted


def render_header(settings: dict[str, object]) -> None:
    provider = str(settings["provider"])
    model_name = str(settings["model_name"])
    embedding_provider = str(settings["embedding_provider"])
    st.markdown(
        f"""
        <div class="hero">
            <div>
                <div class="eyebrow">AI job outreach assistant</div>
                <h1>Career Email Copilot</h1>
                <p>
                    Chat with an AI assistant that reads a job post, retrieves your best portfolio proof
                    from ChromaDB, and drafts a polished cold email.
                </p>
            </div>
            <div class="hero-panel">
                <span>LLM</span>
                <strong>{provider}</strong>
                <span>Model</span>
                <strong>{model_name}</strong>
                <span>Retrieval</span>
                <strong>{embedding_provider} + ChromaDB</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_intro(candidate: CandidateProfile) -> None:
    profile_name = candidate.name.strip() or "there"
    with st.chat_message("assistant"):
        st.markdown(
            f"""
            Hi {profile_name}. I can help you turn a job post into a recruiter-ready cold email.

            Send me a job description below. I will extract the role requirements, search your portfolio
            knowledge base, and return a complete subject line plus email body.
            """
        )


def render_idle_panel() -> None:
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            """
            <div class="feature-card">
                <span>Step 1</span>
                <strong>Add the job post</strong>
                <p>Paste a description, fetch a public URL, or use the sample post.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            """
            <div class="feature-card">
                <span>Step 2</span>
                <strong>Retrieve proof</strong>
                <p>ChromaDB matches the role against your portfolio projects.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            """
            <div class="feature-card">
                <span>Step 3</span>
                <strong>Send a sharper email</strong>
                <p>Groq and LangChain generate a concise, job-specific draft.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_results(*, job, portfolio_matches, email: str, indexed_count: int, groq_active: bool) -> None:
    st.markdown('<div class="chat-spacer"></div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        st.markdown("I found the core requirements and matched them against your portfolio.")
        metric_a, metric_b, metric_c, metric_d = st.columns(4)
        metric_a.metric("Role", job.role)
        metric_b.metric("Company", job.company)
        metric_c.metric("Skills found", len(job.required_skills))
        metric_d.metric("Portfolio indexed", indexed_count)

    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        with st.chat_message("assistant"):
            st.markdown("#### Job intelligence")
            st.write(f"**Location:** {job.location}")
            st.write(f"**Experience:** {job.experience_level}")
            if job.required_skills:
                st.write("**Required skills:** " + ", ".join(job.required_skills))
            if job.preferred_skills:
                st.write("**Preferred skills:** " + ", ".join(job.preferred_skills))
            if job.description_summary:
                st.write(job.description_summary)

            st.markdown("#### Portfolio evidence")
            for item in portfolio_matches:
                label = item.title
                if item.score is not None:
                    label = f"{item.title} | match distance {item.score:.3f}"
                with st.expander(label, expanded=False):
                    st.write(item.description)
                    st.write(f"**Skills:** {item.skills}")
                    if item.outcome:
                        st.write(f"**Outcome:** {item.outcome}")
                    if item.url:
                        st.link_button("Open project", item.url)

    with right:
        with st.chat_message("assistant"):
            status = "Generated with Groq" if groq_active else "Generated in demo mode"
            st.markdown(f"#### Copy-ready email draft")
            st.caption(status)
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
        :root {
            --bg: #f5f7fb;
            --surface: #ffffff;
            --surface-soft: #f8fafc;
            --line: #dbe3ef;
            --text: #0f172a;
            --muted: #64748b;
            --primary: #0f766e;
            --primary-dark: #0b5f59;
            --primary-soft: #dff5f1;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.09), transparent 34rem),
                linear-gradient(180deg, #f8fafc 0%, var(--bg) 100%);
            color: var(--text);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3.5rem;
            max-width: 1280px;
        }
        h1, h2, h3, h4, h5, h6, p, label {
            letter-spacing: 0;
        }
        [data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        [data-testid="stSidebar"] * {
            color: #e5edf7;
        }
        [data-testid="stSidebar"] input {
            color: #0f172a;
            background: #ffffff;
            border-radius: 8px;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: #a8b6ca;
        }
        .sidebar-brand {
            display: flex;
            gap: 0.75rem;
            align-items: center;
            margin: 0.4rem 0 1.4rem;
            padding: 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.06);
        }
        .brand-mark {
            display: grid;
            place-items: center;
            width: 2.5rem;
            height: 2.5rem;
            border-radius: 8px;
            background: var(--primary);
            color: white;
            font-weight: 800;
        }
        .brand-title {
            font-size: 0.98rem;
            font-weight: 800;
            color: #ffffff;
        }
        .brand-subtitle {
            font-size: 0.78rem;
            color: #a8b6ca;
        }
        .hero {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(260px, 340px);
            gap: 1.4rem;
            align-items: stretch;
            margin-bottom: 1.4rem;
            padding: 1.5rem;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.88);
            box-shadow: 0 18px 46px rgba(15, 23, 42, 0.08);
        }
        .hero h1 {
            margin: 0.18rem 0 0.55rem;
            font-size: clamp(2rem, 4vw, 3.4rem);
            line-height: 1.03;
        }
        .hero p {
            max-width: 760px;
            margin: 0;
            color: var(--muted);
            font-size: 1.04rem;
            line-height: 1.6;
        }
        .eyebrow {
            color: var(--primary);
            font-weight: 800;
            font-size: 0.76rem;
            text-transform: uppercase;
        }
        .hero-panel {
            display: grid;
            grid-template-columns: 0.7fr 1.3fr;
            gap: 0.65rem 0.85rem;
            align-content: center;
            padding: 1rem;
            border-radius: 10px;
            background: #0f172a;
            color: #ffffff;
        }
        .hero-panel span {
            color: #9fb0c7;
            font-size: 0.82rem;
        }
        .hero-panel strong {
            color: #ffffff;
            font-size: 0.92rem;
            overflow-wrap: anywhere;
        }
        .section-label {
            margin: 1.2rem 0 0.7rem;
            color: var(--primary);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05);
        }
        [data-testid="stMetricValue"] {
            font-size: 1.85rem;
            color: var(--text);
        }
        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.95rem 1.05rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
        }
        [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
            background: var(--primary);
        }
        .feature-card {
            min-height: 130px;
            padding: 1rem;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
        }
        .feature-card span {
            color: var(--primary);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        .feature-card strong {
            display: block;
            margin-top: 0.35rem;
            font-size: 1.05rem;
        }
        .feature-card p {
            margin: 0.45rem 0 0;
            color: var(--muted);
            line-height: 1.5;
        }
        .chat-spacer {
            height: 1rem;
            border-top: 1px solid var(--line);
            margin-top: 1.2rem;
        }
        textarea {
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            border-radius: 10px !important;
        }
        div.stButton > button,
        div.stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] button {
            border-radius: 10px;
            border: 1px solid var(--primary);
            background: var(--primary);
            color: white;
            font-weight: 800;
        }
        div.stButton > button:hover,
        div.stDownloadButton > button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            border-color: var(--primary-dark);
            background: var(--primary-dark);
            color: white;
        }
        [data-testid="stExpander"] {
            border-radius: 10px;
            border-color: var(--line);
            background: rgba(255, 255, 255, 0.8);
        }
        @media (max-width: 900px) {
            .hero {
                grid-template-columns: 1fr;
                padding: 1.1rem;
            }
            .hero h1 {
                font-size: 2.2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
