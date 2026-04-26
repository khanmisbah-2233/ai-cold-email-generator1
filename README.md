# AI Cold Email Generator

An AI-powered Streamlit application that turns job descriptions into personalized cold emails by matching the role against a ChromaDB-backed portfolio knowledge base.

The LLM layer uses Groq through LangChain. The default model is `llama-3.3-70b-versatile`.

## What It Does

- Accepts a pasted job description, a job-post URL, or a sample job post.
- Uses LangChain with Groq to extract role, company, skills, responsibilities, and seniority.
- Stores portfolio projects in ChromaDB and retrieves the most relevant evidence for each role.
- Generates a tailored cold email with a subject line, concise pitch, and portfolio proof.
- Includes a no-key demo fallback so the UI can still be tested before adding credentials.

## Project Structure

```text
.
|-- app.py                    # Streamlit UI and orchestration
|-- data/
|   `-- portfolio.csv         # Editable portfolio knowledge base
|-- src/
|   |-- chains.py             # LangChain parsing and generation chains
|   |-- config.py             # Paths, defaults, and provider config
|   |-- embeddings.py         # Embedding factory and local hashing embeddings
|   |-- job_loader.py         # Job-post URL extraction
|   |-- llm.py                # Groq chat model factory
|   |-- models.py             # Pydantic data models
|   |-- portfolio.py          # ChromaDB indexing and retrieval
|   `-- text.py               # Text cleanup helpers
|-- .env.example              # Environment variable template
|-- requirements.txt          # Python dependencies
`-- .streamlit/
    `-- config.toml           # Streamlit theme and server defaults
```

## Setup

Python 3.11 or 3.12 is recommended for the smoothest dependency support.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

Add your Groq key to `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

OpenAI embeddings are optional. Without `OPENAI_API_KEY`, the app uses local hashing embeddings for ChromaDB retrieval.

## Run

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Streamlit Cloud Deployment

In Streamlit Cloud, open **App settings**, then **Secrets**, and add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

The app is configured with `server.headless = true` and `browser.gatherUsageStats = false` so it can run in Streamlit Cloud without the first-run prompt.

ChromaDB needs a recent SQLite build on Streamlit Cloud. The deployment installs `pysqlite3-binary` on Linux and swaps it in before ChromaDB imports. The ChromaDB/OpenTelemetry/protobuf versions are pinned in `requirements.txt` so cloud and local builds use the same compatible stack.

## Portfolio Data

Edit `data/portfolio.csv` with your own projects, case studies, experience bullets, or portfolio links.

Required columns:

- `title`
- `category`
- `skills`
- `description`
- `outcome`
- `url`

The app indexes this CSV into a persistent local ChromaDB collection under `chroma_db/`. After editing the CSV, delete the local `chroma_db/` folder or set `REBUILD_PORTFOLIO_INDEX=true` in `.env` for the next run.

## Demo Flow

1. Start the Streamlit app.
2. Confirm `.env` contains `GROQ_API_KEY` and `GROQ_MODEL`.
3. Paste a job description or use the sample post.
4. Click **Generate tailored email**.
5. Review the parsed job, retrieved portfolio matches, and generated email.
6. Download the final email as a `.txt` file.

## Notes

- Groq is the primary LLM provider for parsing and email generation.
- Demo mode creates a structured template email without calling an external LLM.
- The app avoids fabricating details by instructing the model to use only the provided job description and portfolio evidence.
