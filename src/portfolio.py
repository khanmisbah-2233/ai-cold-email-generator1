"""ChromaDB-backed portfolio indexing and retrieval."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

if sys.platform.startswith("linux"):
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass

import chromadb
import pandas as pd
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .models import JobSummary, RetrievedPortfolioItem
from .text import compact_text


REQUIRED_COLUMNS = {"title", "category", "skills", "description", "outcome", "url"}
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True


class PortfolioStore:
    """Manage portfolio data in a persistent ChromaDB collection."""

    def __init__(
        self,
        *,
        csv_path: Path,
        persist_directory: Path,
        collection_name: str,
        embedding_function: Embeddings,
    ) -> None:
        self.csv_path = csv_path
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self._vector_store: Chroma | None = None

    def ensure_index(self, *, rebuild: bool = False) -> int:
        """Create the ChromaDB collection if needed and return document count."""
        if rebuild:
            self._delete_collection()

        vector_store = self.vector_store
        existing_count = self.count()
        if existing_count > 0 and not rebuild:
            return existing_count

        documents, ids = self._load_documents()
        if not documents:
            raise ValueError("No portfolio rows were found to index.")

        vector_store.add_documents(documents=documents, ids=ids)
        return self.count()

    @property
    def vector_store(self) -> Chroma:
        if self._vector_store is None:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
            self._vector_store = Chroma(
                client=client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
            )
        return self._vector_store

    def count(self) -> int:
        try:
            return int(self.vector_store._collection.count())
        except Exception:
            return 0

    def search(self, job: JobSummary, *, k: int = 3) -> list[RetrievedPortfolioItem]:
        """Retrieve portfolio entries that best match a parsed job."""
        self.ensure_index()
        results = self.vector_store.similarity_search_with_score(job.query_text(), k=k)
        matches: list[RetrievedPortfolioItem] = []
        for document, score in results:
            metadata = document.metadata
            matches.append(
                RetrievedPortfolioItem(
                    title=str(metadata.get("title", "")),
                    category=str(metadata.get("category", "")),
                    skills=str(metadata.get("skills", "")),
                    description=str(metadata.get("description", "")),
                    outcome=str(metadata.get("outcome", "")),
                    url=str(metadata.get("url", "")),
                    score=float(score),
                    content=document.page_content,
                )
            )
        return matches

    def _delete_collection(self) -> None:
        client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._vector_store = None

    def _load_documents(self) -> tuple[list[Document], list[str]]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Portfolio CSV not found: {self.csv_path}")

        frame = pd.read_csv(self.csv_path).fillna("")
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"Portfolio CSV is missing columns: {missing_columns}")

        documents: list[Document] = []
        ids: list[str] = []
        seen_ids: set[str] = set()

        for index, row in frame.iterrows():
            title = compact_text(str(row["title"]))
            if not title:
                continue

            metadata = {
                "title": title,
                "category": compact_text(str(row["category"])),
                "skills": compact_text(str(row["skills"])),
                "description": compact_text(str(row["description"])),
                "outcome": compact_text(str(row["outcome"])),
                "url": compact_text(str(row["url"])),
            }
            content = self._content_from_metadata(metadata)
            digest = hashlib.sha256(f"{index}|{content}".encode("utf-8")).hexdigest()[:32]
            while digest in seen_ids:
                digest = hashlib.sha256(f"{digest}|duplicate".encode("utf-8")).hexdigest()[:32]
            seen_ids.add(digest)

            documents.append(Document(page_content=content, metadata=metadata))
            ids.append(digest)

        return documents, ids

    @staticmethod
    def _content_from_metadata(metadata: dict[str, str]) -> str:
        return "\n".join(
            [
                f"Project: {metadata['title']}",
                f"Category: {metadata['category']}",
                f"Skills: {metadata['skills']}",
                f"Description: {metadata['description']}",
                f"Outcome: {metadata['outcome']}",
                f"URL: {metadata['url']}",
            ]
        )
