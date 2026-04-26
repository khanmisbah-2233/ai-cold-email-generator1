"""Embedding helpers for ChromaDB."""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}")


class HashingEmbeddings(Embeddings):
    """Small local embedding model based on stable feature hashing.

    It is not a replacement for semantic embedding APIs, but it gives the app a
    no-key retrieval path and works well for skill-heavy portfolio matching.
    """

    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [token.lower() for token in TOKEN_PATTERN.findall(text or "")]
        features = tokens + [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]

        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.5 if " " in feature else 1.0
            vector[bucket] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def create_embedding_function(
    provider: str,
    *,
    openai_api_key: str | None = None,
    openai_model: str = "text-embedding-3-small",
) -> Embeddings:
    """Create a LangChain-compatible embedding function."""
    if provider == "OpenAI":
        if not openai_api_key:
            raise ValueError("OpenAI embeddings require OPENAI_API_KEY.")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=openai_model, api_key=openai_api_key)

    return HashingEmbeddings()
