"""LLM factory functions."""

from __future__ import annotations


class LLMConfigurationError(RuntimeError):
    """Raised when a provider is selected without required configuration."""


def create_chat_model(
    provider: str,
    *,
    model_name: str,
    api_key: str | None = None,
    temperature: float = 0.05,
    base_url: str | None = None,
):
    """Create a LangChain chat model for the selected provider."""
    if provider == "Demo mode":
        return None

    if provider == "Groq":
        if not api_key:
            raise LLMConfigurationError("Groq is selected, but GROQ_API_KEY is missing.")
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name, temperature=temperature, api_key=api_key)

    raise LLMConfigurationError(f"Unsupported LLM provider: {provider}")
