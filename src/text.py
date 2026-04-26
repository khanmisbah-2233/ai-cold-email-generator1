"""Text cleanup utilities."""

from __future__ import annotations

import re


def clean_text(value: str) -> str:
    """Normalize whitespace while preserving sentence boundaries."""
    value = value or ""
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def compact_text(value: str) -> str:
    """Collapse all whitespace into single spaces."""
    return re.sub(r"\s+", " ", value or "").strip()


def truncate_text(value: str, limit: int = 12_000) -> str:
    """Trim very large inputs to a model-friendly size."""
    value = clean_text(value)
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0].strip()
