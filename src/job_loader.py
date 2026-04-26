"""Load and clean job descriptions from external URLs."""

from __future__ import annotations

from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .text import clean_text


def fetch_job_posting(url: str, *, timeout: int = 20) -> str:
    """Fetch readable text from a job posting URL."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a valid http or https URL.")

    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; AIColdEmailGenerator/1.0; "
                "+https://localhost)"
            )
        },
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    cleaned = clean_text(text)
    if len(cleaned) < 120:
        raise ValueError("The URL did not return enough readable job-post text.")
    return cleaned
