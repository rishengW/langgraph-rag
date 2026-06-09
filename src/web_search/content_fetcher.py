# REFACTOR: Lightweight web-search page fetching and extraction helpers.
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
import time
from typing import Any

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from ..rag.document_loader import (
    LoaderFactory,
    SourceDocumentCache,
    default_loader_factory,
    load_source_documents,
)


@dataclass(frozen=True)
class FetchedPage:
    """Fetched web page content prepared for direct LLM prompting."""

    url: str
    title: str
    text: str
    fetch_time_ms: float
    error: str | None = None


class WebPageFetchError(RuntimeError):
    """Raised when page content cannot be prepared for prompt context."""


def fetch_pages(
    urls: Sequence[str],
    *,
    timeout: float = 15.0,
    max_tokens_per_page: int = 8000,
    cache_ttl_seconds: int = 300,
    max_concurrent_loads: int = 4,
    document_cache: SourceDocumentCache | None = None,
    loader_factory: LoaderFactory = default_loader_factory,
) -> list[FetchedPage]:
    """Fetch URLs with the shared loader and return prompt-ready page text.

    Args:
        urls: Ordered URLs to load.
        timeout: Per-page loader timeout in seconds.
        max_tokens_per_page: Rough token budget for each returned page.
        cache_ttl_seconds: Shared source document cache TTL.
        max_concurrent_loads: Existing loader concurrency limit.
        document_cache: Optional cache override for tests or callers.
        loader_factory: Optional loader factory override.

    Returns:
        Fetched pages in the original URL order, with error entries for URLs
        that produced no document.
    """

    ordered_urls = [url for url in urls if url]
    if not ordered_urls:
        return []

    started_at = time.monotonic()
    documents, load_error = _load_documents(
        ordered_urls,
        timeout=timeout,
        cache_ttl_seconds=cache_ttl_seconds,
        max_concurrent_loads=max_concurrent_loads,
        document_cache=document_cache,
        loader_factory=loader_factory,
    )
    elapsed_ms = (time.monotonic() - started_at) * 1000
    grouped_documents = _group_documents_by_source(documents)
    return [
        _page_from_documents(
            url,
            grouped_documents.get(url, []),
            elapsed_ms=elapsed_ms,
            max_tokens_per_page=max_tokens_per_page,
            load_error=load_error,
        )
        for url in ordered_urls
    ]


def extract_text(html: str) -> str:
    """Extract readable article-style text from HTML or raw page text."""

    soup = BeautifulSoup(html or "", "html.parser")
    for element in soup(
        ["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]
    ):
        element.decompose()

    candidates = [
        soup.find("article"),
        soup.find("main"),
        soup.find(attrs={"role": "main"}),
        soup.body,
        soup,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        text = normalize_whitespace(candidate.get_text(" ", strip=True))
        if text:
            return text
    return ""


def estimate_tokens(text: str) -> int:
    """Estimate token count with the project's rough chars / 4 heuristic."""

    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Trim text to an approximate token budget."""

    token_budget = max(0, int(max_tokens))
    if estimate_tokens(text) <= token_budget:
        return text
    max_chars = token_budget * 4
    return normalize_whitespace(text[:max_chars])


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace into prompt-friendly text."""

    return re.sub(r"\s+", " ", text or "").strip()


def _load_documents(
    urls: Sequence[str],
    *,
    timeout: float,
    cache_ttl_seconds: int,
    max_concurrent_loads: int,
    document_cache: SourceDocumentCache | None,
    loader_factory: LoaderFactory,
) -> tuple[list[Document], str | None]:
    try:
        documents = load_source_documents(
            urls,
            page_load_timeout=max(1, int(timeout)),
            max_concurrent_loads=max_concurrent_loads,
            page_load_cache_ttl_seconds=cache_ttl_seconds,
            document_cache=document_cache,
            loader_factory=loader_factory,
        )
    except Exception as exc:
        return [], str(exc)
    return documents, None


def _group_documents_by_source(documents: Sequence[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = {}
    for document in documents:
        source = _document_url(document)
        if source:
            grouped.setdefault(source, []).append(document)
    return grouped


def _page_from_documents(
    url: str,
    documents: Sequence[Document],
    *,
    elapsed_ms: float,
    max_tokens_per_page: int,
    load_error: str | None,
) -> FetchedPage:
    if not documents:
        error = load_error or "No document loaded for URL."
        return FetchedPage(url=url, title="", text="", fetch_time_ms=elapsed_ms, error=error)

    title = _document_title(documents[0])
    text = "\n\n".join(extract_text(document.page_content) for document in documents)
    text = truncate_to_token_budget(normalize_whitespace(text), max_tokens_per_page)
    error = None if text else "Loaded document did not contain readable text."
    return FetchedPage(url=url, title=title, text=text, fetch_time_ms=elapsed_ms, error=error)


def _document_url(document: Document) -> str:
    metadata: dict[str, Any] = document.metadata or {}
    return str(metadata.get("source") or metadata.get("url") or "")


def _document_title(document: Document) -> str:
    metadata: dict[str, Any] = document.metadata or {}
    title = metadata.get("title")
    if title:
        return normalize_whitespace(str(title))
    soup = BeautifulSoup(document.page_content or "", "html.parser")
    if soup.title is None or soup.title.string is None:
        return ""
    return normalize_whitespace(soup.title.string)


__all__ = [
    "FetchedPage",
    "WebPageFetchError",
    "estimate_tokens",
    "extract_text",
    "fetch_pages",
    "normalize_whitespace",
    "truncate_to_token_budget",
]
