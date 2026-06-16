# REFACTOR: Lightweight web-search page fetching and extraction helpers.
from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from ..rag.document_loader import (
    LoaderFactory,
    SourceDocumentCache,
    default_loader_factory,
    load_source_documents,
)
from .fetch_policy import FetchPolicy, resolve_fetch_policy
from .playwright_loader import playwright_loader_factory

logger = logging.getLogger(__name__)

DEFAULT_MIN_READABLE_CHARS = 200
DEFAULT_MIN_READABLE_TOKENS = 50


@dataclass(frozen=True)
class FetchedPage:
    """Fetched web page content prepared for direct LLM prompting."""

    url: str
    title: str
    text: str
    fetch_time_ms: float
    error: str | None = None
    extracted_chars: int = 0
    extracted_tokens: int = 0
    fetch_method: str = "http"


class WebPageFetchError(RuntimeError):
    """Raised when page content cannot be prepared for prompt context."""


def fetch_pages(
    urls: Sequence[str],
    *,
    timeout: float = 15.0,
    max_tokens_per_page: int = 8000,
    cache_ttl_seconds: int = 300,
    max_concurrent_loads: int = 4,
    min_readable_chars: int = DEFAULT_MIN_READABLE_CHARS,
    min_readable_tokens: int = DEFAULT_MIN_READABLE_TOKENS,
    document_cache: SourceDocumentCache | None = None,
    loader_factory: LoaderFactory = default_loader_factory,
    js_fallback_enabled: bool = False,
    js_fallback_domains: Sequence[str] | None = None,
    js_force_domains: Sequence[str] | None = None,
    js_loader_factory: LoaderFactory | None = None,
) -> list[FetchedPage]:
    """Fetch URLs with the shared loader and return prompt-ready page text.

    Args:
        urls: Ordered URLs to load.
        timeout: Per-page loader timeout in seconds.
        max_tokens_per_page: Rough token budget for each returned page.
        cache_ttl_seconds: Shared source document cache TTL.
        max_concurrent_loads: Existing loader concurrency limit.
        min_readable_chars: Minimum normalized characters required.
        min_readable_tokens: Minimum estimated tokens required.
        document_cache: Optional cache override for tests or callers.
        loader_factory: Optional loader factory override.
        js_fallback_enabled: Enables optional browser fallback for policy domains.
        js_fallback_domains: Domains that may retry with JS on low text.
        js_force_domains: Domains that should skip HTTP and use JS first.
        js_loader_factory: Optional JS loader override for tests.

    Returns:
        Fetched pages in the original URL order, with error entries for URLs
        that produced no document.
    """

    ordered_urls = [url for url in urls if url]
    if not ordered_urls:
        return []

    policies = {
        url: resolve_fetch_policy(
            url,
            js_fallback_enabled=js_fallback_enabled,
            js_fallback_domains=js_fallback_domains,
            js_force_domains=js_force_domains,
        )
        for url in ordered_urls
    }
    http_urls = [url for url in ordered_urls if not policies[url].force_js]
    force_js_urls = [url for url in ordered_urls if policies[url].force_js]
    effective_loader_factory = _policy_http_loader_factory(policies, loader_factory)

    started_at = time.monotonic()
    documents, load_error = _load_documents(
        http_urls,
        timeout=timeout,
        cache_ttl_seconds=cache_ttl_seconds,
        max_concurrent_loads=max_concurrent_loads,
        document_cache=document_cache,
        loader_factory=effective_loader_factory,
    )
    js_factory = js_loader_factory or playwright_loader_factory
    js_documents, js_load_error = _load_js_documents(
        force_js_urls,
        timeout=timeout,
        loader_factory=js_factory,
    )
    documents.extend(js_documents)
    elapsed_ms = (time.monotonic() - started_at) * 1000
    grouped_documents = _group_documents_by_source(documents)
    pages = [
        _page_from_documents(
            url,
            grouped_documents.get(url, []),
            elapsed_ms=elapsed_ms,
            max_tokens_per_page=max_tokens_per_page,
            min_readable_chars=min_readable_chars,
            min_readable_tokens=min_readable_tokens,
            load_error=js_load_error if policies[url].force_js else load_error,
            fetch_method="js" if policies[url].force_js else "http",
        )
        for url in ordered_urls
    ]
    pages = _retry_pages_with_js_fallback(
        pages,
        policies=policies,
        timeout=timeout,
        max_tokens_per_page=max_tokens_per_page,
        min_readable_chars=min_readable_chars,
        min_readable_tokens=min_readable_tokens,
        js_loader_factory=js_factory,
    )
    for page in pages:
        logger.info(
            "Fetched web page content: url=%s extracted_chars=%d "
            "extracted_tokens=%d prompt_chars=%d prompt_tokens=%d error=%s "
            "fetch_method=%s",
            page.url,
            page.extracted_chars,
            page.extracted_tokens,
            len(page.text or ""),
            estimate_tokens(page.text),
            page.error,
            page.fetch_method,
        )
    return pages


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
        soup.find(attrs={"role": "main"}),  # type: ignore[call-overload]
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


def is_readable_text(
    text: str,
    *,
    min_chars: int = DEFAULT_MIN_READABLE_CHARS,
    min_tokens: int = DEFAULT_MIN_READABLE_TOKENS,
) -> bool:
    """Return whether extracted text is large enough to ground an answer."""

    normalized = normalize_whitespace(text)
    if not normalized:
        return False

    min_chars = max(0, int(min_chars))
    min_tokens = max(0, int(min_tokens))
    checks: list[bool] = []
    if min_chars > 0:
        checks.append(len(normalized) >= min_chars)
    if min_tokens > 0:
        checks.append(estimate_tokens(normalized) >= min_tokens)
    return any(checks) if checks else True


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
    if not urls:
        return [], None

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


def _load_js_documents(
    urls: Sequence[str],
    *,
    timeout: float,
    loader_factory: LoaderFactory,
) -> tuple[list[Document], str | None]:
    documents: list[Document] = []
    errors: list[str] = []
    for url in urls:
        started_at = time.monotonic()
        loaded, error = _load_documents(
            [url],
            timeout=timeout,
            cache_ttl_seconds=0,
            max_concurrent_loads=1,
            document_cache=None,
            loader_factory=loader_factory,
        )
        if error:
            errors.append(f"{url}: {error}")
        documents.extend(_normalize_js_documents(loaded, url, started_at))
    return documents, "; ".join(errors) or None


def _normalize_js_documents(
    documents: Sequence[Document],
    url: str,
    started_at: float,
) -> list[Document]:
    normalized: list[Document] = []
    for document in documents:
        metadata: dict[str, Any] = dict(document.metadata or {})
        metadata.setdefault(
            "rendered_url",
            metadata.get("url") or metadata.get("source") or url,
        )
        metadata["source"] = url
        metadata["url"] = url
        metadata.setdefault("fetch_method", "js")
        metadata.setdefault("fetch_time_ms", (time.monotonic() - started_at) * 1000)
        normalized.append(Document(page_content=document.page_content, metadata=metadata))
    return normalized


def _policy_http_loader_factory(
    policies: dict[str, FetchPolicy],
    loader_factory: LoaderFactory,
) -> LoaderFactory:
    if loader_factory is not default_loader_factory:
        return loader_factory

    def build_loader(url: str, page_timeout: int) -> Any:
        policy = policies.get(url)
        if not policy or not policy.request_headers:
            return default_loader_factory(url, page_timeout)
        return WebBaseLoader(
            url,
            requests_kwargs={
                "timeout": page_timeout,
                "headers": dict(policy.request_headers),
            },
        )

    return build_loader


def _retry_pages_with_js_fallback(
    pages: Sequence[FetchedPage],
    *,
    policies: dict[str, FetchPolicy],
    timeout: float,
    max_tokens_per_page: int,
    min_readable_chars: int,
    min_readable_tokens: int,
    js_loader_factory: LoaderFactory,
) -> list[FetchedPage]:
    retried_pages: list[FetchedPage] = []
    for page in pages:
        policy = policies.get(page.url, FetchPolicy())
        if not _should_retry_with_js(page, policy):
            retried_pages.append(page)
            continue
        retried_pages.append(
            _fetch_js_fallback_page(
                page,
                timeout=timeout,
                max_tokens_per_page=max_tokens_per_page,
                min_readable_chars=min_readable_chars,
                min_readable_tokens=min_readable_tokens,
                loader_factory=js_loader_factory,
            )
        )
    return retried_pages


def _should_retry_with_js(page: FetchedPage, policy: FetchPolicy) -> bool:
    return bool(
        policy.retry_js_on_low_text
        and not policy.force_js
        and not page.text
        and page.fetch_method == "http"
    )


def _fetch_js_fallback_page(
    page: FetchedPage,
    *,
    timeout: float,
    max_tokens_per_page: int,
    min_readable_chars: int,
    min_readable_tokens: int,
    loader_factory: LoaderFactory,
) -> FetchedPage:
    started_at = time.monotonic()
    documents, load_error = _load_js_documents(
        [page.url],
        timeout=timeout,
        loader_factory=loader_factory,
    )
    elapsed_ms = (time.monotonic() - started_at) * 1000
    fallback_page = _page_from_documents(
        page.url,
        documents,
        elapsed_ms=elapsed_ms,
        max_tokens_per_page=max_tokens_per_page,
        min_readable_chars=min_readable_chars,
        min_readable_tokens=min_readable_tokens,
        load_error=load_error,
        fetch_method="js_fallback",
    )
    if fallback_page.text:
        return fallback_page
    return replace(
        page,
        error=_combined_errors(page.error, fallback_page.error),
        fetch_method="http+js_failed",
    )


def _combined_errors(primary: str | None, fallback: str | None) -> str | None:
    if primary and fallback:
        return f"{primary} JS fallback failed: {fallback}"
    if fallback:
        return f"JS fallback failed: {fallback}"
    return primary


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
    min_readable_chars: int,
    min_readable_tokens: int,
    load_error: str | None,
    fetch_method: str = "http",
) -> FetchedPage:
    if not documents:
        error = load_error or "No document loaded for URL."
        return FetchedPage(
            url=url,
            title="",
            text="",
            fetch_time_ms=elapsed_ms,
            error=error,
            fetch_method=fetch_method,
        )

    title = _document_title(documents[0])
    extracted_text = normalize_whitespace(
        "\n\n".join(extract_text(document.page_content) for document in documents)
    )
    extracted_chars = len(extracted_text)
    extracted_tokens = estimate_tokens(extracted_text)
    if fetch_method.startswith("js") and _looks_like_loader_error(extracted_text):
        return FetchedPage(
            url=url,
            title=title,
            text="",
            fetch_time_ms=elapsed_ms,
            error=f"JS loader returned error text: {_short_error_text(extracted_text)}",
            extracted_chars=extracted_chars,
            extracted_tokens=extracted_tokens,
            fetch_method=fetch_method,
        )
    if not is_readable_text(
        extracted_text,
        min_chars=min_readable_chars,
        min_tokens=min_readable_tokens,
    ):
        error = (
            "Loaded document text below readability threshold "
            f"({extracted_chars} chars, {extracted_tokens} tokens)."
        )
        return FetchedPage(
            url=url,
            title=title,
            text="",
            fetch_time_ms=elapsed_ms,
            error=error,
            extracted_chars=extracted_chars,
            extracted_tokens=extracted_tokens,
            fetch_method=fetch_method,
        )

    text = truncate_to_token_budget(extracted_text, max_tokens_per_page)
    readable_error = None if text else "Loaded document did not contain readable text."
    return FetchedPage(
        url=url,
        title=title,
        text=text,
        fetch_time_ms=elapsed_ms,
        error=readable_error,
        extracted_chars=extracted_chars,
        extracted_tokens=extracted_tokens,
        fetch_method=fetch_method,
    )


def _looks_like_loader_error(text: str) -> bool:
    normalized = normalize_whitespace(text).lower()
    return normalized.startswith("error:")


def _short_error_text(text: str) -> str:
    return normalize_whitespace(text)[:180]


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
    "DEFAULT_MIN_READABLE_CHARS",
    "DEFAULT_MIN_READABLE_TOKENS",
    "WebPageFetchError",
    "estimate_tokens",
    "extract_text",
    "fetch_pages",
    "is_readable_text",
    "normalize_whitespace",
    "truncate_to_token_budget",
]
