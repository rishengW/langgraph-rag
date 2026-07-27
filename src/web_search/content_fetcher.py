# REFACTOR: Lightweight web-search page fetching and extraction helpers.
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date
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
from .common import host_authority_class, is_page_text_relevant, page_relevance_score
from .date_extractor import extract_publication_date
from .fetch_policy import FetchPolicy, resolve_fetch_policy
from .page_structure import (
    DEFAULT_MAX_LINK_DENSITY,
    DEFAULT_MIN_CONTENT_WORDS,
    PageStructure,
    assess_page_structure,
)
from .pdf_loader import PdfPageLoader, is_pdf_url, looks_like_pdf_payload
from .playwright_loader import playwright_loader_factory

logger = logging.getLogger(__name__)

DEFAULT_MIN_READABLE_CHARS = 200
DEFAULT_MIN_READABLE_TOKENS = 50
SHORT_OFFICIAL_MIN_CHARS = 40
SHORT_OFFICIAL_LEAD_MAX_CHARS = 600
SHORT_OFFICIAL_MIN_RELEVANCE_SCORE = 15


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
    publication_date: date | None = None
    structure: PageStructure = field(default_factory=PageStructure)


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
    relevance_query: str = "",
    document_cache: SourceDocumentCache | None = None,
    loader_factory: LoaderFactory = default_loader_factory,
    js_fallback_enabled: bool = False,
    js_fallback_domains: Sequence[str] | None = None,
    js_force_domains: Sequence[str] | None = None,
    js_loader_factory: LoaderFactory | None = None,
    js_retry_budget: int = 2,
    max_link_density: float = DEFAULT_MAX_LINK_DENSITY,
    min_content_words: int = DEFAULT_MIN_CONTENT_WORDS,
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
        relevance_query: Original question used to retain concise, strongly relevant
            official pages. Empty queries use the standard length gate only.
        document_cache: Optional cache override for tests or callers.
        loader_factory: Optional loader factory override.
        js_fallback_enabled: Enables optional browser fallback for policy domains.
        js_fallback_domains: Domains prioritized for the JS retry.
        js_force_domains: Domains that should skip HTTP and use JS first.
        js_loader_factory: Optional JS loader override for tests.
        js_retry_budget: Maximum browser renders for this batch.
        max_link_density: Anchor-text share above which a page is a listing.
        min_content_words: Content units below which a page is too thin.

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
            relevance_query=relevance_query,
            load_error=js_load_error if policies[url].force_js else load_error,
            fetch_method="js" if policies[url].force_js else "http",
            max_link_density=max_link_density,
            min_content_words=min_content_words,
        )
        for url in ordered_urls
    ]
    pages = _retry_pages_as_pdf(
        pages,
        timeout=timeout,
        max_tokens_per_page=max_tokens_per_page,
        min_readable_chars=min_readable_chars,
        min_readable_tokens=min_readable_tokens,
        relevance_query=relevance_query,
        policies=policies,
        max_link_density=max_link_density,
        min_content_words=min_content_words,
    )
    pages = _retry_pages_with_js_fallback(
        pages,
        policies=policies,
        timeout=timeout,
        max_tokens_per_page=max_tokens_per_page,
        min_readable_chars=min_readable_chars,
        min_readable_tokens=min_readable_tokens,
        relevance_query=relevance_query,
        js_loader_factory=js_factory,
        js_fallback_enabled=js_fallback_enabled,
        js_retry_budget=js_retry_budget,
        max_link_density=max_link_density,
        min_content_words=min_content_words,
    )
    for page in pages:
        logger.info(
            "Fetched web page content: url=%s extracted_chars=%d "
            "extracted_tokens=%d prompt_chars=%d prompt_tokens=%d error=%s "
            "fetch_method=%s publication_date=%s shape=%s link_density=%.3f "
            "content_words=%d",
            page.url,
            page.extracted_chars,
            page.extracted_tokens,
            len(page.text or ""),
            estimate_tokens(page.text),
            page.error,
            page.fetch_method,
            page.publication_date.isoformat() if page.publication_date else None,
            page.structure.shape,
            page.structure.link_density,
            page.structure.content_words,
        )
    return pages


def extract_text(html: str) -> str:
    """Extract readable article-style text from HTML or raw page text."""

    soup = BeautifulSoup(html or "", "html.parser")
    structured_texts = _json_ld_text_candidates(soup)
    for element in soup(
        ["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]
    ):
        element.decompose()

    selectors = (
        "article",
        "main",
        '[role="main"]',
        '[itemprop="articleBody"]',
        ".article-content",
        ".article-body",
        ".content-body",
        ".TRS_Editor",
    )
    semantic_texts: list[str] = []
    for selector in selectors:
        for candidate in soup.select(selector):
            text = normalize_whitespace(candidate.get_text(" ", strip=True))
            if text and text not in semantic_texts:
                semantic_texts.append(text)
    preferred = [*structured_texts, *semantic_texts]
    if preferred:
        return max(preferred, key=len)

    for fallback_candidate in (soup.body, soup):
        if fallback_candidate is None:
            continue
        text = normalize_whitespace(fallback_candidate.get_text(" ", strip=True))
        if text:
            return text
    return ""


def _json_ld_text_candidates(soup: BeautifulSoup) -> list[str]:
    candidates: list[str] = []
    for element in soup.select('script[type="application/ld+json"]'):
        raw = element.string or element.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            value = json.loads(raw)
        except (TypeError, ValueError):
            continue
        for text in _walk_json_ld_text(value):
            normalized = normalize_whitespace(text)
            if normalized and normalized not in candidates:
                candidates.append(normalized)
    return candidates


def _walk_json_ld_text(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, dict):
        for key in ("articleBody", "text", "description"):
            text = value.get(key)
            if isinstance(text, str) and len(normalize_whitespace(text)) >= 40:
                texts.append(text)
        for nested in value.values():
            if isinstance(nested, (dict, list)):
                texts.extend(_walk_json_ld_text(nested))
    elif isinstance(value, list):
        for nested in value:
            texts.extend(_walk_json_ld_text(nested))
    return texts


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


def is_readable_page(
    text: str,
    *,
    url: str = "",
    title: str = "",
    query: str = "",
    min_chars: int = DEFAULT_MIN_READABLE_CHARS,
    min_tokens: int = DEFAULT_MIN_READABLE_TOKENS,
) -> bool:
    """Accept long text normally or a concise, evidence-rich official page.

    The short-page exception requires a formally controlled or recognized
    first-party host, a non-trivial body, and concentrated relevance in the
    title and lead. ``is_page_text_relevant`` retains the strict requested-year
    and quantity-answer checks used by the fetched-page relevance gate.
    """

    normalized = normalize_whitespace(text)
    if is_readable_text(normalized, min_chars=min_chars, min_tokens=min_tokens):
        return True
    if not query.strip() or len(normalized) < SHORT_OFFICIAL_MIN_CHARS:
        return False
    if host_authority_class(url) == "standard":
        return False

    lead = normalized[:SHORT_OFFICIAL_LEAD_MAX_CHARS]
    if not is_page_text_relevant(lead, query, title=title):
        return False
    return page_relevance_score(lead, query, title=title) >= SHORT_OFFICIAL_MIN_RELEVANCE_SCORE


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
        # PDFs are a common shape for official notices and whitepapers. The
        # HTML loader returns binary noise for them, so route them through
        # pypdf extraction instead of discarding them at discovery time.
        if is_pdf_url(url):
            return PdfPageLoader(
                url,
                page_timeout,
                headers=policy.request_headers if policy else None,
            )
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


PDF_DETECTED_FETCH_METHOD = "http_pdf_detected"


def _pdf_loader_factory(headers: dict[str, str] | None) -> LoaderFactory:
    """Build a PDF loader factory bound to one page's request headers."""

    def build_loader(url: str, page_timeout: int) -> Any:
        return PdfPageLoader(url, page_timeout, headers=headers)

    return build_loader


def _retry_pages_as_pdf(
    pages: Sequence[FetchedPage],
    *,
    timeout: float,
    max_tokens_per_page: int,
    min_readable_chars: int,
    min_readable_tokens: int,
    relevance_query: str,
    policies: dict[str, FetchPolicy],
    max_link_density: float,
    min_content_words: int,
) -> list[FetchedPage]:
    """Re-fetch responses that turned out to be PDF bytes behind an HTML URL."""

    retried: list[FetchedPage] = []
    for page in pages:
        if page.fetch_method != PDF_DETECTED_FETCH_METHOD:
            retried.append(page)
            continue
        headers = policies.get(page.url, FetchPolicy()).request_headers or None
        started_at = time.monotonic()
        documents, load_error = _load_documents(
            [page.url],
            timeout=timeout,
            cache_ttl_seconds=0,
            max_concurrent_loads=1,
            document_cache=None,
            loader_factory=_pdf_loader_factory(headers),
        )
        elapsed_ms = (time.monotonic() - started_at) * 1000
        logger.info("Re-fetching %s as PDF after detecting a PDF response body", page.url)
        retried.append(
            _page_from_documents(
                page.url,
                documents,
                elapsed_ms=elapsed_ms,
                max_tokens_per_page=max_tokens_per_page,
                min_readable_chars=min_readable_chars,
                min_readable_tokens=min_readable_tokens,
                relevance_query=relevance_query,
                load_error=load_error,
                fetch_method="pdf",
                max_link_density=max_link_density,
                min_content_words=min_content_words,
            )
        )
    return retried


def _retry_pages_with_js_fallback(
    pages: Sequence[FetchedPage],
    *,
    policies: dict[str, FetchPolicy],
    timeout: float,
    max_tokens_per_page: int,
    min_readable_chars: int,
    min_readable_tokens: int,
    relevance_query: str,
    js_loader_factory: LoaderFactory,
    js_fallback_enabled: bool = False,
    js_retry_budget: int = 2,
    max_link_density: float = DEFAULT_MAX_LINK_DENSITY,
    min_content_words: int = DEFAULT_MIN_CONTENT_WORDS,
) -> list[FetchedPage]:
    """Re-render unreadable HTTP pages with a browser, newest signal first.

    The retry trigger is the measurement we already have (no readable text, or a
    login/enable-JavaScript shell) rather than a domain allowlist. Configured
    JS domains keep priority when the budget cannot cover every candidate.
    """

    budget = max(0, int(js_retry_budget)) if js_fallback_enabled else 0
    candidates = [
        index
        for index, page in enumerate(pages)
        if _should_retry_with_js(page, policies.get(page.url, FetchPolicy()))
    ]
    candidates.sort(
        key=lambda index: (
            not policies.get(pages[index].url, FetchPolicy()).retry_js_on_low_text,
            index,
        )
    )
    selected = set(candidates[:budget])
    if len(candidates) > budget:
        logger.info(
            "JS retry budget %d reached; skipping %d candidate page(s)",
            budget,
            len(candidates) - budget,
        )

    retried_pages: list[FetchedPage] = []
    for index, page in enumerate(pages):
        if index not in selected:
            retried_pages.append(page)
            continue
        retried_pages.append(
            _fetch_js_fallback_page(
                page,
                timeout=timeout,
                max_tokens_per_page=max_tokens_per_page,
                min_readable_chars=min_readable_chars,
                min_readable_tokens=min_readable_tokens,
                relevance_query=relevance_query,
                loader_factory=js_loader_factory,
                max_link_density=max_link_density,
                min_content_words=min_content_words,
            )
        )
    return retried_pages


def _should_retry_with_js(page: FetchedPage, policy: FetchPolicy) -> bool:
    """Return whether an HTTP result is worth one browser render."""

    if policy.force_js or page.fetch_method != "http":
        return False
    structure = getattr(page, "structure", None)
    return not page.text or (structure is not None and structure.shape == "gateway")


def _fetch_js_fallback_page(
    page: FetchedPage,
    *,
    timeout: float,
    max_tokens_per_page: int,
    min_readable_chars: int,
    min_readable_tokens: int,
    relevance_query: str,
    loader_factory: LoaderFactory,
    max_link_density: float = DEFAULT_MAX_LINK_DENSITY,
    min_content_words: int = DEFAULT_MIN_CONTENT_WORDS,
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
        relevance_query=relevance_query,
        load_error=load_error,
        fetch_method="js_fallback",
        max_link_density=max_link_density,
        min_content_words=min_content_words,
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
    relevance_query: str,
    load_error: str | None,
    fetch_method: str = "http",
    max_link_density: float = DEFAULT_MAX_LINK_DENSITY,
    min_content_words: int = DEFAULT_MIN_CONTENT_WORDS,
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

    # Report how the content was really obtained, not which stage requested it.
    if fetch_method == "http" and all(
        str((document.metadata or {}).get("fetch_method") or "") == "pdf" for document in documents
    ):
        fetch_method = "pdf"

    if fetch_method == "http" and any(
        looks_like_pdf_payload(document.page_content or "") for document in documents
    ):
        # The HTML loader decoded PDF bytes into text. Signal the PDF retry
        # instead of handing megabytes of binary noise to the prompt.
        return FetchedPage(
            url=url,
            title="",
            text="",
            fetch_time_ms=elapsed_ms,
            error="Response body is a PDF; retrying with the PDF loader.",
            fetch_method=PDF_DETECTED_FETCH_METHOD,
        )

    title = _document_title(documents[0])
    publication_date = _document_publication_date(documents)
    extracted_text = normalize_whitespace(
        "\n\n".join(_document_text(document) for document in documents)
    )
    extracted_chars = len(extracted_text)
    extracted_tokens = estimate_tokens(extracted_text)
    # Measure structure even when the page later fails a gate: the shape is what
    # tells the JS retry and the answer node *why* the page is unusable.
    structure = assess_page_structure(
        "\n".join(document.page_content or "" for document in documents),
        extracted_text,
        max_link_density=max_link_density,
        min_content_words=min_content_words,
    )
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
            publication_date=publication_date,
            structure=structure,
        )
    if not is_readable_page(
        extracted_text,
        url=url,
        title=title,
        query=relevance_query,
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
            publication_date=publication_date,
            structure=structure,
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
        publication_date=publication_date,
        structure=structure,
    )


def _document_text(document: Document) -> str:
    """Return readable text, skipping HTML extraction for non-HTML documents.

    PDF text is already plain, and running it through the HTML parser would
    silently drop content that looks like markup (e.g. ``<2026``).
    """

    metadata: dict[str, Any] = document.metadata or {}
    if str(metadata.get("fetch_method") or "") == "pdf":
        return normalize_whitespace(document.page_content or "")
    return extract_text(document.page_content)


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


def _document_publication_date(documents: Sequence[Document]) -> date | None:
    for document in documents:
        metadata: dict[str, Any] = dict(document.metadata or {})
        published_on = extract_publication_date(document.page_content or "", metadata)
        if published_on is not None:
            return published_on
    return None


__all__ = [
    "FetchedPage",
    "DEFAULT_MIN_READABLE_CHARS",
    "DEFAULT_MIN_READABLE_TOKENS",
    "WebPageFetchError",
    "estimate_tokens",
    "extract_text",
    "fetch_pages",
    "is_readable_page",
    "is_readable_text",
    "normalize_whitespace",
    "truncate_to_token_budget",
]
