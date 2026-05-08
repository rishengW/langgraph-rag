"""Web search helpers for discovering RAG source URLs."""

from __future__ import annotations

import logging
import hashlib
from dataclasses import replace
from typing import Iterable

from .config import Settings


logger = logging.getLogger(__name__)


def _load_ddgs():
    """Import the installed DuckDuckGo search client.

    The package was renamed from `duckduckgo-search` to `ddgs` in newer
    releases. Supporting both keeps the app usable across environments.
    """

    try:
        from ddgs import DDGS

        return DDGS
    except ImportError:
        from duckduckgo_search import DDGS

        return DDGS


def _normalize_urls(urls: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []

    for url in urls:
        clean_url = (url or "").strip()
        if not clean_url or clean_url in seen:
            continue
        if not clean_url.startswith(("http://", "https://")):
            continue

        seen.add(clean_url)
        normalized.append(clean_url)

    return normalized


def discover_urls_from_web(question: str, settings: Settings) -> list[str]:
    """Search the web for pages that can be used as RAG sources."""

    if not settings.web_search_enabled:
        return []

    DDGS = _load_ddgs()

    search_kwargs = {
        "region": settings.web_search_region,
        "max_results": settings.web_search_max_results,
    }
    if settings.web_search_timelimit:
        search_kwargs["timelimit"] = settings.web_search_timelimit

    with DDGS(verify=settings.web_search_verify_ssl) as ddgs:
        results = list(ddgs.text(question, **search_kwargs))

    urls = _normalize_urls(result.get("href", "") for result in results)
    logger.info("Discovered %s URL(s) from web search", len(urls))
    return urls


def settings_for_discovered_urls(settings: Settings, urls: list[str]) -> Settings:
    """Create isolated settings for a search-derived source set."""

    fingerprint = hashlib.sha256("\n".join(urls).encode("utf-8")).hexdigest()[:12]
    return replace(
        settings,
        source_urls=urls,
        chroma_dir=settings.chroma_dir / "web-search" / fingerprint,
        collection_name=f"{settings.collection_name}-web-{fingerprint}",
    )
