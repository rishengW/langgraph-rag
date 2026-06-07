from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from .common import normalize_urls, select_top_urls
from .factory import get_search_provider
from .protocol import WebSearchProvider


if TYPE_CHECKING:
    from ..config import Settings


logger = logging.getLogger(__name__)


def discover_urls_from_web(
    question: str,
    settings: Settings,
    provider: WebSearchProvider | None = None,
) -> list[str]:
    """Search the web for pages that can be used as RAG sources."""

    if not settings.web_search_enabled:
        return []

    search_provider = provider or get_search_provider(settings.web_search_provider, settings)
    urls = normalize_urls(search_provider.search(question, settings.web_search_max_results))
    selected = select_top_urls(
        urls,
        getattr(settings, "web_search_top_k", 0) or 0,
    )

    logger.info(
        "Discovered %s URL(s) from %s; keeping top %s after filtering",
        len(urls),
        search_provider.provider_name,
        len(selected),
    )
    return selected


def settings_for_discovered_urls(settings: Settings, urls: list[str]) -> Settings:
    """Create isolated settings for a search-derived source set."""

    fingerprint = hashlib.sha256("\n".join(urls).encode("utf-8")).hexdigest()[:12]
    return replace(
        settings,
        source_urls=urls,
        chroma_dir=settings.chroma_dir / "web-search" / fingerprint,
        collection_name=f"{settings.collection_name}-web-{fingerprint}",
    )
