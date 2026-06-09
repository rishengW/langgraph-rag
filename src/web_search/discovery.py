from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from time import monotonic
from typing import TYPE_CHECKING

from .baidu import BaiduVerificationError
from .bing import BingVerificationError
from .common import normalize_urls, select_top_urls
from .factory import get_search_provider, normalize_provider_name
from .protocol import WebSearchProvider


if TYPE_CHECKING:
    from ..config import Settings


logger = logging.getLogger(__name__)

BAIDU_VERIFICATION_COOLDOWN_SECONDS = 300.0
BING_VERIFICATION_COOLDOWN_SECONDS = 300.0
DEFAULT_PROVIDER_ORDER = ("bing", "baidu", "duckduckgo")
_provider_cooldowns: dict[str, float] = {}


def _fallback_provider_names(provider_name: str) -> list[str]:
    normalized = normalize_provider_name(provider_name)
    names = [normalized]
    for fallback in DEFAULT_PROVIDER_ORDER:
        if fallback not in names:
            names.append(fallback)
    return names


def _provider_cooldown_remaining(provider_name: str) -> float:
    normalized = normalize_provider_name(provider_name)
    until = _provider_cooldowns.get(normalized, 0.0)
    remaining = until - monotonic()
    if remaining <= 0:
        _provider_cooldowns.pop(normalized, None)
        return 0.0
    return remaining


def _cool_down_provider(provider_name: str, seconds: float) -> None:
    if seconds <= 0:
        return
    _provider_cooldowns[normalize_provider_name(provider_name)] = monotonic() + seconds


def discover_urls_from_web(
    question: str,
    settings: Settings,
    provider: WebSearchProvider | None = None,
) -> list[str]:
    """Search the web for pages that can be used as RAG sources."""

    if not settings.web_search_enabled:
        return []

    if provider is not None:
        return _discover_with_provider(question, settings, provider)

    for provider_name in _fallback_provider_names(settings.web_search_provider):
        remaining = _provider_cooldown_remaining(provider_name)
        if remaining:
            logger.info(
                "Skipping %s web search for %.0fs after a recent verification/captcha failure",
                provider_name,
                remaining,
            )
            continue

        search_provider = get_search_provider(provider_name, settings)
        selected = _discover_with_provider(question, settings, search_provider)
        if selected:
            return selected

    return []


def _discover_with_provider(
    question: str,
    settings: Settings,
    search_provider: WebSearchProvider,
) -> list[str]:
    try:
        urls = normalize_urls(search_provider.search(question, settings.web_search_max_results))
    except (BaiduVerificationError, BingVerificationError) as exc:
        cooldown_seconds = _verification_cooldown_seconds(search_provider.provider_name)
        _cool_down_provider(search_provider.provider_name, cooldown_seconds)
        logger.warning(
            "Web search provider %s was blocked by verification/captcha; "
            "cooling down for %.0fs: %s",
            search_provider.provider_name,
            cooldown_seconds,
            exc,
        )
        return []
    except Exception as exc:
        logger.warning("Web search provider %s failed: %s", search_provider.provider_name, exc)
        return []

    selected = select_top_urls(urls, getattr(settings, "web_search_top_k", 0) or 0)
    logger.info(
        "Discovered %s URL(s) from %s; keeping %s usable URL(s) after quality filtering",
        len(urls),
        search_provider.provider_name,
        len(selected),
    )
    # REFACTOR: Treat provider results that fail URL quality gates as a provider miss.
    if not selected and urls:
        logger.info(
            "Web search provider %s returned no usable URLs after filtering; "
            "falling back when another provider is available",
            search_provider.provider_name,
        )
    return selected


def _verification_cooldown_seconds(provider_name: str) -> float:
    if normalize_provider_name(provider_name) == "bing":
        return BING_VERIFICATION_COOLDOWN_SECONDS
    return BAIDU_VERIFICATION_COOLDOWN_SECONDS


def settings_for_discovered_urls(settings: Settings, urls: list[str]) -> Settings:
    """Create isolated settings for a search-derived source set."""

    fingerprint = hashlib.sha256("\n".join(urls).encode("utf-8")).hexdigest()[:12]
    return replace(
        settings,
        source_urls=urls,
        chroma_dir=settings.chroma_dir / "web-search" / fingerprint,
        collection_name=f"{settings.collection_name}-web-{fingerprint}",
    )
