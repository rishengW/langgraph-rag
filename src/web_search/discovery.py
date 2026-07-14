from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import replace
from time import monotonic
from typing import TYPE_CHECKING

from .baidu import BaiduVerificationError
from .bing import BingVerificationError
from .common import (
    DEFAULT_MIN_USABLE_URL_SCORE,
    SearchResult,
    canonical_url_key,
    normalize_results,
    result_quality_score,
    text_relevance_delta,
)
from .factory import get_search_provider, normalize_provider_name
from .protocol import RankedSearchResult, WebSearchProvider
from .query_prep import build_search_query

if TYPE_CHECKING:
    from ..config import Settings


logger = logging.getLogger(__name__)

BAIDU_VERIFICATION_COOLDOWN_SECONDS = 300.0
BING_VERIFICATION_COOLDOWN_SECONDS = 300.0
DEFAULT_PROVIDER_ORDER = ("bing", "baidu", "duckduckgo")
CHINESE_PROVIDER_ORDER = ("baidu", "bing", "duckduckgo")
_provider_cooldowns: dict[str, float] = {}

_CJK_CHARACTER_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_CHARACTER_RE = re.compile(r"[A-Za-z]")


def _fallback_provider_names(provider_name: str, question: str = "") -> list[str]:
    if _is_predominantly_chinese(question):
        return list(CHINESE_PROVIDER_ORDER)

    normalized = normalize_provider_name(provider_name)
    names = [normalized]
    for fallback in DEFAULT_PROVIDER_ORDER:
        if fallback not in names:
            names.append(fallback)
    return names


def _is_predominantly_chinese(question: str) -> bool:
    chinese_count = len(_CJK_CHARACTER_RE.findall(question or ""))
    latin_count = len(_LATIN_CHARACTER_RE.findall(question or ""))
    return chinese_count > 0 and chinese_count >= latin_count


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

    return [
        result.url
        for result in discover_search_results_from_web(question, settings, provider)
    ]


def discover_search_results_from_web(
    question: str,
    settings: Settings,
    provider: WebSearchProvider | None = None,
) -> list[RankedSearchResult]:
    """Search the web while retaining provider and semantic ranking signals."""

    if not settings.web_search_enabled:
        return []

    if provider is not None:
        return _discover_results_with_provider(question, settings, provider)

    for provider_name in _fallback_provider_names(settings.web_search_provider, question):
        remaining = _provider_cooldown_remaining(provider_name)
        if remaining:
            logger.info(
                "Skipping %s web search for %.0fs after a recent verification/captcha failure",
                provider_name,
                remaining,
            )
            continue

        search_provider = get_search_provider(provider_name, settings)
        selected = _discover_results_with_provider(question, settings, search_provider)
        if selected:
            return selected

    return []


def _discover_with_provider(
    question: str,
    settings: Settings,
    search_provider: WebSearchProvider,
) -> list[str]:
    return [
        result.url
        for result in _discover_results_with_provider(question, settings, search_provider)
    ]


def _discover_results_with_provider(
    question: str,
    settings: Settings,
    search_provider: WebSearchProvider,
) -> list[RankedSearchResult]:
    search_query = build_search_query(question, settings)
    logger.info(
        "Web search: original=%r → final_query=%r",
        question,
        search_query,
    )
    try:
        results = normalize_results(
            _provider_search_results(search_provider, search_query, settings.web_search_max_results)
        )
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

    selected = _rank_usable_results(
        results,
        provider_name=search_provider.provider_name,
        top_k=getattr(settings, "web_search_top_k", 0) or 0,
        min_score=getattr(
            settings,
            "web_search_min_url_score",
            DEFAULT_MIN_USABLE_URL_SCORE,
        ),
        query=search_query,
    )
    logger.info(
        "Discovered %s URL(s) from %s; keeping %s usable URL(s) after quality filtering",
        len(results),
        search_provider.provider_name,
        len(selected),
    )
    # REFACTOR: Treat provider results that fail URL quality gates as a provider miss.
    if not selected and results:
        logger.info(
            "Web search provider %s returned no usable URLs after filtering; "
            "falling back when another provider is available",
            search_provider.provider_name,
        )
    return selected


def _rank_usable_results(
    results: list[SearchResult],
    *,
    provider_name: str,
    top_k: int,
    min_score: int,
    query: str,
) -> list[RankedSearchResult]:
    """Filter provider results without discarding their relevance metadata."""

    best_by_url: dict[tuple[str, str], RankedSearchResult] = {}
    for provider_rank, result in enumerate(results):
        quality_score = result_quality_score(result, query=query)
        if quality_score < min_score:
            continue
        ranked = RankedSearchResult(
            url=result.url,
            title=result.title,
            snippet=result.snippet,
            provider=normalize_provider_name(provider_name),
            provider_rank=provider_rank,
            relevance_score=text_relevance_delta(result.relevance_text, query),
            quality_score=quality_score,
        )
        key = canonical_url_key(result.url)
        current = best_by_url.get(key)
        if current is None or (
            ranked.quality_score,
            -ranked.provider_rank,
        ) > (
            current.quality_score,
            -current.provider_rank,
        ):
            best_by_url[key] = ranked

    ranked_results = sorted(
        best_by_url.values(),
        key=lambda item: (-item.quality_score, item.provider_rank, item.url),
    )
    if top_k > 0:
        return ranked_results[:top_k]
    return ranked_results


def _provider_search_results(
    search_provider: WebSearchProvider,
    search_query: str,
    max_results: int,
) -> list[SearchResult]:
    """Prefer snippet-aware results; fall back to bare URLs for older providers."""

    search_results = getattr(search_provider, "search_results", None)
    if callable(search_results):
        return list(search_results(search_query, max_results))
    return [SearchResult(url=url) for url in search_provider.search(search_query, max_results)]


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
