from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import replace
from threading import RLock
from time import monotonic
from typing import TYPE_CHECKING

from .baidu import BaiduVerificationError
from .bing import BingVerificationError
from .common import (
    DEFAULT_MIN_USABLE_URL_SCORE,
    SearchResult,
    canonical_url_key,
    normalize_results,
    prefetch_rejection_reason,
    result_quality_score,
    text_relevance_delta,
)
from .factory import (
    configured_api_provider_names,
    get_search_provider,
    is_search_provider_configured,
    normalize_provider_name,
)
from .protocol import RankedSearchResult, WebSearchProvider
from .query_prep import build_search_query

if TYPE_CHECKING:
    from ..config import Settings


logger = logging.getLogger(__name__)

BAIDU_VERIFICATION_COOLDOWN_SECONDS = 300.0
BING_VERIFICATION_COOLDOWN_SECONDS = 300.0
DEFAULT_PROVIDER_ORDER = ("bing", "baidu", "duckduckgo")
CHINESE_PROVIDER_ORDER = ("baidu", "bing", "duckduckgo")
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 8.0
DEFAULT_API_PROVIDER_TIMEOUT_SECONDS = 20.0
DEFAULT_SEARCH_DEADLINE_SECONDS = 30.0
DEFAULT_PROVIDER_FANOUT = 2
PROVIDER_FAILURE_THRESHOLD = 2
PROVIDER_FAILURE_COOLDOWN_SECONDS = 60.0
_provider_cooldowns: dict[str, float] = {}
_provider_failure_counts: dict[str, int] = {}
_provider_state_lock = RLock()

_CJK_CHARACTER_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_CHARACTER_RE = re.compile(r"[A-Za-z]")


def _fallback_provider_names(provider_name: str, question: str = "") -> list[str]:
    if _is_predominantly_chinese(question):
        return list(CHINESE_PROVIDER_ORDER)

    normalized = normalize_provider_name(provider_name)
    # DuckDuckGo is deliberately the last-resort provider. Its package client
    # may fan out to several upstream engines and make a chat turn take minutes.
    names = [] if normalized == "duckduckgo" else [normalized]
    for fallback in (*DEFAULT_PROVIDER_ORDER[:-1], "duckduckgo"):
        if fallback not in names:
            names.append(fallback)
    return names


def _is_predominantly_chinese(question: str) -> bool:
    chinese_count = len(_CJK_CHARACTER_RE.findall(question or ""))
    latin_count = len(_LATIN_CHARACTER_RE.findall(question or ""))
    return chinese_count > 0 and chinese_count >= latin_count


def _provider_cooldown_remaining(provider_name: str) -> float:
    normalized = normalize_provider_name(provider_name)
    with _provider_state_lock:
        until = _provider_cooldowns.get(normalized, 0.0)
        remaining = until - monotonic()
        if remaining <= 0:
            _provider_cooldowns.pop(normalized, None)
            return 0.0
        return remaining


def _cool_down_provider(provider_name: str, seconds: float) -> None:
    if seconds <= 0:
        return
    with _provider_state_lock:
        normalized = normalize_provider_name(provider_name)
        _provider_cooldowns[normalized] = monotonic() + seconds
        _provider_failure_counts.pop(normalized, None)


def _record_provider_success(provider_name: str) -> None:
    with _provider_state_lock:
        _provider_failure_counts.pop(normalize_provider_name(provider_name), None)


def _record_provider_failure(provider_name: str) -> None:
    normalized = normalize_provider_name(provider_name)
    with _provider_state_lock:
        failures = _provider_failure_counts.get(normalized, 0) + 1
        _provider_failure_counts[normalized] = failures
        if failures < PROVIDER_FAILURE_THRESHOLD:
            return
        _cool_down_provider(normalized, PROVIDER_FAILURE_COOLDOWN_SECONDS)
    logger.warning(
        "Web search provider %s circuit opened for %.0fs after %d consecutive failures",
        normalized,
        PROVIDER_FAILURE_COOLDOWN_SECONDS,
        failures,
    )


def discover_urls_from_web(
    question: str,
    settings: Settings,
    provider: WebSearchProvider | None = None,
) -> list[str]:
    """Search the web for pages that can be used as RAG sources."""

    return [result.url for result in discover_search_results_from_web(question, settings, provider)]


def discover_search_results_from_web(
    question: str,
    settings: Settings,
    provider: WebSearchProvider | None = None,
) -> list[RankedSearchResult]:
    """Search the web while retaining provider and semantic ranking signals."""

    if not settings.web_search_enabled:
        return []

    search_query = build_search_query(question, settings)
    logger.info(
        "Web search: original=%r -> final_query=%r",
        question,
        search_query,
    )
    deadline_at = monotonic() + _positive_seconds(
        getattr(settings, "web_search_deadline_seconds", DEFAULT_SEARCH_DEADLINE_SECONDS),
        DEFAULT_SEARCH_DEADLINE_SECONDS,
    )

    if provider is not None:
        return _discover_with_deadline(
            question,
            search_query,
            settings,
            [provider],
            deadline_at=deadline_at,
        )

    provider_names = _provider_names_for_question(settings, question)
    fanout = _positive_int(
        getattr(settings, "web_search_provider_fanout", DEFAULT_PROVIDER_FANOUT),
        DEFAULT_PROVIDER_FANOUT,
    )
    for start in range(0, len(provider_names), fanout):
        stage_names = provider_names[start : start + fanout]
        selected = _discover_with_deadline(
            question,
            search_query,
            settings,
            _available_providers(stage_names, settings),
            deadline_at=deadline_at,
        )
        if selected:
            return selected

    return []


def _provider_names_for_question(settings: Settings, question: str) -> list[str]:
    explicit = [
        normalize_provider_name(name)
        for name in getattr(settings, "web_search_providers", []) or []
    ]
    api_names = configured_api_provider_names(settings)
    fallback_names = _fallback_provider_names(settings.web_search_provider, question)
    if _is_predominantly_chinese(question):
        # Supported APIs avoid CAPTCHA and provide richer Mandarin snippets.
        # Explicit priorities still win when operators need a fixed order.
        return _dedupe_provider_names([*explicit, *api_names, *fallback_names])
    return _dedupe_provider_names([*explicit, *fallback_names[:1], *api_names, *fallback_names[1:]])


def _dedupe_provider_names(names: list[str]) -> list[str]:
    return list(dict.fromkeys(normalize_provider_name(name) for name in names if name))


def _discover_with_provider(
    question: str,
    settings: Settings,
    search_provider: WebSearchProvider,
) -> list[str]:
    return [
        result.url
        for result in _discover_results_with_provider(question, settings, search_provider)
    ]


def _available_providers(
    provider_names: list[str],
    settings: Settings,
) -> list[WebSearchProvider]:
    providers: list[WebSearchProvider] = []
    for provider_name in provider_names:
        remaining = _provider_cooldown_remaining(provider_name)
        if remaining:
            logger.info(
                "Skipping %s web search for %.0fs while its provider circuit is open",
                provider_name,
                remaining,
            )
            continue
        if not is_search_provider_configured(provider_name, settings):
            logger.info(
                "Skipping unconfigured %s web search provider (API key not set)",
                provider_name,
            )
            continue
        try:
            providers.append(get_search_provider(provider_name, settings))
        except ValueError as exc:
            logger.warning("Skipping web search provider %s: %s", provider_name, exc)
    return providers


def _discover_with_deadline(
    question: str,
    search_query: str,
    settings: Settings,
    providers: list[WebSearchProvider],
    *,
    deadline_at: float,
) -> list[RankedSearchResult]:
    """Run one provider stage concurrently without waiting past its budget."""

    remaining = deadline_at - monotonic()
    if not providers or remaining <= 0:
        if remaining <= 0:
            logger.warning("Web search deadline reached before another provider could run")
        return []

    provider_budget = max(_provider_timeout_seconds(provider, settings) for provider in providers)
    timeout = min(remaining, provider_budget)
    if len(providers) == 1:
        search_provider = providers[0]
        started_at = monotonic()
        single_results = _discover_results_with_provider(
            question,
            settings,
            search_provider,
            search_query,
            0,
        )
        elapsed = monotonic() - started_at
        if elapsed > timeout:
            logger.warning(
                "Web search provider %s completed in %.1fs after its %.1fs budget",
                search_provider.provider_name,
                elapsed,
                timeout,
            )
        return _merge_ranked_results(
            single_results,
            provider_priority=[search_provider.provider_name],
            top_k=getattr(settings, "web_search_top_k", 0) or 0,
        )

    executor = ThreadPoolExecutor(
        max_workers=len(providers),
        thread_name_prefix="web-search-provider",
    )
    futures: dict[Future[list[RankedSearchResult]], WebSearchProvider] = {
        executor.submit(
            _discover_results_with_provider,
            question,
            settings,
            provider,
            search_query,
            0,
        ): provider
        for provider in providers
    }
    try:
        done, not_done = wait(futures, timeout=timeout)
        discovered: list[RankedSearchResult] = []
        for future in done:
            search_provider = futures[future]
            try:
                discovered.extend(future.result())
            except Exception as exc:
                logger.warning(
                    "Web search provider %s failed outside its adapter: %s",
                    search_provider.provider_name,
                    exc,
                )
        for future in not_done:
            search_provider = futures[future]
            future.cancel()
            _record_provider_failure(search_provider.provider_name)
            logger.warning(
                "Web search provider %s exceeded its %.1fs stage deadline",
                search_provider.provider_name,
                timeout,
            )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return _merge_ranked_results(
        discovered,
        provider_priority=[provider.provider_name for provider in providers],
        top_k=getattr(settings, "web_search_top_k", 0) or 0,
    )


def _merge_ranked_results(
    results: list[RankedSearchResult],
    *,
    provider_priority: list[str],
    top_k: int,
) -> list[RankedSearchResult]:
    """Deduplicate a provider blend and apply one final cross-provider ranking."""

    priorities = {
        normalize_provider_name(name): index for index, name in enumerate(provider_priority)
    }
    default_priority = len(priorities)
    best_by_url: dict[tuple[str, str], RankedSearchResult] = {}
    for result in results:
        key = canonical_url_key(result.url)
        current = best_by_url.get(key)
        if current is None or _ranked_result_key(
            result,
            priorities,
            default_priority,
        ) < _ranked_result_key(current, priorities, default_priority):
            best_by_url[key] = result

    selected = sorted(
        best_by_url.values(),
        key=lambda result: _ranked_result_key(
            result,
            priorities,
            default_priority,
        ),
    )
    if top_k > 0:
        return selected[:top_k]
    return selected


def _ranked_result_key(
    result: RankedSearchResult,
    priorities: dict[str, int],
    default_priority: int,
) -> tuple[int, int, int, int, str]:
    return (
        -result.quality_score,
        -result.relevance_score,
        priorities.get(normalize_provider_name(result.provider), default_priority),
        result.provider_rank,
        result.url,
    )


def _positive_seconds(value: int | float | str | None, default: float) -> float:
    if value is None:
        return default
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return default
    return seconds if seconds > 0 else default


def _provider_timeout_seconds(provider: WebSearchProvider, settings: Settings) -> float:
    provider_name = normalize_provider_name(provider.provider_name)
    if provider_name in {"serper", "brave", "tavily", "bing_api"}:
        return _positive_seconds(
            getattr(
                settings,
                "web_search_api_timeout_seconds",
                DEFAULT_API_PROVIDER_TIMEOUT_SECONDS,
            ),
            DEFAULT_API_PROVIDER_TIMEOUT_SECONDS,
        )
    return _positive_seconds(
        getattr(
            settings,
            "web_search_provider_timeout_seconds",
            DEFAULT_PROVIDER_TIMEOUT_SECONDS,
        ),
        DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    )


def _positive_int(value: int | str | None, default: int) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _discover_results_with_provider(
    question: str,
    settings: Settings,
    search_provider: WebSearchProvider,
    search_query: str | None = None,
    top_k: int | None = None,
) -> list[RankedSearchResult]:
    search_query = search_query or build_search_query(question, settings)
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
        _record_provider_failure(search_provider.provider_name)
        logger.warning("Web search provider %s failed: %s", search_provider.provider_name, exc)
        return []

    _record_provider_success(search_provider.provider_name)

    selected = _rank_usable_results(
        results,
        provider_name=search_provider.provider_name,
        top_k=(getattr(settings, "web_search_top_k", 0) or 0) if top_k is None else top_k,
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
        rejection_reason = prefetch_rejection_reason(result, query)
        if rejection_reason is not None:
            logger.info(
                "Rejected search result before fetch: provider=%s reason=%s url=%s",
                provider_name,
                rejection_reason,
                result.url,
            )
            continue
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
