"""Backward-compatible web search imports.

The provider implementations live in :mod:`src.web_search`; this module keeps
the original ``src.core.web_search`` import path available for CLI/API callers
and older tests.
"""

from __future__ import annotations

from .._compat import warn_deprecated_import
from .config import Settings
from ..web_search import (
    BaiduWebSearch,
    DuckDuckGoWebSearch,
    WebSearchProvider,
    discover_urls_from_web,
    get_search_provider,
    settings_for_discovered_urls,
)
from ..web_search.baidu import (
    candidate_baidu_hrefs as _candidate_baidu_hrefs,
    is_baidu_result_redirect as _is_baidu_result_redirect,
    is_baidu_url as _is_baidu_url,
)
from ..web_search.common import (
    BAIDU_BASE_URL,
    DUCKDUCKGO_BASE_URL,
    NOISE_HOSTNAMES,
    SEARCH_USER_AGENT,
    is_noise_url as _is_noise_url,
    normalize_urls as _normalize_urls,
    search_request as _search_request,
    select_top_urls,
    urlopen_context,
)
from ..web_search.duckduckgo import (
    load_ddgs as _load_ddgs,
    unwrap_duckduckgo_redirect as _unwrap_duckduckgo_redirect,
)

warn_deprecated_import("src.core.web_search", "src.web_search")


def _urlopen_context(settings: Settings):
    return urlopen_context(settings.web_search_verify_ssl)


def _ddgs_search(question: str, settings: Settings) -> list[str]:
    return DuckDuckGoWebSearch(
        region=settings.web_search_region,
        timelimit=settings.web_search_timelimit,
        verify_ssl=settings.web_search_verify_ssl,
    ).search_ddgs(question, settings.web_search_max_results)


def _duckduckgo_html_search(question: str, settings: Settings) -> list[str]:
    return DuckDuckGoWebSearch(
        region=settings.web_search_region,
        timelimit=settings.web_search_timelimit,
        verify_ssl=settings.web_search_verify_ssl,
    ).search_html(question, settings.web_search_max_results)


def _discover_urls_from_duckduckgo(question: str, settings: Settings) -> list[str]:
    return DuckDuckGoWebSearch(
        region=settings.web_search_region,
        timelimit=settings.web_search_timelimit,
        verify_ssl=settings.web_search_verify_ssl,
    ).search(question, settings.web_search_max_results)


def _resolve_baidu_redirect(url: str, settings: Settings) -> str:
    return BaiduWebSearch(verify_ssl=settings.web_search_verify_ssl).resolve_redirect(url)


def _discover_urls_from_baidu(question: str, settings: Settings) -> list[str]:
    return BaiduWebSearch(verify_ssl=settings.web_search_verify_ssl).search(
        question,
        settings.web_search_max_results,
    )


def _select_top_urls(urls: list[str], settings: Settings) -> list[str]:
    return select_top_urls(urls, getattr(settings, "web_search_top_k", 0) or 0)


__all__ = [
    "BAIDU_BASE_URL",
    "DUCKDUCKGO_BASE_URL",
    "NOISE_HOSTNAMES",
    "SEARCH_USER_AGENT",
    "BaiduWebSearch",
    "DuckDuckGoWebSearch",
    "WebSearchProvider",
    "_candidate_baidu_hrefs",
    "_ddgs_search",
    "_discover_urls_from_baidu",
    "_discover_urls_from_duckduckgo",
    "_duckduckgo_html_search",
    "_is_baidu_result_redirect",
    "_is_baidu_url",
    "_is_noise_url",
    "_load_ddgs",
    "_normalize_urls",
    "_resolve_baidu_redirect",
    "_search_request",
    "_select_top_urls",
    "_unwrap_duckduckgo_redirect",
    "_urlopen_context",
    "discover_urls_from_web",
    "get_search_provider",
    "settings_for_discovered_urls",
]
