from __future__ import annotations

from typing import Any

from .api_providers import (
    BingApiWebSearch,
    BraveWebSearch,
    SerperWebSearch,
    TavilyWebSearch,
)
from .baidu import BaiduVerificationError, BaiduWebSearch
from .bing import BingVerificationError, BingWebSearch
from .common import SearchResult
from .content_fetcher import FetchedPage, fetch_pages, is_readable_text
from .discovery import (
    discover_search_results_from_web,
    discover_urls_from_web,
    settings_for_discovered_urls,
)
from .duckduckgo import DuckDuckGoWebSearch
from .factory import (
    configured_api_provider_names,
    get_search_provider,
    is_search_provider_configured,
    normalize_provider_name,
)
from .fetch_policy import FetchPolicy, resolve_fetch_policy
from .prompt_builder import build_web_search_prompt
from .protocol import RankedSearchResult, WebSearchProvider
from .query_prep import build_search_query, prepare_search_query, rewrite_search_query_llm

_TOOL_EXPORTS = {
    "WebSearchInput",
    "build_web_search_tool",
    "format_web_search_results",
}


def __getattr__(name: str) -> Any:
    """Lazily expose legacy tool symbols from their canonical package."""

    if name in _TOOL_EXPORTS:
        from ..tools import live_web_search as tool_module

        return getattr(tool_module, name)
    raise AttributeError(name)

__all__ = [
    "BaiduWebSearch",
    "BaiduVerificationError",
    "BingVerificationError",
    "BingWebSearch",
    "BingApiWebSearch",
    "BraveWebSearch",
    "DuckDuckGoWebSearch",
    "FetchedPage",
    "FetchPolicy",
    "SearchResult",
    "SerperWebSearch",
    "TavilyWebSearch",
    "RankedSearchResult",
    "WebSearchProvider",
    "WebSearchInput",
    "build_web_search_prompt",
    "build_web_search_tool",
    "discover_urls_from_web",
    "discover_search_results_from_web",
    "fetch_pages",
    "format_web_search_results",
    "get_search_provider",
    "configured_api_provider_names",
    "is_search_provider_configured",
    "is_readable_text",
    "normalize_provider_name",
    "build_search_query",
    "prepare_search_query",
    "resolve_fetch_policy",
    "rewrite_search_query_llm",
    "settings_for_discovered_urls",
]
