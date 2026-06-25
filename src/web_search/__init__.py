from __future__ import annotations

from .baidu import BaiduVerificationError, BaiduWebSearch
from .bing import BingVerificationError, BingWebSearch

# REFACTOR: Export the search result value object for snippet-aware ranking.
from .common import SearchResult

# REFACTOR: Export lightweight web-search fetch and prompt primitives.
from .content_fetcher import FetchedPage, fetch_pages, is_readable_text
from .discovery import discover_urls_from_web, settings_for_discovered_urls
from .duckduckgo import DuckDuckGoWebSearch
from .factory import get_search_provider, normalize_provider_name
from .fetch_policy import FetchPolicy, resolve_fetch_policy

# REFACTOR: Export direct-answer prompt assembly for discovered pages.
from .prompt_builder import build_web_search_prompt
from .protocol import WebSearchProvider

# REFACTOR: Export the LangChain live web search tool builder.
from .query_prep import build_search_query, prepare_search_query, rewrite_search_query_llm
from .tool import WebSearchInput, build_web_search_tool, format_web_search_results

__all__ = [
    "BaiduWebSearch",
    "BaiduVerificationError",
    "BingVerificationError",
    "BingWebSearch",
    "DuckDuckGoWebSearch",
    "FetchedPage",
    "FetchPolicy",
    "SearchResult",
    "WebSearchProvider",
    "WebSearchInput",
    "build_web_search_prompt",
    "build_web_search_tool",
    "discover_urls_from_web",
    "fetch_pages",
    "format_web_search_results",
    "get_search_provider",
    "is_readable_text",
    "normalize_provider_name",
    "build_search_query",
    "prepare_search_query",
    "resolve_fetch_policy",
    "rewrite_search_query_llm",
    "settings_for_discovered_urls",
]
