from __future__ import annotations

from .baidu import BaiduVerificationError, BaiduWebSearch
from .bing import BingVerificationError, BingWebSearch
# REFACTOR: Export lightweight web-search fetch and prompt primitives.
from .content_fetcher import FetchedPage, fetch_pages
from .discovery import discover_urls_from_web, settings_for_discovered_urls
from .duckduckgo import DuckDuckGoWebSearch
from .factory import get_search_provider, normalize_provider_name
# REFACTOR: Export direct-answer prompt assembly for discovered pages.
from .prompt_builder import build_web_search_prompt
from .protocol import WebSearchProvider
# REFACTOR: Export the LangChain live web search tool builder.
from .tool import WebSearchInput, build_web_search_tool, format_web_search_results

__all__ = [
    "BaiduWebSearch",
    "BaiduVerificationError",
    "BingVerificationError",
    "BingWebSearch",
    "DuckDuckGoWebSearch",
    "FetchedPage",
    "WebSearchProvider",
    "WebSearchInput",
    "build_web_search_prompt",
    "build_web_search_tool",
    "discover_urls_from_web",
    "fetch_pages",
    "format_web_search_results",
    "get_search_provider",
    "normalize_provider_name",
    "settings_for_discovered_urls",
]
