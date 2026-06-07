from __future__ import annotations

from .baidu import BaiduWebSearch
from .discovery import discover_urls_from_web, settings_for_discovered_urls
from .duckduckgo import DuckDuckGoWebSearch
from .factory import get_search_provider, normalize_provider_name
from .protocol import WebSearchProvider

__all__ = [
    "BaiduWebSearch",
    "DuckDuckGoWebSearch",
    "WebSearchProvider",
    "discover_urls_from_web",
    "get_search_provider",
    "normalize_provider_name",
    "settings_for_discovered_urls",
]
