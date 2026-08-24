"""Deprecated compatibility facade for the live web-search LangChain tool."""

from __future__ import annotations

import warnings

warnings.warn(
    "src.web_search.tool is deprecated; import the web-search tool from "
    "src.tools.web_search instead",
    DeprecationWarning,
    stacklevel=2,
)

from ..tools.web_search import (  # noqa: E402
    WebSearchDiscovery,
    WebSearchInput,
    build_web_search_tool,
    format_web_search_results,
)

__all__ = [
    "WebSearchDiscovery",
    "WebSearchInput",
    "build_web_search_tool",
    "format_web_search_results",
]