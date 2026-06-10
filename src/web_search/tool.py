# REFACTOR: LangChain tool wrapper for live web search providers.
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from .discovery import discover_urls_from_web
from .protocol import WebSearchProvider

if TYPE_CHECKING:
    from ..config import Settings


WebSearchDiscovery = Callable[[str, "Settings", WebSearchProvider | None], list[str]]


class WebSearchInput(BaseModel):
    """Input schema for the live web search tool."""

    query: str = Field(..., description="Search query to send to the live web provider.")
    max_results: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Optional provider result limit for this search.",
    )


def build_web_search_tool(
    settings: "Settings",
    *,
    provider: WebSearchProvider | None = None,
    discovery: WebSearchDiscovery = discover_urls_from_web,
) -> BaseTool:
    """Create a LangChain tool that returns provider-ranked live web URLs.

    Args:
        settings: Runtime settings controlling provider, limits, and enablement.
        provider: Optional provider override for tests or dependency injection.
        discovery: URL discovery callable, injectable for focused tests.

    Returns:
        A LangChain structured tool suitable for graph ToolNode execution.
    """

    def _run_web_search(query: str, max_results: int | None = None) -> str:
        search_settings = _settings_for_limit(settings, max_results)
        urls = discovery(query, search_settings, provider)
        return format_web_search_results(query, urls)

    return StructuredTool.from_function(
        func=_run_web_search,
        name="live_web_search",
        description=(
            "Search the live web for current or external information. "
            "IMPORTANT: Formulate the query as search-engine keywords, NOT a "
            "natural-language question. Extract core concepts and named entities, "
            "drop filler words (what, is, the, of, does, etc.), and include the "
            "current year when the question is about recent events or releases. "
            "For example, instead of 'what is the latest model of deepseek' use "
            "'DeepSeek latest model 2026' or 'DeepSeek new model release 2026'. "
            "Returns provider-ranked result URLs."
        ),
        args_schema=WebSearchInput,
    )


def format_web_search_results(query: str, urls: list[str]) -> str:
    """Format live web search URLs for tool-call output."""

    if not urls:
        return f"No live web search results found for: {query}"

    lines = [f"Live web search results for: {query}"]
    lines.extend(f"{index}. {url}" for index, url in enumerate(urls, start=1))
    return "\n".join(lines)


def _settings_for_limit(settings: "Settings", max_results: int | None) -> "Settings":
    if max_results is None:
        return settings
    return replace(settings, web_search_max_results=max_results)


__all__ = [
    "WebSearchDiscovery",
    "WebSearchInput",
    "build_web_search_tool",
    "format_web_search_results",
]
