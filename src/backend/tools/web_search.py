"""LangChain tool adapter for live web-search domain providers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ..web_search.discovery import discover_search_results_from_web
from ..web_search.protocol import RankedSearchResult, WebSearchProvider

if TYPE_CHECKING:
    from src.config import Settings


WebSearchDiscovery = Callable[
    [str, "Settings", WebSearchProvider | None],
    Sequence[str | RankedSearchResult],
]


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
    settings: Settings,
    *,
    provider: WebSearchProvider | None = None,
    discovery: WebSearchDiscovery = discover_search_results_from_web,
) -> BaseTool:
    """Create a LangChain tool that returns provider-ranked live web URLs."""

    def _run_web_search(
        query: str, max_results: int | None = None
    ) -> tuple[str, dict[str, object]]:
        search_settings = _settings_for_limit(settings, max_results)
        discovered = list(discovery(query, search_settings, provider))
        results = _coerce_ranked_results(discovered)
        urls = [result.url for result in results]
        artifact: dict[str, object] = {
            "query": query,
            "results": [result.as_dict() for result in results],
        }
        return format_web_search_results(query, urls), artifact

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
        response_format="content_and_artifact",
    )


def format_web_search_results(query: str, urls: list[str]) -> str:
    """Format live web-search URLs for tool-call output."""

    if not urls:
        return f"No live web search results found for: {query}"
    lines = [f"Live web search results for: {query}"]
    lines.extend(f"{index}. {url}" for index, url in enumerate(urls, start=1))
    return "\n".join(lines)


def _settings_for_limit(settings: Settings, max_results: int | None) -> Settings:
    if max_results is None:
        return settings
    return replace(settings, web_search_max_results=max_results)


def _coerce_ranked_results(
    results: Sequence[str | RankedSearchResult],
) -> list[RankedSearchResult]:
    ranked: list[RankedSearchResult] = []
    for provider_rank, result in enumerate(results):
        if isinstance(result, RankedSearchResult):
            ranked.append(result)
            continue
        url = str(result).strip()
        if url:
            ranked.append(RankedSearchResult(url=url, provider_rank=provider_rank))
    return ranked


__all__ = [
    "WebSearchDiscovery",
    "WebSearchInput",
    "build_web_search_tool",
    "format_web_search_results",
]