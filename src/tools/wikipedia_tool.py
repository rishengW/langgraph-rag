from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import quote

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._http import JsonRequester, request_json

if TYPE_CHECKING:
    from ..config import Settings

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

class WikipediaInput(BaseModel):
    """Input schema for Wikipedia search."""

    query: str = Field(..., min_length=1, description="Wikipedia search query.")
    max_results: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Maximum matching articles to summarize.",
    )


def build_wikipedia_tool(
    settings: Settings,
    *,
    requester: JsonRequester | None = None,
) -> BaseTool:
    """Create a Wikipedia search and summary tool."""

    def _run_wikipedia(query: str, max_results: int = 1) -> str:
        return search_wikipedia(
            query,
            max_results=max_results,
            max_summary_chars=settings.wikipedia_max_summary_chars,
            user_agent=settings.wikipedia_user_agent,
            requester=requester,
        )

    return StructuredTool.from_function(
        func=_run_wikipedia,
        name="search_wikipedia",
        description=(
            "Search Wikipedia and return concise article summaries with URLs. "
            "Use for encyclopedic background, people, places, organizations, "
            "concepts, and historical facts."
        ),
        args_schema=WikipediaInput,
    )


def search_wikipedia(
    query: str,
    *,
    max_results: int = 1,
    max_summary_chars: int = 1500,
    user_agent: str = "langgraph-rag/1.0 (contact: configure WIKIPEDIA_USER_AGENT)",
    requester: JsonRequester | None = None,
) -> str:
    term = query.strip()
    if not term:
        return "Wikipedia search requires a non-empty query."

    try:
        search_payload = request_json(
            WIKIPEDIA_API_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": term,
                "srlimit": max_results,
                "format": "json",
                "utf8": 1,
            },
            requester=requester,
            headers=_wikipedia_headers(user_agent),
        )
    except Exception as exc:
        return f"Wikipedia search failed for {term!r}: {exc}"

    matches = (
        search_payload.get("query", {}).get("search", [])
        if isinstance(search_payload.get("query"), dict)
        else []
    )
    if not matches:
        return f"No Wikipedia results found for: {term}"

    page_ids = [
        str(match.get("pageid"))
        for match in matches[:max_results]
        if match.get("pageid") is not None
    ]
    if not page_ids:
        return f"No Wikipedia results found for: {term}"

    try:
        extract_payload = request_json(
            WIKIPEDIA_API_URL,
            params={
                "action": "query",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "redirects": 1,
                "pageids": "|".join(page_ids),
                "format": "json",
                "utf8": 1,
            },
            requester=requester,
            headers=_wikipedia_headers(user_agent),
        )
    except Exception as exc:
        return f"Wikipedia summary failed for {term!r}: {exc}"

    pages = (
        extract_payload.get("query", {}).get("pages", {})
        if isinstance(extract_payload.get("query"), dict)
        else {}
    )
    lines = [f"Wikipedia results for: {term}"]
    for page_id in page_ids:
        page = pages.get(page_id, {}) if isinstance(pages, dict) else {}
        title = str(page.get("title") or page_id)
        extract = _trim_summary(str(page.get("extract") or ""), max_summary_chars)
        lines.append(f"- {title}: {extract or 'No summary available.'}")
        lines.append(f"  URL: {_page_url(title)}")
    return "\n".join(lines)


def _trim_summary(summary: str, max_chars: int) -> str:
    normalized = " ".join(summary.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 3)].rstrip() + "..."


def _page_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))


def _wikipedia_headers(user_agent: str) -> dict[str, str]:
    agent = user_agent.strip() or (
        "langgraph-rag/1.0 (contact: configure WIKIPEDIA_USER_AGENT)"
    )
    return {"User-Agent": agent, "Api-User-Agent": agent}


__all__ = [
    "WikipediaInput",
    "build_wikipedia_tool",
    "search_wikipedia",
]
