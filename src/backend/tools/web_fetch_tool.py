from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.config import Settings


class WebFetchInput(BaseModel):
    """Input schema for the raw web page fetch tool."""

    url: str = Field(
        ...,
        min_length=1,
        description="The http(s) URL of the page to fetch.",
    )


def build_web_fetch_tool(
    settings: Settings,
) -> BaseTool:
    """Create a tool that fetches one URL and returns its readable content.

    Unlike ``summarize_url``, the page content is returned as-is (truncated to
    the configured per-page token budget) with no model pass, so the agent can
    read the original text and draw its own conclusions.
    """

    def _run_fetch(url: str) -> str:
        return web_fetch(url, settings=settings)

    return StructuredTool.from_function(
        func=_run_fetch,
        name="web_fetch",
        description=(
            "Fetch a single web page by URL and return its original readable "
            "content (truncated to the per-page token budget). Use when the "
            "user gives a specific URL and wants the page's actual text, or "
            "when a summary is not enough. For open-ended research across "
            "many pages, use live web search instead."
        ),
        args_schema=WebFetchInput,
    )


def web_fetch(url: str, *, settings: Settings) -> str:
    """Fetch one URL and return its readable text content verbatim."""

    target = (url or "").strip()
    if not target:
        return "web_fetch requires a non-empty URL."
    if not target.lower().startswith(("http://", "https://")):
        return f"Unsupported URL {url!r}; provide an http(s) URL."

    from ..web_search.content_fetcher import fetch_pages, is_readable_text

    try:
        pages = fetch_pages(
            [target],
            timeout=settings.page_load_timeout,
            max_tokens_per_page=settings.web_search_max_page_tokens,
            cache_ttl_seconds=settings.page_load_cache_ttl_seconds,
            max_concurrent_loads=1,
            min_readable_chars=settings.web_search_min_page_chars,
            min_readable_tokens=settings.web_search_min_page_tokens,
            js_fallback_enabled=settings.web_search_js_fallback_enabled,
            js_fallback_domains=settings.web_search_js_fallback_domains,
            js_force_domains=settings.web_search_js_force_domains,
        )
    except Exception as exc:
        return f"Could not fetch {target}: {exc}"

    page = pages[0] if pages else None
    if page is None or not is_readable_text(
        page.text or "",
        min_chars=settings.web_search_min_page_chars,
        min_tokens=settings.web_search_min_page_tokens,
    ):
        detail = page.error if page and page.error else "no readable content"
        return f"Could not read content from {target}: {detail}."

    header = f"Content of {page.title or target} ({page.url}):"
    return f"{header}\n\n{page.text}"


__all__ = [
    "WebFetchInput",
    "build_web_fetch_tool",
    "web_fetch",
]
