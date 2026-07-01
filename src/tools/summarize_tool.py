from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..config import Settings

_DEFAULT_FOCUS = "Summarize the main points of the page."


class SummarizeUrlInput(BaseModel):
    """Input schema for the URL summarization tool."""

    url: str = Field(
        ...,
        min_length=1,
        description="The http(s) URL of the page to fetch and summarize.",
    )
    focus: str | None = Field(
        default=None,
        description=(
            "Optional instruction for what to focus on, e.g. 'pricing', "
            "'key findings', or a specific question about the page."
        ),
    )


def build_summarize_url_tool(
    settings: Settings,
) -> BaseTool:
    """Create a tool that fetches one URL and summarizes it with the LLM.

    Reuses the project's web-search content fetcher (HTTP load, readable-text
    extraction, optional JS fallback) and the configured chat model, so it
    behaves consistently with the lightweight web-answer path.
    """

    def _run_summarize(url: str, focus: str | None = None) -> str:
        return summarize_url(url, focus=focus, settings=settings)

    return StructuredTool.from_function(
        func=_run_summarize,
        name="summarize_url",
        description=(
            "Fetch a single web page by URL and return a concise summary of "
            "its content. Use when the user gives a specific URL and asks what "
            "it says, to summarize an article, or to extract specific "
            "information from one page. For open-ended research across many "
            "pages, use live web search instead."
        ),
        args_schema=SummarizeUrlInput,
    )


def summarize_url(
    url: str,
    *,
    focus: str | None = None,
    settings: Settings,
) -> str:
    target = (url or "").strip()
    if not target:
        return "Summarize requires a non-empty URL."
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
        detail = (page.error if page and page.error else "no readable content")
        return f"Could not read content from {target}: {detail}."

    prompt = _build_prompt(page.title or target, page.url, page.text, focus)

    from ..llm.provider import build_chat_model
    from ..utils.retry import invoke_with_retry

    try:
        result = invoke_with_retry(
            build_chat_model(settings),
            [HumanMessage(content=prompt)],
            max_retries=settings.dashscope_max_retries,
        )
    except Exception as exc:
        return f"Fetched {target} but could not reach the model to summarize it: {exc}"

    content = getattr(result, "content", "")
    text = content if isinstance(content, str) else str(content)
    text = text.strip()
    if not text:
        return f"Fetched {target} but the model returned an empty summary."
    return f"Summary of {page.title or target} ({page.url}):\n\n{text}"


def _build_prompt(title: str, url: str, body: str, focus: str | None) -> str:
    instruction = (focus or "").strip() or _DEFAULT_FOCUS
    return (
        "You are summarizing the content of a single web page. Use ONLY the "
        "page content below; do not add outside knowledge. If the page does "
        "not contain the requested information, say so.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Page title: {title}\n"
        f"Page URL: {url}\n\n"
        "Page content:\n"
        f"{body}\n\n"
        "Summary:"
    )


__all__ = [
    "SummarizeUrlInput",
    "build_summarize_url_tool",
    "summarize_url",
]
