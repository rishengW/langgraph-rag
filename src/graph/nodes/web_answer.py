"""Lightweight web-search answer node for direct page-context prompting."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from ...config import Settings
from ...utils.retry import invoke_with_retry
from .common import message_text, new_chat_model, qa_question_resolver

logger = logging.getLogger(__name__)

QuestionResolver = Callable[[dict[str, Any]], str]

# Match URLs but stop at whitespace, brackets/parens, and any CJK/full-width
# punctuation or ideographs. Without the unicode ranges, a URL written inline in
# Chinese prose (e.g. "https://deepseek.net/zh）虽为官方渠道") would greedily
# swallow the surrounding sentence and produce an unreachable garbage URL.
_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\u3000-\u303f\uff00-\uffef\u4e00-\u9fff]+")


def web_answer_factory(
    settings: Settings,
    question_resolver: QuestionResolver = qa_question_resolver,
):
    """Return a node that answers directly from fetched web pages.

    The node intentionally avoids retrievers, embeddings, Chroma, grading, and
    query rewriting. It delegates page loading and prompt assembly to the web
    search lightweight-path modules.
    """

    def web_answer(state: dict[str, Any]) -> dict[str, list[AIMessage]]:
        logger.info("GENERATE WEB ANSWER")
        question = question_resolver(state)
        urls = _extract_source_urls(state, settings)

        from ...web_search.content_fetcher import fetch_pages, is_readable_text
        from ...web_search.prompt_builder import build_web_search_prompt

        pages = fetch_pages(
            urls,
            timeout=settings.page_load_timeout,
            max_tokens_per_page=settings.web_search_max_page_tokens,
            cache_ttl_seconds=settings.page_load_cache_ttl_seconds,
            max_concurrent_loads=settings.page_load_max_concurrency,
            min_readable_chars=settings.web_search_min_page_chars,
            min_readable_tokens=settings.web_search_min_page_tokens,
            js_fallback_enabled=settings.web_search_js_fallback_enabled,
            js_fallback_domains=settings.web_search_js_fallback_domains,
            js_force_domains=settings.web_search_js_force_domains,
        )

        # Guard against ungrounded answers: if no fetched page produced
        # readable text (all URLs unreachable, empty, or junk), do NOT prompt
        # the model. With an empty context the LLM falls back to its training
        # data and silently answers from stale/parametric knowledge, which is
        # exactly the failure we want to avoid in a retrieval-grounded app.
        readable_pages = [
            page
            for page in pages
            if is_readable_text(
                page.text or "",
                min_chars=settings.web_search_min_page_chars,
                min_tokens=settings.web_search_min_page_tokens,
            )
        ]
        if not readable_pages:
            attempted = ", ".join(page.url for page in pages if page.url) or "none"
            logger.warning(
                "No readable web source content for question; skipping LLM call "
                "to avoid an ungrounded answer (attempted URLs: %s)",
                attempted,
            )
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I couldn't retrieve readable content from the web "
                            "sources for this question, so I don't have grounded "
                            "information to answer it and won't guess.\n\n"
                            f"Attempted sources: {attempted}\n\n"
                            "Try rephrasing the question, or provide a specific "
                            "source URL you'd like me to read."
                        )
                    )
                ]
            }

        prompt = build_web_search_prompt(question, readable_pages)

        try:
            result = invoke_with_retry(
                new_chat_model(settings),
                [HumanMessage(content=prompt)],
                max_retries=settings.dashscope_max_retries,
            )
            content = message_text(result)
        except Exception as exc:
            logger.error("Web answer error: %s", exc)
            content = (
                "I could not reach the chat model to synthesize an answer from "
                "the fetched web sources."
            )

        return {"messages": [AIMessage(content=content)]}

    return web_answer


def _extract_source_urls(state: dict[str, Any], settings: Settings) -> list[str]:
    # 1. URLs explicitly placed in graph state by the entry point.
    explicit_urls = _clean_urls(state.get("source_urls") or [])
    if explicit_urls:
        return explicit_urls

    # 2. The discovered/configured URLs the entry point set on Settings. These
    #    are the clean, provider-ranked URLs from the session's web-search
    #    discovery (e.g. the top-K kept after quality filtering) and are the
    #    authoritative source set for the session. They take precedence over
    #    the agent's in-graph live_web_search tool output, which is a narrower,
    #    lower-quality re-search that can otherwise shadow the curated URLs and
    #    send the answer node to chase junk/unreachable pages.
    settings_urls = _clean_urls(getattr(settings, "source_urls", None) or [])
    if settings_urls:
        return settings_urls

    # 3. URLs emitted by the live_web_search tool at runtime, used only when the
    #    session has no curated URLs. Only tool messages are parsed -- never
    #    AI/agent prose, which may contain URLs glued to surrounding text
    #    (e.g. "https://x.com/zh）虽为官方渠道").
    return _clean_urls(_urls_from_tool_messages(state.get("messages") or []))


def _urls_from_tool_messages(messages: Sequence[Any]) -> list[str]:
    urls: list[str] = []
    for message in messages:
        if _message_role(message) != "tool":
            continue
        urls.extend(_URL_RE.findall(message_text(message)))
    return urls


def _message_role(message: Any) -> str:
    role = getattr(message, "type", None)
    if role:
        return str(role)
    if isinstance(message, (tuple, list)) and message:
        return str(message[0])
    return message.__class__.__name__.lower()


def _clean_urls(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for value in values:
        url = str(value).strip().rstrip(".,;）)】」》")
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


__all__ = ["web_answer_factory"]
