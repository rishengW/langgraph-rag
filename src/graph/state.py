from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class WebSearchResultMetadata(TypedDict, total=False):
    """Serializable ranking signals retained from a provider result."""

    url: str
    title: str
    snippet: str
    provider: str
    provider_rank: int
    relevance_score: int
    quality_score: int


class RAGState(TypedDict, total=False):
    """Unified graph state for both single-shot QA and multi-turn chat."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    rewrite_count: int
    max_rewrites: int
    current_question: str
    current_question_index: int
    source_urls: list[str]
    source_mode: Literal["explicit", "web_search", "defaults"]
    source_note: str | None
    errors: list[str]
    # Bounded web-search fan-out. ``decompose`` and ``expand`` publish the
    # exact queries for the next search pass; the search node records one URL
    # list per query so merge can reward results returned by multiple queries.
    sub_questions: list[str]
    expanded_queries: list[str]
    search_queries: list[str]
    web_search_results: list[list[str]]
    web_search_result_metadata: list[list[WebSearchResultMetadata]]
    # REFACTOR: Lightweight web-search fallback. When ``web_answer`` produces
    # no readable page text, ``web_answer_no_readable_content`` is set to True
    # so the post-``web_answer`` edge can run one expanded search pass, and
    # ``web_answer_attempts`` bounds that loop. The second failure terminates
    # the graph with the grounded refusal.
    web_answer_attempts: int
    web_answer_no_readable_content: bool
    # REFACTOR: Conditional-expansion one-shot switch. When the first
    # single-query ``web_answer`` run produces no readable content and this
    # flag is False, the post-``web_answer`` edge routes to ``expand`` (and
    # through ``decompose`` -> ``web_search`` -> ``merge`` -> ``web_answer``)
    # to retry with N x k search queries. Once expansion has been attempted
    # the edge routes to the agent fallback so the lightweight graph does
    # not loop between ``web_answer`` and ``expand`` forever.
    expansion_attempted: bool


AgentState = RAGState
ChatState = RAGState

__all__ = ["AgentState", "ChatState", "RAGState", "WebSearchResultMetadata"]
