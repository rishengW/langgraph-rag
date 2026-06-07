from __future__ import annotations

from typing import Annotated, Literal, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


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


AgentState = RAGState
ChatState = RAGState

__all__ = ["AgentState", "ChatState", "RAGState"]

