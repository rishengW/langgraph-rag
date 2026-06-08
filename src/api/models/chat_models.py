"""Pydantic models for the multi-turn chat API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class StartChatRequest(BaseModel):
    urls: Optional[str | list[str]] = Field(
        None,
        description=(
            "Comma-separated URLs or a URL list to use as RAG sources. "
            "If empty, chat uses web search whenever it has a question to search."
        ),
    )
    web_search: bool = Field(
        True,
        description=(
            "Deprecated compatibility flag. Chat web search is controlled by "
            "server settings and is always attempted when URLs are not explicit."
        ),
    )
    seed_question: Optional[str] = Field(
        None,
        description=(
            "Optional first question to use for web-search discovery. "
            "If omitted, the configured default URLs are used instead."
        ),
    )


class StartChatResponse(BaseModel):
    thread_id: str
    source_urls: list[str]
    source_mode: str
    source_note: Optional[str] = None


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1)


class MessageResponse(BaseModel):
    thread_id: str
    answer: str
    error: Optional[str] = None


class HistoryTurn(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    thread_id: str
    turns: list[HistoryTurn]
    source_urls: list[str]
    source_mode: str


__all__ = [
    "HistoryResponse",
    "HistoryTurn",
    "MessageRequest",
    "MessageResponse",
    "StartChatRequest",
    "StartChatResponse",
]
