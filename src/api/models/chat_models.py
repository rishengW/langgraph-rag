"""Pydantic models for the multi-turn chat API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StartChatRequest(BaseModel):
    urls: str | list[str] | None = Field(
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
    seed_question: str | None = Field(
        None,
        description=(
            "Compatibility seed used only by the heavyweight web-search path. "
            "Lightweight chat performs search inside the graph on each message."
        ),
    )


class StartChatResponse(BaseModel):
    thread_id: str
    source_urls: list[str]
    source_mode: str
    source_note: str | None = None


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1)


class MessageResponse(BaseModel):
    thread_id: str
    answer: str
    error: str | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class HistoryTurn(BaseModel):
    role: str
    content: str
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    thread_id: str
    turns: list[HistoryTurn]
    source_urls: list[str]
    source_mode: str


class UploadedFile(BaseModel):
    filename: str
    relative_path: str
    size_bytes: int


class UploadResponse(BaseModel):
    thread_id: str
    files: list[UploadedFile]
    errors: list[str] = Field(default_factory=list)


__all__ = [
    "HistoryResponse",
    "HistoryTurn",
    "MessageRequest",
    "MessageResponse",
    "StartChatRequest",
    "StartChatResponse",
    "UploadResponse",
    "UploadedFile",
]
