"""Stable transport-neutral contracts for RAG application services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from ..security import Principal


@dataclass(frozen=True, slots=True)
class SourceReference:
    """A source used to ground an answer."""

    url: str
    title: str | None = None
    citation_id: str | None = None


@dataclass(frozen=True, slots=True)
class RagRequest:
    """One stateless RAG request independent of HTTP or MCP."""

    question: str
    urls: str | list[str] | None = None
    rebuild: bool = False
    web_search: bool = True
    debug: bool = False
    request_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """Transport-neutral result retaining the legacy QA response fields."""

    answer: str | None
    error: str | None
    success: bool
    messages: list[str] | None
    source_urls: list[str] | None
    source_mode: str | None
    source_note: str | None
    request_id: str
    sources: tuple[SourceReference, ...] = ()


@dataclass(frozen=True, slots=True)
class TurnRequest:
    """One chat turn independent of its HTTP or streaming transport."""

    thread_id: str
    message: str
    stream_tokens: bool = True
    request_id: str = field(default_factory=lambda: uuid4().hex)
    principal: Principal = field(default_factory=Principal.local_process)


@dataclass(frozen=True, slots=True)
class TurnResult:
    """Completed non-streaming chat turn with legacy-compatible fields."""

    thread_id: str
    answer: str = ""
    error: str | None = None
    artifacts: tuple[dict[str, Any], ...] = ()
    request_id: str = ""


@dataclass(frozen=True, slots=True)
class StartSessionRequest:
    """Inputs used to create one chat session."""

    urls: str | list[str] | None = None
    web_search: bool = True
    seed_question: str | None = None
    request_id: str = field(default_factory=lambda: uuid4().hex)
    principal: Principal = field(default_factory=Principal.local_process)


@dataclass(frozen=True, slots=True)
class SessionResult:
    """Transport-neutral chat session metadata."""

    thread_id: str
    source_urls: tuple[str, ...]
    source_mode: str
    source_note: str | None = None


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """One visible conversation-history entry."""

    role: str
    content: str
    artifacts: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class SessionHistory:
    """Visible history and source metadata for one chat session."""

    thread_id: str
    turns: tuple[HistoryEntry, ...]
    source_urls: tuple[str, ...]
    source_mode: str


@dataclass(frozen=True, slots=True)
class SessionDeletion:
    """Result of deleting one chat session."""

    thread_id: str
    status: str = "deleted"
