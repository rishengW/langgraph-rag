# REFACTOR: Defines session persistence metadata and storage protocols.
from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .models import ChatSession


class SessionStorageError(RuntimeError):
    """Raised when session metadata cannot be persisted or loaded."""


@dataclass(frozen=True)
class SessionMetadata:
    """Serializable metadata for a chat session.

    Runtime-only values such as the compiled graph and LangGraph checkpoint are
    intentionally excluded.
    """

    thread_id: str
    source_urls: list[str] = field(default_factory=list)
    source_mode: str = "defaults"
    created_at: float = 0.0
    last_accessed_at: float = 0.0
    chroma_dir: str | None = None
    isolated_chroma: bool = False
    config: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_session(cls, session: ChatSession) -> SessionMetadata:
        """Build serializable metadata from an in-memory chat session."""

        return cls(
            thread_id=session.thread_id,
            source_urls=list(session.source_urls),
            source_mode=session.source_mode,
            created_at=session.created_at,
            last_accessed_at=session.last_accessed_at or session.created_at,
            chroma_dir=str(session.settings.chroma_dir),
            isolated_chroma=session.isolated_chroma,
            config={
                "collection_name": session.settings.collection_name,
            },
        )


class StorageBackend(Protocol):
    """Persistence interface for chat session metadata."""

    def save(self, metadata: SessionMetadata) -> None:
        """Persist or replace one session metadata record."""

    def load(self, thread_id: str) -> SessionMetadata | None:
        """Return metadata for one session, if present."""

    def delete(self, thread_id: str) -> bool:
        """Delete one session metadata record."""

    def list_ids(self) -> list[str]:
        """Return persisted session IDs."""

    def list_metadata(self) -> list[SessionMetadata]:
        """Return all persisted session metadata records."""


class InMemoryStorage:
    """Thread-safe storage backend for tests and local metadata snapshots."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metadata: dict[str, SessionMetadata] = {}

    def save(self, metadata: SessionMetadata) -> None:
        """Persist or replace one session metadata record."""

        with self._lock:
            self._metadata[metadata.thread_id] = _copy_metadata(metadata)

    def load(self, thread_id: str) -> SessionMetadata | None:
        """Return metadata for one session, if present."""

        with self._lock:
            metadata = self._metadata.get(thread_id)
            return _copy_metadata(metadata) if metadata is not None else None

    def delete(self, thread_id: str) -> bool:
        """Delete one session metadata record."""

        with self._lock:
            return self._metadata.pop(thread_id, None) is not None

    def list_ids(self) -> list[str]:
        """Return persisted session IDs."""

        with self._lock:
            return list(self._metadata.keys())

    def list_metadata(self) -> list[SessionMetadata]:
        """Return all persisted session metadata records."""

        with self._lock:
            return [_copy_metadata(item) for item in self._metadata.values()]


def _copy_metadata(metadata: SessionMetadata) -> SessionMetadata:
    return replace(
        metadata,
        source_urls=list(metadata.source_urls),
        config=dict(metadata.config),
    )


__all__ = [
    "InMemoryStorage",
    "SessionMetadata",
    "SessionStorageError",
    "StorageBackend",
]
