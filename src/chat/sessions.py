"""In-process registry of chat sessions.

Each session owns:

* a compiled chat graph bound to a specific source URL set (the retriever
  tool inside the graph holds the Chroma collection for those sources);
* the ``Settings`` that produced the graph (URL list, Chroma path,
  collection name);
* a creation timestamp, useful for introspection and future TTL cleanup.

LangGraph's ``MemorySaver`` is what actually stores conversation state per
``thread_id``. The session registry tracks the *graph object* itself
because each graph is bound to a different retriever (different source
URLs ⇒ different Chroma index).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..core.config import Settings


logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    thread_id: str
    graph: Any
    settings: Settings
    source_urls: list[str] = field(default_factory=list)
    source_mode: str = "defaults"  # "explicit" | "web_search" | "defaults"
    created_at: float = field(default_factory=time.time)


class ChatSessionRegistry:
    """Thread-safe in-memory map of ``thread_id → ChatSession``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, ChatSession] = {}

    def create(
        self,
        graph: Any,
        settings: Settings,
        source_urls: list[str],
        source_mode: str,
        thread_id: str | None = None,
    ) -> ChatSession:
        thread_id = thread_id or uuid.uuid4().hex
        session = ChatSession(
            thread_id=thread_id,
            graph=graph,
            settings=settings,
            source_urls=list(source_urls),
            source_mode=source_mode,
        )
        with self._lock:
            self._sessions[thread_id] = session
        logger.info(
            "Created chat session %s with %d source URL(s) (mode=%s)",
            thread_id,
            len(source_urls),
            source_mode,
        )
        return session

    def get(self, thread_id: str) -> ChatSession | None:
        with self._lock:
            return self._sessions.get(thread_id)

    def delete(self, thread_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(thread_id, None)
        if session is None:
            return False
        logger.info("Deleted chat session %s", thread_id)
        return True

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
