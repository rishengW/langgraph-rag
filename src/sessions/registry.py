from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from ..config import Settings
from .models import ChatSession


logger = logging.getLogger(__name__)

SessionCleanup = Callable[[ChatSession], None]
TimeProvider = Callable[[], float]

ISOLATED_SOURCE_MODES = {"explicit", "web_search"}


def _safe_isolated_chroma_dir(chroma_dir: Path) -> bool:
    try:
        resolved = chroma_dir.resolve()
    except (OSError, RuntimeError):
        resolved = chroma_dir.absolute()

    return bool(resolved.name) and resolved.parent.name.lower() == "chat"


def _release_chroma_system(chroma_dir: Path) -> None:
    try:
        from ..core.retriever import _release_chroma_system as release
    except Exception as exc:
        logger.debug("Chroma release helper is unavailable: %s", exc)
        return

    try:
        release(chroma_dir)
    except Exception as exc:
        logger.warning("Failed to release Chroma system for %s: %s", chroma_dir, exc)


def _remove_chroma_dir(chroma_dir: Path) -> None:
    try:
        from ..core.retriever import _rmtree_with_retry as remove_with_retry
    except Exception as exc:
        logger.debug("Chroma retry deletion helper is unavailable: %s", exc)
        shutil.rmtree(chroma_dir)
        return

    try:
        remove_with_retry(chroma_dir)
    except Exception as exc:
        logger.warning("Retry deletion failed for %s: %s", chroma_dir, exc)
        shutil.rmtree(chroma_dir)


def cleanup_isolated_chroma(session: ChatSession) -> None:
    """Release and delete the Chroma directory for isolated chat sessions."""

    if not session.isolated_chroma:
        return

    chroma_dir = Path(session.settings.chroma_dir)
    if not _safe_isolated_chroma_dir(chroma_dir):
        logger.warning(
            "Skipping Chroma cleanup for session %s because %s is not an isolated chat path",
            session.thread_id,
            chroma_dir,
        )
        return

    _release_chroma_system(chroma_dir)
    if chroma_dir.exists():
        _remove_chroma_dir(chroma_dir)


class ChatSessionRegistry:
    """Thread-safe in-memory map of ``thread_id`` to ``ChatSession``."""

    def __init__(
        self,
        ttl_seconds: float | None = 3600,
        cleanup_interval: float = 300,
        *,
        cleanup: SessionCleanup | None = None,
        time_func: TimeProvider = time.time,
    ) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, ChatSession] = {}
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._cleanup = cleanup or cleanup_isolated_chroma
        self._time = time_func
        self._stop_cleanup = threading.Event()
        self._cleanup_thread: threading.Thread | None = None

    @property
    def ttl_seconds(self) -> float | None:
        return self._ttl_seconds

    def create(
        self,
        graph: Any,
        settings: Settings,
        source_urls: list[str],
        source_mode: str,
        thread_id: str | None = None,
        isolated_chroma: bool | None = None,
    ) -> ChatSession:
        thread_id = thread_id or uuid.uuid4().hex
        now = self._time()
        session = ChatSession(
            thread_id=thread_id,
            graph=graph,
            settings=settings,
            source_urls=list(source_urls),
            source_mode=source_mode,
            created_at=now,
            last_accessed_at=now,
            isolated_chroma=(
                source_mode in ISOLATED_SOURCE_MODES
                if isolated_chroma is None
                else isolated_chroma
            ),
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

    def get(self, thread_id: str, *, touch: bool = True) -> ChatSession | None:
        with self._lock:
            session = self._sessions.get(thread_id)
            if session is not None and touch:
                session.touch(self._time())
            return session

    def delete(self, thread_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(thread_id, None)
        if session is None:
            return False

        self._cleanup_session(session)
        logger.info("Deleted chat session %s", thread_id)
        return True

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    def cleanup_expired(self) -> int:
        if self._ttl_seconds is None:
            return 0

        now = self._time()
        expired: list[ChatSession] = []
        with self._lock:
            for thread_id, session in list(self._sessions.items()):
                last_accessed = session.last_accessed_at or session.created_at
                if now - last_accessed > self._ttl_seconds:
                    expired_session = self._sessions.pop(thread_id, None)
                    if expired_session is not None:
                        expired.append(expired_session)

        for session in expired:
            self._cleanup_session(session)
            logger.info("Expired chat session %s", session.thread_id)

        return len(expired)

    def start_background_cleanup(self) -> None:
        if self._ttl_seconds is None:
            return

        with self._lock:
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                return

            self._stop_cleanup.clear()
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                name="chat-session-cleanup",
                daemon=True,
            )
            self._cleanup_thread.start()

    def stop_background_cleanup(self, timeout: float | None = None) -> None:
        self._stop_cleanup.set()
        thread = self._cleanup_thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)

    def _cleanup_loop(self) -> None:
        interval = max(0.1, float(self._cleanup_interval))
        while not self._stop_cleanup.wait(interval):
            self.cleanup_expired()

    def _cleanup_session(self, session: ChatSession) -> None:
        try:
            self._cleanup(session)
        except Exception as exc:
            logger.warning("Cleanup failed for chat session %s: %s", session.thread_id, exc)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


__all__ = [
    "ChatSessionRegistry",
    "cleanup_isolated_chroma",
]
