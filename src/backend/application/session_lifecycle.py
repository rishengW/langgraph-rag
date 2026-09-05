"""Transport-neutral chat session creation, refresh, history, and deletion."""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.config import Settings
from src.errors import RAGError, ResourceNotFoundError
from src.utils.urls import parse_url_input

from ..graph.artifacts import extract_artifacts_from_messages
from ..security import Principal, ResourceOwner
from ..sessions import ChatSession, ChatSessionRegistry
from .errors import SessionLifecycleError
from .models import (
    HistoryEntry,
    SessionDeletion,
    SessionHistory,
    SessionResult,
    StartSessionRequest,
)

logger = logging.getLogger(__name__)

DiscoverUrls = Callable[[str, Settings], list[str]]
BuildGraph = Callable[[Settings, bool, Any, str], Any]
BuildLightweightGraph = Callable[[Settings, Any, str], Any]
SettingsForSession = Callable[[Settings, list[str], str, bool], Settings]
CondenseQuestion = Callable[[list[Any], str, Settings], str]
SessionHook = Callable[[str], None]
UploadDirectory = Callable[[Settings, str], Path]
PurgeMemory = Callable[[Settings, str], None]


@dataclass(frozen=True, slots=True)
class SessionLifecycleDependencies:
    """Infrastructure callbacks used by session lifecycle operations."""

    discover_urls: DiscoverUrls
    build_graph: BuildGraph
    build_lightweight_graph: BuildLightweightGraph
    settings_for_session: SettingsForSession
    condense_question: CondenseQuestion
    on_session_start: SessionHook | None = None
    upload_directory: UploadDirectory | None = None
    purge_memory: PurgeMemory | None = None


class SessionLifecycleService:
    """Manage chat sessions without depending on an inbound transport."""

    def __init__(
        self,
        *,
        settings: Settings,
        sessions: ChatSessionRegistry,
        graph_factory_lock: asyncio.Lock,
        checkpointer: Any,
        dependencies: SessionLifecycleDependencies,
    ) -> None:
        self._settings = settings
        self._sessions = sessions
        self._graph_factory_lock = graph_factory_lock
        self._checkpointer = checkpointer
        self._dependencies = dependencies

    def require_session(
        self,
        thread_id: str,
        principal: Principal | None = None,
    ) -> ChatSession:
        """Return an owned session or one non-disclosing not-found error."""

        trusted_principal = principal or Principal.local_process()
        session = self._sessions.get_owned(thread_id, trusted_principal)
        if session is None:
            raise ResourceNotFoundError("Resource not found.")
        return session

    def searches_for_start(self, request: StartSessionRequest) -> int:
        """Return the server-controlled search reservation for session creation."""

        return int(request.urls is None and self._settings.web_search_enabled)

    def searches_for_turn(self, session: ChatSession) -> int:
        """Return the server-controlled search reservation for one continuation."""

        return int(session.source_mode == "web_search" or self._source_refresh_allowed(session))

    async def start(self, request: StartSessionRequest) -> SessionResult:
        """Create a graph-backed session with legacy source fallback behavior."""

        urls = parse_url_input(request.urls)
        discovered = False
        search_error = False
        graph_owned_web = (
            urls is None
            and self._settings.web_search_enabled
            and self._settings.web_search_lightweight
        )
        if urls is None and self._settings.web_search_enabled and not graph_owned_web:
            seed = (request.seed_question or "").strip()
            if seed:
                try:
                    found = self._dependencies.discover_urls(seed, self._settings)
                    if found:
                        urls = found
                        discovered = True
                    else:
                        search_error = True
                except Exception as exc:
                    search_error = True
                    logger.warning(
                        "Web search failed during chat start (cause=%s)",
                        type(exc).__name__,
                    )

        if request.urls:
            source_mode = "explicit"
        elif graph_owned_web or discovered:
            source_mode = "web_search"
        else:
            source_mode = "defaults"

        if graph_owned_web:
            urls = []
            isolated = False
        elif urls is None:
            urls = list(self._settings.source_urls)
            isolated = False
        else:
            isolated = True

        thread_id = uuid4().hex
        session_settings = self._initial_settings(
            urls,
            thread_id,
            isolated,
            graph_owned_web=graph_owned_web,
            discovered=discovered,
        )
        build_failed = False
        try:
            graph = await self._build_initial_graph(
                session_settings,
                isolated,
                thread_id,
                lightweight=graph_owned_web,
            )
        except RAGError:
            raise
        except Exception as exc:
            if not discovered:
                raise self._creation_error(request, exc) from exc
            build_failed = True
            urls = list(self._settings.source_urls)
            isolated = False
            source_mode = "defaults"
            session_settings = self._dependencies.settings_for_session(
                self._settings,
                urls,
                thread_id,
                False,
            )
            try:
                graph = await self._build_initial_graph(
                    session_settings,
                    False,
                    thread_id,
                    lightweight=False,
                )
            except RAGError:
                raise
            except Exception as fallback_exc:
                raise self._creation_error(request, fallback_exc) from fallback_exc

        session = self._sessions.create(
            graph=graph,
            settings=session_settings,
            source_urls=urls,
            source_mode=source_mode,
            thread_id=thread_id,
            isolated_chroma=isolated,
            owner=ResourceOwner.from_principal(request.principal),
        )
        try:
            self._register_checkpoint_owner(session)
        except Exception as exc:
            self._sessions.delete_owned(thread_id, request.principal)
            raise self._creation_error(request, exc) from exc
        self._run_start_hook(thread_id)
        note = (
            "web_search_failed"
            if source_mode == "defaults" and (search_error or build_failed)
            else None
        )
        return SessionResult(
            thread_id=session.thread_id,
            source_urls=tuple(session.source_urls),
            source_mode=session.source_mode,
            source_note=note,
        )

    async def refresh(self, session: ChatSession, query: str) -> ChatSession:
        """Refresh heavyweight web-search sources for one serialized turn."""

        if not self._source_refresh_allowed(session):
            return session
        search_query = await self._condense_query(session, query)
        try:
            urls = self._dependencies.discover_urls(search_query, self._settings)
        except Exception as exc:
            logger.warning(
                "Web search failed during chat turn for %s (cause=%s)",
                session.thread_id,
                type(exc).__name__,
            )
            return session
        if not urls or (session.source_mode == "web_search" and urls == session.source_urls):
            return session

        session_settings = self._dependencies.settings_for_session(
            replace(self._settings, web_search_enabled=False),
            urls,
            session.thread_id,
            True,
        )
        try:
            async with self._graph_factory_lock:
                graph = await asyncio.to_thread(
                    self._dependencies.build_graph,
                    session_settings,
                    True,
                    self._checkpointer,
                    session.thread_id,
                )
        except RAGError:
            raise
        except Exception as exc:
            logger.warning(
                "Failed to rebuild web-search graph for %s (cause=%s)",
                session.thread_id,
                type(exc).__name__,
            )
            return session
        return (
            self._sessions.update_sources(
                session.thread_id,
                graph=graph,
                settings=session_settings,
                source_urls=urls,
                source_mode="web_search",
                isolated_chroma=True,
            )
            or session
        )

    def sync_graph_owned_sources(self, session: ChatSession, values: Any) -> None:
        """Persist graph-discovered URLs after a completely successful turn."""

        if session.source_mode != "web_search" or not session.settings.web_search_lightweight:
            return
        if not isinstance(values, Mapping):
            return
        raw_urls = values.get("source_urls")
        if not isinstance(raw_urls, list):
            return
        urls = [str(url).strip() for url in raw_urls if str(url).strip()]
        if urls == session.source_urls:
            return
        self._sessions.update_sources(
            session.thread_id,
            graph=session.graph,
            settings=session.settings,
            source_urls=urls,
            source_mode="web_search",
            isolated_chroma=False,
        )

    async def history(
        self,
        thread_id: str,
        principal: Principal | None = None,
    ) -> SessionHistory:
        """Return visible checkpoint history for one authorized session."""

        session = self.require_session(thread_id, principal)
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await asyncio.to_thread(session.graph.get_state, config)
        values = getattr(snapshot, "values", {}) or {}
        messages = values.get("messages", []) if isinstance(values, Mapping) else []
        return SessionHistory(
            thread_id=thread_id,
            turns=serialize_history(messages),
            source_urls=tuple(session.source_urls),
            source_mode=session.source_mode,
        )

    async def delete(
        self,
        thread_id: str,
        principal: Principal | None = None,
    ) -> SessionDeletion:
        """Delete authorized session state and session-owned side data."""

        trusted_principal = principal or Principal.local_process()
        session = self.require_session(thread_id, trusted_principal)
        if not self._sessions.delete_owned(thread_id, trusted_principal):
            raise ResourceNotFoundError("Resource not found.")
        self._delete_checkpoints(session)
        if self._dependencies.upload_directory is not None:
            try:
                upload_dir = self._dependencies.upload_directory(
                    self._settings,
                    thread_id,
                )
                if upload_dir.exists():
                    await asyncio.to_thread(shutil.rmtree, upload_dir, True)
            except Exception as exc:
                logger.warning(
                    "Failed to remove uploads for %s (cause=%s)",
                    thread_id,
                    type(exc).__name__,
                )
        if self._settings.memory_enabled and self._dependencies.purge_memory is not None:
            try:
                await asyncio.to_thread(
                    self._dependencies.purge_memory,
                    self._settings,
                    thread_id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to purge session memory for %s (cause=%s)",
                    thread_id,
                    type(exc).__name__,
                )
        return SessionDeletion(thread_id=thread_id)

    def _register_checkpoint_owner(self, session: ChatSession) -> None:
        register_owner = getattr(self._checkpointer, "register_owner", None)
        if not callable(register_owner):
            return
        if session.owner is None:
            raise ResourceNotFoundError("Resource not found.")
        register_owner(session.thread_id, session.owner)

    def _delete_checkpoints(self, session: ChatSession) -> None:
        delete_thread = getattr(self._checkpointer, "delete_thread", None)
        if not callable(delete_thread):
            return
        try:
            if getattr(self._checkpointer, "ownership_enforced", False):
                if session.owner is None:
                    raise ResourceNotFoundError("Resource not found.")
                delete_thread(session.thread_id, owner=session.owner)
            else:
                delete_thread(session.thread_id)
        except Exception as exc:
            logger.warning(
                "Failed to remove checkpoints for %s (cause=%s)",
                session.thread_id,
                type(exc).__name__,
            )

    def _initial_settings(
        self,
        urls: list[str],
        thread_id: str,
        isolated: bool,
        *,
        graph_owned_web: bool,
        discovered: bool,
    ) -> Settings:
        if graph_owned_web:
            return replace(self._settings, source_urls=[])
        base = replace(self._settings, web_search_enabled=False) if discovered else self._settings
        return self._dependencies.settings_for_session(
            base,
            urls,
            thread_id,
            isolated,
        )

    async def _build_initial_graph(
        self,
        settings: Settings,
        rebuild: bool,
        thread_id: str,
        *,
        lightweight: bool,
    ) -> Any:
        async with self._graph_factory_lock:
            if lightweight:
                return await asyncio.to_thread(
                    self._dependencies.build_lightweight_graph,
                    settings,
                    self._checkpointer,
                    thread_id,
                )
            return await asyncio.to_thread(
                self._dependencies.build_graph,
                settings,
                rebuild,
                self._checkpointer,
                thread_id,
            )

    def _source_refresh_allowed(self, session: ChatSession) -> bool:
        return (
            session.source_mode != "explicit"
            and self._settings.web_search_enabled
            and not self._settings.web_search_lightweight
        )

    async def _condense_query(self, session: ChatSession, message: str) -> str:
        try:
            config = {"configurable": {"thread_id": session.thread_id}}
            snapshot = await asyncio.to_thread(session.graph.get_state, config)
            values = getattr(snapshot, "values", {}) or {}
            prior = values.get("messages", []) if isinstance(values, Mapping) else []
        except Exception:
            return message
        if not prior:
            return message
        try:
            standalone = await asyncio.to_thread(
                self._dependencies.condense_question,
                list(prior),
                message,
                self._settings,
            )
        except Exception:
            logger.warning(
                "Follow-up condense failed for %s; using raw message",
                session.thread_id,
            )
            return message
        return standalone or message

    def _run_start_hook(self, thread_id: str) -> None:
        if self._dependencies.on_session_start is None:
            return
        try:
            self._dependencies.on_session_start(thread_id)
        except Exception:
            logger.debug("session-start hook failed", exc_info=True)

    @staticmethod
    def _creation_error(
        request: StartSessionRequest,
        exc: BaseException,
    ) -> SessionLifecycleError:
        logger.error(
            "Unexpected session creation failure (request_id=%s, cause=%s)",
            request.request_id,
            type(exc).__name__,
            exc_info=exc,
        )
        return SessionLifecycleError(
            "Failed to build chat graph.",
            request_id=request.request_id,
            internal_cause=exc,
        )


def serialize_history(messages: Iterable[Any]) -> tuple[HistoryEntry, ...]:
    """Convert checkpoint messages into visible history entries."""

    turns: list[HistoryEntry] = []
    pending_artifacts: list[dict[str, Any]] = []
    for message in messages:
        kind = getattr(message, "type", None) or message.__class__.__name__.lower()
        content = getattr(message, "content", str(message))
        if kind.startswith("human") or kind == "user":
            role = "user"
        elif kind.startswith("ai") or kind == "assistant":
            role = "assistant"
        elif kind in ("tool", "function"):
            role = "tool"
        elif kind.startswith("system"):
            role = "system"
        else:
            role = kind
        if role == "tool":
            pending_artifacts.extend(extract_artifacts_from_messages([message]))
            continue
        if role == "system":
            continue
        if role == "user":
            pending_artifacts = []
        if role == "assistant" and (
            getattr(message, "tool_calls", None) or not str(content or "").strip()
        ):
            continue
        turns.append(
            HistoryEntry(
                role=role,
                content=content if isinstance(content, str) else str(content),
                artifacts=tuple(pending_artifacts) if role == "assistant" else (),
            )
        )
        if role == "assistant":
            pending_artifacts = []
    return tuple(turns)


__all__ = [
    "SessionLifecycleDependencies",
    "SessionLifecycleService",
    "serialize_history",
]
