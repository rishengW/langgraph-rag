"""Transport-neutral facade for chat session and turn operations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

from ..graph.events import GraphEvent
from .models import (
    SessionDeletion,
    SessionHistory,
    SessionResult,
    StartSessionRequest,
    TurnRequest,
    TurnResult,
)
from .session_lifecycle import SessionLifecycleService
from .turn_execution import TurnExecutionService


class ChatApplicationService:
    """Coordinate lifecycle preparation with serialized turn execution."""

    def __init__(
        self,
        *,
        lifecycle: SessionLifecycleService,
        turns: TurnExecutionService,
    ) -> None:
        self._lifecycle = lifecycle
        self._turns = turns

    async def start(self, request: StartSessionRequest) -> SessionResult:
        """Create a new chat session."""

        return await self._lifecycle.start(request)

    async def complete_turn(self, request: TurnRequest) -> TurnResult:
        """Refresh and execute one non-streaming turn under its session lock."""

        session = self._lifecycle.require_session(request.thread_id)
        await _acquire_lock(session.turn_lock)
        try:
            session = await self._lifecycle.refresh(session, request.message)
            return await self._turns.execute_locked(session, request)
        finally:
            session.turn_lock.release()

    async def stream_turn(self, request: TurnRequest) -> AsyncIterator[GraphEvent]:
        """Refresh and stream one turn while retaining its session lock."""

        session = self._lifecycle.require_session(request.thread_id)
        await _acquire_lock(session.turn_lock)
        events: AsyncIterator[GraphEvent] | None = None
        try:
            session = await self._lifecycle.refresh(session, request.message)
            events = self._turns.stream_locked(session, request)
            async for event in events:
                yield event
        finally:
            if events is not None:
                close = getattr(events, "aclose", None)
                if close is not None:
                    await close()
            session.turn_lock.release()

    async def history(self, thread_id: str) -> SessionHistory:
        """Return visible history for one chat session."""

        return await self._lifecycle.history(thread_id)

    async def delete(self, thread_id: str) -> SessionDeletion:
        """Delete one chat session and its session-owned side data."""

        return await self._lifecycle.delete(thread_id)


class Lock(Protocol):
    """Minimal blocking lock contract used by chat sessions."""

    def acquire(self) -> bool: ...

    def release(self) -> None: ...


async def _acquire_lock(lock: Lock) -> None:
    worker = asyncio.create_task(asyncio.to_thread(lock.acquire))
    try:
        await asyncio.shield(worker)
    except asyncio.CancelledError:
        await worker
        lock.release()
        raise


__all__ = ["ChatApplicationService"]
