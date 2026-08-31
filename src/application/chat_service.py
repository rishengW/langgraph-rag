"""Transport-neutral facade for chat session and turn operations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Protocol

from ..graph.events import GraphEvent
from ..security import (
    Principal,
    QuotaCharge,
    QuotaLease,
    QuotaManager,
    estimate_token_units,
    estimated_cost_units,
)
from ..sessions import ChatSession
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
    """Coordinate owned lifecycle operations and quota-bound turn execution."""

    def __init__(
        self,
        *,
        lifecycle: SessionLifecycleService,
        turns: TurnExecutionService,
        quotas: QuotaManager | None = None,
        retry_reservation: int = 0,
    ) -> None:
        self._lifecycle = lifecycle
        self._turns = turns
        self._quotas = quotas or QuotaManager()
        self._retry_reservation = max(0, int(retry_reservation))

    async def start(self, request: StartSessionRequest) -> SessionResult:
        """Create an owned chat session under the caller's quota lease."""

        searches = self._lifecycle.searches_for_start(request)
        charge = _operation_charge(
            searches=searches,
            tokens=estimate_token_units(request.seed_question),
            retries=self._retry_reservation,
        )
        with self._quotas.acquire(request.principal, charge):
            return await self._lifecycle.start(request)

    async def complete_turn(self, request: TurnRequest) -> TurnResult:
        """Authorize, refresh, and execute one quota-bound serialized turn."""

        session = self._lifecycle.require_session(request.thread_id, request.principal)
        charge = _turn_charge(
            request.message,
            self._lifecycle.searches_for_turn(session),
            self._retry_reservation,
        )
        with self._quotas.acquire(request.principal, charge):
            await _acquire_lock(session.turn_lock)
            try:
                session = await self._lifecycle.refresh(session, request.message)
                return await self._turns.execute_locked(session, request)
            finally:
                session.turn_lock.release()

    def stream_turn(self, request: TurnRequest) -> AsyncIterator[GraphEvent]:
        """Authorize before transport start and retain quotas for the stream lifetime."""

        session = self._lifecycle.require_session(request.thread_id, request.principal)
        lease = self._quotas.acquire(
            request.principal,
            _turn_charge(
                request.message,
                self._lifecycle.searches_for_turn(session),
                self._retry_reservation,
            ),
        )
        return self._stream_turn(session, request, lease)

    async def _stream_turn(
        self,
        session: ChatSession,
        request: TurnRequest,
        lease: QuotaLease,
    ) -> AsyncIterator[GraphEvent]:
        events: AsyncIterator[GraphEvent] | None = None
        lock_acquired = False
        try:
            await _acquire_lock(session.turn_lock)
            lock_acquired = True
            session = await self._lifecycle.refresh(session, request.message)
            events = self._turns.stream_locked(session, request)
            async for event in events:
                yield event
        finally:
            if events is not None:
                close = getattr(events, "aclose", None)
                if close is not None:
                    await close()
            if lock_acquired:
                session.turn_lock.release()
            lease.release()

    async def history(
        self,
        thread_id: str,
        principal: Principal | None = None,
    ) -> SessionHistory:
        """Return visible history only to its owner under a request quota."""

        trusted_principal = principal or Principal.local_process()
        self._lifecycle.require_session(thread_id, trusted_principal)
        with self._quotas.acquire(trusted_principal):
            return await self._lifecycle.history(thread_id, trusted_principal)

    async def delete(
        self,
        thread_id: str,
        principal: Principal | None = None,
    ) -> SessionDeletion:
        """Delete one owned chat session under a request quota."""

        trusted_principal = principal or Principal.local_process()
        self._lifecycle.require_session(thread_id, trusted_principal)
        with self._quotas.acquire(trusted_principal):
            return await self._lifecycle.delete(thread_id, trusted_principal)


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


def _turn_charge(
    message: str,
    searches: int,
    retry_reservation: int,
) -> QuotaCharge:
    return _operation_charge(
        searches=searches,
        tokens=estimate_token_units(message),
        tool_calls=1,
        retries=retry_reservation,
    )


def _operation_charge(
    *,
    searches: int = 0,
    tokens: int = 0,
    tool_calls: int = 0,
    retries: int = 0,
) -> QuotaCharge:
    charge = QuotaCharge(
        requests=1,
        searches=searches,
        tokens=tokens,
        tool_calls=tool_calls,
        retries=retries,
    )
    return replace(charge, cost_units=estimated_cost_units(charge))


__all__ = ["ChatApplicationService"]
