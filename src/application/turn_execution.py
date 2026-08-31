"""Transport-neutral serialized chat-turn execution and rollback."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, cast

from ..errors import RAGError
from ..graph.artifacts import extract_artifacts_from_messages
from ..graph.events import DoneEvent, ErrorEvent, GraphEvent
from ..graph.executor import GraphExecutor
from ..graph.metrics import MetricsCollector
from ..sessions import ChatSession
from .errors import TurnExecutionError
from .models import TurnRequest, TurnResult

logger = logging.getLogger(__name__)

BuildTurnInputs = Callable[[ChatSession, str], dict[str, Any]]
AfterTurn = Callable[[str], None]
SyncSources = Callable[[ChatSession, Any], None]
ExecutorFactory = Callable[[Any, MetricsCollector | None], GraphExecutor]


def _default_executor_factory(
    graph: Any,
    metrics: MetricsCollector | None,
) -> GraphExecutor:
    return GraphExecutor(graph, metrics=metrics)


@dataclass(frozen=True, slots=True)
class TurnExecutionDependencies:
    """Infrastructure callbacks used to execute one checkpointed turn."""

    build_inputs: BuildTurnInputs
    checkpointer: Any = None
    metrics: MetricsCollector | None = None
    after_turn: AfterTurn | None = None
    sync_sources: SyncSources | None = None
    executor_factory: ExecutorFactory = _default_executor_factory


class TurnExecutionService:
    """Serialize, execute, and atomically complete chat turns."""

    def __init__(self, dependencies: TurnExecutionDependencies) -> None:
        self._dependencies = dependencies

    async def execute(
        self,
        session: ChatSession,
        request: TurnRequest,
    ) -> TurnResult:
        """Execute a non-streaming turn under the session lock."""

        return await self._run_non_streaming(session, request, lock_held=False)

    async def execute_locked(
        self,
        session: ChatSession,
        request: TurnRequest,
    ) -> TurnResult:
        """Execute while a coordinating service already owns the turn lock."""

        return await self._run_non_streaming(session, request, lock_held=True)

    async def _run_non_streaming(
        self,
        session: ChatSession,
        request: TurnRequest,
        *,
        lock_held: bool,
    ) -> TurnResult:
        stop_requested = threading.Event()
        worker = asyncio.create_task(
            asyncio.to_thread(
                self._execute_sync,
                session,
                request,
                lock_held,
                stop_requested,
            )
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            stop_requested.set()
            await asyncio.shield(worker)
            raise

    def _execute_sync(
        self,
        session: ChatSession,
        request: TurnRequest,
        lock_held: bool,
        stop_requested: threading.Event,
    ) -> TurnResult:
        context = nullcontext() if lock_held else session.turn_lock
        with context:
            try:
                snapshot = self._snapshot(request.thread_id)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                return self._failed_result(request, exc)

            try:
                config = _thread_config(request.thread_id)
                previous_count = _message_count(session.graph, config)
                inputs = self._dependencies.build_inputs(session, request.message)
                result = session.graph.invoke(inputs, config)
                messages = _result_messages(result)
                current_messages = (
                    messages[previous_count:] if previous_count <= len(messages) else messages
                )
                artifacts = tuple(extract_artifacts_from_messages(current_messages))
                answer = _last_assistant_answer(current_messages)
            except BaseException as exc:
                self._rollback(request.thread_id, snapshot)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                return self._failed_result(request, exc)

            if stop_requested.is_set():
                self._rollback(request.thread_id, snapshot)
                return TurnResult(
                    thread_id=request.thread_id,
                    error="Turn cancelled.",
                    request_id=request.request_id,
                )
            if not answer:
                self._rollback(request.thread_id, snapshot)
                return TurnResult(
                    thread_id=request.thread_id,
                    error="No assistant reply was produced.",
                    request_id=request.request_id,
                )

            self._complete(session, result, request.thread_id)
            return TurnResult(
                thread_id=request.thread_id,
                answer=answer,
                artifacts=artifacts,
                request_id=request.request_id,
            )

    def _failed_result(
        self,
        request: TurnRequest,
        exc: BaseException,
    ) -> TurnResult:
        return TurnResult(
            thread_id=request.thread_id,
            error=_public_error(exc, request.request_id),
            request_id=request.request_id,
        )

    async def stream(
        self,
        session: ChatSession,
        request: TurnRequest,
    ) -> AsyncIterator[GraphEvent]:
        """Yield graph events in order and roll back incomplete streams."""

        events = self._stream(session, request, lock_held=False)
        try:
            async for event in events:
                yield event
        finally:
            await events.aclose()

    async def stream_locked(
        self,
        session: ChatSession,
        request: TurnRequest,
    ) -> AsyncIterator[GraphEvent]:
        """Stream while a coordinating service already owns the turn lock."""

        events = self._stream(session, request, lock_held=True)
        try:
            async for event in events:
                yield event
        finally:
            await events.aclose()

    async def _stream(
        self,
        session: ChatSession,
        request: TurnRequest,
        *,
        lock_held: bool,
    ) -> AsyncGenerator[GraphEvent, None]:
        queue: asyncio.Queue[object] = asyncio.Queue()
        producer_done = object()
        stop_requested = threading.Event()
        loop = asyncio.get_running_loop()

        def publish(item: object) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                # The application event loop is already shutting down.
                return

        def produce_events() -> None:
            try:
                context = nullcontext() if lock_held else session.turn_lock
                with context:
                    if stop_requested.is_set():
                        return
                    try:
                        snapshot = self._snapshot(request.thread_id)
                    except BaseException as exc:
                        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                            raise
                        publish(_error_event(exc, request.request_id))
                        return

                    completed = False
                    failed = False
                    saw_done = False
                    events: Iterator[GraphEvent] | None = None
                    try:
                        inputs = self._dependencies.build_inputs(
                            session,
                            request.message,
                        )
                        events = self._dependencies.executor_factory(
                            session.graph,
                            self._dependencies.metrics,
                        ).stream(
                            inputs,
                            config=_thread_config(request.thread_id),
                            stream_tokens=request.stream_tokens,
                        )
                        for event in events:
                            if stop_requested.is_set():
                                break
                            if isinstance(event, ErrorEvent):
                                failed = True
                                event = _sanitize_error_event(
                                    event,
                                    request.request_id,
                                )
                            if isinstance(event, DoneEvent):
                                saw_done = True
                            publish(event)
                        else:
                            completed = saw_done and not failed
                    except BaseException as exc:
                        failed = True
                        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                            raise
                        if not stop_requested.is_set():
                            publish(_error_event(exc, request.request_id))
                    finally:
                        if events is not None:
                            close = getattr(events, "close", None)
                            if callable(close):
                                try:
                                    close()
                                except BaseException as exc:
                                    failed = True
                                    completed = False
                                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                                        raise
                                    if not stop_requested.is_set():
                                        publish(_error_event(exc, request.request_id))

                        completed = (
                            completed and saw_done and not failed and not stop_requested.is_set()
                        )
                        if completed:
                            self._complete_from_graph(session, request.thread_id)
                        else:
                            self._rollback(request.thread_id, snapshot)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                if not stop_requested.is_set():
                    publish(_error_event(exc, request.request_id))
            finally:
                publish(producer_done)

        worker = asyncio.create_task(asyncio.to_thread(produce_events))
        try:
            while True:
                item = await queue.get()
                if item is producer_done:
                    break
                yield cast(GraphEvent, item)
        finally:
            stop_requested.set()
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                await asyncio.shield(worker)
                raise

    def _snapshot(self, thread_id: str) -> bytes | None:
        checkpointer = self._dependencies.checkpointer
        if checkpointer is None:
            return None
        return cast(bytes, checkpointer.snapshot_thread(thread_id))

    def _rollback(self, thread_id: str, snapshot: bytes | None) -> None:
        checkpointer = self._dependencies.checkpointer
        if snapshot is None or checkpointer is None:
            return
        try:
            checkpointer.restore_thread(thread_id, snapshot)
        except Exception:
            logger.exception("Could not roll back chat turn for %s", thread_id)

    def _complete(self, session: ChatSession, values: Any, thread_id: str) -> None:
        if self._dependencies.sync_sources is not None:
            try:
                self._dependencies.sync_sources(session, values)
            except Exception:
                logger.debug("Could not synchronize graph-owned sources", exc_info=True)
        self._run_after_turn(thread_id)

    def _complete_from_graph(self, session: ChatSession, thread_id: str) -> None:
        if self._dependencies.sync_sources is not None:
            try:
                snapshot = session.graph.get_state(_thread_config(thread_id))
                values = getattr(snapshot, "values", {}) or {}
                self._dependencies.sync_sources(session, values)
            except Exception:
                logger.debug("Could not synchronize graph-owned sources", exc_info=True)
        self._run_after_turn(thread_id)

    def _run_after_turn(self, thread_id: str) -> None:
        if self._dependencies.after_turn is None:
            return
        try:
            self._dependencies.after_turn(thread_id)
        except Exception:
            logger.debug("after-turn hook failed", exc_info=True)


def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _message_count(graph: Any, config: Mapping[str, Any]) -> int:
    try:
        snapshot = graph.get_state(config)
        values = getattr(snapshot, "values", {}) or {}
        if not isinstance(values, Mapping):
            return 0
        messages = values.get("messages", []) or []
        return len(messages) if isinstance(messages, list) else 0
    except Exception:
        return 0


def _result_messages(result: Any) -> list[Any]:
    if not isinstance(result, Mapping):
        return []
    messages = result.get("messages", []) or []
    return list(messages) if isinstance(messages, list) else []


def _last_assistant_answer(messages: list[Any]) -> str:
    for message in reversed(messages):
        kind = _message_value(message, "type")
        if kind is None:
            kind = message.__class__.__name__.lower()
        content = _message_value(message, "content")
        if _message_value(message, "tool_calls"):
            continue
        if (str(kind).startswith("ai") or kind == "assistant") and str(content or "").strip():
            return content if isinstance(content, str) else str(content)
    return ""


def _message_value(message: Any, name: str) -> Any:
    if isinstance(message, Mapping):
        return message.get(name)
    return getattr(message, name, None)


def _public_error(exc: BaseException, request_id: str) -> str:
    if isinstance(exc, RAGError):
        return str(exc)
    error = TurnExecutionError(
        "Internal server error.",
        request_id=request_id,
        internal_cause=exc,
    )
    logger.error(
        "Unexpected chat turn failure (request_id=%s, cause=%s)",
        request_id,
        type(exc).__name__,
        exc_info=exc,
    )
    return error.public_detail


def _error_event(exc: BaseException, request_id: str) -> ErrorEvent:
    return ErrorEvent(
        message=_public_error(exc, request_id),
        recoverable=False,
    )


def _sanitize_error_event(event: ErrorEvent, request_id: str) -> ErrorEvent:
    error = TurnExecutionError(
        "Internal server error.",
        request_id=request_id,
    )
    return replace(event, message=error.public_detail)


__all__ = ["TurnExecutionDependencies", "TurnExecutionService"]
