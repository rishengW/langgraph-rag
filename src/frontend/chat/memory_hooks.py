"""Best-effort trigger hooks for automatic long-term memory extraction."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Any

from src.config import Settings
from src.backend.memory.extraction import MemoryExtractor
from src.backend.memory.scheduler import ExtractionScheduler
from src.backend.memory.watermark import (
    WATERMARK_KEY,
    InMemoryWatermarkStore,
    WatermarkStore,
    coerce_watermark,
)
from src.backend.sessions import ChatSessionRegistry, SessionMetadata
from src.backend.sessions.storage import StorageBackend

logger = logging.getLogger("src.frontend.chat.memory_hooks")


class SessionWatermarkStore:
    """Persist extraction progress through session metadata and the registry."""

    def __init__(
        self,
        registry: ChatSessionRegistry,
        storage: StorageBackend | None,
    ) -> None:
        self._registry = registry
        self._storage = storage

    def get(self, thread_id: str) -> int:
        """Read persisted metadata first, then the in-memory session."""

        if self._storage is not None:
            metadata = self._storage.load(thread_id)
            if metadata is not None and WATERMARK_KEY in metadata.config:
                raw = metadata.config.get(WATERMARK_KEY)
                value = coerce_watermark(raw)
                if not _is_valid_watermark(raw):
                    logger.warning(
                        "memory_extraction thread_id=%s cause=unusable_watermark",
                        thread_id,
                    )
                return value

        session = self._registry.get(thread_id, touch=False)
        if session is None:
            return 0
        return coerce_watermark(session.extraction_watermark)

    def set(self, thread_id: str, value: int) -> None:
        """Write the value through both persistence layers when available."""

        resolved = coerce_watermark(value)
        session = self._registry.get(thread_id, touch=False)

        if self._storage is not None:
            metadata = self._storage.load(thread_id)
            if metadata is not None:
                config = dict(metadata.config)
                config[WATERMARK_KEY] = resolved
                self._storage.save(replace(metadata, config=config))
            elif session is not None:
                metadata = SessionMetadata.from_session(session)
                config = dict(metadata.config)
                config[WATERMARK_KEY] = resolved
                self._storage.save(replace(metadata, config=config))

        if session is not None:
            session.extraction_watermark = resolved


@dataclass(frozen=True)
class ExtractionRuntime:
    """Objects shared by the API or CLI extraction hooks."""

    settings: Settings
    extractor: MemoryExtractor
    scheduler: ExtractionScheduler
    registry: ChatSessionRegistry | None
    storage: StorageBackend | None
    watermarks: WatermarkStore | None = None


def build_extraction_runtime(
    settings: Settings,
    *,
    checkpointer: Any,
    registry: ChatSessionRegistry | None = None,
    storage: StorageBackend | None = None,
) -> ExtractionRuntime | None:
    """Build extraction services, or stay fully inert when disabled."""

    if (
        not settings.memory_enabled
        or not settings.memory_extraction_enabled
        or checkpointer is None
    ):
        return None

    watermarks: WatermarkStore
    if registry is None:
        watermarks = InMemoryWatermarkStore()
    else:
        watermarks = SessionWatermarkStore(registry, storage)

    extractor = MemoryExtractor(
        settings,
        checkpointer=checkpointer,
        watermarks=watermarks,
    )
    scheduler = ExtractionScheduler(
        max_concurrency=settings.memory_extraction_max_concurrency,
        runner=extractor.run,
    )
    return ExtractionRuntime(
        settings=settings,
        extractor=extractor,
        scheduler=scheduler,
        registry=registry,
        storage=storage,
        watermarks=watermarks,
    )


def on_session_start(
    runtime: ExtractionRuntime | None,
    *,
    new_thread_id: str,
) -> None:
    """Schedule at most one eligible previous session for extraction."""

    if runtime is None or not runtime.settings.memory_extraction_on_session_start:
        return

    try:
        registry = runtime.registry
        storage = runtime.storage
        if registry is None or storage is None:
            return

        live_threads = set(registry.list_ids())
        eligible: list[tuple[float, str, int]] = []
        for metadata in storage.list_metadata():
            thread_id = metadata.thread_id
            if thread_id == new_thread_id or thread_id not in live_threads:
                continue
            turns = runtime.extractor.turn_count(thread_id)
            if turns < 1:
                continue
            # Selection is explicitly based on persisted metadata. The live
            # registry may be newer after a failed metadata write, but that
            # failure must leave the previous session eligible for retry.
            watermark = coerce_watermark(metadata.config.get(WATERMARK_KEY))
            if watermark < turns:
                eligible.append((metadata.last_accessed_at, thread_id, turns))

        if not eligible:
            return

        last_accessed_at, thread_id, turns = max(
            eligible,
            key=lambda item: (item[0], item[1]),
        )
        max_age_seconds = runtime.settings.memory_extraction_max_session_age_hours * 3600
        if time.time() - last_accessed_at > max_age_seconds:
            _runtime_watermark(runtime).set(thread_id, turns)
            return

        runtime.scheduler.submit("session_start", thread_id)
    except Exception as exc:  # noqa: BLE001 - request hooks are best effort
        logger.warning(
            "memory_extraction trigger=session_start thread_id=%s cause=%s",
            new_thread_id,
            type(exc).__name__,
        )


def after_turn(
    runtime: ExtractionRuntime | None,
    *,
    thread_id: str,
) -> None:
    """Schedule extraction when the checkpointed turn count reaches a boundary."""

    if runtime is None:
        return

    try:
        if runtime.extractor.should_extract(thread_id):
            runtime.scheduler.submit("round_complete", thread_id)
    except Exception as exc:  # noqa: BLE001 - request hooks are best effort
        logger.warning(
            "memory_extraction trigger=round_complete thread_id=%s cause=%s",
            thread_id,
            type(exc).__name__,
        )


def _runtime_watermark(runtime: ExtractionRuntime) -> WatermarkStore:
    if runtime.watermarks is not None:
        return runtime.watermarks
    assert runtime.registry is not None
    return SessionWatermarkStore(runtime.registry, runtime.storage)


def _is_valid_watermark(raw: object) -> bool:
    return (
        isinstance(raw, int)
        and not isinstance(raw, bool)
        and raw >= 0
        and coerce_watermark(raw) == raw
    )


__all__ = [
    "ExtractionRuntime",
    "SessionWatermarkStore",
    "after_turn",
    "build_extraction_runtime",
    "on_session_start",
]
