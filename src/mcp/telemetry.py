"""Bounded audit and metrics seams for policy-controlled tool execution."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Literal, Protocol

ToolOutcome = Literal[
    "success",
    "denied",
    "invalid",
    "limited",
    "timeout",
    "cancelled",
    "failed",
]


@dataclass(frozen=True, slots=True)
class ToolAuditEvent:
    """Argument-free, bounded record of one tool-policy decision."""

    tool: str
    outcome: ToolOutcome
    generation: int
    duration_ms: int
    source_server: str | None = None
    principal_id: str = "anonymous"
    tenant_id: str | None = None

    def __post_init__(self) -> None:
        for value in (
            self.tool,
            self.outcome,
            self.source_server or "",
            self.principal_id,
            self.tenant_id or "",
        ):
            if len(value) > 128 or any(ord(char) < 32 for char in value):
                raise ValueError("Invalid bounded tool audit field")
        if self.generation < 1 or not 0 <= self.duration_ms <= 86_400_000:
            raise ValueError("Invalid tool audit generation or duration")


class ToolAuditSink(Protocol):
    """Receives sanitized tool audit records."""

    def emit(self, event: ToolAuditEvent) -> None: ...


class NullToolAuditSink:
    """Default sink used when no durable or logging sink is configured."""

    def emit(self, event: ToolAuditEvent) -> None:
        """Discard one already-sanitized event."""


@dataclass(frozen=True, slots=True)
class ToolMetricsSnapshot:
    """Bounded-cardinality aggregate tool metrics."""

    calls_by_outcome: dict[ToolOutcome, int]
    total_duration_ms: int
    catalog_generation: int


class ToolMetricsRecorder:
    """Thread-safe aggregate recorder without tool or identity labels."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._outcomes: Counter[ToolOutcome] = Counter()
        self._duration_ms = 0
        self._generation = 0

    def record(self, event: ToolAuditEvent) -> None:
        """Record one sanitized outcome; sink failures never affect callers."""

        with self._lock:
            self._outcomes[event.outcome] += 1
            self._duration_ms += event.duration_ms
            self._generation = max(self._generation, event.generation)

    def snapshot(self) -> ToolMetricsSnapshot:
        """Return a detached immutable aggregate snapshot."""

        with self._lock:
            return ToolMetricsSnapshot(
                calls_by_outcome=dict(self._outcomes),
                total_duration_ms=self._duration_ms,
                catalog_generation=self._generation,
            )


class CallbackToolAuditSink:
    """Small adapter for injecting an audit callback in tests or applications."""

    def __init__(self, callback: Callable[[ToolAuditEvent], None]) -> None:
        self._callback = callback

    def emit(self, event: ToolAuditEvent) -> None:
        """Forward a sanitized audit event."""

        self._callback(event)


__all__ = [
    "CallbackToolAuditSink",
    "NullToolAuditSink",
    "ToolAuditEvent",
    "ToolAuditSink",
    "ToolMetricsRecorder",
    "ToolMetricsSnapshot",
    "ToolOutcome",
]
