"""Bounded, redacted observability export for MCP lifecycle and tool activity.

The module intentionally depends only on the Python standard library.  Concrete
production exporters can implement :class:`ObservationSink`; the default sinks
emit bounded JSON records to stderr-backed loggers while an in-process recorder
keeps fixed-dimension metric totals.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from threading import Lock
from types import MappingProxyType
from typing import Literal, Protocol

ObservationSignal = Literal[
    "authentication",
    "authorization",
    "catalog_publication",
    "dependency",
    "lifecycle",
    "limit_rejection",
    "policy_denial",
    "tool_invocation",
    "transport",
]
ObservationOutcome = Literal[
    "authenticated",
    "cancelled",
    "closed",
    "degraded",
    "denied",
    "failed",
    "internal_error",
    "invalid",
    "limited",
    "published",
    "ready",
    "rejected",
    "shutdown_drained",
    "shutdown_grace_expired",
    "started",
    "success",
    "timeout",
    "upstream_failure",
]
ObservationTransport = Literal[
    "http",
    "internal",
    "outbound_http",
    "outbound_stdio",
    "stdio",
]
ObservationCategory = Literal["audit", "log", "metric", "trace"]
ExporterName = Literal["audit", "log", "metric", "trace"]

MAX_OBSERVATION_FIELD_CHARS = 128
MAX_OBSERVATION_RECORD_BYTES = 4_096
MAX_REDACTION_VALUES = 128
MAX_REDACTION_VALUE_CHARS = 65_536
MAX_DURATION_MS = 86_400_000

_VALID_SIGNALS = frozenset(
    {
        "authentication",
        "authorization",
        "catalog_publication",
        "dependency",
        "lifecycle",
        "limit_rejection",
        "policy_denial",
        "tool_invocation",
        "transport",
    }
)
_VALID_OUTCOMES = frozenset(
    {
        "authenticated",
        "cancelled",
        "closed",
        "degraded",
        "denied",
        "failed",
        "internal_error",
        "invalid",
        "limited",
        "published",
        "ready",
        "rejected",
        "shutdown_drained",
        "shutdown_grace_expired",
        "started",
        "success",
        "timeout",
        "upstream_failure",
    }
)
_VALID_TRANSPORTS = frozenset({"http", "internal", "outbound_http", "outbound_stdio", "stdio"})
_VALID_CATEGORIES = frozenset({"audit", "log", "metric", "trace"})

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_ -]?key|authorization|bearer|access[_ -]?token|"
    r"refresh[_ -]?token|client[_ -]?secret|password)\b"
    r"(\s*[\"']?\s*[:=]\s*[\"']?|\s+)(?:bearer\s+)?[^\s,;\"'&]+"
)
_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
_URL_CREDENTIALS = re.compile(r"(?i)(https?://)[^/@\s]+@")
_SECRET_QUERY = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"client[_-]?secret|password)=)[^&#\s]+"
)


@dataclass(frozen=True, slots=True)
class ObservationEvent:
    """One closed, argument-free event safe for every observability channel."""

    signal: ObservationSignal
    outcome: ObservationOutcome
    request_id: str
    principal_id: str
    transport: ObservationTransport = "internal"
    duration_ms: int = 0
    generation: int = 0
    tenant_id: str | None = None
    tool: str | None = None
    source_server: str | None = None

    def __post_init__(self) -> None:
        if self.signal not in _VALID_SIGNALS:
            raise ValueError("Invalid observation signal")
        if self.outcome not in _VALID_OUTCOMES:
            raise ValueError("Invalid observation outcome")
        if self.transport not in _VALID_TRANSPORTS:
            raise ValueError("Invalid observation transport")
        for name, value, required in (
            ("request_id", self.request_id, True),
            ("principal_id", self.principal_id, True),
            ("tenant_id", self.tenant_id, False),
            ("tool", self.tool, False),
            ("source_server", self.source_server, False),
        ):
            _validate_field(name, value, required=required)
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, int)
            or not 0 <= self.duration_ms <= MAX_DURATION_MS
        ):
            raise ValueError("Invalid observation duration")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("Invalid observation generation")

    def to_record(self, category: ObservationCategory) -> dict[str, str | int | None]:
        """Return a closed channel record with fixed metric dimensions.

        Metric exports deliberately omit request, identity, tool, and provider
        fields. Those values remain available to bounded logs, traces, and
        audits for correlation, but they must never become metric labels.
        """

        if category not in _VALID_CATEGORIES:
            raise ValueError("Invalid observation category")
        common: dict[str, str | int | None] = {
            "schema_version": 1,
            "category": category,
            "signal": self.signal,
            "outcome": self.outcome,
            "transport": self.transport,
            "duration_ms": self.duration_ms,
            "generation": self.generation,
        }
        if category == "metric":
            return common
        return {
            **common,
            "request_id": self.request_id,
            "principal_id": self.principal_id,
            "tenant_id": self.tenant_id,
            "tool": self.tool,
            "source_server": self.source_server,
        }


class ObservationSink(Protocol):
    """Export one already-sanitized observation."""

    def emit(self, event: ObservationEvent) -> None: ...


class NullObservationSink:
    """Explicitly discard one observability channel."""

    def emit(self, event: ObservationEvent) -> None:
        """Discard one already-sanitized event."""


class CallbackObservationSink:
    """Adapt a synchronous callback to the exporter protocol."""

    def __init__(self, callback: Callable[[ObservationEvent], None]) -> None:
        if not callable(callback):
            raise TypeError("Observation callback must be callable")
        self._callback = callback

    def emit(self, event: ObservationEvent) -> None:
        """Forward one sanitized event."""

        self._callback(event)


class JsonLoggingSink:
    """Export one bounded JSON object through a dedicated logger."""

    def __init__(self, logger_name: str, category: ObservationCategory) -> None:
        if not logger_name or len(logger_name) > MAX_OBSERVATION_FIELD_CHARS:
            raise ValueError("Observation logger name is invalid")
        if category not in _VALID_CATEGORIES:
            raise ValueError("Observation logger category is invalid")
        self._logger = logging.getLogger(logger_name)
        self._category = category

    def emit(self, event: ObservationEvent) -> None:
        """Serialize with stable keys and enforce the final byte ceiling."""

        serialized = json.dumps(
            event.to_record(self._category),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(serialized.encode("utf-8")) > MAX_OBSERVATION_RECORD_BYTES:
            raise ValueError("Observation record exceeds its byte limit")
        self._logger.info("%s", serialized)


@dataclass(frozen=True, slots=True)
class ObservationMetricsSnapshot:
    """Fixed-dimension metrics; no identity, URL, request, or tool labels."""

    total_events: int
    total_duration_ms: int
    latest_generation: int
    events_by_signal: Mapping[str, int]
    events_by_outcome: Mapping[str, int]
    events_by_transport: Mapping[str, int]


class BoundedMetricsRecorder:
    """Record only values from the closed signal/outcome/transport domains."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._total_events = 0
        self._total_duration_ms = 0
        self._latest_generation = 0
        self._signals: Counter[str] = Counter()
        self._outcomes: Counter[str] = Counter()
        self._transports: Counter[str] = Counter()

    def record(self, event: ObservationEvent) -> None:
        """Add one event without retaining any high-cardinality fields."""

        with self._lock:
            self._total_events += 1
            self._total_duration_ms += event.duration_ms
            self._latest_generation = max(self._latest_generation, event.generation)
            self._signals[event.signal] += 1
            self._outcomes[event.outcome] += 1
            self._transports[event.transport] += 1

    def snapshot(self) -> ObservationMetricsSnapshot:
        """Return detached immutable metric mappings."""

        with self._lock:
            return ObservationMetricsSnapshot(
                total_events=self._total_events,
                total_duration_ms=self._total_duration_ms,
                latest_generation=self._latest_generation,
                events_by_signal=MappingProxyType(dict(self._signals)),
                events_by_outcome=MappingProxyType(dict(self._outcomes)),
                events_by_transport=MappingProxyType(dict(self._transports)),
            )


@dataclass(frozen=True, slots=True)
class ObservabilitySnapshot:
    """Safe administrative snapshot of metrics and exporter health."""

    metrics: ObservationMetricsSnapshot
    export_failures: Mapping[str, int]


class RequiredAuditDeliveryError(RuntimeError):
    """Fixed public-safe failure raised only by explicit durability policy."""

    def __init__(self) -> None:
        super().__init__("Required audit delivery failed")


class MCPObservability:
    """Fan out sanitized events without coupling client success to optional sinks."""

    def __init__(
        self,
        *,
        log_sink: ObservationSink | None = None,
        metric_sink: ObservationSink | None = None,
        trace_sink: ObservationSink | None = None,
        audit_sink: ObservationSink | None = None,
        redaction_values: Sequence[str] = (),
        require_audit_delivery: bool = False,
    ) -> None:
        if not isinstance(require_audit_delivery, bool):
            raise TypeError("Audit delivery policy must be boolean")
        self._log_sink = log_sink or JsonLoggingSink("mcp.events", "log")
        self._metric_sink = metric_sink or JsonLoggingSink("mcp.metrics", "metric")
        self._trace_sink = trace_sink or JsonLoggingSink("mcp.traces", "trace")
        self._audit_sink = audit_sink or JsonLoggingSink("mcp.audit", "audit")
        self._redaction_values = _validated_redaction_values(redaction_values)
        self._require_audit_delivery = require_audit_delivery
        self._metrics = BoundedMetricsRecorder()
        self._failure_lock = Lock()
        self._export_failures: Counter[str] = Counter()

    @property
    def require_audit_delivery(self) -> bool:
        """Return whether audit failure is allowed to fail a protected operation."""

        return self._require_audit_delivery

    def emit(self, event: ObservationEvent) -> ObservationEvent:
        """Export one event; only explicit required audit durability may raise."""

        if not isinstance(event, ObservationEvent):
            raise TypeError("MCP observability accepts only ObservationEvent values")
        sanitized = _redact_event(event, self._redaction_values)
        self._metrics.record(sanitized)

        audit_failed = False
        exporters: tuple[tuple[ExporterName, ObservationSink], ...] = (
            ("log", self._log_sink),
            ("metric", self._metric_sink),
            ("trace", self._trace_sink),
            ("audit", self._audit_sink),
        )
        for name, sink in exporters:
            try:
                sink.emit(sanitized)
            except Exception:
                self._record_export_failure(name)
                audit_failed = audit_failed or name == "audit"

        if audit_failed and self._require_audit_delivery:
            raise RequiredAuditDeliveryError from None
        return sanitized

    def snapshot(self) -> ObservabilitySnapshot:
        """Expose fixed-cardinality totals without exporter exception details."""

        with self._failure_lock:
            failures = MappingProxyType(dict(self._export_failures))
        return ObservabilitySnapshot(metrics=self._metrics.snapshot(), export_failures=failures)

    async def aclose(self) -> None:
        """Close owned exporters in dependency order without coupling failures."""

        exporters: tuple[tuple[ExporterName, ObservationSink], ...] = (
            ("log", self._log_sink),
            ("metric", self._metric_sink),
            ("trace", self._trace_sink),
            ("audit", self._audit_sink),
        )
        closed_ids: set[int] = set()
        for name, sink in exporters:
            if id(sink) in closed_ids:
                continue
            closed_ids.add(id(sink))
            method = getattr(sink, "aclose", None)
            if not callable(method):
                method = getattr(sink, "close", None)
            if not callable(method):
                method = getattr(sink, "shutdown", None)
            if not callable(method):
                continue
            try:
                if inspect.iscoroutinefunction(method):
                    await method()
                else:
                    result = await asyncio.to_thread(method)
                    if inspect.isawaitable(result):
                        await result
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                raise
            except Exception:
                self._record_export_failure(name)

    def _record_export_failure(self, name: ExporterName) -> None:
        with self._failure_lock:
            self._export_failures[name] += 1


_DEFAULT_OBSERVABILITY: MCPObservability | None = None
_DEFAULT_OBSERVABILITY_LOCK = Lock()


def default_observability() -> MCPObservability:
    """Return the lazily initialized process-wide fallback exporter."""

    global _DEFAULT_OBSERVABILITY
    current = _DEFAULT_OBSERVABILITY
    if current is not None:
        return current
    with _DEFAULT_OBSERVABILITY_LOCK:
        current = _DEFAULT_OBSERVABILITY
        if current is None:
            current = MCPObservability()
            _DEFAULT_OBSERVABILITY = current
        return current


def _validated_redaction_values(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str | bytes):
        raise TypeError("Observation redaction values must be a sequence of strings")
    unique: dict[str, None] = {}
    for value in values:
        if not isinstance(value, str):
            raise TypeError("Observation redaction values must contain strings")
        if not value:
            continue
        if len(value) > MAX_REDACTION_VALUE_CHARS:
            raise ValueError("Observation redaction value exceeds its bound")
        unique[value] = None
        if len(unique) > MAX_REDACTION_VALUES:
            raise ValueError("Observation redaction values exceed their count bound")
    return tuple(sorted(unique, key=len, reverse=True))


def _redact_event(event: ObservationEvent, secrets: tuple[str, ...]) -> ObservationEvent:
    return replace(
        event,
        request_id=_redact_text(event.request_id, secrets),
        principal_id=_redact_text(event.principal_id, secrets),
        tenant_id=_redact_optional(event.tenant_id, secrets),
        tool=_redact_optional(event.tool, secrets),
        source_server=_redact_optional(event.source_server, secrets),
    )


def _redact_optional(value: str | None, secrets: tuple[str, ...]) -> str | None:
    return None if value is None else _redact_text(value, secrets)


def _redact_text(value: str, secrets: tuple[str, ...]) -> str:
    redacted = value
    for secret in secrets:
        redacted = redacted.replace(secret, "[REDACTED]")
    redacted = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", redacted)
    redacted = _BEARER_VALUE.sub("Bearer [REDACTED]", redacted)
    redacted = _URL_CREDENTIALS.sub(r"\1[REDACTED]@", redacted)
    redacted = _SECRET_QUERY.sub(lambda match: f"{match.group(1)}[REDACTED]", redacted)
    return redacted[:MAX_OBSERVATION_FIELD_CHARS]


def _validate_field(name: str, value: str | None, *, required: bool) -> None:
    if value is None:
        if required:
            raise ValueError(f"Observation {name} is required")
        return
    if (
        not isinstance(value, str)
        or (required and not value)
        or len(value) > MAX_OBSERVATION_FIELD_CHARS
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"Invalid bounded observation field: {name}")


__all__ = [
    "BoundedMetricsRecorder",
    "CallbackObservationSink",
    "JsonLoggingSink",
    "MAX_OBSERVATION_FIELD_CHARS",
    "MAX_OBSERVATION_RECORD_BYTES",
    "MCPObservability",
    "NullObservationSink",
    "ObservationCategory",
    "ObservationEvent",
    "ObservationMetricsSnapshot",
    "ObservationOutcome",
    "ObservationSignal",
    "ObservationSink",
    "ObservationTransport",
    "ObservabilitySnapshot",
    "RequiredAuditDeliveryError",
    "default_observability",
]
