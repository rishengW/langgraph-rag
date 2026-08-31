"""Bounded, argument-free audit events for inbound MCP calls."""

from __future__ import annotations

from dataclasses import dataclass

from ...mcp.observability import (
    MCPObservability,
    ObservationEvent,
    ObservationOutcome,
    ObservationSignal,
    ObservationTransport,
    default_observability,
)


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """Transport adapter record convertible to the shared closed event schema."""

    request_id: str
    principal_id: str
    tool: str
    outcome: ObservationOutcome
    duration_ms: int
    generation: int = 1
    signal: ObservationSignal = "tool_invocation"
    transport: ObservationTransport = "internal"
    tenant_id: str | None = None
    source_server: str | None = None

    def __post_init__(self) -> None:
        self.to_observation()

    def to_observation(self) -> ObservationEvent:
        """Validate and return the transport-neutral event representation."""

        return ObservationEvent(
            signal=self.signal,
            outcome=self.outcome,
            request_id=self.request_id,
            principal_id=self.principal_id,
            tenant_id=self.tenant_id,
            tool=self.tool,
            source_server=self.source_server,
            transport=self.transport,
            duration_ms=self.duration_ms,
            generation=self.generation,
        )


def emit_audit(
    event: AuditEvent,
    observability: MCPObservability | None = None,
) -> ObservationEvent:
    """Export safe metadata; optional sink failures never alter tool results."""

    if not isinstance(event, AuditEvent):
        raise TypeError("Inbound MCP audit requires an AuditEvent")
    return (observability or default_observability()).emit(event.to_observation())


__all__ = ["AuditEvent", "emit_audit"]
