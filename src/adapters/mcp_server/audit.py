"""Bounded, argument-free audit events for inbound MCP calls."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

logger = logging.getLogger("mcp.audit")


@dataclass(frozen=True, slots=True)
class AuditEvent:
    request_id: str
    principal_id: str
    tool: str
    outcome: str
    duration_ms: int
    generation: int = 1

    def __post_init__(self) -> None:
        for name in ("request_id", "principal_id", "tool", "outcome"):
            value = getattr(self, name)
            if len(value) > 128 or any(ord(char) < 32 for char in value):
                raise ValueError(f"Invalid bounded audit field: {name}")
        if not 0 <= self.duration_ms <= 86_400_000:
            raise ValueError("Invalid audit duration")


def emit_audit(event: AuditEvent) -> None:
    """Emit safe metadata only; audit sink failures never alter a tool result."""

    try:
        logger.info("mcp_audit %s", json.dumps(asdict(event), sort_keys=True))
    except Exception:
        logger.error("mcp_audit_sink_failure", exc_info=False)


__all__ = ["AuditEvent", "emit_audit"]
