"""SSE serialization helpers for typed graph events."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from ..graph.events import GraphEvent


def event_payload(event: GraphEvent) -> dict[str, Any]:
    """Return a JSON-serializable event payload."""

    if is_dataclass(event):
        return asdict(event)
    return {"type": getattr(event, "type", "message"), "value": str(event)}


def format_sse(event: GraphEvent) -> str:
    """Serialize a typed graph event as a Server-Sent Event frame."""

    payload = event_payload(event)
    event_type = str(payload.get("type", "message"))
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {data}\n\n"


__all__ = ["event_payload", "format_sse"]
