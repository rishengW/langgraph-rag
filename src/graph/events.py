from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ToolEventOutcome = Literal[
    "started",
    "success",
    "denied",
    "invalid",
    "limited",
    "timeout",
    "cancelled",
    "failed",
]


@dataclass(frozen=True)
class NodeStartEvent:
    node: str
    type: Literal["node_start"] = "node_start"


@dataclass(frozen=True)
class NodeEndEvent:
    node: str
    output: dict[str, Any] | None = None
    type: Literal["node_end"] = "node_end"


@dataclass(frozen=True)
class ToolStartEvent:
    tool: str
    tool_call_id: str
    node: str
    source_server: str | None = None
    catalog_generation: int = 0
    duration_ms: int = 0
    outcome: ToolEventOutcome = "started"
    type: Literal["tool_start"] = "tool_start"

    def __post_init__(self) -> None:
        _validate_tool_metadata(self)


@dataclass(frozen=True)
class ToolEndEvent:
    tool: str
    tool_call_id: str
    node: str
    source_server: str | None = None
    catalog_generation: int = 0
    duration_ms: int = 0
    outcome: ToolEventOutcome = "success"
    type: Literal["tool_end"] = "tool_end"

    def __post_init__(self) -> None:
        _validate_tool_metadata(self)


@dataclass(frozen=True)
class TokenEvent:
    token: str
    node: str | None = None
    type: Literal["token"] = "token"


# REFACTOR: Carry source node metadata for chunk-derived summary events.
@dataclass(frozen=True)
class RetrieverResultEvent:
    num_docs: int
    sources: list[str] = field(default_factory=list)
    node: str = "retrieve"
    type: Literal["retriever_result"] = "retriever_result"


@dataclass(frozen=True)
class GraderDecisionEvent:
    score: str
    explanation: str = ""
    rewrite_count: int = 0
    node: str = "grade_documents"
    type: Literal["grader_decision"] = "grader_decision"


@dataclass(frozen=True)
class ArtifactEvent:
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    node: str | None = None
    type: Literal["artifact"] = "artifact"


@dataclass(frozen=True)
class ErrorEvent:
    message: str
    recoverable: bool = True
    node: str | None = None
    type: Literal["error"] = "error"


@dataclass(frozen=True)
class DoneEvent:
    output: dict[str, Any] | None = None
    answer: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    type: Literal["done"] = "done"


def _validate_tool_metadata(event: ToolStartEvent | ToolEndEvent) -> None:
    for value in (event.tool, event.tool_call_id, event.node, event.source_server or ""):
        if len(value) > 128 or any(ord(char) < 32 for char in value):
            raise ValueError("Invalid bounded tool event metadata")
    if not 0 <= event.catalog_generation <= 2_147_483_647:
        raise ValueError("Invalid tool event catalog generation")
    if not 0 <= event.duration_ms <= 86_400_000:
        raise ValueError("Invalid tool event duration")


GraphEvent = (
    NodeStartEvent
    | NodeEndEvent
    | ToolStartEvent
    | ToolEndEvent
    | TokenEvent
    | RetrieverResultEvent
    | GraderDecisionEvent
    | ArtifactEvent
    | ErrorEvent
    | DoneEvent
)


__all__ = [
    "ArtifactEvent",
    "DoneEvent",
    "ErrorEvent",
    "GraphEvent",
    "GraderDecisionEvent",
    "NodeEndEvent",
    "NodeStartEvent",
    "RetrieverResultEvent",
    "TokenEvent",
    "ToolEndEvent",
    "ToolEventOutcome",
    "ToolStartEvent",
]
