from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


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


GraphEvent = (
    NodeStartEvent
    | NodeEndEvent
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
]
