# REFACTOR: Metrics groundwork for typed graph event streams.
"""Collect in-process metrics from typed graph execution events."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter

from .events import (
    DoneEvent,
    ErrorEvent,
    GraphEvent,
    GraderDecisionEvent,
    NodeEndEvent,
    NodeStartEvent,
    RetrieverResultEvent,
    TokenEvent,
)


@dataclass(frozen=True)
class NodeMetricsSnapshot:
    call_count: int = 0
    error_count: int = 0
    total_latency_seconds: float = 0.0

    @property
    def average_latency_seconds(self) -> float:
        """Return average observed node latency."""

        if self.call_count == 0:
            return 0.0
        return self.total_latency_seconds / self.call_count


@dataclass(frozen=True)
class MetricsSnapshot:
    nodes: dict[str, NodeMetricsSnapshot] = field(default_factory=dict)
    error_count: int = 0
    query_count: int = 0
    token_count: int = 0
    retriever_query_count: int = 0
    retriever_total_docs: int = 0
    grade_distribution: dict[str, int] = field(default_factory=dict)
    rewrite_count_distribution: dict[int, int] = field(default_factory=dict)

    @property
    def retriever_average_docs(self) -> float:
        """Return average retrieved document count."""

        if self.retriever_query_count == 0:
            return 0.0
        return self.retriever_total_docs / self.retriever_query_count


class MetricsCollector:
    """Record lightweight graph metrics from emitted GraphEvent instances."""

    def __init__(self, clock: Callable[[], float] = perf_counter) -> None:
        self._clock = clock
        self._node_starts: dict[str, float] = {}
        self._nodes: dict[str, NodeMetricsSnapshot] = {}
        self._error_count = 0
        self._query_count = 0
        self._token_count = 0
        self._retriever_query_count = 0
        self._retriever_total_docs = 0
        self._grade_distribution: Counter[str] = Counter()
        self._rewrite_count_distribution: Counter[int] = Counter()

    def record_event(self, event: GraphEvent) -> None:
        """Record metrics for one graph event."""

        if isinstance(event, NodeStartEvent):
            self._node_starts[event.node] = self._clock()
        elif isinstance(event, NodeEndEvent):
            self._record_node_end(event)
        elif isinstance(event, ErrorEvent):
            self._record_error(event)
        elif isinstance(event, DoneEvent):
            self._query_count += 1
        elif isinstance(event, TokenEvent):
            self._token_count += 1
        elif isinstance(event, RetrieverResultEvent):
            self._retriever_query_count += 1
            self._retriever_total_docs += event.num_docs
        elif isinstance(event, GraderDecisionEvent):
            self._grade_distribution[event.score] += 1
            self._rewrite_count_distribution[event.rewrite_count] += 1

    def snapshot(self) -> MetricsSnapshot:
        """Return an immutable metrics snapshot for API or test consumers."""

        return MetricsSnapshot(
            nodes=dict(self._nodes),
            error_count=self._error_count,
            query_count=self._query_count,
            token_count=self._token_count,
            retriever_query_count=self._retriever_query_count,
            retriever_total_docs=self._retriever_total_docs,
            grade_distribution=dict(self._grade_distribution),
            rewrite_count_distribution=dict(self._rewrite_count_distribution),
        )

    def _record_node_end(self, event: NodeEndEvent) -> None:
        start = self._node_starts.pop(event.node, None)
        if start is None:
            start = self._clock()
        elapsed = max(0.0, self._clock() - start)
        previous = self._nodes.get(event.node, NodeMetricsSnapshot())
        self._nodes[event.node] = NodeMetricsSnapshot(
            call_count=previous.call_count + 1,
            error_count=previous.error_count,
            total_latency_seconds=previous.total_latency_seconds + elapsed,
        )

    def _record_error(self, event: ErrorEvent) -> None:
        self._error_count += 1
        if event.node is None:
            return
        previous = self._nodes.get(event.node, NodeMetricsSnapshot())
        self._nodes[event.node] = NodeMetricsSnapshot(
            call_count=previous.call_count,
            error_count=previous.error_count + 1,
            total_latency_seconds=previous.total_latency_seconds,
        )


__all__ = ["MetricsCollector", "MetricsSnapshot", "NodeMetricsSnapshot"]
