from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .events import DoneEvent, GraphEvent, NodeEndEvent


class RunnableGraph(Protocol):
    def invoke(self, inputs: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> Any:
        ...

    def stream(
        self,
        inputs: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class GraphExecutor:
    """Small typed wrapper around a compiled graph.

    Existing callers still use compiled LangGraph objects directly. This class
    is an additive abstraction for code that wants typed graph events.
    """

    graph: RunnableGraph

    def run(
        self,
        inputs: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
    ) -> Any:
        if config is None:
            return self.graph.invoke(inputs)
        return self.graph.invoke(inputs, config=config)

    def stream(
        self,
        inputs: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
    ) -> Iterator[GraphEvent]:
        final_output: dict[str, Any] | None = None
        graph_stream = (
            self.graph.stream(inputs)
            if config is None
            else self.graph.stream(inputs, config=config)
        )

        for output in graph_stream:
            final_output = output
            for node, node_output in output.items():
                yield NodeEndEvent(node=node, output=node_output)

        yield DoneEvent(output=final_output)

    async def astream(
        self,
        inputs: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[GraphEvent]:
        for event in self.stream(inputs, config=config):
            yield event


__all__ = ["GraphExecutor", "RunnableGraph"]
