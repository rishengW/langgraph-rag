from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .events import (
    DoneEvent,
    ErrorEvent,
    GraderDecisionEvent,
    GraphEvent,
    NodeEndEvent,
    NodeStartEvent,
    RetrieverResultEvent,
)
from .metrics import MetricsCollector


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
    metrics: MetricsCollector | None = None

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
        # REFACTOR: Emit typed lifecycle events while preserving chunk streaming.
        final_output: Mapping[str, Any] | None = None
        try:
            graph_stream = (
                self.graph.stream(inputs)
                if config is None
                else self.graph.stream(inputs, config=config)
            )
            for output in graph_stream:
                if not isinstance(output, Mapping):
                    yield from self._emit(
                        ErrorEvent(
                            message="Graph stream yielded a non-mapping chunk",
                            recoverable=False,
                        )
                    )
                    continue
                final_output = output
                yield from self._events_from_chunk(output)
        except Exception as exc:
            yield from self._emit(ErrorEvent(message=str(exc), recoverable=False))

        yield from self._emit(
            DoneEvent(output=dict(final_output or {}), answer=_extract_answer(final_output))
        )

    async def astream(
        self,
        inputs: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[GraphEvent]:
        for event in self.stream(inputs, config=config):
            yield event

    def _events_from_chunk(self, output: Mapping[str, Any]) -> Iterator[GraphEvent]:
        for node, node_output in output.items():
            yield from self._emit(NodeStartEvent(node=node))
            yield from self._summary_events(node, node_output)
            yield from self._emit(NodeEndEvent(node=node, output=_as_output_dict(node_output)))

    def _summary_events(self, node: str, node_output: Any) -> Iterator[GraphEvent]:
        retriever_event = _retriever_event(node, node_output)
        if retriever_event is not None:
            yield from self._emit(retriever_event)

        grader_event = _grader_event(node, node_output)
        if grader_event is not None:
            yield from self._emit(grader_event)

    def _emit(self, event: GraphEvent) -> Iterator[GraphEvent]:
        if self.metrics is not None:
            self.metrics.record_event(event)
        yield event


def _as_output_dict(node_output: Any) -> dict[str, Any] | None:
    if node_output is None:
        return None
    if isinstance(node_output, Mapping):
        return dict(node_output)
    return {"value": node_output}


def _extract_answer(output: Mapping[str, Any] | None) -> str:
    if not output:
        return ""

    for node in ("generate", "agent"):
        answer = _answer_from_node_output(output.get(node))
        if answer:
            return answer

    for node_output in output.values():
        answer = _answer_from_node_output(node_output)
        if answer:
            return answer
    return ""


def _answer_from_node_output(node_output: Any) -> str:
    if isinstance(node_output, Mapping):
        for key in ("answer", "content"):
            value = node_output.get(key)
            if isinstance(value, str) and value.strip():
                return value
        messages = node_output.get("messages")
        return _last_message_text(messages)
    return ""


def _last_message_text(messages: Any) -> str:
    if not isinstance(messages, Iterable) or isinstance(messages, (str, bytes)):
        return ""

    values = list(messages)
    if not values:
        return ""
    return _message_text(values[-1]).strip()


def _message_text(message: Any) -> str:
    if hasattr(message, "content"):
        return str(message.content)
    if isinstance(message, (tuple, list)) and len(message) >= 2:
        return str(message[1])
    return str(message)


def _retriever_event(node: str, node_output: Any) -> RetrieverResultEvent | None:
    if "retriev" not in node.lower():
        return None

    docs = _extract_documents(node_output)
    if docs is not None:
        return RetrieverResultEvent(
            node=node,
            num_docs=len(docs),
            sources=_extract_sources(docs),
        )

    text = _node_output_text(node_output)
    if not text:
        return RetrieverResultEvent(node=node, num_docs=0)
    return RetrieverResultEvent(node=node, num_docs=1, sources=_urls_from_text(text))


def _extract_documents(node_output: Any) -> list[Any] | None:
    if not isinstance(node_output, Mapping):
        return None

    for key in ("documents", "docs"):
        value = node_output.get(key)
        if isinstance(value, list):
            return value

    messages = node_output.get("messages")
    if isinstance(messages, Iterable) and not isinstance(messages, (str, bytes)):
        return _documents_from_messages(messages)
    return None


def _documents_from_messages(messages: Iterable[Any]) -> list[Any] | None:
    docs: list[Any] = []
    for message in messages:
        artifact = getattr(message, "artifact", None)
        if isinstance(artifact, list):
            docs.extend(artifact)
    return docs or None


def _extract_sources(docs: Iterable[Any]) -> list[str]:
    sources = []
    for doc in docs:
        source = _source_from_document(doc)
        if source and source not in sources:
            sources.append(source)
    return sources


def _source_from_document(doc: Any) -> str:
    metadata = getattr(doc, "metadata", None)
    if isinstance(doc, Mapping):
        metadata = doc.get("metadata", metadata)
        for key in ("source", "url"):
            value = doc.get(key)
            if isinstance(value, str):
                return value
    if isinstance(metadata, Mapping):
        for key in ("source", "url"):
            value = metadata.get(key)
            if isinstance(value, str):
                return value
    return ""


def _node_output_text(node_output: Any) -> str:
    if isinstance(node_output, Mapping):
        messages = node_output.get("messages")
        text = _last_message_text(messages)
        if text:
            return text
        content = node_output.get("content")
        return content if isinstance(content, str) else ""
    return _message_text(node_output)


def _urls_from_text(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"https?://[^\s,;]+", text)))


def _grader_event(node: str, node_output: Any) -> GraderDecisionEvent | None:
    if not isinstance(node_output, Mapping):
        return None
    if "grade" not in node.lower() and not _has_grader_fields(node_output):
        return None

    score = _first_str(node_output, ("score", "binary_score", "decision"))
    if not score:
        return None
    return GraderDecisionEvent(
        node=node,
        score=score.strip().lower(),
        explanation=_first_str(node_output, ("explanation", "reason")),
        rewrite_count=int(node_output.get("rewrite_count", 0) or 0),
    )


def _has_grader_fields(node_output: Mapping[str, Any]) -> bool:
    return any(key in node_output for key in ("score", "binary_score", "decision"))


def _first_str(values: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = values.get(key)
        if isinstance(value, str):
            return value
    return ""


__all__ = ["GraphExecutor", "RunnableGraph"]
