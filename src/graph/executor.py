from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ..llm.sanitize import CitationArtifactFilter
from .artifacts import extract_artifacts_from_node_output
from .events import (
    ArtifactEvent,
    DoneEvent,
    ErrorEvent,
    GraderDecisionEvent,
    GraphEvent,
    NodeEndEvent,
    NodeStartEvent,
    RetrieverResultEvent,
    TokenEvent,
)
from .metrics import MetricsCollector

# REFACTOR: Nodes whose LLM output is the user-facing answer. Token deltas are
# streamed only from these so internal structured-output calls (decompose,
# expand, grade_documents, condense, rewrite) do not leak into the visible
# answer stream. ``agent`` is included because in the lightweight chat graph
# the agent can answer directly without a tool call.
ANSWER_NODES = frozenset({"generate", "web_answer", "agent"})
TERMINAL_UPDATE_NODES = frozenset({"fallback_answer"})


class RunnableGraph(Protocol):
    def invoke(self, inputs: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> Any:
        ...

    def stream(
        self,
        inputs: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
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
        *,
        stream_tokens: bool = False,
        token_nodes: frozenset[str] = ANSWER_NODES,
    ) -> Iterator[GraphEvent]:
        # REFACTOR: Emit typed lifecycle events while preserving chunk streaming.
        # When ``stream_tokens`` is True, also emit per-token ``TokenEvent``s
        # from the answer-producing nodes via LangGraph's combined stream modes
        # (``updates`` for node lifecycle + ``messages`` for LLM token chunks).
        if stream_tokens:
            yield from self._stream_with_tokens(inputs, config, token_nodes)
            return

        final_output: Mapping[str, Any] | None = None
        artifacts: list[dict[str, Any]] = []
        seen_artifact_ids: set[str] = set()
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
                yield from self._artifact_events_from_chunk(
                    output,
                    artifacts,
                    seen_artifact_ids,
                )
        except Exception as exc:
            yield from self._emit(ErrorEvent(message=str(exc), recoverable=False))

        yield from self._emit(
            DoneEvent(
                output=dict(final_output or {}),
                answer=_extract_answer(final_output),
                artifacts=list(artifacts),
            )
        )

    def _stream_with_tokens(
        self,
        inputs: Mapping[str, Any],
        config: Mapping[str, Any] | None,
        token_nodes: frozenset[str],
    ) -> Iterator[GraphEvent]:
        """Stream node lifecycle events plus per-token deltas.

        Uses ``stream_mode=["updates", "messages"]`` so the same pass yields
        node-update chunks (for the existing typed lifecycle/summary events and
        the final answer) and LLM message chunks (for token deltas). Token
        deltas are emitted only for nodes in ``token_nodes`` so internal
        structured-output calls (decompose, expand, grading) do not leak into
        the user-visible answer stream.
        """

        final_output: Mapping[str, Any] | None = None
        artifacts: list[dict[str, Any]] = []
        seen_artifact_ids: set[str] = set()
        # One filter per streaming node: a fabricated citation marker can be
        # split across token chunks, so partial markers are buffered until they
        # either complete (and are dropped) or are ruled out (and released).
        token_filters: dict[str, CitationArtifactFilter] = {}
        try:
            stream_kwargs: dict[str, Any] = {"stream_mode": ["updates", "messages"]}
            if config is not None:
                stream_kwargs["config"] = config
            for mode, chunk in self.graph.stream(inputs, **stream_kwargs):
                if mode == "messages":
                    yield from self._token_events_from_messages(
                        chunk,
                        token_nodes,
                        token_filters,
                    )
                    continue
                yield from self._flush_token_filters(token_filters)
                if not isinstance(chunk, Mapping):
                    yield from self._emit(
                        ErrorEvent(
                            message="Graph stream yielded a non-mapping chunk",
                            recoverable=False,
                        )
                    )
                    continue
                final_output = chunk
                yield from self._events_from_chunk(chunk)
                yield from self._artifact_events_from_chunk(
                    chunk,
                    artifacts,
                    seen_artifact_ids,
                )
                if TERMINAL_UPDATE_NODES.intersection(chunk):
                    break
        except Exception as exc:
            yield from self._emit(ErrorEvent(message=str(exc), recoverable=False))

        yield from self._flush_token_filters(token_filters)
        yield from self._emit(
            DoneEvent(
                output=dict(final_output or {}),
                answer=_extract_answer(final_output),
                artifacts=list(artifacts),
            )
        )

    def _flush_token_filters(
        self,
        token_filters: dict[str, CitationArtifactFilter],
    ) -> Iterator[GraphEvent]:
        """Release any text still buffered by the per-node token filters."""

        for node, token_filter in list(token_filters.items()):
            remaining = token_filter.flush()
            if remaining:
                yield from self._emit(TokenEvent(token=remaining, node=node or None))
        token_filters.clear()

    def _token_events_from_messages(
        self,
        chunk: Any,
        token_nodes: frozenset[str],
        token_filters: dict[str, CitationArtifactFilter] | None = None,
    ) -> Iterator[GraphEvent]:
        """Convert a ``messages``-mode chunk into ``TokenEvent``s.

        LangGraph yields ``(message_chunk, metadata)`` tuples in messages mode.
        Only non-empty text deltas from a node in ``token_nodes`` are emitted.

        Genuine streaming deltas arrive as ``AIMessageChunk`` objects. A node
        whose return value is an ``AIMessage`` (e.g. the ``generate`` chain's
        final aggregated output, or a node that returns a whole message without
        streaming) is ALSO surfaced in messages mode as a plain ``AIMessage``
        carrying the full text. Emitting both double-prints the answer, so we
        only stream ``AIMessageChunk`` deltas and drop the aggregated final
        ``AIMessage`` replay.
        """

        if not isinstance(chunk, tuple) or len(chunk) != 2:
            return
        message_chunk, metadata = chunk
        if not _is_streaming_chunk(message_chunk):
            return
        node = ""
        if isinstance(metadata, Mapping):
            node = str(metadata.get("langgraph_node", "") or "")
        if token_nodes and node and node not in token_nodes:
            return
        text = _message_text(message_chunk)
        if not text:
            return
        if token_filters is not None:
            token_filter = token_filters.setdefault(node, CitationArtifactFilter())
            text = token_filter.feed(text)
            if not text:
                return
        yield from self._emit(TokenEvent(token=text, node=node or None))

    async def astream(
        self,
        inputs: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
        *,
        stream_tokens: bool = False,
        token_nodes: frozenset[str] = ANSWER_NODES,
    ) -> AsyncIterator[GraphEvent]:
        for event in self.stream(
            inputs,
            config=config,
            stream_tokens=stream_tokens,
            token_nodes=token_nodes,
        ):
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

    def _artifact_events_from_chunk(
        self,
        output: Mapping[str, Any],
        artifacts: list[dict[str, Any]],
        seen_artifact_ids: set[str],
    ) -> Iterator[GraphEvent]:
        for node, node_output in output.items():
            for artifact in extract_artifacts_from_node_output(node_output):
                artifact_id = str(artifact.get("id") or "")
                if not artifact_id or artifact_id in seen_artifact_ids:
                    continue
                seen_artifact_ids.add(artifact_id)
                artifacts.append(artifact)
                yield from self._emit(ArtifactEvent(artifacts=[artifact], node=node))

    def _emit(self, event: GraphEvent) -> Iterator[GraphEvent]:
        if self.metrics is not None:
            self.metrics.record_event(event)
        yield event


def _is_streaming_chunk(message: Any) -> bool:
    """Return True only for genuine streaming delta chunks.

    LangChain emits incremental tokens as ``AIMessageChunk`` instances. The
    aggregated final message a node returns is a plain ``AIMessage`` (or other
    non-chunk message), which in messages mode replays the full text. We treat
    only ``*Chunk`` message classes as streamable so the final replay does not
    double the answer. The check is name-based to avoid a hard import and to
    cover ``AIMessageChunk`` / ``BaseMessageChunk`` subclasses uniformly.
    """

    if message is None:
        return False
    return any(klass.__name__.endswith("Chunk") for klass in type(message).__mro__)


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


__all__ = ["ANSWER_NODES", "GraphExecutor", "RunnableGraph"]
