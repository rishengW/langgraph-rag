from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from src.graph.events import (
    ArtifactEvent,
    DoneEvent,
    ErrorEvent,
    GraderDecisionEvent,
    NodeEndEvent,
    NodeStartEvent,
    RetrieverResultEvent,
)
from src.graph.executor import GraphExecutor
from src.graph.metrics import MetricsCollector


class FakeGraph:
    def invoke(self, inputs, config=None):
        return {"inputs": inputs, "config": config}

    def stream(self, inputs, config=None):
        yield {"agent": {"messages": ["agent output"], "config": config}}
        yield {"generate": {"messages": ["final output"]}}


def test_graph_executor_run_delegates_to_compiled_graph():
    executor = GraphExecutor(FakeGraph())

    assert executor.run({"question": "hi"}) == {
        "inputs": {"question": "hi"},
        "config": None,
    }


def test_graph_executor_stream_wraps_outputs_as_events():
    executor = GraphExecutor(FakeGraph())

    events = list(executor.stream({"question": "hi"}, config={"thread_id": "t1"}))

    assert isinstance(events[0], NodeStartEvent)
    assert events[0].node == "agent"
    assert isinstance(events[1], NodeEndEvent)
    assert events[1].node == "agent"
    assert events[1].output == {
        "messages": ["agent output"],
        "config": {"thread_id": "t1"},
    }
    assert isinstance(events[2], NodeStartEvent)
    assert events[2].node == "generate"
    assert isinstance(events[3], NodeEndEvent)
    assert events[3].node == "generate"
    assert isinstance(events[4], DoneEvent)
    assert events[4].output == {"generate": {"messages": ["final output"]}}
    assert events[4].answer == "final output"


class TokenStreamGraph:
    """Fake graph supporting combined ``["updates", "messages"]`` streaming.

    Mimics LangGraph 0.6: in combined mode, ``stream`` yields
    ``(mode, chunk)`` tuples -- ``("updates", {node: output})`` for node
    lifecycle and ``("messages", (message_chunk, metadata))`` for LLM tokens.
    """

    def invoke(self, inputs, config=None):
        return {"messages": ["final output"]}

    def stream(self, inputs, config=None, *, stream_mode=None):
        # Tokens from an answer node (kept) interleaved with a non-answer
        # node's tokens (filtered out). The final aggregated AIMessage replay
        # must be dropped so the answer is not double-counted.
        yield ("messages", (AIMessageChunk(content="Hel"), {"langgraph_node": "web_answer"}))
        yield ("messages", (AIMessageChunk(content="lo"), {"langgraph_node": "web_answer"}))
        yield ("messages", (AIMessageChunk(content="IGNORED"), {"langgraph_node": "decompose"}))
        # Aggregated final message (plain AIMessage) — must be skipped.
        yield ("messages", (AIMessage(content="Hello"), {"langgraph_node": "web_answer"}))
        yield ("updates", {"web_answer": {"messages": [AIMessage(content="Hello")]}})


def test_graph_executor_streams_tokens_from_answer_nodes_only():
    from src.graph.events import TokenEvent

    executor = GraphExecutor(TokenStreamGraph())

    events = list(executor.stream({"question": "hi"}, stream_tokens=True))

    tokens = [event for event in events if isinstance(event, TokenEvent)]
    assert [token.token for token in tokens] == ["Hel", "lo"]
    # The decompose-node token must be filtered out of the answer stream.
    assert all(token.node == "web_answer" for token in tokens)
    # Node lifecycle + a terminal DoneEvent are still emitted.
    assert any(isinstance(event, NodeStartEvent) for event in events)
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].answer == "Hello"


class FallbackTerminalGraph:
    def invoke(self, inputs, config=None):
        return {"messages": ["fallback"]}

    def stream(self, inputs, config=None, *, stream_mode=None):
        yield (
            "updates",
            {"fallback_answer": {"messages": [AIMessage(content="fallback")]}},
        )
        raise AssertionError("terminal fallback update must close the stream")


def test_graph_executor_closes_token_stream_after_terminal_fallback_update():
    events = list(
        GraphExecutor(FallbackTerminalGraph()).stream(
            {"question": "hi"},
            stream_tokens=True,
        )
    )

    assert not any(isinstance(event, ErrorEvent) for event in events)
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].answer == "fallback"


class SummaryGraph:
    def invoke(self, inputs, config=None):
        return {"inputs": inputs, "config": config}

    def stream(self, inputs, config=None):
        yield {
            "retrieve": {
                "messages": [
                    ToolMessage(
                        content="retrieved context",
                        tool_call_id="call_1",
                        artifact=[
                            {"page_content": "a", "metadata": {"source": "https://example.com/a"}},
                            {"page_content": "b", "metadata": {"source": "https://example.com/b"}},
                        ],
                    )
                ]
            }
        }
        yield {
            "grade_documents": {
                "binary_score": "yes",
                "explanation": "matched context",
                "rewrite_count": 1,
            }
        }
        yield {"generate": {"messages": [AIMessage(content="final answer")]}}


def test_graph_executor_emits_retriever_and_grader_summaries():
    executor = GraphExecutor(SummaryGraph())

    events = list(executor.stream({"question": "hi"}))

    retriever_event = next(event for event in events if isinstance(event, RetrieverResultEvent))
    assert retriever_event.node == "retrieve"
    assert retriever_event.num_docs == 2
    assert retriever_event.sources == ["https://example.com/a", "https://example.com/b"]

    grader_event = next(event for event in events if isinstance(event, GraderDecisionEvent))
    assert grader_event.node == "grade_documents"
    assert grader_event.score == "yes"
    assert grader_event.explanation == "matched context"
    assert grader_event.rewrite_count == 1


class ErrorGraph:
    def invoke(self, inputs, config=None):
        return {"inputs": inputs, "config": config}

    def stream(self, inputs, config=None):
        yield {"agent": {"messages": [AIMessage(content="partial")]}}
        raise RuntimeError("stream failed")


def test_graph_executor_emits_error_and_done_after_stream_failure():
    executor = GraphExecutor(ErrorGraph())

    events = list(executor.stream({"question": "hi"}))

    assert events[0].node == "agent"
    assert isinstance(events[-2], ErrorEvent)
    assert events[-2].message == "stream failed"
    assert events[-2].recoverable is False
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].answer == "partial"


def test_graph_executor_records_metrics_from_emitted_events():
    clock_values = iter([1.0, 1.5, 2.0, 2.25, 3.0, 3.75])
    metrics = MetricsCollector(clock=lambda: next(clock_values))
    executor = GraphExecutor(SummaryGraph(), metrics=metrics)

    list(executor.stream({"question": "hi"}))
    snapshot = metrics.snapshot()

    assert snapshot.query_count == 1
    assert snapshot.nodes["retrieve"].call_count == 1
    assert snapshot.nodes["retrieve"].total_latency_seconds == 0.5
    assert snapshot.nodes["generate"].call_count == 1
    assert snapshot.retriever_query_count == 1
    assert snapshot.retriever_total_docs == 2
    assert snapshot.retriever_average_docs == 2
    assert snapshot.grade_distribution == {"yes": 1}
    assert snapshot.rewrite_count_distribution == {1: 1}


AMAP_MARKER_ARTIFACT = {
    "type": "amap",
    "version": 1,
    "kind": "marker",
    "coordinateSystem": "gcj02",
    "provider": "amap",
    "title": "Shanghai",
    "lng": 121.4737,
    "lat": 31.2304,
}


class ArtifactGraph:
    def invoke(self, inputs, config=None):
        return {"messages": ["final output"]}

    def stream(self, inputs, config=None):
        tool_message = ToolMessage(
            content="map result",
            tool_call_id="call-map",
            artifact=AMAP_MARKER_ARTIFACT,
        )
        # Duplicate update chunks can be replayed by callers; artifacts should
        # be emitted and accumulated only once while lifecycle behavior remains.
        yield {"tools": {"messages": [tool_message]}}
        yield {"tools": {"messages": [tool_message]}}
        yield {"generate": {"messages": [AIMessage(content="final answer")]}}


def test_graph_executor_emits_artifact_before_later_ai_answer_and_done_accumulates():
    events = list(GraphExecutor(ArtifactGraph()).stream({"question": "map"}))

    artifact_events = [event for event in events if isinstance(event, ArtifactEvent)]
    assert len(artifact_events) == 1
    assert artifact_events[0].node == "tools"
    assert len(artifact_events[0].artifacts) == 1
    artifact = artifact_events[0].artifacts[0]
    assert artifact["tool_call_id"] == "call-map"
    assert artifact["kind"] == "marker"
    assert artifact["markers"][0]["position"] == {"lat": 31.2304, "lng": 121.4737}
    assert artifact["url"].startswith("https://uri.amap.com/marker?")

    artifact_index = events.index(artifact_events[0])
    generate_start_index = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, NodeStartEvent) and event.node == "generate"
    )
    assert artifact_index < generate_start_index

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.answer == "final answer"
    assert done.artifacts == [artifact]


class TokenArtifactGraph:
    def invoke(self, inputs, config=None):
        return {"messages": ["final output"]}

    def stream(self, inputs, config=None, *, stream_mode=None):
        tool_message = ToolMessage(
            content="map result",
            tool_call_id="call-map",
            artifact=AMAP_MARKER_ARTIFACT,
        )
        yield ("updates", {"tools": {"messages": [tool_message]}})
        yield ("updates", {"tools": {"messages": [tool_message]}})
        yield ("messages", (AIMessageChunk(content="final "), {"langgraph_node": "generate"}))
        yield ("messages", (AIMessageChunk(content="answer"), {"langgraph_node": "generate"}))
        yield ("updates", {"generate": {"messages": [AIMessage(content="final answer")]}})


def test_graph_executor_token_stream_emits_and_accumulates_artifacts_once():
    from src.graph.events import TokenEvent

    events = list(GraphExecutor(TokenArtifactGraph()).stream({"question": "map"}, stream_tokens=True))

    artifact_events = [event for event in events if isinstance(event, ArtifactEvent)]
    assert len(artifact_events) == 1
    assert artifact_events[0].node == "tools"
    tokens = [event.token for event in events if isinstance(event, TokenEvent)]
    assert tokens == ["final ", "answer"]
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].artifacts == artifact_events[0].artifacts
    assert events[-1].answer == "final answer"
