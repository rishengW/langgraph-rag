from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from src.graph.events import (
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
