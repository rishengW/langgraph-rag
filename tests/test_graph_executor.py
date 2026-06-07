from __future__ import annotations

from src.graph.events import DoneEvent, NodeEndEvent
from src.graph.executor import GraphExecutor


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

    assert isinstance(events[0], NodeEndEvent)
    assert events[0].node == "agent"
    assert events[0].output == {
        "messages": ["agent output"],
        "config": {"thread_id": "t1"},
    }
    assert isinstance(events[1], NodeEndEvent)
    assert events[1].node == "generate"
    assert isinstance(events[2], DoneEvent)
    assert events[2].output == {"generate": {"messages": ["final output"]}}
