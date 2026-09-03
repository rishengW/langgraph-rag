from __future__ import annotations

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from src.chat import api as chat_api


class FakeStreamGraph:
    def __init__(self) -> None:
        self.messages = []

    def get_state(self, config):
        return type("Snapshot", (), {"values": {"messages": list(self.messages)}})()

    def invoke(self, inputs, config=None):
        self.messages.extend(inputs["messages"])
        self.messages.append(AIMessage(content="chat answer"))
        return {"messages": list(self.messages)}

    def stream(self, inputs, config=None, *, stream_mode=None):
        self.messages.extend(inputs["messages"])
        self.messages.append(AIMessage(content="stream answer"))
        if stream_mode and "messages" in stream_mode:
            # Combined ["updates", "messages"] mode: yield (mode, chunk) tuples.
            yield ("messages", (AIMessageChunk(content="stream "), {"langgraph_node": "generate"}))
            yield ("messages", (AIMessageChunk(content="answer"), {"langgraph_node": "generate"}))
            yield ("updates", {"generate": {"messages": [AIMessage(content="stream answer")]}})
            return
        yield {"generate": {"messages": [AIMessage(content="stream answer")]}}


def test_chat_stream_endpoint_emits_sse_and_records_metrics(monkeypatch, isolated_settings):
    settings = isolated_settings(web_search_enabled=False)

    monkeypatch.setattr(chat_api, "load_settings", lambda: settings)
    monkeypatch.setattr(
        chat_api,
        "build_chat_graph",
        lambda session_settings, rebuild_vectorstore=False: FakeStreamGraph(),
    )

    app = chat_api.create_app()
    with TestClient(app) as client:
        start = client.post("/chat", json={"web_search": False})
        thread_id = start.json()["thread_id"]
        response = client.post(
            f"/chat/{thread_id}/message/stream",
            json={"message": "Hello"},
        )
        metrics = client.get("/metrics")

    assert response.status_code == 200
    assert "event: node_start" in response.text
    assert "event: done" in response.text
    assert "stream answer" in response.text
    # Token streaming is on by default: per-token deltas must be emitted.
    assert "event: token" in response.text
    assert metrics.json()["query_count"] == 1
    assert metrics.json()["nodes"]["generate"]["call_count"] == 1


def test_chat_stream_endpoint_tokens_disabled_falls_back_to_updates(
    monkeypatch, isolated_settings
):
    settings = isolated_settings(web_search_enabled=False)

    monkeypatch.setattr(chat_api, "load_settings", lambda: settings)
    monkeypatch.setattr(
        chat_api,
        "build_chat_graph",
        lambda session_settings, rebuild_vectorstore=False: FakeStreamGraph(),
    )

    app = chat_api.create_app()
    with TestClient(app) as client:
        start = client.post("/chat", json={"web_search": False})
        thread_id = start.json()["thread_id"]
        response = client.post(
            f"/chat/{thread_id}/message/stream?tokens=false",
            json={"message": "Hello"},
        )

    assert response.status_code == 200
    assert "event: node_start" in response.text
    assert "event: done" in response.text
    assert "stream answer" in response.text
    # With tokens disabled, no per-token deltas are emitted.
    assert "event: token" not in response.text
