from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.api.dependencies import initialize_chat_app_state, initialize_qa_app_state
from src.api.models import QueryRequest, StartChatRequest
from src.chat import api as chat_api
from src.qa import api as qa_api


def test_models_remain_importable_from_legacy_api_modules():
    assert qa_api.QueryRequest is QueryRequest
    assert chat_api.StartChatRequest is StartChatRequest


def test_state_initializers_preserve_existing_lock_and_registry(mock_settings):
    app = FastAPI()

    initialize_qa_app_state(app)
    rebuild_lock = app.state.rebuild_lock
    initialize_qa_app_state(app, settings=mock_settings, graph="graph")

    assert app.state.settings is mock_settings
    assert app.state.config is mock_settings
    assert app.state.qa_graph == "graph"
    assert app.state.rebuild_lock is rebuild_lock
    initialize_qa_app_state(app, settings=mock_settings)
    assert app.state.qa_graph == "graph"

    initialize_chat_app_state(app)
    registry = app.state.session_registry
    graph_factory_lock = app.state.chat_graph_factory_lock
    initialize_chat_app_state(app, settings=mock_settings)

    assert app.state.settings is mock_settings
    assert app.state.config is mock_settings
    assert app.state.session_registry is registry
    assert app.state.chat_graph_factory_lock is graph_factory_lock


def test_qa_app_uses_app_state_and_promotes_rebuilt_graph(monkeypatch, isolated_settings):
    initial_settings = isolated_settings(
        source_urls=["https://initial.test"],
        web_search_enabled=False,
    )
    built_graphs = []
    run_calls = []

    def fake_build_graph(settings, rebuild_vectorstore=False):
        graph = {
            "source_urls": list(settings.source_urls),
            "rebuild_vectorstore": rebuild_vectorstore,
        }
        built_graphs.append(graph)
        return graph

    def fake_run_rag_query(**kwargs):
        run_calls.append(kwargs)
        return {"answer": "qa answer", "error": None, "messages": []}

    monkeypatch.setattr(qa_api, "load_settings", lambda: initial_settings)
    monkeypatch.setattr(qa_api, "build_graph", fake_build_graph)
    monkeypatch.setattr(qa_api, "run_rag_query", fake_run_rag_query)

    app = qa_api.create_app()
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "graph_ready": True}
        response = client.post(
            "/query",
            json={
                "question": "What changed?",
                "urls": ["https://updated.test"],
                "web_search": False,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "qa answer"
    assert body["source_mode"] == "explicit"
    assert body["source_urls"] == ["https://updated.test"]
    assert built_graphs == [
        {"source_urls": ["https://initial.test"], "rebuild_vectorstore": False},
        {"source_urls": ["https://updated.test"], "rebuild_vectorstore": True},
    ]
    assert run_calls[0]["graph"] is app.state.qa_graph
    assert app.state.settings.source_urls == ["https://updated.test"]


def test_chat_app_uses_app_state_session_registry(monkeypatch, isolated_settings):
    settings = isolated_settings(
        source_urls=["https://chat-default.test"],
        web_search_enabled=False,
    )
    built = []

    class FakeGraph:
        def __init__(self):
            self.messages = []

        def get_state(self, config):
            return SimpleNamespace(values={"messages": list(self.messages)})

        def invoke(self, inputs, config):
            self.messages.extend(inputs["messages"])
            self.messages.append(AIMessage(content="chat answer"))
            return {"messages": list(self.messages)}

    def fake_build_chat_graph(session_settings, rebuild_vectorstore=False):
        built.append((session_settings, rebuild_vectorstore))
        return FakeGraph()

    monkeypatch.setattr(chat_api, "load_settings", lambda: settings)
    monkeypatch.setattr(chat_api, "build_chat_graph", fake_build_chat_graph)

    app = chat_api.create_app()
    with TestClient(app) as client:
        start = client.post("/chat", json={"web_search": False})
        assert start.status_code == 200
        thread_id = start.json()["thread_id"]

        health = client.get("/health")
        assert health.json()["sessions"] == 1

        message = client.post(f"/chat/{thread_id}/message", json={"message": "Hello"})
        assert message.status_code == 200
        assert message.json() == {
            "thread_id": thread_id,
            "answer": "chat answer",
            "error": None,
        }

        history = client.get(f"/chat/{thread_id}/history")
        assert history.status_code == 200
        assert history.json()["turns"] == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "chat answer"},
        ]

        deleted = client.delete(f"/chat/{thread_id}")
        assert deleted.json() == {"status": "deleted", "thread_id": thread_id}
        assert client.get("/health").json()["sessions"] == 0

    assert len(built) == 1
    session_settings, rebuild = built[0]
    assert session_settings.source_urls == ["https://chat-default.test"]
    assert rebuild is False
