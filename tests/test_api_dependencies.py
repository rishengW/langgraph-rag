from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.api.dependencies import initialize_chat_app_state
from src.chat import api as chat_api


def test_state_initializers_preserve_existing_lock_and_registry(mock_settings):
    app = FastAPI()

    initialize_chat_app_state(app)
    registry = app.state.session_registry
    graph_factory_lock = app.state.chat_graph_factory_lock
    initialize_chat_app_state(app, settings=mock_settings)

    assert app.state.settings is mock_settings
    assert app.state.config is mock_settings
    assert app.state.session_registry is registry
    assert app.state.chat_graph_factory_lock is graph_factory_lock


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
        assert health.json() == {"status": "ok"}
        assert len(app.state.session_registry) == 1

        message = client.post(f"/chat/{thread_id}/message", json={"message": "Hello"})
        assert message.status_code == 200
        assert message.json() == {
            "thread_id": thread_id,
            "answer": "chat answer",
            "error": None,
            "artifacts": [],
        }

        history = client.get(f"/chat/{thread_id}/history")
        assert history.status_code == 200
        assert history.json()["turns"] == [
            {"role": "user", "content": "Hello", "artifacts": []},
            {"role": "assistant", "content": "chat answer", "artifacts": []},
        ]

        deleted = client.delete(f"/chat/{thread_id}")
        assert deleted.json() == {"status": "deleted", "thread_id": thread_id}
        assert client.get("/health").json() == {"status": "ok"}
        assert len(app.state.session_registry) == 0

    assert len(built) == 1
    session_settings, rebuild = built[0]
    assert session_settings.source_urls == ["https://chat-default.test"]
    assert rebuild is False


def test_chat_graph_factory_receives_session_word_edit_scope(
    monkeypatch,
    isolated_settings,
    tmp_path,
):
    settings = isolated_settings(
        source_urls=["https://chat-default.test"],
        web_search_enabled=False,
        file_read_root=str(tmp_path / "files"),
        file_read_enabled=True,
        word_edit_enabled=True,
    )
    captured = {}

    class FakeGraph:
        pass

    def fake_build_chat_graph(
        session_settings,
        rebuild_vectorstore=False,
        checkpointer=None,
        *,
        session_root=None,
        thread_id="",
    ):
        captured.update(
            settings=session_settings,
            rebuild_vectorstore=rebuild_vectorstore,
            checkpointer=checkpointer,
            session_root=session_root,
            thread_id=thread_id,
        )
        return FakeGraph()

    monkeypatch.setattr(chat_api, "load_settings", lambda: settings)
    monkeypatch.setattr(chat_api, "build_chat_graph", fake_build_chat_graph)

    with TestClient(chat_api.create_app()) as client:
        response = client.post("/chat", json={"web_search": False})

    assert response.status_code == 200
    thread_id = response.json()["thread_id"]
    assert captured["thread_id"] == thread_id
    assert captured["session_root"] == tmp_path / "files" / "chat_uploads" / thread_id
