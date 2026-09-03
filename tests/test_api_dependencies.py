from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.api.dependencies import initialize_chat_app_state
from src.api.models import StartChatRequest
from src.chat import api as chat_api


def test_models_remain_importable_from_legacy_api_modules():
    assert chat_api.StartChatRequest is StartChatRequest


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
    assert captured["session_root"] == (Path(settings.file_read_root) / "chat_uploads" / thread_id)


def test_chat_web_search_ignores_request_toggle_and_refreshes_turns(
    monkeypatch,
    isolated_settings,
):
    settings = isolated_settings(
        source_urls=["https://chat-default.test"],
        web_search_enabled=True,
        web_search_lightweight=False,
    )
    searches = []
    built = []

    class FakeGraph:
        def __init__(self, source_urls):
            self.source_urls = list(source_urls)
            self.messages = []

        def get_state(self, config):
            return SimpleNamespace(values={"messages": list(self.messages)})

        def invoke(self, inputs, config):
            self.messages.extend(inputs["messages"])
            self.messages.append(AIMessage(content=f"answer from {self.source_urls[-1]}"))
            return {"messages": list(self.messages)}

    def fake_discover_urls_from_web(question, search_settings):
        searches.append((question, list(search_settings.source_urls)))
        if question == "seed question":
            return ["https://seed.test"]
        if question == "fresh question":
            return ["https://fresh.test"]
        return []

    def fake_build_chat_graph(session_settings, rebuild_vectorstore=False, checkpointer=None):
        built.append((session_settings, rebuild_vectorstore, checkpointer))
        return FakeGraph(session_settings.source_urls)

    monkeypatch.setattr(chat_api, "load_settings", lambda: settings)
    monkeypatch.setattr(chat_api, "discover_urls_from_web", fake_discover_urls_from_web)
    monkeypatch.setattr(chat_api, "build_chat_graph", fake_build_chat_graph)

    app = chat_api.create_app()
    with TestClient(app) as client:
        start = client.post(
            "/chat",
            json={"seed_question": "seed question", "web_search": False},
        )
        assert start.status_code == 200
        start_body = start.json()
        thread_id = start_body["thread_id"]

        assert start_body["source_mode"] == "web_search"
        assert start_body["source_urls"] == ["https://seed.test"]

        message = client.post(
            f"/chat/{thread_id}/message",
            json={"message": "fresh question"},
        )
        assert message.status_code == 200
        assert message.json()["answer"] == "answer from https://fresh.test"

        history = client.get(f"/chat/{thread_id}/history")
        assert history.status_code == 200
        assert history.json()["source_mode"] == "web_search"
        assert history.json()["source_urls"] == ["https://fresh.test"]

    assert [call[0] for call in searches] == ["seed question", "fresh question"]
    assert searches[1][1] == ["https://chat-default.test"]
    assert [list(item[0].source_urls) for item in built] == [
        ["https://seed.test"],
        ["https://fresh.test"],
    ]
    assert [item[1] for item in built] == [True, True]
    assert all(item[0].web_search_enabled is False for item in built)


def test_chat_web_search_lightweight_is_graph_owned_and_built_once(
    monkeypatch,
    isolated_settings,
):
    settings = isolated_settings(
        source_urls=["https://chat-default.test"],
        web_search_enabled=True,
        web_search_lightweight=True,
    )
    searches = []
    heavy_built = []
    lightweight_built = []

    class FakeGraph:
        def __init__(self, source_urls):
            self.source_urls = list(source_urls)
            self.current_urls = list(source_urls)
            self.messages = []

        def get_state(self, config):
            return SimpleNamespace(
                values={
                    "messages": list(self.messages),
                    "source_urls": list(self.current_urls),
                }
            )

        def invoke(self, inputs, config):
            self.messages.extend(inputs["messages"])
            self.current_urls = ["https://fresh-light.test"]
            self.messages.append(AIMessage(content="answer from https://fresh-light.test"))
            return {
                "messages": list(self.messages),
                "source_urls": list(self.current_urls),
            }

    def fake_discover_urls_from_web(question, search_settings):
        searches.append((question, list(search_settings.source_urls)))
        raise AssertionError("lightweight chat search must be owned by the graph")

    def fake_build_chat_graph(*args, **kwargs):
        heavy_built.append((args, kwargs))
        return FakeGraph(["https://heavy.test"])

    def fake_build_lightweight_graph(*, settings, checkpointer):
        lightweight_built.append((settings, checkpointer))
        return FakeGraph(settings.source_urls)

    monkeypatch.setattr(chat_api, "load_settings", lambda: settings)
    monkeypatch.setattr(chat_api, "discover_urls_from_web", fake_discover_urls_from_web)
    monkeypatch.setattr(chat_api, "build_chat_graph", fake_build_chat_graph)
    monkeypatch.setattr(chat_api, "build_lightweight_graph", fake_build_lightweight_graph)

    app = chat_api.create_app()
    with TestClient(app) as client:
        start = client.post(
            "/chat",
            json={"seed_question": "seed question", "web_search": False},
        )
        assert start.status_code == 200
        start_body = start.json()
        thread_id = start_body["thread_id"]

        assert start_body["source_mode"] == "web_search"
        assert start_body["source_urls"] == []

        message = client.post(
            f"/chat/{thread_id}/message",
            json={"message": "fresh question"},
        )
        assert message.status_code == 200
        assert message.json()["answer"] == "answer from https://fresh-light.test"

        history = client.get(f"/chat/{thread_id}/history")
        assert history.status_code == 200
        assert history.json()["source_urls"] == ["https://fresh-light.test"]

    assert searches == []
    assert heavy_built == []
    assert [list(item[0].source_urls) for item in lightweight_built] == [
        [],
    ]
    assert len(lightweight_built) == 1
