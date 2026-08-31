from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.chat import api as chat_api
from src.config.loader import load_settings
from src.qa import api as qa_api


class FakeGraph:
    def get_state(self, config):
        return SimpleNamespace(values={"messages": []})

    def invoke(self, inputs, config=None):
        return {"messages": []}


def test_qa_api_key_auth_is_optional_until_configured(monkeypatch, isolated_settings):
    settings = isolated_settings(api_key="secret-token", web_search_enabled=False)

    monkeypatch.setattr(qa_api, "load_settings", lambda *args, **kwargs: settings)
    monkeypatch.setattr(qa_api, "build_graph", lambda *args, **kwargs: FakeGraph())
    monkeypatch.setattr(
        qa_api,
        "run_rag_query",
        lambda **kwargs: {"answer": "secured answer", "error": None, "messages": []},
    )

    app = qa_api.create_app()
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready"}
        assert client.get("/admin/health/dependencies").status_code == 401
        dependency_health = client.get(
            "/admin/health/dependencies",
            headers={"Authorization": "Bearer secret-token"},
        )
        assert dependency_health.status_code == 200
        assert dependency_health.json()["dependencies"]["graph"] == "ready"

        missing = client.post("/query", json={"question": "Hello"})
        invalid = client.post(
            "/query",
            headers={"Authorization": "Bearer wrong"},
            json={"question": "Hello"},
        )
        allowed = client.post(
            "/query",
            headers={"Authorization": "Bearer secret-token"},
            json={"question": "Hello", "web_search": False},
        )

        app.state.qa_graph = None
        not_ready = client.get("/ready")
        unavailable_dependencies = client.get(
            "/admin/health/dependencies",
            headers={"Authorization": "Bearer secret-token"},
        )

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json()["answer"] == "secured answer"
    assert not_ready.status_code == 503
    assert not_ready.json() == {"status": "not_ready"}
    assert "dependencies" not in not_ready.json()
    assert unavailable_dependencies.status_code == 503
    assert unavailable_dependencies.json()["dependencies"]["graph"] == "not_ready"


def test_chat_mutations_require_api_key_when_configured(monkeypatch, isolated_settings):
    settings = isolated_settings(api_key="chat-secret", web_search_enabled=False)

    monkeypatch.setattr(chat_api, "load_settings", lambda *args, **kwargs: settings)
    monkeypatch.setattr(chat_api, "build_chat_graph", lambda *args, **kwargs: FakeGraph())

    app = chat_api.create_app()
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").json() == {"status": "ready"}
        assert client.get("/admin/health/dependencies").status_code == 401
        dependencies = client.get(
            "/admin/health/dependencies",
            headers={"Authorization": "Bearer chat-secret"},
        )
        assert dependencies.status_code == 200
        assert dependencies.json()["dependencies"]["session_registry"] == "ready"

        missing = client.post("/chat", json={"web_search": False})
        allowed = client.post(
            "/chat",
            headers={"Authorization": "Bearer chat-secret"},
            json={"web_search": False},
        )

    assert missing.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json()["source_mode"] == "defaults"


def test_api_key_auth_is_disabled_when_unset(monkeypatch, isolated_settings):
    settings = isolated_settings(api_key="", web_search_enabled=False)

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(qa_api, "load_settings", lambda *args, **kwargs: settings)
    monkeypatch.setattr(qa_api, "build_graph", lambda *args, **kwargs: FakeGraph())
    monkeypatch.setattr(
        qa_api,
        "run_rag_query",
        lambda **kwargs: {"answer": "local answer", "error": None, "messages": []},
    )

    app = qa_api.create_app()
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"question": "Hello", "web_search": False},
        )
        admin_health = client.get("/admin/health/dependencies")

    assert response.status_code == 200
    assert response.json()["answer"] == "local answer"
    assert admin_health.status_code == 404


def test_configured_cors_origins_are_applied(
    tmp_path,
    monkeypatch,
    isolated_settings,
):
    config_file = tmp_path / "api.yaml"
    config_file.write_text(
        "\n".join(
            [
                "cors_allow_origins:",
                "  - https://client.example",
            ]
        ),
        encoding="utf-8",
    )
    settings = isolated_settings(web_search_enabled=False)

    monkeypatch.setattr(qa_api, "load_settings", lambda *args, **kwargs: settings)
    monkeypatch.setattr(qa_api, "build_graph", lambda *args, **kwargs: FakeGraph())

    app = qa_api.create_app(config_file=config_file)
    with TestClient(app) as client:
        response = client.options(
            "/query",
            headers={
                "Origin": "https://client.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://client.example"


def test_rag_env_selects_environment_config_overlay(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        "\n".join(
            [
                "qwen_model: qwen-default",
                "api_port: 8000",
                "cors_allow_origins: []",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "staging.yaml").write_text(
        "\n".join(
            [
                "qwen_model: qwen-staging",
                "api_port: 8100",
                "cors_allow_origins:",
                "  - https://staging.example",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-secret")
    monkeypatch.setenv("API_KEY", "gateway-secret")
    monkeypatch.setenv("RAG_ENV", "staging")

    settings = load_settings(env_file=tmp_path / ".env-missing")

    assert settings.qwen_model == "qwen-staging"
    assert settings.api_port == 8100
    assert settings.api_key == "gateway-secret"
    assert settings.cors_allow_origins == ["https://staging.example"]
