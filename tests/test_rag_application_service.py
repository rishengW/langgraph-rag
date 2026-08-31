from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.application import (
    RagApplicationError,
    RagApplicationService,
    RagGraphState,
    RagRequest,
    RagServiceDependencies,
)
from src.qa import api as qa_api


def _service(*, settings, graph, dependencies, promoted):
    state = {"graph": graph, "settings": settings}

    def clear() -> None:
        state["graph"] = None

    def promote(new_graph, new_settings) -> None:
        state["graph"] = new_graph
        state["settings"] = new_settings
        promoted.append((new_graph, new_settings))

    return RagApplicationService(
        settings=settings,
        graph=graph,
        rebuild_lock=asyncio.Lock(),
        graph_state=RagGraphState(
            current_graph=lambda: state["graph"],
            current_settings=lambda: state["settings"],
            clear=clear,
            promote=promote,
        ),
        dependencies=dependencies,
    )


def test_service_invocation_preserves_explicit_source_debug_and_promotion(mock_settings):
    promoted = []
    calls = []

    def build_graph(settings, rebuild):
        return {"urls": list(settings.source_urls), "rebuild": rebuild}

    def run_query(**kwargs):
        calls.append(kwargs)
        return {
            "answer": "service answer",
            "error": None,
            "messages": [SimpleNamespace(content="debug message")],
        }

    service = _service(
        settings=mock_settings,
        graph={"urls": list(mock_settings.source_urls)},
        promoted=promoted,
        dependencies=RagServiceDependencies(
            build_graph=build_graph,
            build_lightweight_graph=lambda settings: {"lightweight": settings.source_urls},
            discover_urls=lambda question, settings: [],
            settings_for_discovered_urls=lambda settings, urls: settings,
            run_query=run_query,
        ),
    )

    answer = asyncio.run(
        service.ask(
            RagRequest(
                question="What changed?",
                urls=["https://updated.test"],
                web_search=False,
                debug=True,
                request_id="request-direct",
            )
        )
    )

    assert answer.answer == "service answer"
    assert answer.source_mode == "explicit"
    assert answer.messages == ["debug message"]
    assert answer.source_urls == ["https://updated.test"]
    assert answer.sources[0].citation_id == "source-1"
    assert calls[0]["graph"] == {"urls": ["https://updated.test"], "rebuild": True}
    assert promoted[-1][1].source_urls == ["https://updated.test"]


def test_service_sanitizes_unexpected_dependency_errors(mock_settings):
    def fail_query(**kwargs):
        raise RuntimeError("provider-secret-detail")

    service = _service(
        settings=mock_settings,
        graph=object(),
        promoted=[],
        dependencies=RagServiceDependencies(
            build_graph=lambda settings, rebuild: object(),
            build_lightweight_graph=lambda settings: object(),
            discover_urls=lambda question, settings: [],
            settings_for_discovered_urls=lambda settings, urls: settings,
            run_query=fail_query,
        ),
    )

    with pytest.raises(RagApplicationError) as raised:
        asyncio.run(
            service.ask(
                RagRequest(
                    question="fail safely",
                    web_search=False,
                    request_id="request-error",
                )
            )
        )

    assert raised.value.public_detail == "Internal server error. Request ID: request-error"
    assert "provider-secret-detail" not in raised.value.public_detail
    assert isinstance(raised.value.internal_cause, RuntimeError)


def test_query_http_adapter_preserves_legacy_response_shape(monkeypatch, isolated_settings):
    settings = isolated_settings(
        source_urls=["https://default.test"],
        web_search_enabled=False,
    )
    graph = object()
    monkeypatch.setattr(qa_api, "load_settings", lambda: settings)
    monkeypatch.setattr(qa_api, "build_graph", lambda *args, **kwargs: graph)
    monkeypatch.setattr(
        qa_api,
        "run_rag_query",
        lambda **kwargs: {
            "answer": "HTTP answer",
            "error": None,
            "messages": [SimpleNamespace(content="HTTP debug")],
        },
    )

    with TestClient(qa_api.create_app()) as client:
        response = client.post(
            "/query",
            json={"question": "Hello", "web_search": False, "debug": True},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "HTTP answer",
        "error": None,
        "success": True,
        "messages": ["HTTP debug"],
        "source_urls": ["https://default.test"],
        "source_mode": "defaults",
        "source_note": None,
    }


def test_failed_rebuild_restores_previous_global_graph(mock_settings):
    previous_graph = object()
    promoted = []

    def fail_build(settings, rebuild):
        raise RuntimeError("rebuild failed")

    service = _service(
        settings=mock_settings,
        graph=previous_graph,
        promoted=promoted,
        dependencies=RagServiceDependencies(
            build_graph=fail_build,
            build_lightweight_graph=lambda settings: object(),
            discover_urls=lambda question, settings: [],
            settings_for_discovered_urls=lambda settings, urls: settings,
            run_query=lambda **kwargs: {},
        ),
    )

    with pytest.raises(RagApplicationError):
        asyncio.run(
            service.ask(
                RagRequest(
                    question="rebuild",
                    urls=["https://updated.test"],
                    web_search=False,
                )
            )
        )

    assert promoted == [(previous_graph, mock_settings)]
