from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.chat import api as chat_api
from src.errors import (
    AllSourcesFailedError,
    ConfigurationError,
    DocumentLoadError,
    LLMUnavailableError,
    RAGError,
    ResourceNotFoundError,
    RetrieverError,
    WebSearchError,
)
from src.qa import api as qa_api


@pytest.mark.parametrize(
    ("error_cls", "expected_status", "expected_code"),
    [
        (RAGError, 500, "RAG_ERROR"),
        (ConfigurationError, 500, "CONFIGURATION_ERROR"),
        (LLMUnavailableError, 503, "LLM_UNAVAILABLE"),
        (RetrieverError, 503, "RETRIEVER_ERROR"),
        (WebSearchError, 502, "WEB_SEARCH_ERROR"),
        (DocumentLoadError, 502, "DOCUMENT_LOAD_ERROR"),
        (AllSourcesFailedError, 422, "ALL_SOURCES_FAILED"),
        (ResourceNotFoundError, 404, "RESOURCE_NOT_FOUND"),
    ],
)
def test_qa_app_formats_typed_rag_errors(
    error_cls: type[RAGError],
    expected_status: int,
    expected_code: str,
) -> None:
    app = qa_api.create_app()

    @app.get("/raise-rag-error")
    async def raise_rag_error() -> None:
        raise error_cls("planned failure")

    response = TestClient(app).get("/raise-rag-error")

    assert response.status_code == expected_status
    assert response.json() == {
        "detail": {
            "code": expected_code,
            "message": "planned failure",
        },
    }


def test_chat_unknown_thread_uses_typed_not_found_response() -> None:
    app = chat_api.create_app()

    response = TestClient(app).post(
        "/chat/missing-thread/message",
        json={"message": "hello"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": {
            "code": "RESOURCE_NOT_FOUND",
            "message": "Resource not found.",
        },
    }
