"""FastAPI error handlers for typed RAG errors."""

from __future__ import annotations

from http import HTTPStatus
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..errors import (
    AllSourcesFailedError,
    ConfigurationError,
    DocumentLoadError,
    LLMUnavailableError,
    RAGError,
    ResourceNotFoundError,
    RetrieverError,
    WebSearchError,
)

# REFACTOR: Centralize RAG error code to HTTP status mapping.
RAG_ERROR_STATUS_CODES: dict[str, int] = {
    RAGError.code: HTTPStatus.INTERNAL_SERVER_ERROR,
    ConfigurationError.code: HTTPStatus.INTERNAL_SERVER_ERROR,
    LLMUnavailableError.code: HTTPStatus.SERVICE_UNAVAILABLE,
    RetrieverError.code: HTTPStatus.SERVICE_UNAVAILABLE,
    WebSearchError.code: HTTPStatus.BAD_GATEWAY,
    DocumentLoadError.code: HTTPStatus.BAD_GATEWAY,
    AllSourcesFailedError.code: HTTPStatus.UNPROCESSABLE_ENTITY,
    ResourceNotFoundError.code: HTTPStatus.NOT_FOUND,
}


def error_detail(exc: RAGError) -> dict[str, str]:
    """Build the stable API error payload for a typed RAG exception."""

    return {"code": exc.code, "message": str(exc)}


async def rag_error_handler(request: Request, exc: RAGError) -> JSONResponse:
    """Return a JSON response for typed RAG exceptions."""

    status_code = RAG_ERROR_STATUS_CODES.get(
        exc.code,
        HTTPStatus.INTERNAL_SERVER_ERROR,
    )
    return JSONResponse(
        status_code=int(status_code),
        content={"detail": error_detail(exc)},
    )


async def _registered_rag_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Adapt FastAPI's broad exception handler protocol to RAGError."""

    return await rag_error_handler(request, cast(RAGError, exc))


def register_error_handlers(app: FastAPI) -> None:
    """Register shared typed error handlers on a FastAPI app."""

    app.add_exception_handler(RAGError, _registered_rag_error_handler)



__all__ = [
    "RAG_ERROR_STATUS_CODES",
    "error_detail",
    "rag_error_handler",
    "register_error_handlers",
]
