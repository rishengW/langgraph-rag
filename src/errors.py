"""Typed error hierarchy for RAG application failures."""

from __future__ import annotations

from typing import ClassVar


# REFACTOR: Add stable application error codes for API and future stream consumers.
class RAGError(Exception):
    """Base exception for all RAG-related errors."""

    code: ClassVar[str] = "RAG_ERROR"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class ConfigurationError(RAGError):
    """Raised when application configuration is invalid or unavailable."""

    code: ClassVar[str] = "CONFIGURATION_ERROR"


class LLMUnavailableError(RAGError):
    """Raised when an LLM provider cannot satisfy a request."""

    code: ClassVar[str] = "LLM_UNAVAILABLE"


class RetrieverError(RAGError):
    """Raised when retrieval or vector store access fails."""

    code: ClassVar[str] = "RETRIEVER_ERROR"


class WebSearchError(RAGError):
    """Raised when web-search discovery fails."""

    code: ClassVar[str] = "WEB_SEARCH_ERROR"


class DocumentLoadError(RAGError):
    """Raised when a source document cannot be loaded."""

    code: ClassVar[str] = "DOCUMENT_LOAD_ERROR"


class AllSourcesFailedError(RAGError):
    """Raised when no configured or requested source can be loaded."""

    code: ClassVar[str] = "ALL_SOURCES_FAILED"


class ResourceNotFoundError(RAGError):
    """Raised when a requested API resource does not exist."""

    code: ClassVar[str] = "RESOURCE_NOT_FOUND"


__all__ = [
    "AllSourcesFailedError",
    "ConfigurationError",
    "DocumentLoadError",
    "LLMUnavailableError",
    "RAGError",
    "ResourceNotFoundError",
    "RetrieverError",
    "WebSearchError",
]
