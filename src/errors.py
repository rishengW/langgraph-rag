"""Typed error hierarchy for RAG application failures."""

from __future__ import annotations


# REFACTOR: Add stable application error codes for API and future stream consumers.
class RAGError(Exception):
    """Base exception for all RAG-related errors."""

    code = "RAG_ERROR"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class ConfigurationError(RAGError):
    """Raised when application configuration is invalid or unavailable."""

    code = "CONFIGURATION_ERROR"


class LLMUnavailableError(RAGError):
    """Raised when an LLM provider cannot satisfy a request."""

    code = "LLM_UNAVAILABLE"


class RetrieverError(RAGError):
    """Raised when retrieval or vector store access fails."""

    code = "RETRIEVER_ERROR"


class WebSearchError(RAGError):
    """Raised when web-search discovery fails."""

    code = "WEB_SEARCH_ERROR"


class DocumentLoadError(RAGError):
    """Raised when a source document cannot be loaded."""

    code = "DOCUMENT_LOAD_ERROR"


class AllSourcesFailedError(RAGError):
    """Raised when no configured or requested source can be loaded."""

    code = "ALL_SOURCES_FAILED"


class ResourceNotFoundError(RAGError):
    """Raised when a requested API resource does not exist or is not authorized."""

    code = "RESOURCE_NOT_FOUND"


class QuotaExceededError(RAGError):
    """Raised when a trusted principal or tenant exceeds a configured quota."""

    code = "QUOTA_EXCEEDED"

    def __init__(self) -> None:
        super().__init__("Request quota exceeded.")


__all__ = [
    "AllSourcesFailedError",
    "ConfigurationError",
    "DocumentLoadError",
    "LLMUnavailableError",
    "QuotaExceededError",
    "RAGError",
    "ResourceNotFoundError",
    "RetrieverError",
    "WebSearchError",
]
