"""FastAPI app.state initialization and dependency providers."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from ..config import Settings
from ..chat.sessions import ChatSessionRegistry
from ..graph.metrics import MetricsCollector


_UNSET: Any = object()


def initialize_qa_app_state(
    app: FastAPI,
    *,
    settings: Any = _UNSET,
    graph: Any = _UNSET,
    rebuild_lock: asyncio.Lock | None = None,
    metrics: MetricsCollector | None = None,
) -> None:
    """Initialize or refresh QA state on a FastAPI app instance."""

    if settings is not _UNSET or not hasattr(app.state, "settings"):
        resolved_settings = None if settings is _UNSET else settings
        app.state.settings = resolved_settings
        app.state.config = resolved_settings
    if graph is not _UNSET or not hasattr(app.state, "qa_graph"):
        resolved_graph = None if graph is _UNSET else graph
        app.state.qa_graph = resolved_graph
        app.state.graph = resolved_graph
    if rebuild_lock is not None or not hasattr(app.state, "rebuild_lock"):
        app.state.rebuild_lock = rebuild_lock or asyncio.Lock()
    if metrics is not None or not hasattr(app.state, "metrics"):
        app.state.metrics = metrics or MetricsCollector()


def update_qa_graph_state(app: FastAPI, *, settings: Settings, graph: Any) -> None:
    """Promote a rebuilt QA graph and matching settings to app.state."""

    app.state.settings = settings
    app.state.config = settings
    app.state.qa_graph = graph
    app.state.graph = graph


def clear_qa_graph(app: FastAPI) -> None:
    """Temporarily drop the app-level QA graph while preserving settings."""

    app.state.qa_graph = None
    app.state.graph = None


def initialize_chat_app_state(
    app: FastAPI,
    *,
    settings: Any = _UNSET,
    session_registry: ChatSessionRegistry | None = None,
    graph_factory_lock: asyncio.Lock | None = None,
    metrics: MetricsCollector | None = None,
) -> None:
    """Initialize or refresh chat state on a FastAPI app instance."""

    if settings is not _UNSET or not hasattr(app.state, "settings"):
        resolved_settings = None if settings is _UNSET else settings
        app.state.settings = resolved_settings
        app.state.config = resolved_settings
    if session_registry is not None or not hasattr(app.state, "session_registry"):
        app.state.session_registry = session_registry or ChatSessionRegistry()
    if graph_factory_lock is not None or not hasattr(app.state, "chat_graph_factory_lock"):
        app.state.chat_graph_factory_lock = graph_factory_lock or asyncio.Lock()
    if metrics is not None or not hasattr(app.state, "metrics"):
        app.state.metrics = metrics or MetricsCollector()


def get_config(request: Request) -> Settings:
    settings = getattr(request.app.state, "config", None)
    if settings is None:
        raise HTTPException(status_code=503, detail="Application settings are not initialized.")
    return settings


def get_settings(request: Request) -> Settings:
    """Backward-compatible alias for callers that use settings terminology."""

    return get_config(request)


def get_qa_graph(request: Request) -> Any:
    graph = getattr(request.app.state, "qa_graph", None)
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Graph not initialized. Try again in a moment.",
        )
    return graph


def get_rebuild_lock(request: Request) -> asyncio.Lock:
    lock = getattr(request.app.state, "rebuild_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.rebuild_lock = lock
    return lock


def get_session_registry(request: Request) -> ChatSessionRegistry:
    registry = getattr(request.app.state, "session_registry", None)
    if registry is None:
        registry = ChatSessionRegistry()
        request.app.state.session_registry = registry
    return registry


def get_chat_graph_factory_lock(request: Request) -> asyncio.Lock:
    lock = getattr(request.app.state, "chat_graph_factory_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        request.app.state.chat_graph_factory_lock = lock
    return lock


def get_metrics(request: Request) -> MetricsCollector:
    metrics = getattr(request.app.state, "metrics", None)
    if metrics is None:
        metrics = MetricsCollector()
        request.app.state.metrics = metrics
    return metrics


__all__ = [
    "clear_qa_graph",
    "get_chat_graph_factory_lock",
    "get_config",
    "get_qa_graph",
    "get_rebuild_lock",
    "get_metrics",
    "get_session_registry",
    "get_settings",
    "initialize_chat_app_state",
    "initialize_qa_app_state",
    "update_qa_graph_state",
]
