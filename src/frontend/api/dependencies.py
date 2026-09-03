"""FastAPI app.state initialization and dependency providers."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..chat.sessions import ChatSessionRegistry
from src.config import Settings
from src.backend.graph.metrics import MetricsCollector
from src.backend.security import QuotaManager

_UNSET: Any = object()


def configure_cors(app: FastAPI, allow_origins: list[str]) -> None:
    """Install CORS middleware when origins are configured."""

    if not allow_origins:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )


def initialize_chat_app_state(
    app: FastAPI,
    *,
    settings: Any = _UNSET,
    session_registry: ChatSessionRegistry | None = None,
    graph_factory_lock: asyncio.Lock | None = None,
    metrics: MetricsCollector | None = None,
    quota_manager: QuotaManager | None = None,
) -> None:
    """Initialize or refresh chat state on a FastAPI app instance."""

    if not hasattr(app.state, "accepting_requests"):
        app.state.accepting_requests = False
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
    if quota_manager is not None or not hasattr(app.state, "quota_manager"):
        app.state.quota_manager = quota_manager or QuotaManager()


def get_config(request: Request) -> Settings:
    settings = getattr(request.app.state, "config", None)
    if settings is None:
        raise HTTPException(status_code=503, detail="Application settings are not initialized.")
    return cast(Settings, settings)


def get_settings(request: Request) -> Settings:
    """Backward-compatible alias for callers that use settings terminology."""

    return get_config(request)


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


def get_quota_manager(request: Request) -> QuotaManager:
    quotas = getattr(request.app.state, "quota_manager", None)
    if quotas is None:
        quotas = QuotaManager()
        request.app.state.quota_manager = quotas
    return cast(QuotaManager, quotas)


async def close_lifecycle_resource(
    resource: Any,
    *,
    timeout_seconds: float = 5.0,
) -> bool:
    """Close one optional app resource without blocking shutdown indefinitely."""

    method = next(
        (
            candidate
            for name in ("aclose", "close", "shutdown")
            if callable(candidate := getattr(resource, name, None))
        ),
        None,
    )
    if method is None:
        return True

    async def invoke() -> None:
        if inspect.iscoroutinefunction(method):
            await method()
            return
        result = await asyncio.to_thread(method)
        if inspect.isawaitable(result):
            await result

    try:
        await asyncio.wait_for(invoke(), timeout=timeout_seconds)
        return True
    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
        raise
    except Exception:
        return False


def _llm_api_key_configured(settings: Any) -> bool:
    """Check whether the active LLM provider's API key is set."""

    if settings is None:
        return False
    provider = getattr(settings, "llm_provider", "dashscope") or "dashscope"
    provider = provider.strip().lower()
    if provider == "deepseek":
        return bool(getattr(settings, "deepseek_api_key", ""))
    return bool(getattr(settings, "dashscope_api_key", ""))


def _chat_dependency_checks(request: Request) -> dict[str, bool]:
    settings = getattr(request.app.state, "settings", None)
    return {
        "request_acceptance": bool(getattr(request.app.state, "accepting_requests", False)),
        "settings": settings is not None,
        "llm_credentials": _llm_api_key_configured(settings),
        "session_registry": getattr(request.app.state, "session_registry", None) is not None,
        "checkpoint_store": getattr(request.app.state, "chat_checkpointer", None) is not None,
        "session_store": getattr(request.app.state, "chat_session_storage", None) is not None,
    }


def _public_readiness_response(checks: dict[str, bool]) -> JSONResponse:
    ready = all(checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready"},
        headers={"Cache-Control": "no-store"},
    )


def _dependency_health_response(
    checks: dict[str, bool],
    *,
    optional: dict[str, str] | None = None,
) -> JSONResponse:
    ready = all(checks.values())
    dependencies = {name: "ready" if healthy else "not_ready" for name, healthy in checks.items()}
    dependencies.update(optional or {})
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "dependencies": dependencies,
        },
        headers={"Cache-Control": "no-store"},
    )


def liveness_response() -> JSONResponse:
    """Return dependency-free process liveness."""

    return JSONResponse(
        content={"status": "ok"},
        headers={"Cache-Control": "no-store"},
    )


def chat_readiness_response(request: Request) -> JSONResponse:
    """Return only bounded public chat readiness."""

    return _public_readiness_response(_chat_dependency_checks(request))


def chat_dependency_health_response(request: Request) -> JSONResponse:
    """Return chat dependency details for an authenticated admin route."""

    extraction_runtime = getattr(request.app.state, "extraction_runtime", None)
    return _dependency_health_response(
        _chat_dependency_checks(request),
        optional={"background_workers": "ready" if extraction_runtime is not None else "disabled"},
    )


__all__ = [
    "chat_dependency_health_response",
    "chat_readiness_response",
    "close_lifecycle_resource",
    "configure_cors",
    "get_chat_graph_factory_lock",
    "get_config",
    "get_metrics",
    "get_quota_manager",
    "get_session_registry",
    "get_settings",
    "initialize_chat_app_state",
    "liveness_response",
]
