"""Selected stdio or stateless Streamable HTTP transport."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from types import FrameType
from typing import Any
from uuid import uuid4

import uvicorn
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .audit import AuditEvent, emit_audit
from .lifecycle import MCPRuntime


class _NoSignalUvicornServer(uvicorn.Server):
    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def transport_security(settings: Any) -> TransportSecuritySettings:
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(settings.effective_allowed_hosts),
        allowed_origins=list(settings.effective_allowed_origins),
    )


def _probe_headers() -> dict[str, str]:
    return {"Cache-Control": "no-store"}


def build_http_app(runtime: MCPRuntime) -> Starlette:
    """Build the isolated MCP app with bounded operational health routes."""

    app = runtime.server.streamable_http_app(
        streamable_http_path=runtime.settings.path,
        json_response=True,
        stateless_http=True,
        max_request_body_size=runtime.settings.max_request_body_bytes,
        transport_security=transport_security(runtime.settings),
        host=runtime.settings.host,
    )

    async def liveness(_request: Request) -> JSONResponse:
        return JSONResponse(runtime.public_liveness(), headers=_probe_headers())

    async def readiness(_request: Request) -> JSONResponse:
        return JSONResponse(
            runtime.public_readiness(),
            status_code=200 if runtime.ready else 503,
            headers=_probe_headers(),
        )

    async def dependency_health(request: Request) -> JSONResponse:
        # Anonymous-development mode deliberately has no administrative health
        # surface. With configured auth, the SDK middleware has already
        # verified the bearer token and populated trusted scopes.
        if runtime.token_verifier is None:
            return JSONResponse(
                {"detail": "Not found."},
                status_code=404,
                headers=_probe_headers(),
            )
        authenticated = bool(getattr(request.user, "is_authenticated", False))
        scopes = set(getattr(request.auth, "scopes", ()))
        if not authenticated or "rag:invoke" not in scopes:
            return JSONResponse(
                {"detail": "Missing or invalid bearer token."},
                status_code=401,
                headers={
                    **_probe_headers(),
                    "WWW-Authenticate": "Bearer",
                },
            )
        return JSONResponse(
            runtime.restricted_dependency_health(),
            status_code=200 if runtime.ready else 503,
            headers=_probe_headers(),
        )

    # The SDK applies authentication middleware globally but enforces scopes
    # only on its protocol route. These route endpoints intentionally keep
    # liveness/readiness public and perform an explicit admin scope check for
    # dependency details.
    app.router.routes[0:0] = [
        Route("/health", liveness, methods=["GET"]),
        Route("/ready", readiness, methods=["GET"]),
        Route(
            "/admin/health/dependencies",
            dependency_health,
            methods=["GET"],
        ),
    ]
    return app


def _install_signal_handlers(on_signal: Callable[[], None]) -> Callable[[], None]:
    previous: dict[signal.Signals, Any] = {}

    def handler(_signum: int, _frame: FrameType | None) -> None:
        on_signal()

    for name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)

    def restore() -> None:
        for signum, prior in previous.items():
            signal.signal(signum, prior)

    return restore


async def _run_until_complete_or_signal(
    runtime: MCPRuntime,
    operation: Awaitable[None],
    *,
    request_stop: Callable[[], None] | None = None,
) -> None:
    loop = asyncio.get_running_loop()
    signal_received = asyncio.Event()

    def notify_signal() -> None:
        loop.call_soon_threadsafe(signal_received.set)

    restore = _install_signal_handlers(notify_signal)
    operation_task: asyncio.Future[None] = asyncio.ensure_future(operation)
    signal_task = asyncio.create_task(signal_received.wait())
    shutdown_deadline: float | None = None
    try:
        wait_set: set[asyncio.Future[Any]] = {operation_task, signal_task}
        done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
        if signal_task in done and signal_received.is_set():
            shutdown_deadline = loop.time() + runtime.settings.shutdown_grace_seconds
            runtime.begin_draining()
            if request_stop is not None:
                request_stop()
            await runtime.shutdown()

        if operation_task.done():
            await operation_task
        else:
            timeout = runtime.settings.shutdown_grace_seconds
            if shutdown_deadline is not None:
                timeout = max(0.0, shutdown_deadline - loop.time())
            if timeout <= 0:
                raise TimeoutError
            await asyncio.wait_for(asyncio.shield(operation_task), timeout=timeout)
    except TimeoutError:
        operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
    finally:
        signal_task.cancel()
        await asyncio.gather(signal_task, return_exceptions=True)
        restore()
        if runtime.lifecycle_status != "closed":
            await runtime.shutdown()


def _emit_transport_start(runtime: MCPRuntime) -> None:
    emit_audit(
        AuditEvent(
            request_id=uuid4().hex,
            principal_id="server",
            tool="transport",
            outcome="started",
            duration_ms=0,
            signal="transport",
            transport=runtime.settings.transport,
        ),
        runtime.observability,
    )


async def run_stdio(runtime: MCPRuntime) -> None:
    _emit_transport_start(runtime)
    await _run_until_complete_or_signal(runtime, runtime.server.run_stdio_async())


async def run_http(runtime: MCPRuntime) -> None:
    app = build_http_app(runtime)
    config = uvicorn.Config(
        app,
        host=runtime.settings.host,
        port=runtime.settings.port,
        log_level="info",
        access_log=False,
    )
    server = _NoSignalUvicornServer(config)
    _emit_transport_start(runtime)
    await _run_until_complete_or_signal(
        runtime,
        server.serve(),
        request_stop=lambda: setattr(server, "should_exit", True),
    )


async def run_selected_transport(runtime: MCPRuntime) -> None:
    if runtime.settings.transport == "stdio":
        await run_stdio(runtime)
    else:
        await run_http(runtime)


__all__ = [
    "build_http_app",
    "run_http",
    "run_selected_transport",
    "run_stdio",
    "transport_security",
]
