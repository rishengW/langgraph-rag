"""Selected stdio or stateless Streamable HTTP transport."""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from types import FrameType
from typing import Any

import uvicorn
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette

from .lifecycle import MCPRuntime

logger = logging.getLogger(__name__)


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


def build_http_app(runtime: MCPRuntime) -> Starlette:
    """Build only the SDK ASGI app; the project FastAPI app is never imported."""

    return runtime.server.streamable_http_app(
        streamable_http_path=runtime.settings.path,
        json_response=True,
        stateless_http=True,
        max_request_body_size=runtime.settings.max_request_body_bytes,
        transport_security=transport_security(runtime.settings),
        host=runtime.settings.host,
    )


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
    try:
        wait_set: set[asyncio.Future[Any]] = {operation_task, signal_task}
        done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
        if signal_task in done and signal_received.is_set():
            runtime.gate.stop_accepting()
            runtime.ready = False
            if request_stop is not None:
                request_stop()
            drained = await runtime.shutdown()
            if not drained:
                operation_task.cancel()
        await asyncio.wait_for(
            asyncio.shield(operation_task),
            timeout=runtime.settings.shutdown_grace_seconds,
        )
    except TimeoutError:
        operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
    finally:
        signal_task.cancel()
        await asyncio.gather(signal_task, return_exceptions=True)
        restore()
        if runtime.ready:
            await runtime.shutdown()


async def run_stdio(runtime: MCPRuntime) -> None:
    logger.info("Starting inbound MCP stdio transport")
    await _run_until_complete_or_signal(runtime, runtime.server.run_stdio_async())


async def run_http(runtime: MCPRuntime) -> None:
    app = build_http_app(runtime)
    config = uvicorn.Config(
        app,
        host=runtime.settings.host,
        port=runtime.settings.port,
        log_level="info",
        access_log=True,
    )
    server = _NoSignalUvicornServer(config)
    logger.info(
        "Starting inbound MCP HTTP transport host=%s port=%d path=%s",
        runtime.settings.host,
        runtime.settings.port,
        runtime.settings.path,
    )
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
