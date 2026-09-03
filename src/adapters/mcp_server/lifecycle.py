"""Atomic initialization, bounded health, and shutdown for inbound MCP."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from uuid import uuid4

from mcp.server import MCPServer

from ...application import (
    RagAnswer,
    RagApplicationService,
    RagGraphState,
    RagRequest,
    RagServiceDependencies,
)
from ...config import Settings, load_settings
from ...core.graph_executor import run_rag_query
from ...graph.builder import build_graph, build_lightweight_graph
from ...mcp.observability import MCPObservability
from ...web_search import discover_urls_from_web, settings_for_discovered_urls
from .audit import AuditEvent, emit_audit
from .auth import (
    SharedBearerTokenVerifier,
    resolve_http_auth,
    sdk_auth_settings,
)
from .config import MCPSettings
from .tools import RequestGate, create_mcp_server
from .url_policy import URLValidator

logger = logging.getLogger(__name__)

LifecycleStatus = Literal["ready", "draining", "closed"]
DependencyStatus = Literal["ready", "draining", "closed", "degraded"]
_CLOSE_METHODS = ("aclose", "close", "shutdown")


async def _close_resource(resource: Any, *, timeout_seconds: float, name: str) -> bool:
    """Close one optional sync/async lifecycle resource within a hard bound."""

    method = next(
        (
            candidate
            for method_name in _CLOSE_METHODS
            if callable(candidate := getattr(resource, method_name, None))
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

    if timeout_seconds <= 0:
        return False
    try:
        await asyncio.wait_for(invoke(), timeout=timeout_seconds)
        return True
    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
        raise
    except Exception as exc:
        logger.warning(
            "Lifecycle resource cleanup failed resource=%s error_type=%s",
            name[:128],
            type(exc).__name__[:128],
        )
        return False


class LifecycleRequestGate(RequestGate):
    """Track all admitted tasks, including concurrency waiters, for shutdown."""

    def __init__(self, settings: MCPSettings) -> None:
        super().__init__(settings)
        self._lifecycle_tasks: set[asyncio.Task[Any]] = set()
        self._anonymous_lifecycle_tasks = 0
        self._lifecycle_drained = asyncio.Event()
        self._lifecycle_drained.set()

    @property
    def active_count(self) -> int:
        return len(self._lifecycle_tasks) + self._anonymous_lifecycle_tasks

    def cancel_active(self) -> int:
        current = asyncio.current_task()
        tasks = tuple(
            task for task in self._lifecycle_tasks if task is not current and not task.done()
        )
        for task in tasks:
            task.cancel()
        return len(tasks)

    async def wait_for_drain(self, timeout: float) -> bool:
        if timeout <= 0:
            return self._lifecycle_drained.is_set()
        try:
            async with asyncio.timeout(timeout):
                await self._lifecycle_drained.wait()
            return True
        except TimeoutError:
            return False

    @asynccontextmanager
    async def slot(self, principal_id: str, request_id: str) -> AsyncIterator[None]:
        # Let the base gate produce the stable NOT_READY error for rejected work.
        if not self.accepting:
            async with super().slot(principal_id, request_id):
                yield
            return

        task = asyncio.current_task()
        if task is None:
            self._anonymous_lifecycle_tasks += 1
        else:
            self._lifecycle_tasks.add(task)
        self._lifecycle_drained.clear()
        try:
            async with super().slot(principal_id, request_id):
                yield
        finally:
            if task is None:
                self._anonymous_lifecycle_tasks -= 1
            else:
                self._lifecycle_tasks.discard(task)
            if self.active_count == 0:
                self._lifecycle_drained.set()


class RagServiceHost:
    """Construct each service call from the latest atomically promoted graph."""

    def __init__(
        self,
        *,
        settings: Settings,
        graph: Any,
        url_validator: URLValidator,
        max_source_urls: int,
    ) -> None:
        self._settings = settings
        self._graph = graph
        self._lock = asyncio.Lock()
        self._url_validator = url_validator
        self._max_source_urls = max_source_urls

    async def ask(self, request: RagRequest) -> RagAnswer:
        service = RagApplicationService(
            settings=self._settings,
            graph=self._graph,
            rebuild_lock=self._lock,
            graph_state=RagGraphState(
                current_graph=lambda: self._graph,
                current_settings=lambda: self._settings,
                clear=self._clear,
                promote=self._promote,
            ),
            dependencies=RagServiceDependencies(
                build_graph=lambda graph_settings, rebuild: build_graph(
                    settings=graph_settings, rebuild_vectorstore=rebuild
                ),
                build_lightweight_graph=lambda graph_settings: build_lightweight_graph(
                    graph_settings
                ),
                discover_urls=self._discover_safe_urls,
                settings_for_discovered_urls=settings_for_discovered_urls,
                run_query=run_rag_query,
            ),
        )
        return await service.ask(request)

    async def aclose(self) -> None:
        """Release the currently promoted graph when the inbound process stops."""

        graph = self._graph
        self._graph = None
        await _close_resource(graph, timeout_seconds=5.0, name="rag_graph")

    def _clear(self) -> None:
        self._graph = None

    def _promote(self, graph: Any, settings: Settings) -> None:
        self._graph = graph
        self._settings = settings

    def _discover_safe_urls(self, question: str, settings: Settings) -> list[str]:
        discovered = discover_urls_from_web(question, settings)
        return self._url_validator.validate_sync(
            discovered,
            maximum=self._max_source_urls,
        )


@dataclass(slots=True)
class MCPRuntime:
    settings: MCPSettings
    server: MCPServer[Any]
    gate: LifecycleRequestGate
    service: RagServiceHost | Any
    observability: MCPObservability
    token_verifier: SharedBearerTokenVerifier | None = None
    lifecycle_status: LifecycleStatus = "ready"
    _managed_resources: tuple[tuple[str, Any], ...] = field(default_factory=tuple, repr=False)
    _dependency_status: dict[str, DependencyStatus] = field(default_factory=dict, repr=False)
    _shutdown_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _shutdown_complete: bool = field(default=False, repr=False)
    _shutdown_drained: bool = field(default=True, repr=False)

    @property
    def ready(self) -> bool:
        """Return whether new MCP work may be accepted."""

        return self.lifecycle_status == "ready" and self.gate.accepting

    def begin_draining(self) -> None:
        """Publish non-readiness before a listener is asked to stop."""

        if self.lifecycle_status == "ready":
            self.lifecycle_status = "draining"
        self.gate.stop_accepting()
        for name in self._dependency_status:
            if self._dependency_status[name] == "ready":
                self._dependency_status[name] = "draining"

    def public_liveness(self) -> dict[str, str]:
        """Return a dependency-free liveness contract."""

        return {"status": "ok"}

    def public_readiness(self) -> dict[str, str]:
        """Return readiness without dependency names or diagnostics."""

        return {"status": "ready" if self.ready else "not_ready"}

    def restricted_dependency_health(self) -> dict[str, object]:
        """Return bounded dependency details for an authenticated admin route."""

        dependencies = [
            {
                "name": name,
                "required": True,
                "status": self._dependency_status.get(name, self.lifecycle_status),
            }
            for name, _resource in self._managed_resources
        ]
        dependencies.extend(
            (
                {
                    "name": "request_gate",
                    "required": True,
                    "status": "ready" if self.gate.accepting else self.lifecycle_status,
                },
                {
                    "name": "mcp_protocol",
                    "required": True,
                    "status": self.lifecycle_status,
                },
            )
        )
        return {
            "status": self.lifecycle_status,
            "active_requests": min(self.gate.active_count, self.settings.max_concurrency),
            "dependencies": dependencies,
        }

    async def shutdown(self) -> bool:
        """Drain, cancel, and close resources within the configured grace period."""

        async with self._shutdown_lock:
            if self._shutdown_complete:
                return self._shutdown_drained

            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.settings.shutdown_grace_seconds
            self.begin_draining()

            # Preserve a usable event-loop window for dependency cleanup. For
            # very short grace periods, cancel immediately rather than letting
            # timer granularity consume the entire shutdown deadline.
            cleanup_reserve = min(
                self.settings.shutdown_grace_seconds,
                max(0.05, self.settings.shutdown_grace_seconds / 2),
            )
            drain_timeout = max(0.0, deadline - loop.time() - cleanup_reserve)
            drained = await self.gate.wait_for_drain(drain_timeout)
            if not drained:
                self.gate.cancel_active()
                await asyncio.sleep(0)

            try:
                emit_audit(
                    AuditEvent(
                        request_id=uuid4().hex,
                        principal_id="server",
                        tool="lifecycle",
                        outcome="shutdown_drained" if drained else "shutdown_grace_expired",
                        duration_ms=0,
                        signal="lifecycle",
                        transport=self.settings.transport,
                    ),
                    self.observability,
                )
            except Exception as exc:
                logger.warning(
                    "Shutdown audit delivery failed error_type=%s",
                    type(exc).__name__[:128],
                )

            for name, resource in self._managed_resources:
                remaining = max(0.0, deadline - loop.time())
                closed = await _close_resource(
                    resource,
                    timeout_seconds=remaining,
                    name=name,
                )
                self._dependency_status[name] = "closed" if closed else "degraded"

            self.lifecycle_status = "closed"
            self._shutdown_complete = True
            self._shutdown_drained = drained
            return drained


async def initialize_runtime(
    settings: MCPSettings,
    *,
    service: Any | None = None,
    url_validator: URLValidator | None = None,
    observability: MCPObservability | None = None,
) -> MCPRuntime:
    """Initialize every required dependency before publishing a server generation."""

    if not settings.enabled:
        raise ValueError("Inbound MCP is disabled; set MCP_ENABLED=true to start it")

    from ...deployment import validate_single_instance_deployment

    validate_single_instance_deployment(environment=settings.environment)
    validator = url_validator or URLValidator()
    resolved_auth = resolve_http_auth(settings)

    service_host: Any | None = None
    resolved_observability: MCPObservability | None = None
    try:
        if service is None:
            rag_settings = load_settings()
            rag_settings = replace(
                rag_settings,
                web_search_max_results=settings.max_search_results,
                web_search_top_k=min(
                    rag_settings.web_search_top_k,
                    settings.max_returned_sources,
                ),
            )
            if not rag_settings.web_search_enabled:
                raise ValueError("WEB_SEARCH_ENABLED must be true for rag_web_search_answer")
            validator.validate_sync(
                rag_settings.source_urls,
                maximum=settings.max_source_urls,
            )
            graph = build_graph(settings=rag_settings, rebuild_vectorstore=False)
            service_host = RagServiceHost(
                settings=rag_settings,
                graph=graph,
                url_validator=validator,
                max_source_urls=settings.max_source_urls,
            )
        else:
            service_host = service

        auth_settings = None
        token_verifier = None
        redaction_values: tuple[str, ...] = ()
        if resolved_auth is not None:
            redaction_values = (resolved_auth.token,)
        if observability is not None and not isinstance(observability, MCPObservability):
            raise TypeError("Inbound MCP observability must be MCPObservability")
        resolved_observability = observability or MCPObservability(
            redaction_values=redaction_values
        )
        if resolved_auth is not None:
            auth_settings = sdk_auth_settings(settings)
            token_verifier = SharedBearerTokenVerifier(
                resolved_auth,
                resolved_observability,
            )

        lifecycle_gate = LifecycleRequestGate(settings)
        server, _published_gate = create_mcp_server(
            service=service_host,
            settings=settings,
            url_validator=validator,
            gate=lifecycle_gate,
            observability=resolved_observability,
            redaction_values=redaction_values,
            auth_settings=auth_settings,
            token_verifier=token_verifier,
        )
        managed_resources = (
            ("rag_service", service_host),
            ("observability", resolved_observability),
        )
        dependency_status: dict[str, DependencyStatus] = {
            name: "ready" for name, _resource in managed_resources
        }
        runtime = MCPRuntime(
            settings=settings,
            server=server,
            gate=lifecycle_gate,
            service=service_host,
            observability=resolved_observability,
            token_verifier=token_verifier,
            _managed_resources=managed_resources,
            _dependency_status=dependency_status,
        )
        emit_audit(
            AuditEvent(
                request_id=uuid4().hex,
                principal_id="server",
                tool="lifecycle",
                outcome="ready",
                duration_ms=0,
                signal="lifecycle",
                transport=settings.transport,
            ),
            resolved_observability,
        )
        return runtime
    except BaseException:
        cleanup_deadline = asyncio.get_running_loop().time() + settings.shutdown_grace_seconds
        if service_host is not None:
            await _close_resource(
                service_host,
                timeout_seconds=max(0.0, cleanup_deadline - asyncio.get_running_loop().time()),
                name="rag_service",
            )
        if resolved_observability is not None:
            await _close_resource(
                resolved_observability,
                timeout_seconds=max(0.0, cleanup_deadline - asyncio.get_running_loop().time()),
                name="observability",
            )
        raise


__all__ = ["MCPRuntime", "RagServiceHost", "initialize_runtime"]
