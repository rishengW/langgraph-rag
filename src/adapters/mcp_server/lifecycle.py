"""Atomic initialization and bounded shutdown for inbound MCP."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import Any
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
                    mode="qa", settings=graph_settings, rebuild_vectorstore=rebuild
                ),
                build_lightweight_graph=lambda graph_settings: build_lightweight_graph(
                    graph_settings, mode="qa"
                ),
                discover_urls=self._discover_safe_urls,
                settings_for_discovered_urls=settings_for_discovered_urls,
                run_query=run_rag_query,
            ),
        )
        return await service.ask(request)

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
    gate: RequestGate
    service: RagServiceHost | Any
    ready: bool = True

    async def shutdown(self) -> bool:
        """Reject new calls and wait no longer than the configured grace period."""

        self.ready = False
        self.gate.stop_accepting()
        drained = await self.gate.wait_for_drain(self.settings.shutdown_grace_seconds)
        emit_audit(
            AuditEvent(
                request_id=uuid4().hex,
                principal_id="server",
                tool="lifecycle",
                outcome="shutdown_drained" if drained else "shutdown_grace_expired",
                duration_ms=0,
            )
        )
        return drained


async def initialize_runtime(
    settings: MCPSettings,
    *,
    service: Any | None = None,
    url_validator: URLValidator | None = None,
) -> MCPRuntime:
    """Initialize every required dependency before publishing a server generation."""

    if not settings.enabled:
        raise ValueError("Inbound MCP is disabled; set MCP_ENABLED=true to start it")
    validator = url_validator or URLValidator()
    resolved_auth = resolve_http_auth(settings)

    service_host: Any
    if service is None:
        rag_settings = load_settings()
        rag_settings = replace(
            rag_settings,
            web_search_max_results=settings.max_search_results,
            web_search_top_k=min(rag_settings.web_search_top_k, settings.max_returned_sources),
        )
        if not rag_settings.web_search_enabled:
            raise ValueError("WEB_SEARCH_ENABLED must be true for rag_web_search_answer")
        validator.validate_sync(
            rag_settings.source_urls,
            maximum=settings.max_source_urls,
        )
        graph = build_graph(mode="qa", settings=rag_settings, rebuild_vectorstore=False)
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
        auth_settings = sdk_auth_settings(settings)
        token_verifier = SharedBearerTokenVerifier(resolved_auth)
        redaction_values = (resolved_auth.token,)

    server, gate = create_mcp_server(
        service=service_host,
        settings=settings,
        url_validator=validator,
        redaction_values=redaction_values,
        auth_settings=auth_settings,
        token_verifier=token_verifier,
    )
    runtime = MCPRuntime(settings=settings, server=server, gate=gate, service=service_host)
    emit_audit(
        AuditEvent(
            request_id=uuid4().hex,
            principal_id="server",
            tool="lifecycle",
            outcome="ready",
            duration_ms=0,
        )
    )
    return runtime


__all__ = ["MCPRuntime", "RagServiceHost", "initialize_runtime"]
