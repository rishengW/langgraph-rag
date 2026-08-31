"""Bounded, reconnecting lifecycle for approved outbound MCP providers."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from ...errors import RAGError
from ...mcp.catalog import validate_snapshot
from ...mcp.models import ProviderHealth, ProviderStatus, ToolCatalogSnapshot
from ...mcp.observability import (
    MCPObservability,
    ObservationEvent,
    ObservationOutcome,
    ObservationSignal,
    ObservationTransport,
)
from ...mcp.policy import current_tool_principal
from ...mcp.secrets import ResolvedSecret
from .config import OutboundMCPServerSettings, OutboundMCPSettings
from .lifecycle import ResolvedOutboundCredential
from .security import (
    ApprovedProcessTemplate,
    AsyncResolver,
    ExecutableAllowlist,
    OutboundEndpointPolicy,
    PrivateEndpointApproval,
    ValidatedEndpoint,
)


class OutboundMCPRuntimeError(RAGError):
    """Sanitized outbound connection, invocation, or lifecycle failure."""

    code = "OUTBOUND_MCP_RUNTIME_ERROR"


@dataclass(frozen=True, slots=True)
class OutboundRedirect:
    """A redirect response returned without being automatically followed."""

    location: str = field(repr=False)


@runtime_checkable
class OutboundConnection(Protocol):
    """One initialized remote or stdio MCP session.

    HTTP connectors must expose the actual effective URL and connected peer so
    the provider can verify that redirects were disabled and DNS was pinned.
    """

    @property
    def effective_url(self) -> str | None: ...

    @property
    def peer_address(self) -> str | None: ...

    async def snapshot(self) -> ToolCatalogSnapshot: ...

    async def invoke(self, tool_name: str, arguments: Mapping[str, object]) -> object: ...

    async def close(self) -> None: ...


@runtime_checkable
class OutboundConnector(Protocol):
    """Transport seam that cannot choose an unvalidated destination or command."""

    async def connect_http(
        self,
        target: ValidatedEndpoint,
        *,
        authorization: ResolvedSecret | None,
        follow_redirects: bool,
    ) -> OutboundConnection | OutboundRedirect: ...

    async def connect_stdio(
        self,
        template: ApprovedProcessTemplate,
    ) -> OutboundConnection: ...

    async def close(self) -> None: ...


ConnectorFactory = Callable[[OutboundMCPServerSettings], OutboundConnector]


class ManagedOutboundMCPProvider:
    """Tool provider with fail-closed startup and bounded runtime behavior."""

    def __init__(
        self,
        settings: OutboundMCPServerSettings,
        connector: OutboundConnector,
        *,
        credential: ResolvedOutboundCredential | None = None,
        resolver: AsyncResolver | None = None,
        private_approval: PrivateEndpointApproval | None = None,
        process_allowlist: ExecutableAllowlist | None = None,
        observability: MCPObservability | None = None,
    ) -> None:
        if not isinstance(settings, OutboundMCPServerSettings):
            raise TypeError("Managed outbound provider requires server settings")
        if not isinstance(connector, OutboundConnector):
            raise TypeError("Managed outbound provider requires an OutboundConnector")
        if credential is not None and credential.server_name != settings.name:
            raise ValueError("Outbound credential does not match its server")
        if settings.transport == "stdio" and process_allowlist is None:
            raise ValueError("Outbound stdio requires an executable allowlist")
        if observability is not None and not isinstance(observability, MCPObservability):
            raise TypeError("Outbound MCP observability must be MCPObservability")

        redaction_values: tuple[str, ...] = ()
        if credential is not None and credential.authorization is not None:
            redaction_values = (credential.authorization.reveal(),)
        self.name = settings.name
        self._settings = settings
        self._connector = connector
        self._credential = credential
        self._observability = observability or MCPObservability(redaction_values=redaction_values)
        self._process_allowlist = process_allowlist
        self._endpoint_policy: OutboundEndpointPolicy | None = None
        if settings.transport == "streamable_http":
            if settings.endpoint is None:
                raise ValueError("Outbound HTTP settings require an endpoint")
            policy_arguments: dict[str, object] = {
                "allowed_redirect_origins": settings.allowed_redirect_origins,
                "max_redirects": settings.max_redirects,
                "private_approval": private_approval,
            }
            if resolver is not None:
                policy_arguments["resolver"] = resolver
            self._endpoint_policy = OutboundEndpointPolicy(
                settings.endpoint,
                **policy_arguments,  # type: ignore[arg-type]
            )

        self._snapshot: ToolCatalogSnapshot | None = None
        self._connection: OutboundConnection | None = None
        self._status: ProviderStatus = "closed"
        self._accepting = True
        self._closed = False
        self._connector_closed = False
        self._state_lock = asyncio.Lock()
        self._reconnect_lock = asyncio.Lock()
        self._invocation_slots = asyncio.Semaphore(settings.max_concurrent_invocations)
        self._active_tasks: set[asyncio.Task[Any]] = set()

    async def start(self) -> None:
        """Open, validate, and observe one complete provider before publication."""

        request_id = uuid4().hex
        started = time.monotonic()
        outcome: ObservationOutcome = "failed"
        try:
            async with self._state_lock:
                if self._closed:
                    raise OutboundMCPRuntimeError("Outbound MCP provider is closed")
                if self._status == "ready":
                    outcome = "ready"
                    return
                try:
                    async with asyncio.timeout(self._settings.connect_timeout_seconds):
                        connection, snapshot = await self._open_connection_and_snapshot()
                except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                    outcome = "cancelled"
                    await self._close_connector_once()
                    raise
                except Exception as exc:
                    self._accepting = False
                    self._closed = True
                    await self._close_connector_once()
                    raise OutboundMCPRuntimeError("Outbound MCP provider startup failed") from exc
                self._connection = connection
                self._snapshot = snapshot
                self._status = "ready"
                outcome = "ready"
        finally:
            self._observe(
                signal="dependency",
                outcome=outcome,
                request_id=request_id,
                started=started,
                principal_id="server",
            )

    async def snapshot(self) -> ToolCatalogSnapshot:
        """Return the immutable snapshot prepared during startup."""

        if self._snapshot is None or self._status != "ready":
            raise OutboundMCPRuntimeError("Outbound MCP provider is not ready")
        return self._snapshot

    async def health(self) -> ProviderHealth:
        """Return bounded state without endpoint, process, or provider details."""

        generation = self._snapshot.generation if self._snapshot is not None else 0
        return ProviderHealth(
            provider=self.name,
            status=self._status,
            generation=generation,
        )

    async def invoke(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
        *,
        timeout_seconds: float | None = None,
        retry_safe: bool = False,
    ) -> object:
        """Invoke within hard bounds and emit only argument-free observations."""

        principal = current_tool_principal()
        request_id = principal.request_id or uuid4().hex
        started = time.monotonic()
        outcome: ObservationOutcome = "invalid"
        raw_observed_tool = tool_name if isinstance(tool_name, str) else "invalid_tool"
        observed_tool = (
            "".join(
                character if 32 <= ord(character) != 127 else "?" for character in raw_observed_tool
            )[:128]
            or "invalid_tool"
        )
        try:
            if (
                not isinstance(tool_name, str)
                or not tool_name
                or len(tool_name) > 128
                or any(ord(character) < 32 or ord(character) == 127 for character in tool_name)
            ):
                raise OutboundMCPRuntimeError("Outbound MCP tool name is invalid")
            if not isinstance(arguments, Mapping):
                raise OutboundMCPRuntimeError("Outbound MCP arguments must be an object")
            if not isinstance(retry_safe, bool):
                raise TypeError("retry_safe must be boolean")
            _bounded_json_size(
                arguments,
                maximum=self._settings.max_request_bytes,
                kind="request",
            )
            effective_timeout = _effective_timeout(
                timeout_seconds,
                maximum=self._settings.invocation_timeout_seconds,
            )
            task = asyncio.current_task()
            if task is None:
                raise OutboundMCPRuntimeError("Outbound MCP invocation has no task context")

            outcome = "rejected"
            async with self._state_lock:
                if not self._accepting or self._status != "ready" or self._connection is None:
                    raise OutboundMCPRuntimeError("Outbound MCP provider is not accepting work")
                self._active_tasks.add(task)

            outcome = "upstream_failure"
            try:
                async with asyncio.timeout(effective_timeout):
                    async with self._invocation_slots:
                        result = await self._invoke_with_reconnect(
                            tool_name,
                            arguments,
                            retry_safe=retry_safe,
                        )
                        outcome = "limited"
                        _bounded_json_size(
                            result,
                            maximum=self._settings.max_result_bytes,
                            kind="result",
                        )
                        outcome = "success"
                        return result
            except TimeoutError:
                outcome = "timeout"
                raise OutboundMCPRuntimeError("Outbound MCP invocation timed out") from None
            finally:
                async with self._state_lock:
                    self._active_tasks.discard(task)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            self._observe(
                signal=("limit_rejection" if outcome == "limited" else "tool_invocation"),
                outcome=outcome,
                request_id=request_id,
                started=started,
                tool=observed_tool,
                principal_id=principal.principal_id,
                tenant_id=principal.tenant_id,
            )

    async def close(self) -> None:
        """Stop work, close dependencies, and emit one bounded shutdown event."""

        request_id = uuid4().hex
        started = time.monotonic()
        try:
            async with self._state_lock:
                if self._closed and self._connector_closed:
                    return
                self._accepting = False
                self._closed = True
                self._status = "closed"
                active = tuple(
                    task for task in self._active_tasks if task is not asyncio.current_task()
                )
                connection = self._connection
                self._connection = None

            phase_timeout = self._settings.shutdown_timeout_seconds / 5
            if active:
                _, pending = await asyncio.wait(active, timeout=phase_timeout * 2)
                for task in pending:
                    task.cancel()
                if pending:
                    _, still_running = await asyncio.wait(pending, timeout=phase_timeout)
                    for task in still_running:
                        task.add_done_callback(_consume_task_result)

            if connection is not None:
                await _close_with_timeout(connection.close, timeout_seconds=phase_timeout)
            await self._close_connector_once(timeout_seconds=phase_timeout)
        finally:
            self._observe(
                signal="lifecycle",
                outcome="closed",
                request_id=request_id,
                started=started,
                principal_id="server",
            )

    def _observe(
        self,
        *,
        signal: ObservationSignal,
        outcome: ObservationOutcome,
        request_id: str,
        started: float,
        principal_id: str,
        tenant_id: str | None = None,
        tool: str | None = None,
    ) -> None:
        transport: ObservationTransport = (
            "outbound_http" if self._settings.transport == "streamable_http" else "outbound_stdio"
        )
        generation = self._snapshot.generation if self._snapshot is not None else 0
        self._observability.emit(
            ObservationEvent(
                signal=signal,
                outcome=outcome,
                request_id=request_id,
                principal_id=principal_id,
                tenant_id=tenant_id,
                tool=tool,
                source_server=self.name,
                transport=transport,
                duration_ms=min(86_400_000, int((time.monotonic() - started) * 1000)),
                generation=generation,
            )
        )

    async def _invoke_with_reconnect(
        self,
        tool_name: str,
        arguments: Mapping[str, object],
        *,
        retry_safe: bool,
    ) -> object:
        connection = self._connection
        if connection is None:
            raise OutboundMCPRuntimeError("Outbound MCP provider is not connected")

        for attempt in range(self._settings.max_reconnect_attempts + 1):
            try:
                return await connection.invoke(tool_name, arguments)
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                raise
            except (ConnectionError, EOFError) as exc:
                if attempt >= self._settings.max_reconnect_attempts:
                    raise OutboundMCPRuntimeError("Outbound MCP connection was lost") from exc
                delay = min(
                    self._settings.reconnect_backoff_seconds * (2**attempt),
                    5.0,
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                connection = await self._reconnect(connection)
                if not retry_safe:
                    raise OutboundMCPRuntimeError(
                        "Outbound MCP connection was restored; invocation was not retried"
                    ) from exc
            except OutboundMCPRuntimeError:
                raise
            except Exception as exc:
                raise OutboundMCPRuntimeError("Outbound MCP invocation failed") from exc
        raise OutboundMCPRuntimeError("Outbound MCP reconnect limit was exceeded")

    async def _reconnect(self, failed_connection: OutboundConnection) -> OutboundConnection:
        async with self._reconnect_lock:
            if not self._accepting:
                raise OutboundMCPRuntimeError("Outbound MCP provider is shutting down")
            if self._connection is not failed_connection and self._connection is not None:
                return self._connection

            await _close_with_timeout(
                failed_connection.close,
                timeout_seconds=self._settings.shutdown_timeout_seconds,
            )
            try:
                connection, snapshot = await self._open_connection_and_snapshot()
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                raise
            except Exception as exc:
                raise OutboundMCPRuntimeError("Outbound MCP reconnect failed") from exc
            if self._snapshot is None or not _snapshots_compatible(self._snapshot, snapshot):
                await _close_with_timeout(
                    connection.close,
                    timeout_seconds=self._settings.shutdown_timeout_seconds,
                )
                raise OutboundMCPRuntimeError(
                    "Outbound MCP reconnect changed the published tool generation"
                )
            self._connection = connection
            return connection

    async def _open_connection_and_snapshot(
        self,
    ) -> tuple[OutboundConnection, ToolCatalogSnapshot]:
        connection: OutboundConnection | None = None
        try:
            if self._settings.transport == "streamable_http":
                connection = await self._open_http_connection()
            else:
                connection = await self._open_stdio_connection()
            snapshot = await connection.snapshot()
            validate_snapshot(snapshot)
            return connection, snapshot
        except BaseException:
            if connection is not None:
                await _close_with_timeout(
                    connection.close,
                    timeout_seconds=self._settings.shutdown_timeout_seconds,
                )
            raise

    async def _open_http_connection(self) -> OutboundConnection:
        policy = self._endpoint_policy
        if policy is None:
            raise OutboundMCPRuntimeError("Outbound MCP endpoint policy is unavailable")
        target = await policy.validate_initial()
        authorization = None if self._credential is None else self._credential.authorization

        redirect_number = 0
        while True:
            result = await self._connector.connect_http(
                target,
                authorization=authorization,
                follow_redirects=False,
            )
            if isinstance(result, OutboundRedirect):
                redirect_number += 1
                target = await policy.validate_redirect(
                    result.location,
                    redirect_number=redirect_number,
                )
                continue
            if not isinstance(result, OutboundConnection):
                raise OutboundMCPRuntimeError("Outbound MCP connector returned an invalid session")
            if result.effective_url != target.url or result.peer_address is None:
                raise OutboundMCPRuntimeError(
                    "Outbound MCP connector did not preserve the validated endpoint"
                )
            policy.validate_connection_peer(target, result.peer_address)
            return result

    async def _open_stdio_connection(self) -> OutboundConnection:
        allowlist = self._process_allowlist
        template_id = self._settings.command_template
        if allowlist is None or template_id is None:
            raise OutboundMCPRuntimeError("Outbound MCP process policy is unavailable")
        template = allowlist.resolve(template_id)
        connection = await self._connector.connect_stdio(template)
        if not isinstance(connection, OutboundConnection):
            raise OutboundMCPRuntimeError("Outbound MCP connector returned an invalid session")
        if connection.effective_url is not None or connection.peer_address is not None:
            await _close_with_timeout(
                connection.close,
                timeout_seconds=self._settings.shutdown_timeout_seconds,
            )
            raise OutboundMCPRuntimeError("Outbound MCP stdio connector returned network metadata")
        return connection

    async def _close_connector_once(self, *, timeout_seconds: float | None = None) -> None:
        if self._connector_closed:
            return
        self._connector_closed = True
        await _close_with_timeout(
            self._connector.close,
            timeout_seconds=(
                self._settings.shutdown_timeout_seconds
                if timeout_seconds is None
                else timeout_seconds
            ),
        )


def build_managed_outbound_providers(
    settings: OutboundMCPSettings,
    credentials: Sequence[ResolvedOutboundCredential],
    connector_factory: ConnectorFactory,
    *,
    resolver: AsyncResolver | None = None,
    private_approvals: Mapping[str, PrivateEndpointApproval] | None = None,
    process_allowlist: ExecutableAllowlist | None = None,
    observability: MCPObservability | None = None,
) -> dict[str, ManagedOutboundMCPProvider]:
    """Construct detached providers without opening a network/process transport."""

    if not isinstance(settings, OutboundMCPSettings):
        raise TypeError("Outbound provider construction requires OutboundMCPSettings")
    if not callable(connector_factory):
        raise TypeError("Outbound provider construction requires a connector factory")
    if not settings.enabled:
        if credentials:
            raise ValueError("Disabled outbound MCP cannot receive credentials")
        return {}

    credentials_by_name: dict[str, ResolvedOutboundCredential] = {}
    for credential in credentials:
        if not isinstance(credential, ResolvedOutboundCredential):
            raise TypeError("Outbound credentials contain an invalid value")
        if credential.server_name in credentials_by_name:
            raise ValueError("Outbound credentials contain duplicate servers")
        credentials_by_name[credential.server_name] = credential
    expected_names = {server.name for server in settings.servers}
    if set(credentials_by_name) != expected_names:
        raise ValueError("Outbound credential set does not match configuration")

    approvals = {} if private_approvals is None else dict(private_approvals)
    if not set(approvals).issubset(expected_names):
        raise ValueError("Private endpoint approval does not match configuration")

    return {
        server.name: ManagedOutboundMCPProvider(
            server,
            connector_factory(server),
            credential=credentials_by_name[server.name],
            resolver=resolver,
            private_approval=approvals.get(server.name),
            process_allowlist=process_allowlist,
            observability=observability,
        )
        for server in settings.servers
    }


def _effective_timeout(requested: float | None, *, maximum: float) -> float:
    if requested is None:
        return maximum
    if isinstance(requested, bool) or not isinstance(requested, (int, float)) or requested <= 0:
        raise OutboundMCPRuntimeError("Outbound MCP invocation timeout is invalid")
    return min(float(requested), maximum)


def _bounded_json_size(value: object, *, maximum: int, kind: str) -> None:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        raise OutboundMCPRuntimeError(f"Outbound MCP {kind} is not valid JSON") from None
    if len(serialized) > maximum:
        raise OutboundMCPRuntimeError(f"Outbound MCP {kind} exceeds its byte limit")


def _snapshots_compatible(first: ToolCatalogSnapshot, second: ToolCatalogSnapshot) -> bool:
    return (
        tuple(first.tools_by_name) == tuple(second.tools_by_name)
        and first.descriptors == second.descriptors
    )


async def _close_with_timeout(
    close: Callable[[], Any],
    *,
    timeout_seconds: float,
) -> None:
    try:
        async with asyncio.timeout(timeout_seconds):
            await close()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        return


__all__ = [
    "ConnectorFactory",
    "ManagedOutboundMCPProvider",
    "OutboundConnection",
    "OutboundConnector",
    "OutboundMCPRuntimeError",
    "OutboundRedirect",
    "build_managed_outbound_providers",
]
