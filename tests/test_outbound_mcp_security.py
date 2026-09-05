"""Focused security and lifecycle coverage for MCP server spec task 5.3."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.tools import BaseTool, StructuredTool

from src.backend.adapters.mcp_client import (
    ApprovedProcessTemplate,
    ExecutableAllowlist,
    ManagedOutboundMCPProvider,
    OutboundEndpointPolicy,
    OutboundMCPRuntimeError,
    OutboundMCPServerSettings,
    OutboundRedirect,
    OutboundSecurityError,
    PrivateEndpointApproval,
    SecureProcessLauncher,
    ValidatedEndpoint,
    load_outbound_mcp_settings,
)
from src.backend.mcp import ToolCatalogSnapshot, compose_snapshot, descriptor_from_tool
from src.errors import ConfigurationError

_PUBLIC_A = "93.184.216.34"
_PUBLIC_B = "142.250.72.14"


def _echo(text: str) -> str:
    return text


def _snapshot(server_name: str = "approved") -> ToolCatalogSnapshot:
    tool: BaseTool = StructuredTool.from_function(
        func=_echo,
        name=f"mcp__{server_name}__read",
        description="Read approved remote content.",
    )
    return compose_snapshot(
        (
            (
                tool,
                descriptor_from_tool(
                    tool,
                    source="mcp",
                    server_name=server_name,
                    risk_level="read",
                ),
            ),
        ),
        generation=1,
    )


class RecordingConnection:
    def __init__(
        self,
        *,
        effective_url: str | None,
        peer_address: str | None,
        invocation_results: Sequence[object] = (),
        snapshot_error: Exception | None = None,
        block_invocation: bool = False,
    ) -> None:
        self._effective_url = effective_url
        self._peer_address = peer_address
        self._invocation_results = list(invocation_results)
        self._snapshot_error = snapshot_error
        self._block_invocation = block_invocation
        self.invocation_started = asyncio.Event()
        self.invocation_cancelled = False
        self.invoke_calls = 0
        self.close_calls = 0

    @property
    def effective_url(self) -> str | None:
        return self._effective_url

    @property
    def peer_address(self) -> str | None:
        return self._peer_address

    async def snapshot(self) -> ToolCatalogSnapshot:
        if self._snapshot_error is not None:
            raise self._snapshot_error
        return _snapshot()

    async def invoke(self, tool_name: str, arguments: Mapping[str, object]) -> object:
        self.invoke_calls += 1
        self.invocation_started.set()
        if self._block_invocation:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.invocation_cancelled = True
                raise
        if not self._invocation_results:
            return {"tool": tool_name, "arguments": dict(arguments)}
        result = self._invocation_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    async def close(self) -> None:
        self.close_calls += 1


class RecordingConnector:
    def __init__(self, http_results: Sequence[object] = ()) -> None:
        self._http_results = list(http_results)
        self.targets: list[ValidatedEndpoint] = []
        self.follow_redirects: list[bool] = []
        self.stdio_templates: list[ApprovedProcessTemplate] = []
        self.close_calls = 0

    async def connect_http(
        self,
        target: ValidatedEndpoint,
        *,
        authorization: object,
        follow_redirects: bool,
    ) -> object:
        del authorization
        self.targets.append(target)
        self.follow_redirects.append(follow_redirects)
        if not self._http_results:
            raise ConnectionError("connector exhausted")
        return self._http_results.pop(0)

    async def connect_stdio(self, template: ApprovedProcessTemplate) -> RecordingConnection:
        self.stdio_templates.append(template)
        return RecordingConnection(effective_url=None, peer_address=None)

    async def close(self) -> None:
        self.close_calls += 1


async def _resolver_for(host: str, port: int) -> Sequence[str]:
    assert port == 443
    addresses = {
        "approved.example": (_PUBLIC_A,),
        "redirect.example": (_PUBLIC_B,),
        "private.example": ("10.20.0.5",),
    }
    return addresses[host]


def test_configuration_requires_https_and_rejects_arbitrary_process_fields(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="endpoint is invalid"):
        OutboundMCPServerSettings(name="unsafe", endpoint="http://approved.example/mcp")

    document = {
        "version": 1,
        "enabled": True,
        "servers": [
            {
                "name": "unsafe",
                "transport": "stdio",
                "command": str(Path(sys.executable)),
                "args": ["server.py"],
            }
        ],
    }
    path = tmp_path / "outbound.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Unknown outbound MCP server fields"):
        load_outbound_mcp_settings(config_file=path, environment={})

    approved = OutboundMCPServerSettings(
        name="local_approved",
        transport="stdio",
        command_template="approved_local_server",
    )
    assert approved.endpoint is None
    assert approved.to_config()["command_template"] == "approved_local_server"
    assert "executable" not in approved.to_config()


def test_endpoint_policy_validates_redirects_dns_and_connected_peer() -> None:
    async def run() -> None:
        policy = OutboundEndpointPolicy(
            "https://approved.example/mcp",
            allowed_redirect_origins=("https://redirect.example",),
            max_redirects=1,
            resolver=_resolver_for,
        )
        initial = await policy.validate_initial()
        redirected = await policy.validate_redirect(
            "https://redirect.example/mcp/v2",
            redirect_number=1,
        )

        assert initial.addresses == (_PUBLIC_A,)
        assert redirected.addresses == (_PUBLIC_B,)
        policy.validate_connection_peer(redirected, _PUBLIC_B)
        with pytest.raises(OutboundSecurityError, match="not pre-resolved"):
            policy.validate_connection_peer(redirected, _PUBLIC_A)
        with pytest.raises(OutboundSecurityError, match="not allowlisted"):
            await policy.validate_redirect("https://evil.example/mcp", redirect_number=1)
        with pytest.raises(OutboundSecurityError, match="limit exceeded"):
            await policy.validate_redirect(
                "https://redirect.example/again",
                redirect_number=2,
            )

    asyncio.run(run())


def test_private_destination_requires_narrow_host_and_cidr_approval() -> None:
    async def run() -> None:
        default_policy = OutboundEndpointPolicy(
            "https://private.example/mcp",
            resolver=_resolver_for,
        )
        with pytest.raises(OutboundSecurityError, match="prohibited address"):
            await default_policy.validate_initial()

        approval = PrivateEndpointApproval(
            approval_id="change-1234",
            allowed_hosts=frozenset({"private.example"}),
            allowed_cidrs=("10.20.0.0/24",),
        )
        approved_policy = OutboundEndpointPolicy(
            "https://private.example/mcp",
            resolver=_resolver_for,
            private_approval=approval,
        )
        target = await approved_policy.validate_initial()
        assert target.addresses == ("10.20.0.5",)
        approved_policy.validate_connection_peer(target, "10.20.0.5")

        wrong_host = PrivateEndpointApproval(
            approval_id="change-1235",
            allowed_hosts=frozenset({"other.example"}),
            allowed_cidrs=("10.20.0.0/24",),
        )
        with pytest.raises(OutboundSecurityError, match="prohibited address"):
            await OutboundEndpointPolicy(
                "https://private.example/mcp",
                resolver=_resolver_for,
                private_approval=wrong_host,
            ).validate_initial()

    asyncio.run(run())


_PROHIBITED_IPS = st.one_of(
    st.integers(min_value=0, max_value=(1 << 24) - 1).map(
        lambda offset: str(ipaddress.IPv4Address(int(ipaddress.IPv4Address("10.0.0.0")) + offset))
    ),
    st.integers(min_value=0, max_value=(1 << 24) - 1).map(
        lambda offset: str(ipaddress.IPv4Address(int(ipaddress.IPv4Address("127.0.0.0")) + offset))
    ),
    st.integers(min_value=0, max_value=(1 << 16) - 1).map(
        lambda offset: str(
            ipaddress.IPv4Address(int(ipaddress.IPv4Address("169.254.0.0")) + offset)
        )
    ),
    st.integers(min_value=0, max_value=(1 << 28) - 1).map(
        lambda offset: str(ipaddress.IPv4Address(int(ipaddress.IPv4Address("224.0.0.0")) + offset))
    ),
    st.sampled_from(["0.0.0.0", "::", "::1", "fe80::1", "fc00::1", "ff02::1"]),
)


@settings(max_examples=40, deadline=None)
@given(address=_PROHIBITED_IPS)
def test_url_destination_safety_property(address: str) -> None:
    """Property 7: URL destination safety.

    **Validates: Requirements 6.3, 6.4, 6.5, 6.6, 6.7, 9.5**
    """

    async def resolver(host: str, port: int) -> Sequence[str]:
        assert host == "approved.example"
        assert port == 443
        return (address,)

    policy = OutboundEndpointPolicy(
        "https://approved.example/mcp",
        resolver=resolver,
    )
    with pytest.raises(OutboundSecurityError):
        asyncio.run(policy.validate_initial())


def test_provider_enforces_redirect_contract_invocation_bounds_and_clean_shutdown() -> None:
    async def run() -> None:
        connection = RecordingConnection(
            effective_url="https://redirect.example/mcp",
            peer_address=_PUBLIC_B,
            invocation_results=({"answer": "bounded"},),
        )
        connector = RecordingConnector(
            (
                OutboundRedirect("https://redirect.example/mcp"),
                connection,
            )
        )
        provider = ManagedOutboundMCPProvider(
            OutboundMCPServerSettings(
                name="approved",
                endpoint="https://approved.example/mcp",
                allowed_redirect_origins=("https://redirect.example",),
                max_redirects=1,
                max_request_bytes=128,
                max_result_bytes=128,
            ),
            connector,
            resolver=_resolver_for,
        )

        await provider.start()
        assert (await provider.health()).status == "ready"
        assert connector.follow_redirects == [False, False]
        assert [target.origin.host for target in connector.targets] == [
            "approved.example",
            "redirect.example",
        ]
        assert await provider.invoke("mcp__approved__read", {"text": "question"}) == {
            "answer": "bounded"
        }
        with pytest.raises(OutboundMCPRuntimeError, match="request exceeds"):
            await provider.invoke("mcp__approved__read", {"text": "x" * 256})

        await provider.close()
        assert connection.close_calls == 1
        assert connector.close_calls == 1
        assert (await provider.health()).status == "closed"

    asyncio.run(run())


def test_provider_reconnects_and_retries_only_when_explicitly_safe() -> None:
    async def run() -> None:
        first = RecordingConnection(
            effective_url="https://approved.example/mcp",
            peer_address=_PUBLIC_A,
            invocation_results=(ConnectionError("lost"),),
        )
        second = RecordingConnection(
            effective_url="https://approved.example/mcp",
            peer_address=_PUBLIC_A,
            invocation_results=({"answer": "after reconnect"},),
        )
        connector = RecordingConnector((first, second))
        provider = ManagedOutboundMCPProvider(
            OutboundMCPServerSettings(
                name="approved",
                endpoint="https://approved.example/mcp",
                max_reconnect_attempts=1,
                reconnect_backoff_seconds=0,
            ),
            connector,
            resolver=_resolver_for,
        )

        await provider.start()
        result = await provider.invoke(
            "mcp__approved__read",
            {"text": "safe"},
            retry_safe=True,
        )
        assert result == {"answer": "after reconnect"}
        assert first.invoke_calls == 1
        assert first.close_calls == 1
        assert second.invoke_calls == 1
        await provider.close()
        assert second.close_calls == 1

    asyncio.run(run())


def test_provider_timeout_and_shutdown_cancel_upstream_work() -> None:
    async def run() -> None:
        connection = RecordingConnection(
            effective_url="https://approved.example/mcp",
            peer_address=_PUBLIC_A,
            block_invocation=True,
        )
        connector = RecordingConnector((connection,))
        provider = ManagedOutboundMCPProvider(
            OutboundMCPServerSettings(
                name="approved",
                endpoint="https://approved.example/mcp",
                invocation_timeout_seconds=1.0,
                shutdown_timeout_seconds=0.05,
            ),
            connector,
            resolver=_resolver_for,
        )
        await provider.start()

        with pytest.raises(OutboundMCPRuntimeError, match="timed out"):
            await provider.invoke(
                "mcp__approved__read",
                {},
                timeout_seconds=0.05,
            )
        assert connection.invocation_cancelled is True

        connection.invocation_cancelled = False
        connection.invocation_started = asyncio.Event()
        task = asyncio.create_task(provider.invoke("mcp__approved__read", {}))
        await connection.invocation_started.wait()
        await provider.close()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert connection.invocation_cancelled is True
        assert connection.close_calls == 1
        assert connector.close_calls == 1

    asyncio.run(run())


def test_startup_failure_closes_initialized_session_and_connector_without_leakage() -> None:
    async def run() -> None:
        marker = "provider-secret-detail"
        connection = RecordingConnection(
            effective_url="https://approved.example/mcp",
            peer_address=_PUBLIC_A,
            snapshot_error=RuntimeError(marker),
        )
        connector = RecordingConnector((connection,))
        provider = ManagedOutboundMCPProvider(
            OutboundMCPServerSettings(
                name="approved",
                endpoint="https://approved.example/mcp",
            ),
            connector,
            resolver=_resolver_for,
        )

        with pytest.raises(OutboundMCPRuntimeError) as captured:
            await provider.start()
        assert marker not in str(captured.value)
        assert connection.close_calls == 1
        assert connector.close_calls == 1
        await provider.close()
        assert connector.close_calls == 1

    asyncio.run(run())


def test_process_allowlist_uses_fixed_exec_without_shell_and_closes_process() -> None:
    class Process:
        def __init__(self) -> None:
            self._returncode: int | None = None
            self.terminate_calls = 0
            self.kill_calls = 0

        @property
        def returncode(self) -> int | None:
            return self._returncode

        def terminate(self) -> None:
            self.terminate_calls += 1
            self._returncode = 0

        def kill(self) -> None:
            self.kill_calls += 1
            self._returncode = -9

        async def wait(self) -> int:
            return 0 if self._returncode is None else self._returncode

    async def run() -> None:
        marker = "fixed-env-secret"
        template = ApprovedProcessTemplate(
            template_id="approved_local_server",
            executable=Path(sys.executable),
            arguments=("-m", "approved_server"),
            environment={"FIXED_TOKEN": marker},
        )
        default_policy = ExecutableAllowlist({template.template_id: template})
        with pytest.raises(OutboundSecurityError, match="disabled in hosted production"):
            default_policy.resolve(template.template_id)

        approved_policy = ExecutableAllowlist(
            {template.template_id: template},
            hosted_production=True,
            allow_hosted_production=True,
        )
        process = Process()
        received: list[ApprovedProcessTemplate] = []

        async def spawn(value: ApprovedProcessTemplate) -> Process:
            received.append(value)
            return process

        launcher = SecureProcessLauncher(approved_policy, spawn=spawn)
        managed = await launcher.start(
            template.template_id,
            startup_timeout_seconds=1,
            shutdown_timeout_seconds=1,
        )
        assert received == [template]
        assert received[0].arguments == ("-m", "approved_server")
        assert marker not in repr(template)
        await managed.close()
        assert process.terminate_calls == 1
        assert process.kill_calls == 0

    asyncio.run(run())
