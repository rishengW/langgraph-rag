"""Focused Task 5.2 tests for required/optional provider publication."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.tools import BaseTool, StructuredTool

from src.adapters.mcp_client import (
    OutboundMCPServerSettings,
    OutboundMCPSettings,
    outbound_provider_registrations,
)
from src.mcp import (
    ProviderHealth,
    ToolCatalog,
    ToolCatalogError,
    ToolCatalogSnapshot,
    ToolProviderRegistration,
    compose_snapshot,
    descriptor_from_tool,
)


def _echo(text: str) -> str:
    return text


def _tool(name: str) -> BaseTool:
    return StructuredTool.from_function(
        func=_echo,
        name=name,
        description=f"Echo text through {name}.",
    )


class RecordingProvider:
    """Deterministic provider with observable lifecycle transitions."""

    def __init__(
        self,
        name: str,
        tool_names: Sequence[str],
        *,
        fail_at: str | None = None,
        failure_message: str = "provider failed",
        health_status: str = "ready",
        close_delay: float = 0.0,
    ) -> None:
        tools = tuple(_tool(tool_name) for tool_name in tool_names)
        entries = tuple(
            (
                tool,
                descriptor_from_tool(
                    tool,
                    source="mcp",
                    server_name=name,
                    risk_level="read",
                ),
            )
            for tool in tools
        )
        self.name = name
        self.fail_at = fail_at
        self.failure_message = failure_message
        self.health_status = health_status
        self.close_delay = close_delay
        self.started = False
        self.closed = False
        self.close_calls = 0
        self._snapshot = compose_snapshot(entries, generation=1)

    async def start(self) -> None:
        self.started = True
        if self.fail_at == "start":
            raise RuntimeError(self.failure_message)

    async def snapshot(self) -> ToolCatalogSnapshot:
        if self.fail_at == "snapshot":
            raise RuntimeError(self.failure_message)
        return self._snapshot

    async def health(self) -> ProviderHealth:
        if self.fail_at == "health":
            raise RuntimeError(self.failure_message)
        return ProviderHealth(
            provider=self.name,
            status=self.health_status,  # type: ignore[arg-type]
            generation=1,
        )

    async def close(self) -> None:
        self.close_calls += 1
        self.closed = True
        if self.close_delay:
            await asyncio.sleep(self.close_delay)
        if self.fail_at == "close":
            raise RuntimeError(self.failure_message)


def _registration(
    provider: RecordingProvider,
    *,
    required: bool,
) -> ToolProviderRegistration:
    return ToolProviderRegistration(
        name=provider.name,
        provider=provider,
        required=required,
    )


def test_required_failure_is_atomic_and_retains_previous_generation() -> None:
    async def run() -> None:
        catalog = ToolCatalog()
        active_provider = RecordingProvider("active", ("mcp__active__read",))
        active = await catalog.publish((_registration(active_provider, required=True),))
        prior_readiness = catalog.readiness

        prepared = RecordingProvider("prepared", ("mcp__prepared__read",))
        marker = "secret-required-provider-detail"
        failed = RecordingProvider(
            "required_failure",
            ("mcp__required_failure__read",),
            fail_at="start",
            failure_message=marker,
        )

        with pytest.raises(ToolCatalogError) as captured:
            await catalog.publish(
                (
                    _registration(prepared, required=True),
                    _registration(failed, required=True),
                )
            )

        assert marker not in str(captured.value)
        assert catalog.current is active
        assert catalog.readiness is prior_readiness
        assert catalog.public_readiness() == {
            "ready": True,
            "status": "ready",
            "generation": 1,
        }
        assert prepared.closed is True
        assert failed.closed is True
        assert active_provider.closed is False

    asyncio.run(run())


def test_optional_failure_omits_all_dependent_tools_and_degrades_readiness(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def run() -> None:
        required = RecordingProvider("required", ("mcp__required__read",))
        marker = "secret-optional-provider-detail"
        optional_failure = RecordingProvider(
            "optional_failure",
            (
                "mcp__optional_failure__read",
                "mcp__optional_failure__search",
            ),
            fail_at="health",
            failure_message=marker,
        )
        optional_ready = RecordingProvider(
            "optional_ready",
            ("mcp__optional_ready__read",),
        )
        catalog = ToolCatalog()

        caplog.set_level(logging.WARNING, logger="src.mcp.catalog")
        published = await catalog.publish(
            (
                _registration(required, required=True),
                _registration(optional_failure, required=False),
                _registration(optional_ready, required=False),
            )
        )

        assert tuple(published.tools_by_name) == (
            "mcp__required__read",
            "mcp__optional_ready__read",
        )
        assert optional_failure.closed is True
        assert optional_ready.closed is False
        assert catalog.public_readiness() == {
            "ready": True,
            "status": "degraded",
            "generation": 1,
        }
        assert [
            (health.provider, health.status, health.detail) for health in catalog.provider_health
        ] == [
            ("required", "ready", ""),
            ("optional_failure", "degraded", "unavailable"),
            ("optional_ready", "ready", ""),
        ]
        assert "optional_failure" not in str(catalog.public_readiness())
        assert catalog.restricted_dependency_health() == {
            "status": "degraded",
            "generation": 1,
            "providers": [
                {
                    "name": "required",
                    "status": "ready",
                    "generation": 1,
                    "detail": "",
                },
                {
                    "name": "optional_failure",
                    "status": "degraded",
                    "generation": 0,
                    "detail": "unavailable",
                },
                {
                    "name": "optional_ready",
                    "status": "ready",
                    "generation": 1,
                    "detail": "",
                },
            ],
        }
        assert marker not in caplog.text

    asyncio.run(run())


def test_optional_catalog_collision_quarantines_the_whole_provider() -> None:
    async def run() -> None:
        required = RecordingProvider("required", ("mcp__required__read",))
        optional = RecordingProvider(
            "optional",
            ("mcp__optional__unique", "mcp__optional__collision"),
        )
        colliding = optional._snapshot.tools[1].model_copy(  # noqa: SLF001 - focused fixture
            update={"name": "mcp__required__read"}
        )
        optional._snapshot = compose_snapshot(  # noqa: SLF001 - focused fixture
            (
                (
                    optional._snapshot.tools[0],  # noqa: SLF001
                    optional._snapshot.descriptors[0],  # noqa: SLF001
                ),
                (
                    colliding,
                    descriptor_from_tool(
                        colliding,
                        source="mcp",
                        server_name="required",
                        risk_level="read",
                    ),
                ),
            ),
            generation=1,
        )
        catalog = ToolCatalog()

        published = await catalog.publish(
            (
                _registration(required, required=True),
                _registration(optional, required=False),
            )
        )

        assert tuple(published.tools_by_name) == ("mcp__required__read",)
        assert "mcp__optional__unique" not in published.tools_by_name
        assert optional.closed is True
        assert catalog.readiness.status == "degraded"

    asyncio.run(run())


def test_active_request_lease_keeps_its_retired_generation_alive() -> None:
    async def run() -> None:
        catalog = ToolCatalog()
        first_provider = RecordingProvider("first", ("mcp__first__read",))
        first = await catalog.publish((_registration(first_provider, required=True),))

        async with catalog.acquire_snapshot() as leased:
            second_provider = RecordingProvider("second", ("mcp__second__read",))
            second = await catalog.publish((_registration(second_provider, required=True),))

            assert leased is first
            assert leased.generation == 1
            assert catalog.current is second
            assert first_provider.closed is False

        assert first_provider.closed is True
        assert second_provider.closed is False

        await catalog.close()
        assert catalog.readiness.status == "closed"
        assert second_provider.close_calls == 1

    asyncio.run(run())


def test_replacement_and_shutdown_cleanup_are_bounded_and_failure_isolated() -> None:
    async def run() -> None:
        catalog = ToolCatalog(cleanup_timeout_seconds=0.01)
        slow = RecordingProvider(
            "slow",
            ("mcp__slow__read",),
            close_delay=1.0,
        )
        await catalog.publish((_registration(slow, required=True),))

        replacement = RecordingProvider("replacement", ("mcp__replacement__read",))
        started = asyncio.get_running_loop().time()
        published = await catalog.publish((_registration(replacement, required=True),))
        elapsed = asyncio.get_running_loop().time() - started

        assert elapsed < 0.25
        assert tuple(published.tools_by_name) == ("mcp__replacement__read",)
        assert slow.close_calls == 1
        assert slow.closed is True

        healthy = RecordingProvider("healthy", ("mcp__healthy__read",))
        failing = RecordingProvider(
            "failing",
            ("mcp__failing__read",),
            fail_at="close",
        )
        await catalog.publish(
            (
                _registration(healthy, required=True),
                _registration(failing, required=True),
            )
        )
        await catalog.close()

        assert catalog.readiness.status == "closed"
        assert failing.close_calls == 1
        assert healthy.close_calls == 1
        assert healthy.closed is True

    asyncio.run(run())


def test_outbound_configuration_binds_required_flags_without_starting_providers() -> None:
    required = RecordingProvider("required", ("mcp__required__read",))
    optional = RecordingProvider("optional", ("mcp__optional__read",))
    settings_value = OutboundMCPSettings(
        enabled=True,
        servers=(
            OutboundMCPServerSettings(
                name="required",
                endpoint="https://required.example/mcp",
                required=True,
            ),
            OutboundMCPServerSettings(
                name="optional",
                endpoint="https://optional.example/mcp",
                required=False,
            ),
        ),
    )

    registrations = outbound_provider_registrations(
        settings_value,
        {"required": required, "optional": optional},
    )

    assert [(item.name, item.required) for item in registrations] == [
        ("required", True),
        ("optional", False),
    ]
    assert required.started is False
    assert optional.started is False


@settings(max_examples=30, deadline=None)
@given(
    required_fails=st.booleans(),
    optional_failures=st.lists(st.booleans(), min_size=0, max_size=5),
)
def test_atomic_dependency_publication_property(
    required_fails: bool,
    optional_failures: list[bool],
) -> None:
    """Property 9: Atomic dependency and configuration publication.

    **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6**
    """

    async def run() -> None:
        catalog = ToolCatalog()
        baseline_provider = RecordingProvider("baseline", ("mcp__baseline__read",))
        baseline = await catalog.publish((_registration(baseline_provider, required=True),))
        required = RecordingProvider(
            "required",
            ("mcp__required__read",),
            fail_at="snapshot" if required_fails else None,
        )
        optionals = [
            RecordingProvider(
                f"optional_{index}",
                (f"mcp__optional_{index}__read",),
                fail_at="start" if fails else None,
            )
            for index, fails in enumerate(optional_failures)
        ]
        registrations = (
            _registration(required, required=True),
            *(_registration(provider, required=False) for provider in optionals),
        )

        if required_fails:
            with pytest.raises(ToolCatalogError):
                await catalog.publish(registrations)
            assert catalog.current is baseline
            assert tuple(baseline.tools_by_name) == ("mcp__baseline__read",)
            assert catalog.public_readiness() == {
                "ready": True,
                "status": "ready",
                "generation": 1,
            }
            assert required.closed is True
            assert all(not provider.started for provider in optionals)
            assert baseline_provider.closed is False
            return

        published = await catalog.publish(registrations)
        expected_names = ["mcp__required__read"]
        expected_names.extend(
            f"mcp__optional_{index}__read"
            for index, fails in enumerate(optional_failures)
            if not fails
        )
        assert list(published.tools_by_name) == expected_names
        assert published.generation == baseline.generation + 1
        assert tuple(baseline.tools_by_name) == ("mcp__baseline__read",)
        assert catalog.current is published
        assert catalog.readiness.status == ("degraded" if any(optional_failures) else "ready")
        assert catalog.readiness.ready is True
        assert baseline_provider.closed is True
        assert all(
            provider.closed is fails
            for provider, fails in zip(optionals, optional_failures, strict=True)
        )

    asyncio.run(run())
