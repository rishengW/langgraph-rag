"""Focused Task 5.1 tests for the separate outbound MCP configuration boundary."""

from __future__ import annotations

import asyncio
import json
import string
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.adapters.mcp_client import (
    OutboundMCPServerSettings,
    OutboundMCPSettings,
    load_outbound_mcp_settings,
    resolve_outbound_credentials,
)
from src.adapters.mcp_server.config import load_mcp_settings
from src.errors import ConfigurationError
from src.mcp import (
    EnvironmentSecretProvider,
    SecretReference,
    SecretResolver,
)


def _document(*, enabled: bool = True, secret_reference: object | None = None) -> dict[str, object]:
    server: dict[str, object] = {
        "name": "approved_docs",
        "transport": "streamable_http",
        "endpoint": "https://mcp.internal.example/service",
        "required": True,
    }
    if secret_reference is not None:
        server["authorization_secret_ref"] = secret_reference
    return {"version": 1, "enabled": enabled, "servers": [server]}


def _write_document(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_outbound_config_is_explicit_and_separate_from_inbound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert load_outbound_mcp_settings(environment={}) == OutboundMCPSettings()

    path = _write_document(tmp_path / "outbound.json", _document())
    environment = {
        "OUTBOUND_MCP_CONFIG_FILE": str(path),
        "OUTBOUND_MCP_ENABLED": "true",
    }
    outbound = load_outbound_mcp_settings(environment=environment)

    monkeypatch.delenv("MCP_ENABLED", raising=False)
    monkeypatch.setenv("OUTBOUND_MCP_CONFIG_FILE", str(path))
    monkeypatch.setenv("OUTBOUND_MCP_ENABLED", "true")
    inbound = load_mcp_settings(env_file="missing.env", config_file=None)

    assert outbound.enabled is True
    assert [server.name for server in outbound.servers] == ["approved_docs"]
    assert inbound.enabled is False


def test_outbound_enablement_fails_closed_without_a_dedicated_document() -> None:
    with pytest.raises(ConfigurationError, match="requires OUTBOUND_MCP_CONFIG_FILE"):
        load_outbound_mcp_settings(environment={"OUTBOUND_MCP_ENABLED": "true"})

    with pytest.raises(ConfigurationError, match="Unknown outbound MCP"):
        load_outbound_mcp_settings(environment={"OUTBOUND_MCP_TOKEN": "raw-secret"})


@pytest.mark.parametrize(
    "secret_configuration",
    [
        "raw-secret-value",
        {"provider": "env", "identifier": "TOKEN", "value": "raw-secret-value"},
    ],
)
def test_outbound_config_rejects_non_reference_secret_values(
    tmp_path: Path,
    secret_configuration: object,
) -> None:
    marker = "raw-secret-value"
    path = _write_document(
        tmp_path / "outbound.json",
        _document(secret_reference=secret_configuration),
    )

    with pytest.raises(ConfigurationError) as captured:
        load_outbound_mcp_settings(config_file=path, environment={})

    assert marker not in str(captured.value)


def test_runtime_resolves_only_typed_references_and_redacts_values(tmp_path: Path) -> None:
    marker = "runtime-only-secret-marker"
    path = _write_document(
        tmp_path / "outbound.json",
        _document(
            secret_reference={
                "provider": "env",
                "identifier": "APPROVED_DOCS_MCP_TOKEN",
            }
        ),
    )
    settings_value = load_outbound_mcp_settings(config_file=path, environment={})
    provider = EnvironmentSecretProvider(
        {"APPROVED_DOCS_MCP_TOKEN"},
        environment={"APPROVED_DOCS_MCP_TOKEN": marker},
    )
    resolver = SecretResolver({"env": provider})

    credentials = asyncio.run(resolve_outbound_credentials(settings_value, resolver))

    assert credentials[0].authorization is not None
    assert credentials[0].authorization.reveal() == marker
    assert marker not in repr(credentials)
    assert marker not in json.dumps(settings_value.to_config())
    with pytest.raises(TypeError, match="only SecretReference"):
        asyncio.run(resolver.resolve(marker))  # type: ignore[arg-type]


def test_disabled_outbound_settings_never_resolve_credentials() -> None:
    reference = SecretReference(provider="env", identifier="MISSING_TOKEN")
    settings_value = OutboundMCPSettings(
        enabled=False,
        servers=(
            OutboundMCPServerSettings(
                name="approved_docs",
                endpoint="https://mcp.internal.example/service",
                authorization_secret_ref=reference,
            ),
        ),
    )
    resolver = SecretResolver({"env": EnvironmentSecretProvider({"MISSING_TOKEN"}, environment={})})

    assert asyncio.run(resolve_outbound_credentials(settings_value, resolver)) == ()


@settings(max_examples=25, deadline=None)
@given(
    suffix=st.text(
        alphabet=string.ascii_letters + string.digits,
        min_size=1,
        max_size=64,
    )
)
def test_secret_reference_confinement_property(suffix: str) -> None:
    """Property 10: Secret-reference confinement.

    **Validates: Requirements 1.1, 1.4, 9.1, 9.2**
    """

    secret = f"runtime-secret::{suffix}"
    reference = SecretReference(provider="env", identifier="OUTBOUND_RUNTIME_TOKEN")
    settings_value = OutboundMCPSettings(
        enabled=True,
        servers=(
            OutboundMCPServerSettings(
                name="property_server",
                endpoint="https://mcp.internal.example/service",
                authorization_secret_ref=reference,
            ),
        ),
    )
    resolver = SecretResolver(
        {
            "env": EnvironmentSecretProvider(
                {reference.identifier},
                environment={reference.identifier: secret},
            )
        }
    )

    credentials = asyncio.run(resolve_outbound_credentials(settings_value, resolver))
    persisted = json.dumps(settings_value.to_config(), sort_keys=True)

    assert secret not in persisted
    assert secret not in repr(settings_value)
    assert secret not in repr(credentials)
    assert credentials[0].authorization is not None
    assert credentials[0].authorization.reveal() == secret
