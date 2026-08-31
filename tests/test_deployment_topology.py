from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hypothesis import given
from hypothesis import strategies as st

from src.adapters.mcp_server.config import MCPSettings
from src.adapters.mcp_server.lifecycle import initialize_runtime
from src.chat import api as chat_api
from src.deployment import (
    REPLICA_COUNT_ENV_NAMES,
    REQUIRED_SHARED_STATE_CAPABILITIES,
    WORKER_COUNT_ENV_NAMES,
    DeploymentTopologyConfigurationError,
    SharedStateCapabilityName,
    SharedStateCapabilityRegistration,
    UnsupportedDeploymentTopologyError,
    validate_single_instance_deployment,
)
from src.qa import api as qa_api


def _set_single_instance_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scaled_name: str,
) -> None:
    worker_value = "2" if scaled_name in WORKER_COUNT_ENV_NAMES else "1"
    replica_value = "2" if scaled_name in REPLICA_COUNT_ENV_NAMES else "1"
    for name in WORKER_COUNT_ENV_NAMES:
        monkeypatch.setenv(name, worker_value)
    for name in REPLICA_COUNT_ENV_NAMES:
        monkeypatch.setenv(name, replica_value)


def _validation_succeeds() -> bool:
    return True


def _ready_capability(
    name: SharedStateCapabilityName,
) -> SharedStateCapabilityRegistration:
    return SharedStateCapabilityRegistration(
        name=name,
        implementation=f"test.{name.value}",
        configured=True,
        validator=_validation_succeeds,
    )


def _ready_capabilities() -> tuple[SharedStateCapabilityRegistration, ...]:
    return tuple(_ready_capability(name) for name in REQUIRED_SHARED_STATE_CAPABILITIES)


def test_production_accepts_one_worker_and_replica_with_matching_aliases() -> None:
    topology = validate_single_instance_deployment(
        environment=" Production ",
        environ={
            "RAG_WORKER_COUNT": "1",
            "WEB_CONCURRENCY": "1",
            "UVICORN_WORKERS": "1",
            "RAG_REPLICA_COUNT": "1",
        },
    )

    assert topology.environment == "production"
    assert topology.worker_count == 1
    assert topology.replica_count == 1
    assert topology.validated_shared_state is None


@pytest.mark.parametrize("worker_name", WORKER_COUNT_ENV_NAMES)
def test_every_worker_count_declaration_rejects_scaled_production(worker_name: str) -> None:
    with pytest.raises(UnsupportedDeploymentTopologyError, match="workers=2, replicas=1"):
        validate_single_instance_deployment(
            environment="production",
            environ={worker_name: "2", "RAG_REPLICA_COUNT": "1"},
        )


def test_replica_count_rejects_scaled_production_and_names_local_blockers() -> None:
    with pytest.raises(UnsupportedDeploymentTopologyError) as exc_info:
        validate_single_instance_deployment(
            environment="production",
            environ={"RAG_WORKER_COUNT": "1", "RAG_REPLICA_COUNT": "2"},
        )

    message = str(exc_info.value)
    assert "workers=1, replicas=2" in message
    assert "local SQLite" in message
    assert "local Chroma" in message
    assert "process-local locks" in message
    assert "implemented, explicitly configured, and validated" in message


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RAG_WORKER_COUNT", "0"),
        ("RAG_WORKER_COUNT", "not-a-number"),
        ("RAG_REPLICA_COUNT", "-1"),
        ("RAG_REPLICA_COUNT", ""),
    ],
)
def test_invalid_count_declarations_fail_closed(name: str, value: str) -> None:
    with pytest.raises(DeploymentTopologyConfigurationError, match=name):
        validate_single_instance_deployment(
            environment="production",
            environ={name: value},
        )


def test_conflicting_worker_count_aliases_fail_closed() -> None:
    with pytest.raises(DeploymentTopologyConfigurationError, match="Conflicting worker"):
        validate_single_instance_deployment(
            environment="production",
            environ={"RAG_WORKER_COUNT": "1", "WEB_CONCURRENCY": "2"},
        )


def test_development_retains_multi_process_flexibility() -> None:
    topology = validate_single_instance_deployment(
        environment="development",
        environ={"RAG_WORKER_COUNT": "3", "RAG_REPLICA_COUNT": "4"},
    )

    assert topology.worker_count == 3
    assert topology.replica_count == 4
    assert topology.validated_shared_state is None


def test_scaled_production_accepts_only_a_complete_validated_capability_generation() -> None:
    validation_calls: list[SharedStateCapabilityName] = []

    def registration(name: SharedStateCapabilityName) -> SharedStateCapabilityRegistration:
        def validate() -> bool:
            validation_calls.append(name)
            return True

        return SharedStateCapabilityRegistration(
            name=name,
            implementation=f"production.{name.value}",
            configured=True,
            validator=validate,
        )

    capabilities = tuple(registration(name) for name in REQUIRED_SHARED_STATE_CAPABILITIES)
    topology = validate_single_instance_deployment(
        environment="production",
        environ={"RAG_WORKER_COUNT": "2", "RAG_REPLICA_COUNT": "3"},
        shared_state_capabilities=capabilities,
    )

    assert topology.is_scaled is True
    assert topology.validated_shared_state is not None
    assert topology.validated_shared_state.names == REQUIRED_SHARED_STATE_CAPABILITIES
    assert validation_calls == list(REQUIRED_SHARED_STATE_CAPABILITIES)


@pytest.mark.parametrize("missing_name", REQUIRED_SHARED_STATE_CAPABILITIES)
def test_scaled_production_rejects_each_missing_capability_before_validation(
    missing_name: SharedStateCapabilityName,
) -> None:
    validation_calls = 0

    def validate() -> bool:
        nonlocal validation_calls
        validation_calls += 1
        return True

    capabilities = tuple(
        replace(_ready_capability(name), validator=validate)
        for name in REQUIRED_SHARED_STATE_CAPABILITIES
        if name is not missing_name
    )

    with pytest.raises(UnsupportedDeploymentTopologyError, match="missing registrations"):
        validate_single_instance_deployment(
            environment="production",
            environ={"RAG_WORKER_COUNT": "2", "RAG_REPLICA_COUNT": "1"},
            shared_state_capabilities=capabilities,
        )

    assert validation_calls == 0


@pytest.mark.parametrize("failure", ["implementation", "configuration", "validator"])
def test_scaled_production_rejects_incomplete_capability_evidence(failure: str) -> None:
    capabilities = list(_ready_capabilities())
    target = capabilities[0]
    if failure == "implementation":
        capabilities[0] = replace(target, implementation=None)
    elif failure == "configuration":
        capabilities[0] = replace(target, configured=False)
    else:
        capabilities[0] = replace(target, validator=None)

    with pytest.raises(UnsupportedDeploymentTopologyError):
        validate_single_instance_deployment(
            environment="production",
            environ={"RAG_WORKER_COUNT": "2", "RAG_REPLICA_COUNT": "1"},
            shared_state_capabilities=capabilities,
        )


def test_scaled_production_sanitizes_failed_runtime_validation() -> None:
    secret_marker = "postgresql://admin:secret@shared-state.internal/db"

    def fail_validation() -> bool:
        raise RuntimeError(secret_marker)

    capabilities = list(_ready_capabilities())
    capabilities[0] = replace(capabilities[0], validator=fail_validation)

    with pytest.raises(UnsupportedDeploymentTopologyError) as exc_info:
        validate_single_instance_deployment(
            environment="production",
            environ={"RAG_WORKER_COUNT": "2", "RAG_REPLICA_COUNT": "1"},
            shared_state_capabilities=capabilities,
        )

    message = str(exc_info.value)
    assert "validation failed" in message
    assert secret_marker not in message


def test_duplicate_capability_registration_fails_closed_before_validation() -> None:
    capabilities = (*_ready_capabilities(), _ready_capabilities()[0])

    with pytest.raises(
        DeploymentTopologyConfigurationError,
        match="Duplicate shared-state capability registrations",
    ):
        validate_single_instance_deployment(
            environment="production",
            environ={"RAG_WORKER_COUNT": "1", "RAG_REPLICA_COUNT": "2"},
            shared_state_capabilities=capabilities,
        )


def test_single_instance_does_not_require_or_run_shared_state_validators() -> None:
    def unexpected_validation() -> bool:
        raise AssertionError("single-instance startup must not validate unused shared backends")

    capabilities = tuple(
        replace(capability, validator=unexpected_validation) for capability in _ready_capabilities()
    )
    topology = validate_single_instance_deployment(
        environment="production",
        environ={"RAG_WORKER_COUNT": "1", "RAG_REPLICA_COUNT": "1"},
        shared_state_capabilities=capabilities,
    )

    assert topology.validated_shared_state is None


@given(
    counts=st.one_of(
        st.tuples(st.integers(min_value=2, max_value=256), st.integers(1, 256)),
        st.tuples(st.integers(1, 256), st.integers(min_value=2, max_value=256)),
    )
)
def test_property_12_local_authority_never_accepts_scaled_production(
    counts: tuple[int, int],
) -> None:
    """Property 12: local authority requires a single production instance.

    **Validates: Requirements 11.1, 11.4, 11.6**
    """

    worker_count, replica_count = counts
    with pytest.raises(UnsupportedDeploymentTopologyError):
        validate_single_instance_deployment(
            environment="production",
            environ={
                "RAG_WORKER_COUNT": str(worker_count),
                "RAG_REPLICA_COUNT": str(replica_count),
            },
        )


@given(
    registered_names=st.sets(
        st.sampled_from(REQUIRED_SHARED_STATE_CAPABILITIES),
        max_size=len(REQUIRED_SHARED_STATE_CAPABILITIES) - 1,
    )
)
def test_property_12_incomplete_shared_state_proof_never_enables_scaling(
    registered_names: set[SharedStateCapabilityName],
) -> None:
    """Property 12: every shared-state prerequisite is mandatory.

    **Validates: Requirements 11.4, 11.5**
    """

    capabilities = tuple(
        _ready_capability(name)
        for name in REQUIRED_SHARED_STATE_CAPABILITIES
        if name in registered_names
    )
    with pytest.raises(UnsupportedDeploymentTopologyError):
        validate_single_instance_deployment(
            environment="production",
            environ={"RAG_WORKER_COUNT": "2", "RAG_REPLICA_COUNT": "2"},
            shared_state_capabilities=capabilities,
        )


@pytest.mark.parametrize(
    "app_factory",
    [
        pytest.param(qa_api.create_app, id="qa"),
        pytest.param(chat_api.create_app, id="chat"),
    ],
)
@pytest.mark.parametrize("scaled_name", ["RAG_WORKER_COUNT", "RAG_REPLICA_COUNT"])
def test_fastapi_startup_rejects_scaled_production_before_local_resources(
    monkeypatch: pytest.MonkeyPatch,
    app_factory: Callable[[], FastAPI],
    scaled_name: str,
) -> None:
    _set_single_instance_environment(monkeypatch, scaled_name=scaled_name)
    monkeypatch.setenv("RAG_ENV", "production")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    app = app_factory()
    with (
        pytest.raises(UnsupportedDeploymentTopologyError),
        TestClient(app),
    ):
        pytest.fail("scaled production startup unexpectedly reached the serving state")


@pytest.mark.parametrize("scaled_name", ["RAG_WORKER_COUNT", "RAG_REPLICA_COUNT"])
def test_mcp_startup_rejects_scaled_production_before_tool_publication(
    monkeypatch: pytest.MonkeyPatch,
    scaled_name: str,
) -> None:
    _set_single_instance_environment(monkeypatch, scaled_name=scaled_name)

    async def run() -> None:
        settings = MCPSettings(enabled=True, environment="production")
        with pytest.raises(UnsupportedDeploymentTopologyError):
            await initialize_runtime(settings, service=object())

    asyncio.run(run())


def test_mcp_startup_does_not_trust_environment_only_shared_state_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_single_instance_environment(monkeypatch, scaled_name="RAG_REPLICA_COUNT")
    for name in REQUIRED_SHARED_STATE_CAPABILITIES:
        monkeypatch.setenv(f"RAG_SHARED_{name.value.upper()}", "implemented,configured,validated")

    async def run() -> None:
        settings = MCPSettings(enabled=True, environment="production")
        with pytest.raises(UnsupportedDeploymentTopologyError):
            await initialize_runtime(settings, service=object())

    asyncio.run(run())


def test_production_launch_configuration_declares_one_instance() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    env_example = Path(".env.example").read_text(encoding="utf-8")

    for declaration in ("RAG_WORKER_COUNT=1", "RAG_REPLICA_COUNT=1"):
        assert declaration in dockerfile
        assert declaration in env_example
