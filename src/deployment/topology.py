"""Fail-closed validation for production deployment topology capabilities."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum

DEFAULT_DEPLOYMENT_ENVIRONMENT = "development"
WORKER_COUNT_ENV_NAMES = (
    "RAG_WORKER_COUNT",
    "WEB_CONCURRENCY",
    "UVICORN_WORKERS",
)
REPLICA_COUNT_ENV_NAMES = ("RAG_REPLICA_COUNT",)
LOCAL_SINGLE_INSTANCE_BLOCKERS = (
    "local SQLite session and checkpoint stores",
    "local Chroma collections",
    "local uploads and artifacts",
    "process-local locks, quota, and catalog state",
)
MAX_CAPABILITY_IMPLEMENTATION_CHARS = 128


class SharedStateCapabilityName(StrEnum):
    """Closed names for every prerequisite required by production scaling."""

    TRANSACTIONAL_PERSISTENCE = "transactional_persistence"
    MIGRATIONS = "migrations"
    DISTRIBUTED_LOCKS = "distributed_locks"
    OBJECT_STORAGE = "object_storage"
    SERVER_VECTOR_STORE = "server_vector_store"
    SHARED_QUOTA_STATE = "shared_quota_state"
    SHARED_CATALOG_STATE = "shared_catalog_state"
    TENANT_OWNERSHIP = "tenant_ownership"


REQUIRED_SHARED_STATE_CAPABILITIES = tuple(SharedStateCapabilityName)
_CAPABILITY_LABELS = {
    SharedStateCapabilityName.TRANSACTIONAL_PERSISTENCE: (
        "shared transactional session/checkpoint persistence"
    ),
    SharedStateCapabilityName.MIGRATIONS: "shared persistence migrations",
    SharedStateCapabilityName.DISTRIBUTED_LOCKS: "distributed per-resource locks",
    SharedStateCapabilityName.OBJECT_STORAGE: "object storage for uploads/artifacts",
    SharedStateCapabilityName.SERVER_VECTOR_STORE: "server-mode vector store",
    SharedStateCapabilityName.SHARED_QUOTA_STATE: "shared quota/rate/cost state",
    SharedStateCapabilityName.SHARED_CATALOG_STATE: "shared catalog-generation state",
    SharedStateCapabilityName.TENANT_OWNERSHIP: (
        "tenant IDs and ownership checks on every durable object"
    ),
}


class DeploymentTopologyError(RuntimeError):
    """Base error for an unsafe or ambiguous deployment topology."""


class DeploymentTopologyConfigurationError(DeploymentTopologyError):
    """Raised when deployment declarations are invalid or conflicting."""


class UnsupportedDeploymentTopologyError(DeploymentTopologyError):
    """Raised when requested scale exceeds validated runtime capabilities."""


@dataclass(frozen=True, slots=True)
class SharedStateCapabilityRegistration:
    """One code-provided shared-state implementation and its startup validator.

    Registration presence is the implementation signal, ``configured`` must be
    explicitly true, and ``validator`` must return exactly ``True`` after
    checking the configured runtime dependency. The gate never derives these
    attestations from environment booleans.
    """

    name: SharedStateCapabilityName
    implementation: str | None
    configured: bool
    validator: Callable[[], bool] | None = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ValidatedSharedStateCapabilities:
    """Immutable proof retained by an accepted scaled topology."""

    implementations: tuple[tuple[SharedStateCapabilityName, str], ...]

    @property
    def names(self) -> tuple[SharedStateCapabilityName, ...]:
        """Return validated capability names in canonical order."""

        return tuple(name for name, _ in self.implementations)


@dataclass(frozen=True, slots=True)
class DeploymentTopology:
    """Normalized process and replica counts for one deployment."""

    environment: str
    worker_count: int
    replica_count: int
    validated_shared_state: ValidatedSharedStateCapabilities | None = None

    @property
    def is_scaled(self) -> bool:
        """Whether the topology has more than one worker or replica."""

        return self.worker_count > 1 or self.replica_count > 1


def _positive_count(
    environ: Mapping[str, str],
    names: tuple[str, ...],
    *,
    dimension: str,
) -> int:
    configured: list[tuple[str, int]] = []
    for name in names:
        raw_value = environ.get(name)
        if raw_value is None:
            continue
        try:
            value = int(raw_value.strip())
        except (AttributeError, ValueError):
            raise DeploymentTopologyConfigurationError(
                f"{name} must be a positive integer"
            ) from None
        if value < 1:
            raise DeploymentTopologyConfigurationError(f"{name} must be a positive integer")
        configured.append((name, value))

    if not configured:
        return 1

    declared_values = {value for _, value in configured}
    if len(declared_values) > 1:
        declarations = ", ".join(f"{name}={value}" for name, value in configured)
        raise DeploymentTopologyConfigurationError(
            f"Conflicting {dimension} count declarations: {declarations}"
        )
    return configured[0][1]


def load_deployment_topology(
    *,
    environment: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> DeploymentTopology:
    """Read and normalize deployment count declarations without side effects."""

    source = os.environ if environ is None else environ
    raw_environment = source.get("RAG_ENV", DEFAULT_DEPLOYMENT_ENVIRONMENT)
    normalized_environment = (environment if environment is not None else raw_environment).strip()
    normalized_environment = normalized_environment.casefold() or DEFAULT_DEPLOYMENT_ENVIRONMENT
    return DeploymentTopology(
        environment=normalized_environment,
        worker_count=_positive_count(
            source,
            WORKER_COUNT_ENV_NAMES,
            dimension="worker",
        ),
        replica_count=_positive_count(
            source,
            REPLICA_COUNT_ENV_NAMES,
            dimension="replica",
        ),
    )


def _capability_list(names: Iterable[SharedStateCapabilityName]) -> str:
    return ", ".join(_CAPABILITY_LABELS[name] for name in names)


def _unsupported_capabilities(
    *,
    missing: tuple[SharedStateCapabilityName, ...] = (),
    unimplemented: tuple[SharedStateCapabilityName, ...] = (),
    unconfigured: tuple[SharedStateCapabilityName, ...] = (),
    unvalidated: tuple[SharedStateCapabilityName, ...] = (),
) -> UnsupportedDeploymentTopologyError:
    reasons: list[str] = []
    if missing:
        reasons.append(f"missing registrations: {_capability_list(missing)}")
    if unimplemented:
        reasons.append(f"not implemented: {_capability_list(unimplemented)}")
    if unconfigured:
        reasons.append(f"not explicitly configured: {_capability_list(unconfigured)}")
    if unvalidated:
        reasons.append(f"validation failed: {_capability_list(unvalidated)}")
    detail = "; ".join(reasons) or "capability validation did not produce a complete proof"
    return UnsupportedDeploymentTopologyError(
        "Every shared-state prerequisite must be implemented, explicitly configured, "
        f"and validated before production scaling ({detail})."
    )


def validate_shared_state_capabilities(
    capabilities: Iterable[SharedStateCapabilityRegistration] | None,
) -> ValidatedSharedStateCapabilities:
    """Validate one exact, complete shared-state capability generation.

    Structural failures are rejected before any validator runs. Validator
    exceptions are intentionally reduced to capability names so backend details
    and configuration values cannot cross the startup error boundary.
    """

    registrations = tuple(capabilities or ())
    by_name: dict[SharedStateCapabilityName, SharedStateCapabilityRegistration] = {}
    duplicates: list[SharedStateCapabilityName] = []

    for registration in registrations:
        if not isinstance(registration.name, SharedStateCapabilityName):
            raise DeploymentTopologyConfigurationError(
                "Shared-state capability registrations must use canonical capability names"
            )
        if registration.name in by_name:
            duplicates.append(registration.name)
            continue
        by_name[registration.name] = registration

    if duplicates:
        duplicate_names = tuple(
            name for name in REQUIRED_SHARED_STATE_CAPABILITIES if name in duplicates
        )
        raise DeploymentTopologyConfigurationError(
            f"Duplicate shared-state capability registrations: {_capability_list(duplicate_names)}"
        )

    missing = tuple(name for name in REQUIRED_SHARED_STATE_CAPABILITIES if name not in by_name)
    unimplemented: list[SharedStateCapabilityName] = []
    unconfigured: list[SharedStateCapabilityName] = []
    missing_validators: list[SharedStateCapabilityName] = []
    implementations: dict[SharedStateCapabilityName, str] = {}

    for name in REQUIRED_SHARED_STATE_CAPABILITIES:
        registered_capability = by_name.get(name)
        if registered_capability is None:
            continue

        implementation = registered_capability.implementation
        if not isinstance(implementation, str) or not implementation.strip():
            unimplemented.append(name)
        else:
            normalized_implementation = implementation.strip()
            if (
                len(normalized_implementation) > MAX_CAPABILITY_IMPLEMENTATION_CHARS
                or not normalized_implementation.isprintable()
            ):
                raise DeploymentTopologyConfigurationError(
                    f"Invalid implementation identifier for {_CAPABILITY_LABELS[name]}"
                )
            implementations[name] = normalized_implementation

        if registered_capability.configured is not True:
            unconfigured.append(name)
        if registered_capability.validator is None or not callable(registered_capability.validator):
            missing_validators.append(name)

    structurally_unvalidated = tuple(
        name for name in REQUIRED_SHARED_STATE_CAPABILITIES if name in missing_validators
    )
    if missing or unimplemented or unconfigured or structurally_unvalidated:
        raise _unsupported_capabilities(
            missing=missing,
            unimplemented=tuple(unimplemented),
            unconfigured=tuple(unconfigured),
            unvalidated=structurally_unvalidated,
        )

    failed_validation: list[SharedStateCapabilityName] = []
    for name in REQUIRED_SHARED_STATE_CAPABILITIES:
        validator = by_name[name].validator
        if validator is None:  # Narrowed structurally above; retained for strict typing.
            failed_validation.append(name)
            continue
        try:
            if validator() is not True:
                failed_validation.append(name)
        except Exception:
            failed_validation.append(name)

    if failed_validation:
        raise _unsupported_capabilities(unvalidated=tuple(failed_validation))

    return ValidatedSharedStateCapabilities(
        implementations=tuple(
            (name, implementations[name]) for name in REQUIRED_SHARED_STATE_CAPABILITIES
        )
    )


def validate_single_instance_deployment(
    *,
    environment: str | None = None,
    environ: Mapping[str, str] | None = None,
    shared_state_capabilities: Iterable[SharedStateCapabilityRegistration] | None = None,
) -> DeploymentTopology:
    """Validate production scale against one complete shared-state proof.

    Development retains its existing reload and multi-process flexibility.
    Single-instance production remains the default and needs no shared-state
    registrations. A scaled production topology is accepted only after every
    requirement 11.5 capability passes code-provided runtime validation.
    """

    topology = load_deployment_topology(environment=environment, environ=environ)
    if topology.environment != "production" or not topology.is_scaled:
        return topology

    try:
        validated = validate_shared_state_capabilities(shared_state_capabilities)
    except UnsupportedDeploymentTopologyError as exc:
        blockers = ", ".join(LOCAL_SINGLE_INSTANCE_BLOCKERS)
        raise UnsupportedDeploymentTopologyError(
            "Unsupported production deployment topology: "
            f"workers={topology.worker_count}, replicas={topology.replica_count}. "
            f"The runtime remains single-instance while {blockers} remain authoritative. {exc}"
        ) from None

    return replace(topology, validated_shared_state=validated)


__all__ = [
    "DEFAULT_DEPLOYMENT_ENVIRONMENT",
    "LOCAL_SINGLE_INSTANCE_BLOCKERS",
    "MAX_CAPABILITY_IMPLEMENTATION_CHARS",
    "REPLICA_COUNT_ENV_NAMES",
    "REQUIRED_SHARED_STATE_CAPABILITIES",
    "WORKER_COUNT_ENV_NAMES",
    "DeploymentTopology",
    "DeploymentTopologyConfigurationError",
    "DeploymentTopologyError",
    "SharedStateCapabilityName",
    "SharedStateCapabilityRegistration",
    "UnsupportedDeploymentTopologyError",
    "ValidatedSharedStateCapabilities",
    "load_deployment_topology",
    "validate_shared_state_capabilities",
    "validate_single_instance_deployment",
]
