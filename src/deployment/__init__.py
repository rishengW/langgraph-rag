"""Production deployment capability checks."""

from .topology import (
    DEFAULT_DEPLOYMENT_ENVIRONMENT,
    LOCAL_SINGLE_INSTANCE_BLOCKERS,
    MAX_CAPABILITY_IMPLEMENTATION_CHARS,
    REPLICA_COUNT_ENV_NAMES,
    REQUIRED_SHARED_STATE_CAPABILITIES,
    WORKER_COUNT_ENV_NAMES,
    DeploymentTopology,
    DeploymentTopologyConfigurationError,
    DeploymentTopologyError,
    SharedStateCapabilityName,
    SharedStateCapabilityRegistration,
    UnsupportedDeploymentTopologyError,
    ValidatedSharedStateCapabilities,
    load_deployment_topology,
    validate_shared_state_capabilities,
    validate_single_instance_deployment,
)

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
