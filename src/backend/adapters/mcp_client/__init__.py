"""Separately configured and lifecycle-managed outbound MCP client boundary.

This package is not imported or started by the inbound MCP or FastAPI entry
points. Connectors are injected and cannot select destinations, follow redirects,
or launch arbitrary commands outside the validated policy values they receive.
"""

from .config import (
    OutboundMCPServerSettings,
    OutboundMCPSettings,
    OutboundMCPTransport,
    load_outbound_mcp_settings,
)
from .lifecycle import (
    ResolvedOutboundCredential,
    outbound_provider_registrations,
    resolve_outbound_credentials,
)
from .runtime import (
    ConnectorFactory,
    ManagedOutboundMCPProvider,
    OutboundConnection,
    OutboundConnector,
    OutboundMCPRuntimeError,
    OutboundRedirect,
    build_managed_outbound_providers,
)
from .security import (
    ApprovedProcessTemplate,
    AsyncResolver,
    EndpointOrigin,
    ExecutableAllowlist,
    ManagedProcess,
    OutboundEndpointPolicy,
    OutboundSecurityError,
    ParsedEndpoint,
    PrivateEndpointApproval,
    ProcessHandle,
    SecureProcessLauncher,
    SpawnProcess,
    ValidatedEndpoint,
    parse_https_endpoint,
    parse_https_origin,
)

__all__ = [
    "ApprovedProcessTemplate",
    "AsyncResolver",
    "ConnectorFactory",
    "EndpointOrigin",
    "ExecutableAllowlist",
    "ManagedOutboundMCPProvider",
    "ManagedProcess",
    "OutboundConnection",
    "OutboundConnector",
    "OutboundEndpointPolicy",
    "OutboundMCPRuntimeError",
    "OutboundMCPServerSettings",
    "OutboundMCPSettings",
    "OutboundMCPTransport",
    "OutboundRedirect",
    "OutboundSecurityError",
    "ParsedEndpoint",
    "PrivateEndpointApproval",
    "ProcessHandle",
    "ResolvedOutboundCredential",
    "SecureProcessLauncher",
    "SpawnProcess",
    "ValidatedEndpoint",
    "build_managed_outbound_providers",
    "load_outbound_mcp_settings",
    "outbound_provider_registrations",
    "parse_https_endpoint",
    "parse_https_origin",
    "resolve_outbound_credentials",
]
