"""Outbound-only runtime preparation with no inbound or FastAPI lifecycle hooks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from src.errors import ConfigurationError
from ...mcp.models import ToolProvider, ToolProviderRegistration
from ...mcp.secrets import ResolvedSecret, SecretResolver
from .config import OutboundMCPSettings


@dataclass(frozen=True, slots=True)
class ResolvedOutboundCredential:
    """Runtime-only credential associated with one configured outbound server."""

    server_name: str
    authorization: ResolvedSecret | None = field(repr=False)


def outbound_provider_registrations(
    settings: OutboundMCPSettings,
    providers: Mapping[str, ToolProvider],
) -> tuple[ToolProviderRegistration, ...]:
    """Bind configured required flags to detached outbound provider instances.

    Constructing providers must not perform I/O. Their ``start`` methods remain
    inside ``ToolCatalog.publish``, which provides the atomic/quarantine policy.
    """

    if not isinstance(settings, OutboundMCPSettings):
        raise TypeError("Outbound provider binding requires OutboundMCPSettings")
    if not settings.enabled:
        if providers:
            raise ConfigurationError("Disabled outbound MCP cannot bind providers")
        return ()

    configured_names = {server.name for server in settings.servers}
    supplied_names = set(providers)
    if supplied_names != configured_names:
        raise ConfigurationError("Outbound MCP provider set does not match configuration")

    return tuple(
        ToolProviderRegistration(
            name=server.name,
            provider=providers[server.name],
            required=server.required,
        )
        for server in settings.servers
    )


async def resolve_outbound_credentials(
    settings: OutboundMCPSettings,
    resolver: SecretResolver,
) -> tuple[ResolvedOutboundCredential, ...]:
    """Resolve configured references without opening transports or publishing tools."""

    if not isinstance(settings, OutboundMCPSettings):
        raise TypeError("Outbound credential resolution requires OutboundMCPSettings")
    if not isinstance(resolver, SecretResolver):
        raise TypeError("Outbound credential resolution requires SecretResolver")
    if not settings.enabled:
        return ()

    resolved: list[ResolvedOutboundCredential] = []
    for server in settings.servers:
        authorization = None
        if server.authorization_secret_ref is not None:
            authorization = await resolver.resolve(server.authorization_secret_ref)
        resolved.append(
            ResolvedOutboundCredential(
                server_name=server.name,
                authorization=authorization,
            )
        )
    if len(resolved) != len(settings.servers):
        raise ConfigurationError("Outbound MCP credential preparation was incomplete")
    return tuple(resolved)


__all__ = [
    "ResolvedOutboundCredential",
    "outbound_provider_registrations",
    "resolve_outbound_credentials",
]
