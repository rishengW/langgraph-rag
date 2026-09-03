"""Fail-closed endpoint and process policy for outbound MCP transports."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import socket
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

from src.errors import ConfigurationError

AsyncResolver = Callable[[str, int], Awaitable[Sequence[str]]]
IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network

_MAX_ENDPOINT_CHARS = 2_048
_MAX_REDIRECT_ORIGINS = 16
_MAX_REDIRECTS = 5
_MAX_PROCESS_ARGUMENTS = 64
_MAX_PROCESS_ARGUMENT_CHARS = 4_096
_MAX_PROCESS_ENVIRONMENT = 64
_TEMPLATE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_METADATA_HOSTS = frozenset(
    {
        "metadata",
        "metadata.google.internal",
        "metadata.azure.internal",
        "instance-data.ec2.internal",
    }
)
_PROHIBITED_HOST_SUFFIXES = (".internal", ".local", ".localhost")


class OutboundSecurityError(ConfigurationError):
    """Sanitized outbound endpoint or process policy rejection."""

    code = "OUTBOUND_MCP_SECURITY_ERROR"


async def _default_resolver(host: str, port: int) -> Sequence[str]:
    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return tuple(dict.fromkeys(str(record[4][0]) for record in records))


@dataclass(frozen=True, slots=True)
class EndpointOrigin:
    """Canonical HTTPS origin used by the exact endpoint allowlist."""

    host: str
    port: int

    @property
    def value(self) -> str:
        authority = f"[{self.host}]" if ":" in self.host else self.host
        if self.port != 443:
            authority = f"{authority}:{self.port}"
        return f"https://{authority}"


@dataclass(frozen=True, slots=True)
class ParsedEndpoint:
    """Canonical endpoint syntax before DNS policy is applied."""

    url: str
    origin: EndpointOrigin
    path: str


@dataclass(frozen=True, slots=True)
class ValidatedEndpoint:
    """One endpoint plus the exact addresses approved for its next connection."""

    url: str
    origin: EndpointOrigin
    addresses: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrivateEndpointApproval:
    """Narrow deployment approval for exact hosts inside explicit private CIDRs.

    This deliberately has no general ``allow_private`` switch. Loopback, link-local,
    multicast, unspecified, reserved, and metadata destinations remain prohibited.
    """

    approval_id: str
    allowed_hosts: frozenset[str]
    allowed_cidrs: tuple[str, ...]
    _networks: tuple[IPNetwork, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.approval_id, str)
            or not self.approval_id.strip()
            or len(self.approval_id) > 128
            or _contains_control(self.approval_id)
        ):
            raise OutboundSecurityError("Private endpoint approval identifier is invalid")
        hosts = frozenset(_normalize_host(host) for host in self.allowed_hosts)
        if not hosts:
            raise OutboundSecurityError("Private endpoint approval requires exact hosts")
        networks: list[IPNetwork] = []
        for value in self.allowed_cidrs:
            try:
                network = ipaddress.ip_network(value, strict=True)
            except ValueError:
                raise OutboundSecurityError("Private endpoint approval CIDR is invalid") from None
            minimum_prefix = 8 if network.version == 4 else 32
            if network.prefixlen < minimum_prefix:
                raise OutboundSecurityError("Private endpoint approval CIDR is too broad")
            first = network.network_address
            last = network.broadcast_address
            if not first.is_private or not last.is_private:
                raise OutboundSecurityError("Private endpoint approval must contain private CIDRs")
            if any(_is_always_prohibited(address) for address in (first, last)):
                raise OutboundSecurityError(
                    "Private endpoint approval contains prohibited addresses"
                )
            networks.append(network)
        if not networks:
            raise OutboundSecurityError("Private endpoint approval requires explicit CIDRs")
        object.__setattr__(self, "allowed_hosts", hosts)
        object.__setattr__(self, "allowed_cidrs", tuple(str(network) for network in networks))
        object.__setattr__(self, "_networks", tuple(networks))

    def permits(self, host: str, address: IPAddress) -> bool:
        """Return whether this exact host/address pair has narrow approval."""

        return host in self.allowed_hosts and any(address in network for network in self._networks)


def parse_https_endpoint(value: str) -> ParsedEndpoint:
    """Validate and canonicalize one secret-free HTTPS endpoint URL."""

    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= _MAX_ENDPOINT_CHARS
        or _contains_control(value)
        or any(character.isspace() for character in value)
        or "\\" in value
    ):
        raise OutboundSecurityError("Outbound MCP endpoint is invalid")
    try:
        parsed = urlsplit(value)
        port = parsed.port or 443
    except ValueError:
        raise OutboundSecurityError("Outbound MCP endpoint is malformed") from None
    if parsed.scheme.lower() != "https":
        raise OutboundSecurityError("Outbound MCP remote endpoints require HTTPS")
    if not parsed.netloc or not parsed.hostname:
        raise OutboundSecurityError("Outbound MCP endpoint requires a host")
    if parsed.username is not None or parsed.password is not None:
        raise OutboundSecurityError("Outbound MCP endpoint cannot contain credentials")
    if parsed.query or parsed.fragment:
        raise OutboundSecurityError("Outbound MCP endpoint cannot contain query or fragment")
    if not 1 <= port <= 65_535:
        raise OutboundSecurityError("Outbound MCP endpoint port is invalid")

    host = _normalize_host(parsed.hostname)
    if _host_is_always_prohibited(host):
        raise OutboundSecurityError("Outbound MCP endpoint host is prohibited")
    literal = _literal_address(host)
    if literal is not None and _is_always_prohibited(literal):
        raise OutboundSecurityError("Outbound MCP endpoint address is prohibited")

    origin = EndpointOrigin(host=host, port=port)
    path = parsed.path or "/"
    authority = f"[{host}]" if ":" in host else host
    if port != 443:
        authority = f"{authority}:{port}"
    canonical = urlunsplit(("https", authority, path, "", ""))
    return ParsedEndpoint(url=canonical, origin=origin, path=path)


def parse_https_origin(value: str) -> EndpointOrigin:
    """Validate one configured redirect origin with no path scope ambiguity."""

    parsed = parse_https_endpoint(value)
    if parsed.path != "/":
        raise OutboundSecurityError("Outbound MCP redirect allowlist entries must be origins")
    return parsed.origin


class OutboundEndpointPolicy:
    """Resolve and pin every initial/redirected outbound MCP destination."""

    def __init__(
        self,
        endpoint: str,
        *,
        allowed_redirect_origins: Sequence[str] = (),
        max_redirects: int = 3,
        resolver: AsyncResolver = _default_resolver,
        private_approval: PrivateEndpointApproval | None = None,
    ) -> None:
        if (
            isinstance(max_redirects, bool)
            or not isinstance(max_redirects, int)
            or not 0 <= max_redirects <= _MAX_REDIRECTS
        ):
            raise OutboundSecurityError("Outbound MCP redirect limit is invalid")
        origins = tuple(allowed_redirect_origins)
        if len(origins) > _MAX_REDIRECT_ORIGINS:
            raise OutboundSecurityError("Outbound MCP redirect origin count exceeds its hard limit")
        initial = parse_https_endpoint(endpoint)
        redirect_origins = tuple(parse_https_origin(value) for value in origins)
        if len(set(redirect_origins)) != len(redirect_origins):
            raise OutboundSecurityError("Outbound MCP redirect origins must be unique")
        if not callable(resolver):
            raise TypeError("Outbound endpoint resolver must be callable")
        if private_approval is not None and not isinstance(
            private_approval, PrivateEndpointApproval
        ):
            raise TypeError("Private endpoint approval has an invalid type")
        self._initial = initial
        self._allowed_origins = frozenset((initial.origin, *redirect_origins))
        self._max_redirects = max_redirects
        self._resolver = resolver
        self._private_approval = private_approval

    @property
    def max_redirects(self) -> int:
        return self._max_redirects

    async def validate_initial(self) -> ValidatedEndpoint:
        """Resolve the configured initial endpoint before any connector I/O."""

        return await self._resolve(self._initial)

    async def validate_redirect(self, location: str, *, redirect_number: int) -> ValidatedEndpoint:
        """Independently validate one connector-reported redirect target."""

        if (
            isinstance(redirect_number, bool)
            or not isinstance(redirect_number, int)
            or redirect_number < 1
            or redirect_number > self._max_redirects
        ):
            raise OutboundSecurityError("Outbound MCP redirect limit exceeded")
        parsed = parse_https_endpoint(location)
        if parsed.origin not in self._allowed_origins:
            raise OutboundSecurityError("Outbound MCP redirect target is not allowlisted")
        return await self._resolve(parsed)

    def validate_connection_peer(self, endpoint: ValidatedEndpoint, peer_address: str) -> None:
        """Reject DNS rebinding unless the connected peer was in the resolved set."""

        address = _parse_address(peer_address)
        _validate_address(
            endpoint.origin.host,
            address,
            private_approval=self._private_approval,
        )
        if str(address) not in endpoint.addresses:
            raise OutboundSecurityError("Outbound MCP connection peer was not pre-resolved")

    async def _resolve(self, endpoint: ParsedEndpoint) -> ValidatedEndpoint:
        literal = _literal_address(endpoint.origin.host)
        if literal is None:
            try:
                raw_addresses = await self._resolver(endpoint.origin.host, endpoint.origin.port)
            except (OSError, UnicodeError):
                raise OutboundSecurityError(
                    "Outbound MCP endpoint could not be resolved safely"
                ) from None
        else:
            raw_addresses = (str(literal),)
        if not raw_addresses:
            raise OutboundSecurityError("Outbound MCP endpoint did not resolve")

        addresses: list[str] = []
        for raw_address in raw_addresses:
            address = _parse_address(raw_address)
            _validate_address(
                endpoint.origin.host,
                address,
                private_approval=self._private_approval,
            )
            normalized = str(address)
            if normalized not in addresses:
                addresses.append(normalized)
        return ValidatedEndpoint(
            url=endpoint.url,
            origin=endpoint.origin,
            addresses=tuple(addresses),
        )


@dataclass(frozen=True, slots=True)
class ApprovedProcessTemplate:
    """One exact executable/argv/environment tuple in a deployment allowlist."""

    template_id: str
    executable: Path
    arguments: tuple[str, ...] = ()
    cwd: Path | None = None
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not _TEMPLATE_ID_PATTERN.fullmatch(self.template_id):
            raise OutboundSecurityError("Outbound MCP process template identifier is invalid")
        executable = Path(self.executable)
        if not executable.is_absolute():
            raise OutboundSecurityError("Outbound MCP executable allowlist requires absolute paths")
        arguments = tuple(self.arguments)
        if len(arguments) > _MAX_PROCESS_ARGUMENTS:
            raise OutboundSecurityError("Outbound MCP fixed argument count exceeds its hard limit")
        for argument in arguments:
            if (
                not isinstance(argument, str)
                or len(argument) > _MAX_PROCESS_ARGUMENT_CHARS
                or _contains_control(argument)
            ):
                raise OutboundSecurityError("Outbound MCP fixed process argument is invalid")
        cwd = None if self.cwd is None else Path(self.cwd)
        if cwd is not None and not cwd.is_absolute():
            raise OutboundSecurityError("Outbound MCP process working directory must be absolute")
        environment = dict(self.environment)
        if len(environment) > _MAX_PROCESS_ENVIRONMENT:
            raise OutboundSecurityError("Outbound MCP fixed environment exceeds its hard limit")
        for key, value in environment.items():
            if (
                not isinstance(key, str)
                or not isinstance(value, str)
                or not key
                or "=" in key
                or _contains_control(key)
                or "\x00" in value
            ):
                raise OutboundSecurityError("Outbound MCP fixed environment is invalid")
        object.__setattr__(self, "executable", executable.resolve(strict=False))
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "cwd", None if cwd is None else cwd.resolve(strict=False))
        object.__setattr__(self, "environment", MappingProxyType(environment))


class ExecutableAllowlist:
    """Resolve only exact process templates approved by deployment policy."""

    def __init__(
        self,
        templates: Mapping[str, ApprovedProcessTemplate],
        *,
        hosted_production: bool = True,
        allow_hosted_production: bool = False,
    ) -> None:
        if not isinstance(hosted_production, bool) or not isinstance(allow_hosted_production, bool):
            raise OutboundSecurityError("Outbound MCP process environment policy is invalid")
        validated = dict(templates)
        for template_id, template in validated.items():
            if (
                not isinstance(template, ApprovedProcessTemplate)
                or template_id != template.template_id
            ):
                raise OutboundSecurityError("Outbound MCP executable allowlist is invalid")
        self._templates: Mapping[str, ApprovedProcessTemplate] = MappingProxyType(validated)
        self._hosted_production = hosted_production
        self._allow_hosted_production = allow_hosted_production

    def resolve(self, template_id: str) -> ApprovedProcessTemplate:
        """Return one exact template or fail before process creation."""

        if self._hosted_production and not self._allow_hosted_production:
            raise OutboundSecurityError("Outbound MCP stdio is disabled in hosted production")
        template = self._templates.get(template_id)
        if template is None:
            raise OutboundSecurityError("Outbound MCP process template is not allowlisted")
        return template


@runtime_checkable
class ProcessHandle(Protocol):
    """Minimal subprocess handle used for bounded cleanup."""

    @property
    def returncode(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


SpawnProcess = Callable[[ApprovedProcessTemplate], Awaitable[ProcessHandle]]


async def _spawn_process(template: ApprovedProcessTemplate) -> ProcessHandle:
    process = await asyncio.create_subprocess_exec(
        str(template.executable),
        *template.arguments,
        cwd=None if template.cwd is None else str(template.cwd),
        env=dict(template.environment),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    return process


@dataclass(slots=True)
class ManagedProcess:
    """A process whose terminate/kill sequence has one hard shutdown bound."""

    process: ProcessHandle
    shutdown_timeout_seconds: float
    _closed: bool = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process.returncode is not None:
            return
        try:
            self.process.terminate()
        except ProcessLookupError:
            return
        try:
            async with asyncio.timeout(self.shutdown_timeout_seconds):
                await self.process.wait()
                return
        except TimeoutError:
            pass
        try:
            self.process.kill()
        except ProcessLookupError:
            return
        try:
            async with asyncio.timeout(self.shutdown_timeout_seconds):
                await self.process.wait()
        except TimeoutError:
            return


class SecureProcessLauncher:
    """Launch approved fixed argv with no shell, PATH lookup, or inherited environment."""

    def __init__(
        self,
        allowlist: ExecutableAllowlist,
        *,
        spawn: SpawnProcess = _spawn_process,
    ) -> None:
        if not isinstance(allowlist, ExecutableAllowlist):
            raise TypeError("Secure process launcher requires an ExecutableAllowlist")
        if not callable(spawn):
            raise TypeError("Secure process launcher spawn function must be callable")
        self._allowlist = allowlist
        self._spawn = spawn

    async def start(
        self,
        template_id: str,
        *,
        startup_timeout_seconds: float,
        shutdown_timeout_seconds: float,
    ) -> ManagedProcess:
        _validate_positive_timeout(startup_timeout_seconds, "startup")
        _validate_positive_timeout(shutdown_timeout_seconds, "shutdown")
        template = self._allowlist.resolve(template_id)
        try:
            resolved = template.executable.resolve(strict=True)
        except OSError:
            raise OutboundSecurityError(
                "Outbound MCP allowlisted executable is unavailable"
            ) from None
        if (
            resolved != template.executable
            or not resolved.is_file()
            or not os.access(resolved, os.X_OK)
        ):
            raise OutboundSecurityError("Outbound MCP allowlisted executable is unavailable")
        try:
            async with asyncio.timeout(startup_timeout_seconds):
                process = await self._spawn(template)
        except TimeoutError:
            raise OutboundSecurityError("Outbound MCP process startup timed out") from None
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            raise
        except Exception as exc:
            raise OutboundSecurityError("Outbound MCP process startup failed") from exc
        if not isinstance(process, ProcessHandle):
            raise OutboundSecurityError("Outbound MCP process launcher returned an invalid handle")
        return ManagedProcess(
            process=process,
            shutdown_timeout_seconds=shutdown_timeout_seconds,
        )


def _normalize_host(value: str) -> str:
    if not isinstance(value, str):
        raise OutboundSecurityError("Outbound MCP endpoint host is malformed")
    candidate = value.rstrip(".")
    if not candidate or "%" in candidate:
        raise OutboundSecurityError("Outbound MCP endpoint host is malformed")
    try:
        return candidate.encode("idna").decode("ascii").lower()
    except UnicodeError:
        raise OutboundSecurityError("Outbound MCP endpoint host is malformed") from None


def _host_is_always_prohibited(host: str) -> bool:
    return (
        host in _METADATA_HOSTS
        or host == "localhost"
        or any(host.endswith(suffix) for suffix in _PROHIBITED_HOST_SUFFIXES)
    )


def _literal_address(host: str) -> IPAddress | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _parse_address(raw_address: str) -> IPAddress:
    if not isinstance(raw_address, str):
        raise OutboundSecurityError("Outbound MCP endpoint resolved to a malformed address")
    try:
        return ipaddress.ip_address(raw_address.split("%", 1)[0])
    except ValueError:
        raise OutboundSecurityError(
            "Outbound MCP endpoint resolved to a malformed address"
        ) from None


def _is_always_prohibited(address: IPAddress) -> bool:
    return (
        address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        or address.is_reserved
    )


def _validate_address(
    host: str,
    address: IPAddress,
    *,
    private_approval: PrivateEndpointApproval | None,
) -> None:
    if _is_always_prohibited(address):
        raise OutboundSecurityError("Outbound MCP endpoint resolved to a prohibited address")
    if address.is_global:
        return
    if (
        address.is_private
        and private_approval is not None
        and private_approval.permits(host, address)
    ):
        return
    raise OutboundSecurityError("Outbound MCP endpoint resolved to a prohibited address")


def _contains_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _validate_positive_timeout(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 120:
        raise OutboundSecurityError(f"Outbound MCP {name} timeout is invalid")


__all__ = [
    "ApprovedProcessTemplate",
    "AsyncResolver",
    "EndpointOrigin",
    "ExecutableAllowlist",
    "ManagedProcess",
    "OutboundEndpointPolicy",
    "OutboundSecurityError",
    "ParsedEndpoint",
    "PrivateEndpointApproval",
    "ProcessHandle",
    "SecureProcessLauncher",
    "SpawnProcess",
    "ValidatedEndpoint",
    "parse_https_endpoint",
    "parse_https_origin",
]
