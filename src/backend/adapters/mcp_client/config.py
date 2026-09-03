"""Closed configuration for the separately launched outbound MCP client."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from src.errors import ConfigurationError
from ...mcp.secrets import SecretReference
from .security import parse_https_endpoint, parse_https_origin

OutboundMCPTransport = Literal["streamable_http", "stdio"]

_CONFIG_VERSION = 1
_MAX_CONFIG_BYTES = 65_536
_MAX_SERVERS = 32
_MAX_REDIRECT_ORIGINS = 16
_MAX_REDIRECTS = 5
_MAX_RECONNECT_ATTEMPTS = 3
_MAX_CONNECT_TIMEOUT_SECONDS = 60.0
_MAX_INVOCATION_TIMEOUT_SECONDS = 120.0
_MAX_SHUTDOWN_TIMEOUT_SECONDS = 30.0
_MAX_RECONNECT_BACKOFF_SECONDS = 5.0
_MAX_CONCURRENT_INVOCATIONS = 64
_MAX_REQUEST_BYTES = 262_144
_MAX_RESULT_BYTES = 1_048_576
_SERVER_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]{0,47}$")
_COMMAND_TEMPLATE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_CONFIG_FILE_ENV = "OUTBOUND_MCP_CONFIG_FILE"
_ENABLED_ENV = "OUTBOUND_MCP_ENABLED"
_ALLOWED_ENV_NAMES = frozenset({_CONFIG_FILE_ENV, _ENABLED_ENV})


@dataclass(frozen=True, slots=True)
class OutboundMCPServerSettings:
    """Persistable non-secret configuration for one approved remote/process provider."""

    name: str
    endpoint: str | None = None
    required: bool = True
    transport: OutboundMCPTransport = "streamable_http"
    authorization_secret_ref: SecretReference | None = None
    command_template: str | None = None
    allowed_redirect_origins: tuple[str, ...] = ()
    max_redirects: int = 0
    connect_timeout_seconds: float = 10.0
    invocation_timeout_seconds: float = 30.0
    shutdown_timeout_seconds: float = 5.0
    max_reconnect_attempts: int = 1
    reconnect_backoff_seconds: float = 0.1
    max_concurrent_invocations: int = 8
    max_request_bytes: int = 65_536
    max_result_bytes: int = 262_144

    def __post_init__(self) -> None:
        if not _SERVER_NAME_PATTERN.fullmatch(self.name):
            raise ConfigurationError(
                "Outbound MCP server name must use lowercase letters, digits, and underscores"
            )
        if self.transport not in {"streamable_http", "stdio"}:
            raise ConfigurationError(
                "Outbound MCP transport is not supported by this configuration generation"
            )
        if not isinstance(self.required, bool):
            raise ConfigurationError("Outbound MCP required flag must be boolean")
        if self.authorization_secret_ref is not None and not isinstance(
            self.authorization_secret_ref, SecretReference
        ):
            raise ConfigurationError("Outbound MCP authorization must use a Secret_Reference")

        redirect_origins = tuple(self.allowed_redirect_origins)
        if len(redirect_origins) > _MAX_REDIRECT_ORIGINS or any(
            not isinstance(origin, str) for origin in redirect_origins
        ):
            raise ConfigurationError("Outbound MCP redirect origin allowlist is invalid")
        try:
            normalized_redirect_origins = tuple(
                parse_https_origin(origin).value for origin in redirect_origins
            )
        except ConfigurationError as exc:
            raise ConfigurationError("Outbound MCP redirect origin allowlist is invalid") from exc
        if len(set(normalized_redirect_origins)) != len(normalized_redirect_origins):
            raise ConfigurationError("Outbound MCP redirect origins must be unique")

        _bounded_int(self.max_redirects, "redirect limit", minimum=0, maximum=_MAX_REDIRECTS)
        _bounded_number(
            self.connect_timeout_seconds,
            "connect timeout",
            maximum=_MAX_CONNECT_TIMEOUT_SECONDS,
        )
        _bounded_number(
            self.invocation_timeout_seconds,
            "invocation timeout",
            maximum=_MAX_INVOCATION_TIMEOUT_SECONDS,
        )
        _bounded_number(
            self.shutdown_timeout_seconds,
            "shutdown timeout",
            maximum=_MAX_SHUTDOWN_TIMEOUT_SECONDS,
        )
        _bounded_int(
            self.max_reconnect_attempts,
            "reconnect attempt limit",
            minimum=0,
            maximum=_MAX_RECONNECT_ATTEMPTS,
        )
        _bounded_number(
            self.reconnect_backoff_seconds,
            "reconnect backoff",
            minimum=0.0,
            maximum=_MAX_RECONNECT_BACKOFF_SECONDS,
        )
        _bounded_int(
            self.max_concurrent_invocations,
            "concurrent invocation limit",
            minimum=1,
            maximum=_MAX_CONCURRENT_INVOCATIONS,
        )
        _bounded_int(
            self.max_request_bytes,
            "request byte limit",
            minimum=1,
            maximum=_MAX_REQUEST_BYTES,
        )
        _bounded_int(
            self.max_result_bytes,
            "result byte limit",
            minimum=1,
            maximum=_MAX_RESULT_BYTES,
        )

        if self.transport == "streamable_http":
            if not isinstance(self.endpoint, str):
                raise ConfigurationError("Outbound MCP HTTP transport requires an endpoint")
            try:
                normalized_endpoint = parse_https_endpoint(self.endpoint).url
            except ConfigurationError as exc:
                raise ConfigurationError("Outbound MCP endpoint is invalid") from exc
            if self.command_template is not None:
                raise ConfigurationError("Outbound MCP HTTP transport cannot configure a process")
        else:
            if self.endpoint is not None:
                raise ConfigurationError(
                    "Outbound MCP stdio transport cannot configure an endpoint"
                )
            if not isinstance(
                self.command_template, str
            ) or not _COMMAND_TEMPLATE_PATTERN.fullmatch(self.command_template):
                raise ConfigurationError(
                    "Outbound MCP stdio transport requires an approved command template"
                )
            if self.authorization_secret_ref is not None:
                raise ConfigurationError("Outbound MCP stdio authorization headers are invalid")
            if redirect_origins or self.max_redirects != 0:
                raise ConfigurationError("Outbound MCP stdio transport cannot configure redirects")
            normalized_endpoint = None

        object.__setattr__(self, "endpoint", normalized_endpoint)
        object.__setattr__(self, "allowed_redirect_origins", normalized_redirect_origins)
        object.__setattr__(self, "connect_timeout_seconds", float(self.connect_timeout_seconds))
        object.__setattr__(
            self, "invocation_timeout_seconds", float(self.invocation_timeout_seconds)
        )
        object.__setattr__(self, "shutdown_timeout_seconds", float(self.shutdown_timeout_seconds))
        object.__setattr__(self, "reconnect_backoff_seconds", float(self.reconnect_backoff_seconds))

    def to_config(self) -> dict[str, object]:
        value: dict[str, object] = {
            "name": self.name,
            "required": self.required,
            "transport": self.transport,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "invocation_timeout_seconds": self.invocation_timeout_seconds,
            "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
            "max_reconnect_attempts": self.max_reconnect_attempts,
            "reconnect_backoff_seconds": self.reconnect_backoff_seconds,
            "max_concurrent_invocations": self.max_concurrent_invocations,
            "max_request_bytes": self.max_request_bytes,
            "max_result_bytes": self.max_result_bytes,
        }
        if self.transport == "streamable_http":
            value.update(
                {
                    "endpoint": self.endpoint,
                    "allowed_redirect_origins": list(self.allowed_redirect_origins),
                    "max_redirects": self.max_redirects,
                }
            )
        else:
            value["command_template"] = self.command_template
        if self.authorization_secret_ref is not None:
            value["authorization_secret_ref"] = self.authorization_secret_ref.to_config()
        return value


@dataclass(frozen=True, slots=True)
class OutboundMCPSettings:
    """Independent outbound settings; defaults never activate a client lifecycle."""

    enabled: bool = False
    servers: tuple[OutboundMCPServerSettings, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigurationError("Outbound MCP enabled flag must be boolean")
        servers = tuple(self.servers)
        object.__setattr__(self, "servers", servers)
        if len(servers) > _MAX_SERVERS:
            raise ConfigurationError("Outbound MCP server count exceeds its hard limit")
        if any(not isinstance(server, OutboundMCPServerSettings) for server in servers):
            raise ConfigurationError("Outbound MCP servers are invalid")
        names = [server.name for server in servers]
        if len(names) != len(set(names)):
            raise ConfigurationError("Outbound MCP server names must be unique")
        if self.enabled and not servers:
            raise ConfigurationError(
                "Outbound MCP cannot be enabled without an approved server configuration"
            )

    def to_config(self) -> dict[str, object]:
        """Serialize only non-secret references, never resolved runtime values."""

        return {
            "version": _CONFIG_VERSION,
            "enabled": self.enabled,
            "servers": [server.to_config() for server in self.servers],
        }


def load_outbound_mcp_settings(
    *,
    config_file: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> OutboundMCPSettings:
    """Load the dedicated outbound document without touching inbound/FastAPI config.

    No file is discovered implicitly. An operator must provide ``config_file`` or
    ``OUTBOUND_MCP_CONFIG_FILE`` and must explicitly enable this bounded context.
    """

    env = os.environ if environment is None else environment
    unknown_env = sorted(
        name for name in env if name.startswith("OUTBOUND_MCP_") and name not in _ALLOWED_ENV_NAMES
    )
    if unknown_env:
        raise ConfigurationError(
            f"Unknown outbound MCP environment configuration: {', '.join(unknown_env)}"
        )

    configured_path = config_file
    if configured_path is None:
        raw_path = env.get(_CONFIG_FILE_ENV, "").strip()
        configured_path = raw_path or None

    enabled_override: bool | None = None
    if _ENABLED_ENV in env:
        enabled_override = _parse_bool(env[_ENABLED_ENV], _ENABLED_ENV)

    if configured_path is None:
        if enabled_override:
            raise ConfigurationError("OUTBOUND_MCP_ENABLED requires OUTBOUND_MCP_CONFIG_FILE")
        return OutboundMCPSettings()

    settings = _parse_document(_read_document(Path(configured_path)))
    if enabled_override is not None:
        settings = replace(settings, enabled=enabled_override)
    return settings


def _read_document(path: Path) -> object:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigurationError("Outbound MCP configuration file is unavailable") from exc
    if len(raw) > _MAX_CONFIG_BYTES:
        raise ConfigurationError("Outbound MCP configuration exceeds its hard byte limit")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError("Outbound MCP configuration is not valid JSON") from exc


def _parse_document(document: object) -> OutboundMCPSettings:
    if not isinstance(document, dict) or any(not isinstance(key, str) for key in document):
        raise ConfigurationError("Outbound MCP configuration must be a JSON object")
    unknown = sorted(set(document) - {"version", "enabled", "servers"})
    if unknown:
        raise ConfigurationError(f"Unknown outbound MCP configuration fields: {', '.join(unknown)}")
    version = document.get("version")
    if isinstance(version, bool) or version != _CONFIG_VERSION:
        raise ConfigurationError("Outbound MCP configuration version is unsupported")
    enabled = _parse_bool(document.get("enabled", False), "enabled")
    raw_servers = document.get("servers", [])
    if not isinstance(raw_servers, list):
        raise ConfigurationError("Outbound MCP servers must be a JSON array")
    if len(raw_servers) > _MAX_SERVERS:
        raise ConfigurationError("Outbound MCP server count exceeds its hard limit")
    servers = tuple(_parse_server(value) for value in raw_servers)
    return OutboundMCPSettings(enabled=enabled, servers=servers)


def _parse_server(value: object) -> OutboundMCPServerSettings:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ConfigurationError("Each outbound MCP server must be a JSON object")
    allowed = {
        "name",
        "endpoint",
        "required",
        "transport",
        "authorization_secret_ref",
        "command_template",
        "allowed_redirect_origins",
        "max_redirects",
        "connect_timeout_seconds",
        "invocation_timeout_seconds",
        "shutdown_timeout_seconds",
        "max_reconnect_attempts",
        "reconnect_backoff_seconds",
        "max_concurrent_invocations",
        "max_request_bytes",
        "max_result_bytes",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigurationError(f"Unknown outbound MCP server fields: {', '.join(unknown)}")
    name = value.get("name")
    transport = value.get("transport", "streamable_http")
    endpoint = value.get("endpoint")
    command_template = value.get("command_template")
    if not isinstance(name, str) or not isinstance(transport, str):
        raise ConfigurationError("Outbound MCP server requires string name and transport fields")
    if endpoint is not None and not isinstance(endpoint, str):
        raise ConfigurationError("Outbound MCP endpoint must be a string")
    if command_template is not None and not isinstance(command_template, str):
        raise ConfigurationError("Outbound MCP command template must be a string")
    required = _parse_bool(value.get("required", True), "required")
    raw_reference = value.get("authorization_secret_ref")
    reference = None if raw_reference is None else SecretReference.from_config(raw_reference)
    redirect_origins = _parse_string_array(
        value.get("allowed_redirect_origins", []),
        "allowed_redirect_origins",
        maximum=_MAX_REDIRECT_ORIGINS,
    )
    return OutboundMCPServerSettings(
        name=name,
        endpoint=endpoint,
        required=required,
        transport=transport,  # type: ignore[arg-type]
        authorization_secret_ref=reference,
        command_template=command_template,
        allowed_redirect_origins=redirect_origins,
        max_redirects=_parse_int(
            value.get("max_redirects", 0),
            "max_redirects",
            minimum=0,
            maximum=_MAX_REDIRECTS,
        ),
        connect_timeout_seconds=_parse_number(
            value.get("connect_timeout_seconds", 10.0),
            "connect_timeout_seconds",
            maximum=_MAX_CONNECT_TIMEOUT_SECONDS,
        ),
        invocation_timeout_seconds=_parse_number(
            value.get("invocation_timeout_seconds", 30.0),
            "invocation_timeout_seconds",
            maximum=_MAX_INVOCATION_TIMEOUT_SECONDS,
        ),
        shutdown_timeout_seconds=_parse_number(
            value.get("shutdown_timeout_seconds", 5.0),
            "shutdown_timeout_seconds",
            maximum=_MAX_SHUTDOWN_TIMEOUT_SECONDS,
        ),
        max_reconnect_attempts=_parse_int(
            value.get("max_reconnect_attempts", 1),
            "max_reconnect_attempts",
            minimum=0,
            maximum=_MAX_RECONNECT_ATTEMPTS,
        ),
        reconnect_backoff_seconds=_parse_number(
            value.get("reconnect_backoff_seconds", 0.1),
            "reconnect_backoff_seconds",
            minimum=0.0,
            maximum=_MAX_RECONNECT_BACKOFF_SECONDS,
        ),
        max_concurrent_invocations=_parse_int(
            value.get("max_concurrent_invocations", 8),
            "max_concurrent_invocations",
            minimum=1,
            maximum=_MAX_CONCURRENT_INVOCATIONS,
        ),
        max_request_bytes=_parse_int(
            value.get("max_request_bytes", 65_536),
            "max_request_bytes",
            minimum=1,
            maximum=_MAX_REQUEST_BYTES,
        ),
        max_result_bytes=_parse_int(
            value.get("max_result_bytes", 262_144),
            "max_result_bytes",
            minimum=1,
            maximum=_MAX_RESULT_BYTES,
        ),
    )


def _parse_string_array(value: object, name: str, *, maximum: int) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or len(value) > maximum
        or any(not isinstance(item, str) for item in value)
    ):
        raise ConfigurationError(f"Outbound MCP {name} must be a bounded string array")
    return tuple(item for item in value if isinstance(item, str))


def _parse_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ConfigurationError(f"Outbound MCP {name} must be boolean")


def _parse_int(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ConfigurationError(f"Outbound MCP {name} is outside its hard bounds")
    return value


def _parse_number(
    value: object,
    name: str,
    *,
    minimum: float = 0.001,
    maximum: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not minimum <= float(value) <= maximum
    ):
        raise ConfigurationError(f"Outbound MCP {name} is outside its hard bounds")
    return float(value)


def _bounded_int(value: object, name: str, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ConfigurationError(f"Outbound MCP {name} is outside its hard bounds")


def _bounded_number(
    value: object,
    name: str,
    *,
    minimum: float = 0.001,
    maximum: float,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not minimum <= float(value) <= maximum
    ):
        raise ConfigurationError(f"Outbound MCP {name} is outside its hard bounds")


__all__ = [
    "OutboundMCPServerSettings",
    "OutboundMCPSettings",
    "OutboundMCPTransport",
    "load_outbound_mcp_settings",
]
