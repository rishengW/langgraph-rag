"""Strict configuration for the separately launched inbound MCP server."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from dotenv import load_dotenv

from ...config import DEFAULT_CONFIG_FILE, load_selected_yaml_config, parse_bool

MCPTransport = Literal["stdio", "http"]
MCPEnvironment = Literal["development", "staging", "production"]

_HARD_MAX_QUESTION_CHARS = 16_000
_HARD_MAX_SOURCE_URLS = 20
_HARD_MAX_SEARCH_RESULTS = 50
_HARD_MAX_RETURNED_SOURCES = 20
_HARD_MAX_WARNINGS = 20
_HARD_MAX_ANSWER_CHARS = 100_000
_HARD_MAX_OUTPUT_BYTES = 262_144
_HARD_MAX_DEADLINE_SECONDS = 600.0
_HARD_MAX_SHUTDOWN_GRACE_SECONDS = 60.0
_HARD_MAX_CONCURRENCY = 64
_HARD_MAX_RATE_PER_MINUTE = 600
_HARD_MAX_REQUEST_BODY_BYTES = 262_144
_ENV_REFERENCE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_RESERVED_OPERATION_PATHS = frozenset({"/health", "/ready", "/admin/health/dependencies"})

_ENV_FIELDS: dict[str, str] = {
    "enabled": "MCP_ENABLED",
    "transport": "MCP_TRANSPORT",
    "environment": "MCP_ENVIRONMENT",
    "host": "MCP_HOST",
    "port": "MCP_PORT",
    "path": "MCP_PATH",
    "allow_anonymous_http": "MCP_ALLOW_ANONYMOUS_HTTP",
    "auth_secret_env": "MCP_AUTH_SECRET_ENV",
    "http_principal_id": "MCP_HTTP_PRINCIPAL_ID",
    "public_base_url": "MCP_PUBLIC_BASE_URL",
    "allowed_hosts": "MCP_ALLOWED_HOSTS",
    "allowed_origins": "MCP_ALLOWED_ORIGINS",
    "max_question_chars": "MCP_MAX_QUESTION_CHARS",
    "max_source_urls": "MCP_MAX_SOURCE_URLS",
    "max_search_results": "MCP_MAX_SEARCH_RESULTS",
    "max_returned_sources": "MCP_MAX_RETURNED_SOURCES",
    "max_warnings": "MCP_MAX_WARNINGS",
    "max_answer_chars": "MCP_MAX_ANSWER_CHARS",
    "max_output_bytes": "MCP_MAX_OUTPUT_BYTES",
    "deadline_seconds": "MCP_DEADLINE_SECONDS",
    "shutdown_grace_seconds": "MCP_SHUTDOWN_GRACE_SECONDS",
    "max_concurrency": "MCP_MAX_CONCURRENCY",
    "rate_limit_per_minute": "MCP_RATE_LIMIT_PER_MINUTE",
    "max_request_body_bytes": "MCP_MAX_REQUEST_BODY_BYTES",
}
_ALLOWED_ENV_NAMES = frozenset(_ENV_FIELDS.values())


@dataclass(frozen=True, slots=True)
class MCPSettings:
    """Validated inbound-only MCP settings; secret values are never stored here."""

    enabled: bool = False
    transport: MCPTransport = "stdio"
    environment: MCPEnvironment = "development"
    host: str = "127.0.0.1"
    port: int = 8002
    path: str = "/mcp"
    allow_anonymous_http: bool = False
    auth_secret_env: str = "API_KEY"
    http_principal_id: str = "mcp-shared-key"
    public_base_url: str = ""
    allowed_hosts: tuple[str, ...] = ()
    allowed_origins: tuple[str, ...] = ()
    max_question_chars: int = _HARD_MAX_QUESTION_CHARS
    max_source_urls: int = 10
    max_search_results: int = 10
    max_returned_sources: int = 10
    max_warnings: int = 5
    max_answer_chars: int = 32_000
    max_output_bytes: int = 65_536
    deadline_seconds: float = 120.0
    shutdown_grace_seconds: float = 10.0
    max_concurrency: int = 4
    rate_limit_per_minute: int = 30
    max_request_body_bytes: int = 65_536

    def __post_init__(self) -> None:
        if self.transport not in ("stdio", "http"):
            raise ValueError("MCP_TRANSPORT must be 'stdio' or 'http'")
        if self.environment not in ("development", "staging", "production"):
            raise ValueError("MCP_ENVIRONMENT must be development, staging, or production")
        if not self.host or any(char.isspace() for char in self.host) or "/" in self.host:
            raise ValueError("MCP_HOST must be a non-empty host name or address")
        if not 1 <= self.port <= 65_535:
            raise ValueError("MCP_PORT must be between 1 and 65535")
        if not self.path.startswith("/") or "?" in self.path or "#" in self.path:
            raise ValueError("MCP_PATH must be an absolute path without query or fragment")
        if self.path != "/" and self.path.endswith("/"):
            raise ValueError("MCP_PATH must not end with '/'")
        if self.path in _RESERVED_OPERATION_PATHS:
            raise ValueError("MCP_PATH conflicts with a reserved operational endpoint")
        if not _ENV_REFERENCE.fullmatch(self.auth_secret_env):
            raise ValueError("MCP_AUTH_SECRET_ENV must name an environment variable")
        self._validate_text("MCP_HTTP_PRINCIPAL_ID", self.http_principal_id, 1, 128)
        self._validate_range(
            "MCP_MAX_QUESTION_CHARS", self.max_question_chars, 1, _HARD_MAX_QUESTION_CHARS
        )
        self._validate_range("MCP_MAX_SOURCE_URLS", self.max_source_urls, 1, _HARD_MAX_SOURCE_URLS)
        self._validate_range(
            "MCP_MAX_SEARCH_RESULTS", self.max_search_results, 1, _HARD_MAX_SEARCH_RESULTS
        )
        self._validate_range(
            "MCP_MAX_RETURNED_SOURCES", self.max_returned_sources, 1, _HARD_MAX_RETURNED_SOURCES
        )
        self._validate_range("MCP_MAX_WARNINGS", self.max_warnings, 1, _HARD_MAX_WARNINGS)
        self._validate_range(
            "MCP_MAX_ANSWER_CHARS", self.max_answer_chars, 256, _HARD_MAX_ANSWER_CHARS
        )
        self._validate_range(
            "MCP_MAX_OUTPUT_BYTES", self.max_output_bytes, 1024, _HARD_MAX_OUTPUT_BYTES
        )
        self._validate_float(
            "MCP_DEADLINE_SECONDS", self.deadline_seconds, _HARD_MAX_DEADLINE_SECONDS
        )
        self._validate_float(
            "MCP_SHUTDOWN_GRACE_SECONDS",
            self.shutdown_grace_seconds,
            _HARD_MAX_SHUTDOWN_GRACE_SECONDS,
        )
        self._validate_range("MCP_MAX_CONCURRENCY", self.max_concurrency, 1, _HARD_MAX_CONCURRENCY)
        self._validate_range(
            "MCP_RATE_LIMIT_PER_MINUTE", self.rate_limit_per_minute, 1, _HARD_MAX_RATE_PER_MINUTE
        )
        self._validate_range(
            "MCP_MAX_REQUEST_BODY_BYTES",
            self.max_request_body_bytes,
            1024,
            _HARD_MAX_REQUEST_BODY_BYTES,
        )
        self._validate_transport_policy()

    @staticmethod
    def _validate_text(name: str, value: str, minimum: int, maximum: int) -> None:
        if not minimum <= len(value) <= maximum or any(ord(char) < 32 for char in value):
            raise ValueError(f"{name} must contain {minimum} to {maximum} printable characters")

    @staticmethod
    def _validate_range(name: str, value: int, minimum: int, maximum: int) -> None:
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")

    @staticmethod
    def _validate_float(name: str, value: float, maximum: float) -> None:
        if not 0 < value <= maximum:
            raise ValueError(f"{name} must be greater than 0 and no more than {maximum:g}")

    def _validate_transport_policy(self) -> None:
        if self.allow_anonymous_http and (
            self.transport != "http" or self.environment != "development"
        ):
            raise ValueError(
                "Anonymous MCP HTTP requires MCP_TRANSPORT=http and MCP_ENVIRONMENT=development"
            )
        if self.transport == "stdio":
            if self.allowed_hosts or self.allowed_origins or self.public_base_url:
                raise ValueError("HTTP-only MCP settings are not accepted for stdio transport")
            return

        if self.environment in ("staging", "production") and self.allow_anonymous_http:
            raise ValueError("Anonymous MCP HTTP is prohibited outside development")
        if self.environment in ("staging", "production") and not self.allowed_hosts:
            raise ValueError("MCP_ALLOWED_HOSTS is required for staging and production HTTP")
        if self.environment in ("staging", "production") and not self.public_base_url:
            raise ValueError("MCP_PUBLIC_BASE_URL is required for staging and production HTTP")
        for name, entries in (
            ("MCP_ALLOWED_HOSTS", self.allowed_hosts),
            ("MCP_ALLOWED_ORIGINS", self.allowed_origins),
        ):
            if (
                any(not item or "*" in item for item in entries)
                and self.environment != "development"
            ):
                raise ValueError(f"{name} cannot contain wildcards outside development")
        if self.public_base_url:
            parsed = urlsplit(self.public_base_url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError("MCP_PUBLIC_BASE_URL must be an absolute HTTP(S) URL")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(
                    "MCP_PUBLIC_BASE_URL cannot contain credentials, query, or fragment"
                )
            if self.environment in ("staging", "production") and parsed.scheme != "https":
                raise ValueError("MCP_PUBLIC_BASE_URL must use HTTPS outside development")

    @property
    def effective_public_base_url(self) -> str:
        if self.public_base_url:
            return self.public_base_url.rstrip("/")
        return f"http://{self.host}:{self.port}"

    @property
    def effective_allowed_hosts(self) -> tuple[str, ...]:
        if self.allowed_hosts:
            return self.allowed_hosts
        host = self.host
        if host == "::1":
            host = "[::1]"
        return (f"{host}:{self.port}",)

    @property
    def effective_allowed_origins(self) -> tuple[str, ...]:
        return self.allowed_origins


def _csv(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list | tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _coerce(name: str, value: Any) -> Any:
    if name in ("enabled", "allow_anonymous_http"):
        return parse_bool(value)
    if name in (
        "port",
        "max_question_chars",
        "max_source_urls",
        "max_search_results",
        "max_returned_sources",
        "max_warnings",
        "max_answer_chars",
        "max_output_bytes",
        "max_concurrency",
        "rate_limit_per_minute",
        "max_request_body_bytes",
    ):
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(f"{_ENV_FIELDS[name]} must be an integer") from None
    if name in ("deadline_seconds", "shutdown_grace_seconds"):
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(f"{_ENV_FIELDS[name]} must be a number") from None
    if name in ("allowed_hosts", "allowed_origins"):
        return _csv(value)
    return str(value).strip()


def _reject_unknown_configuration(yaml_values: dict[str, Any]) -> None:
    field_names = {item.name for item in fields(MCPSettings)}
    unknown_yaml = sorted(
        key for key in yaml_values if key.startswith("mcp_") and key[4:] not in field_names
    )
    configured_env = {name for name in os.environ if name.startswith("MCP_")}
    unknown_env = sorted(configured_env - _ALLOWED_ENV_NAMES)
    outbound = sorted(
        name
        for name in (*unknown_yaml, *unknown_env)
        if "OUTBOUND" in name.upper() or "CLIENT" in name.upper()
    )
    if outbound:
        raise ValueError("Outbound MCP configuration is not supported in this release")
    unknown = [*unknown_yaml, *unknown_env]
    if unknown:
        raise ValueError(f"Unknown inbound MCP configuration: {', '.join(unknown)}")
    if os.getenv("MCP_BEARER_TOKEN") is not None:
        raise ValueError("MCP_BEARER_TOKEN is not accepted; use MCP_AUTH_SECRET_ENV")


def load_mcp_settings(
    *,
    env_file: str | Path = ".env",
    config_file: str | Path | None = DEFAULT_CONFIG_FILE,
    overrides: dict[str, Any] | None = None,
) -> MCPSettings:
    """Load MCP settings with overrides > env > YAML > defaults precedence."""

    load_dotenv(env_file)
    yaml_values = load_selected_yaml_config(config_file)
    _reject_unknown_configuration(yaml_values)
    values: dict[str, Any] = {}
    for item in fields(MCPSettings):
        yaml_key = f"mcp_{item.name}"
        if yaml_key in yaml_values:
            values[item.name] = _coerce(item.name, yaml_values[yaml_key])
    for name, env_name in _ENV_FIELDS.items():
        raw = os.getenv(env_name)
        if raw is not None:
            values[name] = _coerce(name, raw)
    for name, value in (overrides or {}).items():
        if name not in _ENV_FIELDS:
            raise ValueError(f"Unknown MCP override: {name}")
        values[name] = _coerce(name, value)
    return MCPSettings(**values)


__all__ = ["MCPEnvironment", "MCPSettings", "MCPTransport", "load_mcp_settings"]
