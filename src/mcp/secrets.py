"""Secret-reference boundary shared by independently started MCP contexts."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, Self, runtime_checkable

from ..errors import ConfigurationError

_PROVIDER_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_MAX_SECRET_REFERENCE_CHARS = 512
_MAX_RESOLVED_SECRET_CHARS = 65_536


@dataclass(frozen=True, slots=True)
class SecretReference:
    """A non-secret identifier understood by one approved runtime provider."""

    provider: str
    identifier: str

    def __post_init__(self) -> None:
        if not _PROVIDER_PATTERN.fullmatch(self.provider):
            raise ConfigurationError("Secret reference provider is invalid")
        if (
            not 1 <= len(self.identifier) <= _MAX_SECRET_REFERENCE_CHARS
            or self.identifier != self.identifier.strip()
            or any(ord(char) < 33 or ord(char) == 127 for char in self.identifier)
        ):
            raise ConfigurationError("Secret reference identifier is invalid")

    @classmethod
    def from_config(cls, value: object) -> Self:
        """Parse a closed reference object, never a scalar secret value."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ConfigurationError("Secret-bearing settings require a Secret_Reference object")
        if any(not isinstance(key, str) for key in value):
            raise ConfigurationError("Secret_Reference fields must be strings")
        unknown = sorted(set(value) - {"provider", "identifier"})
        if unknown:
            raise ConfigurationError(f"Unknown Secret_Reference fields: {', '.join(unknown)}")
        provider = value.get("provider")
        identifier = value.get("identifier")
        if not isinstance(provider, str) or not isinstance(identifier, str):
            raise ConfigurationError(
                "Secret_Reference requires string provider and identifier fields"
            )
        return cls(provider=provider, identifier=identifier)

    def to_config(self) -> dict[str, str]:
        """Return the persistable non-secret representation."""

        return {"provider": self.provider, "identifier": self.identifier}


class ResolvedSecret:
    """Runtime-only secret material whose representation is always redacted."""

    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value or len(value) > _MAX_RESOLVED_SECRET_CHARS:
            raise ConfigurationError("Resolved secret value is invalid")
        self.__value = value

    def reveal(self) -> str:
        """Return the value only to the runtime component that consumes it."""

        return self.__value

    def __repr__(self) -> str:
        return "ResolvedSecret(<redacted>)"


@runtime_checkable
class SecretProvider(Protocol):
    """Approved runtime provider interface; configuration never implements this."""

    async def resolve(self, reference: SecretReference) -> ResolvedSecret: ...


class SecretResolver:
    """Dispatch typed references only to explicitly approved provider instances."""

    def __init__(self, providers: Mapping[str, SecretProvider]) -> None:
        validated: dict[str, SecretProvider] = {}
        for name, provider in providers.items():
            if not _PROVIDER_PATTERN.fullmatch(name):
                raise ConfigurationError("Approved secret provider name is invalid")
            if not isinstance(provider, SecretProvider):
                raise ConfigurationError("Approved secret provider is invalid")
            validated[name] = provider
        self._providers: Mapping[str, SecretProvider] = MappingProxyType(validated)

    async def resolve(self, reference: SecretReference) -> ResolvedSecret:
        if not isinstance(reference, SecretReference):
            raise TypeError("SecretResolver accepts only SecretReference values")
        provider = self._providers.get(reference.provider)
        if provider is None:
            raise ConfigurationError("Secret provider is not approved")
        try:
            resolved = await provider.resolve(reference)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError("Approved secret provider failed") from exc
        if not isinstance(resolved, ResolvedSecret):
            raise ConfigurationError("Approved secret provider returned an invalid value")
        return resolved


class EnvironmentSecretProvider:
    """Resolve an allowlisted environment reference at runtime."""

    def __init__(
        self,
        allowed_identifiers: Iterable[str],
        *,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        allowed = frozenset(allowed_identifiers)
        for identifier in allowed:
            SecretReference(provider="env", identifier=identifier)
        self._allowed_identifiers = allowed
        self._environment = os.environ if environment is None else environment

    async def resolve(self, reference: SecretReference) -> ResolvedSecret:
        if reference.provider != "env" or reference.identifier not in self._allowed_identifiers:
            raise ConfigurationError("Environment secret reference is not approved")
        value = self._environment.get(reference.identifier)
        if value is None:
            raise ConfigurationError("Referenced environment secret is unavailable")
        return ResolvedSecret(value)


__all__ = [
    "EnvironmentSecretProvider",
    "ResolvedSecret",
    "SecretProvider",
    "SecretReference",
    "SecretResolver",
]
