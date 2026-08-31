"""Immutable models shared by local and future outbound MCP tool boundaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from langchain_core.tools import BaseTool

from ..errors import RAGError

ToolSource = Literal["builtin", "local", "mcp"]
RiskLevel = Literal["read", "write", "execute", "admin"]
ProviderStatus = Literal["ready", "degraded", "disabled", "closed"]
CatalogReadinessStatus = Literal["ready", "degraded", "closed"]
_MAX_FREEZE_DEPTH = 64
_MAX_PROVIDER_NAME_CHARS = 128


class ToolCatalogError(RAGError):
    """Raised when a tool generation cannot be validated or published."""

    code = "TOOL_CATALOG_ERROR"


class ToolPolicyError(RAGError):
    """Sanitized failure raised by the centralized tool policy boundary."""

    code = "TOOL_POLICY_ERROR"


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    """Validated immutable metadata for one model-visible tool."""

    qualified_name: str
    display_name: str
    source: ToolSource
    server_name: str | None
    description: str
    input_schema: Mapping[str, object]
    risk_level: RiskLevel
    allowed_principals: frozenset[str] = field(default_factory=frozenset)
    allowed_tenants: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_schema", _freeze_mapping(self.input_schema))
        object.__setattr__(
            self,
            "allowed_principals",
            _freeze_identifier_set(self.allowed_principals, "principal"),
        )
        object.__setattr__(
            self,
            "allowed_tenants",
            _freeze_identifier_set(self.allowed_tenants, "tenant"),
        )


@dataclass(frozen=True, slots=True)
class ToolCatalogSnapshot:
    """One atomic catalog generation used for model binding and dispatch."""

    generation: int
    tools: tuple[BaseTool, ...]
    descriptors: tuple[ToolDescriptor, ...]
    tools_by_name: Mapping[str, BaseTool] = field(init=False, repr=False, compare=False)
    descriptors_by_name: Mapping[str, ToolDescriptor] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        tools = tuple(self.tools)
        descriptors = tuple(self.descriptors)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "descriptors", descriptors)
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 1
        ):
            raise ToolCatalogError("Catalog generation must be positive")
        if len(tools) != len(descriptors):
            raise ToolCatalogError("Catalog tools and descriptors are inconsistent")

        tools_by_name: dict[str, BaseTool] = {}
        descriptors_by_name: dict[str, ToolDescriptor] = {}
        for tool, descriptor in zip(tools, descriptors, strict=True):
            if not isinstance(tool, BaseTool) or not isinstance(descriptor, ToolDescriptor):
                raise ToolCatalogError("Catalog entries must contain tools and descriptors")
            if tool.name != descriptor.qualified_name:
                raise ToolCatalogError("Catalog tool and descriptor names are inconsistent")
            if tool.name in tools_by_name:
                raise ToolCatalogError(f"Duplicate tool name in catalog: {tool.name!r}")
            tools_by_name[tool.name] = tool
            descriptors_by_name[tool.name] = descriptor

        object.__setattr__(self, "tools_by_name", MappingProxyType(tools_by_name))
        object.__setattr__(
            self,
            "descriptors_by_name",
            MappingProxyType(descriptors_by_name),
        )


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    """Bounded health metadata for one tool provider."""

    provider: str
    status: ProviderStatus
    generation: int = 0
    detail: str = ""

    def __post_init__(self) -> None:
        _validate_provider_name(self.provider)
        if self.status not in {"ready", "degraded", "disabled", "closed"}:
            raise ToolCatalogError("Provider health status is invalid")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ToolCatalogError("Provider health generation must be non-negative")
        if not isinstance(self.detail, str) or len(self.detail) > 256:
            raise ToolCatalogError("Provider health metadata exceeds its bound")


@dataclass(frozen=True, slots=True)
class ToolCatalogReadiness:
    """Sanitized public readiness for one immutable catalog generation."""

    status: CatalogReadinessStatus = "closed"
    generation: int = 0

    def __post_init__(self) -> None:
        if self.status not in {"ready", "degraded", "closed"}:
            raise ToolCatalogError("Catalog readiness status is invalid")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ToolCatalogError("Catalog readiness generation must be non-negative")

    @property
    def ready(self) -> bool:
        """Return whether the published generation may accept work."""

        return self.status in {"ready", "degraded"}

    def to_public_dict(self) -> dict[str, bool | int | str]:
        """Expose status and generation without dependency diagnostics."""

        return {
            "ready": self.ready,
            "status": self.status,
            "generation": self.generation,
        }


@runtime_checkable
class ToolProvider(Protocol):
    """Lifecycle contract for atomic local or future remote tool providers."""

    async def start(self) -> None: ...

    async def snapshot(self) -> ToolCatalogSnapshot: ...

    async def health(self) -> ProviderHealth: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ToolProviderRegistration:
    """One provider contribution and whether publication requires it."""

    name: str
    provider: ToolProvider
    required: bool = True

    def __post_init__(self) -> None:
        _validate_provider_name(self.name)
        if not isinstance(self.provider, ToolProvider):
            raise ToolCatalogError("Catalog provider does not implement the lifecycle contract")
        if not isinstance(self.required, bool):
            raise ToolCatalogError("Provider required flag must be boolean")


def _validate_provider_name(name: object) -> None:
    if (
        not isinstance(name, str)
        or not name.strip()
        or len(name) > _MAX_PROVIDER_NAME_CHARS
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise ToolCatalogError("Provider name is invalid")


def _freeze_mapping(
    value: Mapping[str, object],
    *,
    seen: set[int] | None = None,
    depth: int = 0,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ToolCatalogError("Tool input schema must be a mapping")
    if depth > _MAX_FREEZE_DEPTH:
        raise ToolCatalogError("Tool input schema is nested too deeply")
    active = seen if seen is not None else set()
    identity = id(value)
    if identity in active:
        raise ToolCatalogError("Tool input schema contains a reference cycle")
    active.add(identity)
    try:
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ToolCatalogError("Tool input schema keys must be strings")
            frozen[key] = _freeze_value(item, seen=active, depth=depth + 1)
        return MappingProxyType(frozen)
    finally:
        active.remove(identity)


def _freeze_value(value: object, *, seen: set[int], depth: int) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value, seen=seen, depth=depth)
    if isinstance(value, (list, tuple)):
        if depth > _MAX_FREEZE_DEPTH:
            raise ToolCatalogError("Tool input schema is nested too deeply")
        identity = id(value)
        if identity in seen:
            raise ToolCatalogError("Tool input schema contains a reference cycle")
        seen.add(identity)
        try:
            return tuple(_freeze_value(item, seen=seen, depth=depth + 1) for item in value)
        finally:
            seen.remove(identity)
    if isinstance(value, (set, frozenset)):
        raise ToolCatalogError("Tool input schema contains a non-JSON collection")
    return value


def _freeze_identifier_set(values: object, kind: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ToolCatalogError(f"Tool {kind} allowlist must be a collection of strings")
    items = tuple(values)
    if not all(isinstance(item, str) for item in items):
        raise ToolCatalogError(f"Tool {kind} allowlist must contain only strings")
    return frozenset(item for item in items if isinstance(item, str))


__all__ = [
    "CatalogReadinessStatus",
    "ProviderHealth",
    "RiskLevel",
    "ToolCatalogError",
    "ToolCatalogReadiness",
    "ToolCatalogSnapshot",
    "ToolDescriptor",
    "ToolPolicyError",
    "ToolProvider",
    "ToolProviderRegistration",
    "ToolSource",
]
