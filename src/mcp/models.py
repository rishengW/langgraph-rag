"""Immutable models shared by local and future outbound MCP tool boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from langchain_core.tools import BaseTool

from ..errors import RAGError

ToolSource = Literal["builtin", "local", "mcp"]
RiskLevel = Literal["read", "write", "execute", "admin"]
ProviderStatus = Literal["ready", "degraded", "disabled", "closed"]


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
        object.__setattr__(self, "allowed_principals", frozenset(self.allowed_principals))
        object.__setattr__(self, "allowed_tenants", frozenset(self.allowed_tenants))


@dataclass(frozen=True, slots=True)
class ToolCatalogSnapshot:
    """One atomic catalog generation used for model binding and dispatch."""

    generation: int
    tools: tuple[BaseTool, ...]
    descriptors: tuple[ToolDescriptor, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "descriptors", tuple(self.descriptors))
        if self.generation < 1:
            raise ToolCatalogError("Catalog generation must be positive")
        if len(self.tools) != len(self.descriptors):
            raise ToolCatalogError("Catalog tools and descriptors are inconsistent")
        for tool, descriptor in zip(self.tools, self.descriptors, strict=True):
            if tool.name != descriptor.qualified_name:
                raise ToolCatalogError("Catalog tool and descriptor names are inconsistent")


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    """Bounded health metadata for one tool provider."""

    provider: str
    status: ProviderStatus
    generation: int = 0
    detail: str = ""

    def __post_init__(self) -> None:
        if len(self.provider) > 128 or len(self.detail) > 256:
            raise ToolCatalogError("Provider health metadata exceeds its bound")


@runtime_checkable
class ToolProvider(Protocol):
    """Lifecycle contract for atomic local or future remote tool providers."""

    async def start(self) -> None: ...

    async def snapshot(self) -> ToolCatalogSnapshot: ...

    async def health(self) -> ProviderHealth: ...

    async def close(self) -> None: ...


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


__all__ = [
    "ProviderHealth",
    "RiskLevel",
    "ToolCatalogError",
    "ToolCatalogSnapshot",
    "ToolDescriptor",
    "ToolPolicyError",
    "ToolProvider",
    "ToolSource",
]
