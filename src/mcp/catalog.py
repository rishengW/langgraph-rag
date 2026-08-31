"""Validation, composition, and atomic publication of immutable tool catalogs."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping, Sequence

from langchain_core.tools import BaseTool

from .models import (
    RiskLevel,
    ToolCatalogError,
    ToolCatalogSnapshot,
    ToolDescriptor,
    ToolProvider,
    ToolSource,
)

MAX_NAME_CHARS = 128
MAX_DESCRIPTION_CHARS = 4_096
MAX_SCHEMA_BYTES = 65_536
MAX_SCHEMA_DEPTH = 16
MAX_SCHEMA_PROPERTIES = 256
MAX_SCHEMA_ENUM_VALUES = 256
_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,127}$")
_MCP_NAME_PATTERN = re.compile(
    r"^mcp__(?P<server>[A-Za-z0-9][A-Za-z0-9_]{0,47})__"
    r"(?P<tool>[A-Za-z][A-Za-z0-9_]{0,63})$"
)
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "$dynamicRef",
        "$recursiveRef",
        "dependentSchemas",
        "if",
        "not",
        "patternProperties",
        "then",
        "unevaluatedProperties",
    }
)


def descriptor_from_tool(
    tool: BaseTool,
    *,
    source: ToolSource,
    risk_level: RiskLevel,
    server_name: str | None = None,
    allowed_principals: frozenset[str] = frozenset(),
    allowed_tenants: frozenset[str] = frozenset(),
) -> ToolDescriptor:
    """Create validated immutable metadata from a LangChain tool."""

    try:
        schema = tool.get_input_schema().model_json_schema()
    except Exception as exc:
        raise ToolCatalogError(f"Tool {tool.name!r} has no valid input schema") from exc
    descriptor = ToolDescriptor(
        qualified_name=tool.name,
        display_name=tool.name.replace("_", " ").strip().title() or tool.name,
        source=source,
        server_name=server_name,
        description=str(tool.description or ""),
        input_schema=schema,
        risk_level=risk_level,
        allowed_principals=allowed_principals,
        allowed_tenants=allowed_tenants,
    )
    validate_descriptor(descriptor)
    return descriptor


def validate_descriptor(descriptor: ToolDescriptor) -> None:
    """Reject unsafe names, metadata, or schemas before publication."""

    name = descriptor.qualified_name
    if not _NAME_PATTERN.fullmatch(name) or len(name) > MAX_NAME_CHARS:
        raise ToolCatalogError(f"Invalid tool name: {name!r}")
    if not descriptor.display_name or len(descriptor.display_name) > MAX_NAME_CHARS:
        raise ToolCatalogError(f"Invalid display name for tool {name!r}")
    if not descriptor.description or len(descriptor.description) > MAX_DESCRIPTION_CHARS:
        raise ToolCatalogError(f"Invalid description for tool {name!r}")
    namespace = _MCP_NAME_PATTERN.fullmatch(name)
    if descriptor.source == "mcp":
        if namespace is None or descriptor.server_name != namespace.group("server"):
            raise ToolCatalogError("Remote tools must use mcp__<server>__<tool> names")
    elif descriptor.server_name is not None or name.startswith("mcp__"):
        raise ToolCatalogError("Only remote tools may use the MCP namespace")
    _validate_schema(descriptor.input_schema, tool_name=name)
    _validate_allowlist(descriptor.allowed_principals, "principal", name)
    _validate_allowlist(descriptor.allowed_tenants, "tenant", name)


def validate_snapshot(snapshot: ToolCatalogSnapshot) -> None:
    """Validate every descriptor and reject collisions atomically."""

    seen: set[str] = set()
    for descriptor in snapshot.descriptors:
        validate_descriptor(descriptor)
        if descriptor.qualified_name in seen:
            raise ToolCatalogError(f"Duplicate tool name in catalog: {descriptor.qualified_name!r}")
        seen.add(descriptor.qualified_name)


def compose_snapshot(
    entries: Sequence[tuple[BaseTool, ToolDescriptor]],
    *,
    generation: int,
    transform: Callable[[BaseTool, ToolDescriptor, int], BaseTool] | None = None,
) -> ToolCatalogSnapshot:
    """Build one validated generation, optionally applying a policy wrapper."""

    descriptors = tuple(descriptor for _, descriptor in entries)
    raw = ToolCatalogSnapshot(
        generation=generation,
        tools=tuple(tool for tool, _ in entries),
        descriptors=descriptors,
    )
    validate_snapshot(raw)
    if transform is None:
        return raw
    wrapped = ToolCatalogSnapshot(
        generation=generation,
        tools=tuple(transform(tool, descriptor, generation) for tool, descriptor in entries),
        descriptors=descriptors,
    )
    validate_snapshot(wrapped)
    return wrapped


class ToolCatalog:
    """Publish complete provider generations atomically under an async lock."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._snapshot: ToolCatalogSnapshot | None = None
        self._providers: tuple[ToolProvider, ...] = ()

    @property
    def current(self) -> ToolCatalogSnapshot | None:
        """Return the active immutable generation, if one is published."""

        return self._snapshot

    async def publish(self, providers: Sequence[ToolProvider]) -> ToolCatalogSnapshot:
        """Start providers and atomically replace the active generation."""

        candidates = tuple(providers)
        started: list[ToolProvider] = []
        try:
            for provider in candidates:
                await provider.start()
                started.append(provider)
            provider_snapshots = [await provider.snapshot() for provider in candidates]
            entries = [
                (tool, descriptor)
                for snapshot in provider_snapshots
                for tool, descriptor in zip(snapshot.tools, snapshot.descriptors, strict=True)
            ]
            async with self._lock:
                generation = 1 if self._snapshot is None else self._snapshot.generation + 1
                published = compose_snapshot(entries, generation=generation)
                previous = self._providers
                self._snapshot = published
                self._providers = candidates
            for provider in previous:
                if provider not in candidates:
                    await provider.close()
            return published
        except BaseException:
            for provider in reversed(started):
                if provider not in self._providers:
                    await provider.close()
            raise

    async def close(self) -> None:
        """Close active providers without mutating retained snapshots."""

        async with self._lock:
            providers = self._providers
            self._providers = ()
        for provider in reversed(providers):
            await provider.close()


def _validate_schema(schema: Mapping[str, object], *, tool_name: str) -> None:
    try:
        serialized = json.dumps(_plain_value(schema), separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ToolCatalogError(f"Tool {tool_name!r} schema is not JSON serializable") from exc
    if len(serialized.encode("utf-8")) > MAX_SCHEMA_BYTES:
        raise ToolCatalogError(f"Tool {tool_name!r} schema exceeds the byte limit")

    property_count = 0
    enum_count = 0

    def visit(value: object, depth: int) -> None:
        nonlocal property_count, enum_count
        if depth > MAX_SCHEMA_DEPTH:
            raise ToolCatalogError(f"Tool {tool_name!r} schema exceeds the depth limit")
        if isinstance(value, Mapping):
            unsupported = _UNSUPPORTED_SCHEMA_KEYS.intersection(value)
            if unsupported:
                keys = ", ".join(sorted(unsupported))
                raise ToolCatalogError(
                    f"Tool {tool_name!r} schema uses unsupported constructs: {keys}"
                )
            reference = value.get("$ref")
            if isinstance(reference, str) and not reference.startswith("#/$defs/"):
                raise ToolCatalogError(f"Tool {tool_name!r} schema has an external reference")
            properties = value.get("properties")
            if isinstance(properties, Mapping):
                property_count += len(properties)
            enum = value.get("enum")
            if isinstance(enum, (list, tuple)):
                enum_count += len(enum)
            for nested in value.values():
                visit(nested, depth + 1)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested, depth + 1)

    visit(schema, 0)
    if property_count > MAX_SCHEMA_PROPERTIES:
        raise ToolCatalogError(f"Tool {tool_name!r} schema exceeds the property limit")
    if enum_count > MAX_SCHEMA_ENUM_VALUES:
        raise ToolCatalogError(f"Tool {tool_name!r} schema exceeds the enum limit")


def _plain_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain_value(item) for item in value]
    return value


def _validate_allowlist(values: frozenset[str], kind: str, tool_name: str) -> None:
    for value in values:
        if not value or len(value) > 128 or any(ord(char) < 32 for char in value):
            raise ToolCatalogError(f"Invalid {kind} allowlist entry for tool {tool_name!r}")


__all__ = [
    "MAX_DESCRIPTION_CHARS",
    "MAX_NAME_CHARS",
    "MAX_SCHEMA_BYTES",
    "MAX_SCHEMA_DEPTH",
    "MAX_SCHEMA_ENUM_VALUES",
    "MAX_SCHEMA_PROPERTIES",
    "ToolCatalog",
    "compose_snapshot",
    "descriptor_from_tool",
    "validate_descriptor",
    "validate_snapshot",
]
