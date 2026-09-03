"""Validation, composition, and atomic publication of immutable tool catalogs."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from uuid import uuid4

from langchain_core.tools import BaseTool

from .models import (
    ProviderHealth,
    RiskLevel,
    ToolCatalogError,
    ToolCatalogReadiness,
    ToolCatalogSnapshot,
    ToolDescriptor,
    ToolProvider,
    ToolProviderRegistration,
    ToolSource,
)
from .observability import (
    MCPObservability,
    ObservationEvent,
    ObservationOutcome,
    ObservationSignal,
    RequiredAuditDeliveryError,
)

logger = logging.getLogger(__name__)

MAX_NAME_CHARS = 128
MAX_DESCRIPTION_CHARS = 4_096
MAX_SCHEMA_BYTES = 65_536
MAX_SCHEMA_DEPTH = 16
MAX_SCHEMA_PROPERTIES = 256
MAX_SCHEMA_ENUM_VALUES = 256
MAX_ALLOWLIST_ENTRIES = 256
_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,127}$")
_MCP_NAME_PATTERN = re.compile(
    r"^mcp__(?P<server>[A-Za-z0-9][A-Za-z0-9_]{0,47})__"
    r"(?P<tool>[A-Za-z][A-Za-z0-9_]{0,63})$"
)
_VALID_SOURCES = frozenset({"builtin", "local", "mcp"})
_VALID_RISKS = frozenset({"read", "write", "execute", "admin"})
_VALID_SCHEMA_TYPES = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)
_SUPPORTED_FORMATS = frozenset(
    {
        "date",
        "date-time",
        "duration",
        "email",
        "hostname",
        "idn-email",
        "idn-hostname",
        "ipv4",
        "ipv6",
        "iri",
        "iri-reference",
        "json-pointer",
        "regex",
        "relative-json-pointer",
        "time",
        "uri",
        "uri-reference",
        "uri-template",
        "uuid",
    }
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

    if not isinstance(tool, BaseTool):
        raise ToolCatalogError("Catalog entries must contain BaseTool instances")
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
    """Reject unsafe names, risk metadata, allowlists, or schemas before publication."""

    if not isinstance(descriptor, ToolDescriptor):
        raise ToolCatalogError("Catalog metadata must contain ToolDescriptor values")
    name = descriptor.qualified_name
    if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
        raise ToolCatalogError(f"Invalid tool name: {name!r}")
    if (
        not isinstance(descriptor.display_name, str)
        or not descriptor.display_name.strip()
        or len(descriptor.display_name) > MAX_NAME_CHARS
        or _contains_control(descriptor.display_name)
    ):
        raise ToolCatalogError(f"Invalid display name for tool {name!r}")
    if (
        not isinstance(descriptor.description, str)
        or not descriptor.description.strip()
        or len(descriptor.description) > MAX_DESCRIPTION_CHARS
        or _contains_control(descriptor.description, allow_text_whitespace=True)
    ):
        raise ToolCatalogError(f"Invalid description for tool {name!r}")
    if not isinstance(descriptor.source, str) or descriptor.source not in _VALID_SOURCES:
        raise ToolCatalogError(f"Invalid source metadata for tool {name!r}")
    if not isinstance(descriptor.risk_level, str) or descriptor.risk_level not in _VALID_RISKS:
        raise ToolCatalogError(f"Invalid risk metadata for tool {name!r}")

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
    """Validate all model/dispatcher entries and reject collisions atomically."""

    if not isinstance(snapshot, ToolCatalogSnapshot):
        raise ToolCatalogError("Catalog publication requires a ToolCatalogSnapshot")
    seen: set[str] = set()
    for tool, descriptor in zip(snapshot.tools, snapshot.descriptors, strict=True):
        validate_descriptor(descriptor)
        _validate_tool_descriptor_consistency(tool, descriptor)
        if descriptor.qualified_name in seen:
            raise ToolCatalogError(f"Duplicate tool name in catalog: {descriptor.qualified_name!r}")
        seen.add(descriptor.qualified_name)
    if set(snapshot.tools_by_name) != seen or set(snapshot.descriptors_by_name) != seen:
        raise ToolCatalogError("Catalog name indexes are inconsistent")


def compose_snapshot(
    entries: Sequence[tuple[BaseTool, ToolDescriptor]],
    *,
    generation: int,
    transform: Callable[[BaseTool, ToolDescriptor, int], BaseTool] | None = None,
) -> ToolCatalogSnapshot:
    """Build one validated generation, optionally applying a policy wrapper."""

    candidates = tuple(entries)
    descriptors = tuple(descriptor for _, descriptor in candidates)
    raw = ToolCatalogSnapshot(
        generation=generation,
        tools=tuple(tool for tool, _ in candidates),
        descriptors=descriptors,
    )
    validate_snapshot(raw)
    if transform is None:
        return raw
    wrapped = ToolCatalogSnapshot(
        generation=generation,
        tools=tuple(transform(tool, descriptor, generation) for tool, descriptor in candidates),
        descriptors=descriptors,
    )
    validate_snapshot(wrapped)
    return wrapped


class ToolCatalog:
    """Publish complete required/optional provider generations atomically."""

    def __init__(self, observability: MCPObservability | None = None) -> None:
        if observability is not None and not isinstance(observability, MCPObservability):
            raise TypeError("Tool catalog observability must be MCPObservability")
        self._observability = observability or MCPObservability()
        self._publication_lock = asyncio.Lock()
        self._snapshot: ToolCatalogSnapshot | None = None
        self._providers: tuple[ToolProvider, ...] = ()
        self._registrations: tuple[ToolProviderRegistration, ...] = ()
        self._readiness = ToolCatalogReadiness()
        self._provider_health: tuple[ProviderHealth, ...] = ()
        self._active_leases: dict[int, int] = {}
        self._retired_providers: dict[int, tuple[ToolProvider, ...]] = {}

    @property
    def current(self) -> ToolCatalogSnapshot | None:
        """Return the active immutable generation, if one is published."""

        return self._snapshot

    @property
    def readiness(self) -> ToolCatalogReadiness:
        """Return bounded readiness without dependency diagnostics."""

        return self._readiness

    @property
    def provider_health(self) -> tuple[ProviderHealth, ...]:
        """Return bounded provider states for a restricted administrative surface."""

        return self._provider_health

    def public_readiness(self) -> dict[str, bool | int | str]:
        """Serialize only the bounded public readiness contract."""

        return self._readiness.to_public_dict()

    @asynccontextmanager
    async def acquire_snapshot(self) -> AsyncIterator[ToolCatalogSnapshot]:
        """Lease the active generation until one request finishes or cancels."""

        async with self._publication_lock:
            snapshot = self._snapshot
            if snapshot is None or not self._readiness.ready:
                raise ToolCatalogError("Tool catalog is not ready")
            generation = snapshot.generation
            self._active_leases[generation] = self._active_leases.get(generation, 0) + 1

        try:
            yield snapshot
        finally:
            async with self._publication_lock:
                remaining = self._active_leases.get(generation, 0) - 1
                if remaining > 0:
                    self._active_leases[generation] = remaining
                else:
                    self._active_leases.pop(generation, None)
                    retired = self._retired_providers.pop(generation, ())
                    for provider in reversed(retired):
                        await _close_provider_quietly(provider, phase="retired")

    async def publish(
        self,
        providers: Sequence[ToolProvider | ToolProviderRegistration],
    ) -> ToolCatalogSnapshot:
        """Publish required providers and quarantine failed optional contributions.

        Legacy bare providers remain required. Explicit registrations allow one
        optional provider's complete contribution to be omitted without exposing
        its exception or partially publishing any of its tools.
        """

        request_id = uuid4().hex
        started = time.monotonic()
        candidates = _normalize_provider_candidates(tuple(providers))
        _validate_provider_candidates(candidates)
        async with self._publication_lock:
            previous_generation = self._snapshot.generation if self._snapshot is not None else 0
            generation = previous_generation + 1
            protected_ids = {id(provider) for provider in self._providers}
            for retired in self._retired_providers.values():
                protected_ids.update(id(provider) for provider in retired)
            if any(id(item.provider) in protected_ids for item in candidates):
                raise ToolCatalogError("Catalog publication requires detached provider instances")

            attempted: list[ToolProvider] = []
            closed_ids: set[int] = set()
            prepared: dict[int, tuple[ToolCatalogSnapshot, ProviderHealth]] = {}
            quarantined: dict[int, ProviderHealth] = {}

            try:
                for registration in (item for item in candidates if item.required):
                    provider = registration.provider
                    attempted.append(provider)
                    try:
                        prepared[id(provider)] = await _prepare_provider(provider)
                    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                        raise
                    except ToolCatalogError:
                        raise
                    except Exception as exc:
                        raise ToolCatalogError(
                            "Required tool provider failed during catalog publication"
                        ) from exc

                # Required contributions must form one complete valid baseline.
                compose_snapshot(
                    _entries_for_prepared(candidates, prepared),
                    generation=generation,
                )

                for registration in (item for item in candidates if not item.required):
                    provider = registration.provider
                    attempted.append(provider)
                    try:
                        candidate_state = await _prepare_provider(provider)
                        proposed = dict(prepared)
                        proposed[id(provider)] = candidate_state
                        # Validate collisions and aggregate limits before accepting
                        # this provider. A failure omits its entire snapshot.
                        compose_snapshot(
                            _entries_for_prepared(candidates, proposed),
                            generation=generation,
                        )
                    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                        raise
                    except Exception as exc:
                        if id(provider) not in protected_ids:
                            await _close_provider_quietly(provider, phase="quarantined")
                            closed_ids.add(id(provider))
                        quarantined[id(provider)] = ProviderHealth(
                            provider=registration.name,
                            status="degraded",
                            detail="unavailable",
                        )
                        logger.warning(
                            "Optional tool provider quarantined provider=%s "
                            "phase=publication error_type=%s",
                            registration.name,
                            type(exc).__name__[:128],
                        )
                        self._observe(
                            signal="dependency",
                            outcome="degraded",
                            request_id=request_id,
                            generation=generation,
                            started=started,
                            source_server=registration.name,
                        )
                    else:
                        prepared = proposed

                published = compose_snapshot(
                    _entries_for_prepared(candidates, prepared),
                    generation=generation,
                )
                publication_outcome: ObservationOutcome = "degraded" if quarantined else "published"
                self._observe(
                    signal="catalog_publication",
                    outcome=publication_outcome,
                    request_id=request_id,
                    generation=generation,
                    started=started,
                )
            except BaseException as exc:
                for provider in reversed(attempted):
                    provider_id = id(provider)
                    if provider_id not in protected_ids and provider_id not in closed_ids:
                        await _close_provider_quietly(provider, phase="unpublished")
                        closed_ids.add(provider_id)
                if not isinstance(
                    exc,
                    (
                        KeyboardInterrupt,
                        SystemExit,
                        asyncio.CancelledError,
                        RequiredAuditDeliveryError,
                    ),
                ):
                    self._observe(
                        signal="catalog_publication",
                        outcome="failed",
                        request_id=request_id,
                        generation=generation,
                        started=started,
                    )
                raise

            active_registrations = tuple(
                registration for registration in candidates if id(registration.provider) in prepared
            )
            health_by_id = {
                provider_id: ProviderHealth(
                    provider=next(
                        item.name for item in candidates if id(item.provider) == provider_id
                    ),
                    status=health.status,
                    generation=published.generation if health.status == "ready" else 0,
                )
                for provider_id, (_, health) in prepared.items()
            }
            health_by_id.update(quarantined)

            previous = self._providers
            self._snapshot = published
            self._registrations = active_registrations
            self._providers = tuple(item.provider for item in active_registrations)
            self._provider_health = tuple(health_by_id[id(item.provider)] for item in candidates)
            self._readiness = ToolCatalogReadiness(
                status="degraded" if quarantined else "ready",
                generation=published.generation,
            )

            candidate_ids = {id(provider) for provider in self._providers}
            retired = tuple(provider for provider in previous if id(provider) not in candidate_ids)
            if retired and self._active_leases.get(previous_generation, 0) > 0:
                self._retired_providers[previous_generation] = retired
            else:
                for provider in reversed(retired):
                    await _close_provider_quietly(provider, phase="retired")
            return published

    def _observe(
        self,
        *,
        signal: ObservationSignal,
        outcome: ObservationOutcome,
        request_id: str,
        generation: int,
        started: float,
        source_server: str | None = None,
    ) -> None:
        self._observability.emit(
            ObservationEvent(
                signal=signal,
                outcome=outcome,
                request_id=request_id,
                principal_id="server",
                tool="tool_catalog",
                source_server=source_server,
                transport="internal",
                duration_ms=min(86_400_000, int((time.monotonic() - started) * 1000)),
                generation=generation,
            )
        )

    async def close(self) -> None:
        """Mark readiness closed before shutting down active and retired providers."""

        request_id = uuid4().hex
        started = time.monotonic()
        async with self._publication_lock:
            active = self._providers
            deferred = tuple(
                provider
                for generation_providers in self._retired_providers.values()
                for provider in generation_providers
            )
            providers_to_close = (*active, *deferred)
            registrations = self._registrations
            current_generation = self._snapshot.generation if self._snapshot is not None else 0

            self._providers = ()
            self._registrations = ()
            self._retired_providers = {}
            self._active_leases = {}
            self._readiness = ToolCatalogReadiness(
                status="closed",
                generation=current_generation,
            )
            self._provider_health = tuple(
                ProviderHealth(
                    provider=registration.name,
                    status="closed",
                    generation=current_generation,
                )
                for registration in registrations
            )

            for provider in reversed(providers_to_close):
                await provider.close()

            self._observe(
                signal="lifecycle",
                outcome="closed",
                request_id=request_id,
                generation=current_generation,
                started=started,
            )


def _normalize_provider_candidates(
    candidates: tuple[ToolProvider | ToolProviderRegistration, ...],
) -> tuple[ToolProviderRegistration, ...]:
    registrations: list[ToolProviderRegistration] = []
    for index, candidate in enumerate(candidates, start=1):
        if isinstance(candidate, ToolProviderRegistration):
            registrations.append(candidate)
            continue
        if not isinstance(candidate, ToolProvider):
            raise ToolCatalogError("Catalog provider does not implement the lifecycle contract")
        raw_name = getattr(candidate, "name", "")
        name = raw_name if isinstance(raw_name, str) and raw_name.strip() else f"provider-{index}"
        registrations.append(ToolProviderRegistration(name=name, provider=candidate, required=True))
    return tuple(registrations)


def _validate_provider_candidates(
    candidates: tuple[ToolProviderRegistration, ...],
) -> None:
    seen: set[int] = set()
    for registration in candidates:
        provider = registration.provider
        if id(provider) in seen:
            raise ToolCatalogError("Duplicate provider instance in catalog publication")
        seen.add(id(provider))


async def _prepare_provider(
    provider: ToolProvider,
) -> tuple[ToolCatalogSnapshot, ProviderHealth]:
    await provider.start()
    snapshot = await provider.snapshot()
    validate_snapshot(snapshot)
    health = await provider.health()
    if not isinstance(health, ProviderHealth):
        raise ToolCatalogError("Tool provider returned invalid health metadata")
    if health.status == "ready":
        return snapshot, health
    if health.status == "disabled" and not snapshot.tools:
        return snapshot, health
    raise ToolCatalogError("Tool provider did not reach a publishable state")


def _entries_for_prepared(
    candidates: tuple[ToolProviderRegistration, ...],
    prepared: Mapping[int, tuple[ToolCatalogSnapshot, ProviderHealth]],
) -> tuple[tuple[BaseTool, ToolDescriptor], ...]:
    entries: list[tuple[BaseTool, ToolDescriptor]] = []
    for registration in candidates:
        state = prepared.get(id(registration.provider))
        if state is None:
            continue
        snapshot, _ = state
        entries.extend(zip(snapshot.tools, snapshot.descriptors, strict=True))
    return tuple(entries)


async def _close_provider_quietly(provider: ToolProvider, *, phase: str) -> None:
    try:
        await provider.close()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        logger.warning(
            "Tool provider cleanup failed provider_type=%s phase=%s error_type=%s",
            type(provider).__name__[:128],
            phase,
            type(exc).__name__[:128],
        )


def _validate_tool_descriptor_consistency(
    tool: BaseTool,
    descriptor: ToolDescriptor,
) -> None:
    if not isinstance(tool, BaseTool):
        raise ToolCatalogError("Catalog entries must contain BaseTool instances")
    if tool.name != descriptor.qualified_name:
        raise ToolCatalogError("Catalog tool and descriptor names are inconsistent")
    if str(tool.description or "") != descriptor.description:
        raise ToolCatalogError(
            f"Tool {descriptor.qualified_name!r} description does not match its descriptor"
        )
    try:
        tool_schema = tool.get_input_schema().model_json_schema()
    except Exception as exc:
        raise ToolCatalogError(
            f"Tool {descriptor.qualified_name!r} has no valid input schema"
        ) from exc
    _validate_schema(tool_schema, tool_name=descriptor.qualified_name)
    if _schema_fingerprint(tool_schema, descriptor.qualified_name) != _schema_fingerprint(
        descriptor.input_schema, descriptor.qualified_name
    ):
        raise ToolCatalogError(
            f"Tool {descriptor.qualified_name!r} input schema does not match its descriptor"
        )


def _validate_schema(schema: Mapping[str, object], *, tool_name: str) -> None:
    if not isinstance(schema, Mapping):
        raise ToolCatalogError(f"Tool {tool_name!r} schema must be an object")
    serialized = _schema_fingerprint(schema, tool_name)
    if len(serialized.encode("utf-8")) > MAX_SCHEMA_BYTES:
        raise ToolCatalogError(f"Tool {tool_name!r} schema exceeds the byte limit")
    if schema.get("type") != "object":
        raise ToolCatalogError(f"Tool {tool_name!r} schema root must have type 'object'")

    property_count = 0
    enum_count = 0
    references: list[str] = []

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
            _validate_schema_type(value.get("type"), tool_name)
            reference = value.get("$ref")
            if reference is not None:
                if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
                    raise ToolCatalogError(f"Tool {tool_name!r} schema has an external reference")
                references.append(reference)
            schema_format = value.get("format")
            if schema_format is not None and (
                not isinstance(schema_format, str) or schema_format not in _SUPPORTED_FORMATS
            ):
                raise ToolCatalogError(f"Tool {tool_name!r} schema uses an unsupported format")

            properties = value.get("properties")
            property_names: set[str] = set()
            if properties is not None:
                if not isinstance(properties, Mapping):
                    raise ToolCatalogError(
                        f"Tool {tool_name!r} schema properties must be an object"
                    )
                property_count += len(properties)
                for prop_name, prop_schema in properties.items():
                    if not isinstance(prop_name, str) or not prop_name:
                        raise ToolCatalogError(
                            f"Tool {tool_name!r} schema has an invalid property name"
                        )
                    if not isinstance(prop_schema, (Mapping, bool)):
                        raise ToolCatalogError(
                            f"Tool {tool_name!r} schema has an invalid property definition"
                        )
                    property_names.add(prop_name)

            required = value.get("required")
            if required is not None:
                if isinstance(required, (str, bytes)) or not isinstance(required, (list, tuple)):
                    raise ToolCatalogError(
                        f"Tool {tool_name!r} schema required fields must be an array"
                    )
                if not all(isinstance(item, str) and item for item in required):
                    raise ToolCatalogError(
                        f"Tool {tool_name!r} schema has an invalid required field"
                    )
                required_names = set(required)
                if len(required_names) != len(required) or not required_names.issubset(
                    property_names
                ):
                    raise ToolCatalogError(
                        f"Tool {tool_name!r} schema required fields are inconsistent"
                    )

            enum = value.get("enum")
            if enum is not None:
                if not isinstance(enum, (list, tuple)) or not enum:
                    raise ToolCatalogError(f"Tool {tool_name!r} schema has an invalid enum")
                enum_count += len(enum)
            additional = value.get("additionalProperties")
            if additional is not None and not isinstance(additional, (bool, Mapping)):
                raise ToolCatalogError(
                    f"Tool {tool_name!r} schema has invalid additionalProperties"
                )
            definitions = value.get("$defs")
            if definitions is not None and not isinstance(definitions, Mapping):
                raise ToolCatalogError(f"Tool {tool_name!r} schema $defs must be an object")
            # "properties" and "$defs" maps are keyed by user-chosen names, so the
            # maps themselves must not be read as schema nodes (a field literally
            # named "format" would otherwise fail the format check). Visit their
            # values, which are the actual schemas.
            for key, nested in value.items():
                if key in ("properties", "$defs"):
                    for sub_schema in nested.values():
                        visit(sub_schema, depth + 1)
                else:
                    visit(nested, depth + 1)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested, depth + 1)

    visit(schema, 0)
    if property_count > MAX_SCHEMA_PROPERTIES:
        raise ToolCatalogError(f"Tool {tool_name!r} schema exceeds the property limit")
    if enum_count > MAX_SCHEMA_ENUM_VALUES:
        raise ToolCatalogError(f"Tool {tool_name!r} schema exceeds the enum limit")
    if any(not _local_reference_exists(schema, reference) for reference in references):
        raise ToolCatalogError(f"Tool {tool_name!r} schema has an unresolved reference")


def _validate_schema_type(value: object, tool_name: str) -> None:
    if value is None:
        return
    values = (value,) if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)) or not values:
        raise ToolCatalogError(f"Tool {tool_name!r} schema has an invalid type")
    if not all(isinstance(item, str) and item in _VALID_SCHEMA_TYPES for item in values):
        raise ToolCatalogError(f"Tool {tool_name!r} schema has an invalid type")
    if len(set(values)) != len(values):
        raise ToolCatalogError(f"Tool {tool_name!r} schema has duplicate types")


def _schema_fingerprint(schema: Mapping[str, object], tool_name: str) -> str:
    try:
        return json.dumps(
            _plain_json_value(schema),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ToolCatalogError(f"Tool {tool_name!r} schema is not JSON serializable") from exc


def _plain_json_value(
    value: object,
    *,
    active: set[int] | None = None,
    depth: int = 0,
) -> object:
    if depth > 64:
        raise ValueError("JSON value is nested too deeply")
    seen = active if active is not None else set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise ValueError("JSON value contains a cycle")
        seen.add(identity)
        try:
            result: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("JSON object keys must be strings")
                result[key] = _plain_json_value(item, active=seen, depth=depth + 1)
            return result
        finally:
            seen.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            raise ValueError("JSON value contains a cycle")
        seen.add(identity)
        try:
            return [_plain_json_value(item, active=seen, depth=depth + 1) for item in value]
        finally:
            seen.remove(identity)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _local_reference_exists(schema: Mapping[str, object], reference: str) -> bool:
    current: object = schema
    for token in reference[2:].split("/"):
        if not isinstance(current, Mapping):
            return False
        current = current.get(token.replace("~1", "/").replace("~0", "~"))
        if current is None:
            return False
    return True


def _contains_control(value: str, *, allow_text_whitespace: bool = False) -> bool:
    allowed = {"\t", "\n", "\r"} if allow_text_whitespace else set()
    return any(ord(character) < 32 and character not in allowed for character in value)


def _validate_allowlist(values: frozenset[str], kind: str, tool_name: str) -> None:
    if len(values) > MAX_ALLOWLIST_ENTRIES:
        raise ToolCatalogError(f"Tool {tool_name!r} {kind} allowlist exceeds the limit")
    for value in values:
        if not isinstance(value, str) or not value or len(value) > 128 or _contains_control(value):
            raise ToolCatalogError(f"Invalid {kind} allowlist entry for tool {tool_name!r}")


__all__ = [
    "MAX_ALLOWLIST_ENTRIES",
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
