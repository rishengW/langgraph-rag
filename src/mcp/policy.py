"""Central authorization, validation, limiting, redaction, and audit pipeline."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import suppress
from contextvars import ContextVar, Token, copy_context
from dataclasses import dataclass, field
from itertools import islice
from threading import Lock
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field, ValidationError

from .models import ToolDescriptor, ToolPolicyError
from .observability import MCPObservability, ObservationEvent, ObservationSignal
from .telemetry import (
    NullToolAuditSink,
    ToolAuditEvent,
    ToolAuditSink,
    ToolMetricsRecorder,
    ToolOutcome,
)

_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_ -]?key|authorization|bearer|access[_ -]?token|"
    r"refresh[_ -]?token|client[_ -]?secret)\b"
    r"(\s*[\"']?\s*[:=]\s*[\"']?|\s+)(?:bearer\s+)?[^\s,;\"'&]+"
)
_SENSITIVE_KEYS = frozenset(
    {
        "apikey",
        "authorization",
        "bearer",
        "accesstoken",
        "refreshtoken",
        "clientsecret",
        "password",
    }
)
_OUTCOME_BY_CODE: Mapping[str, ToolOutcome] = MappingProxyType(
    {
        "TOOL_DENIED": "denied",
        "TOOL_INVALID": "invalid",
        "TOOL_LIMITED": "limited",
        "TOOL_TIMEOUT": "timeout",
    }
)
_TRUNCATION_MARKER = "...[tool output truncated]"
_REDACTED = "[REDACTED]"
_CANONICAL_INBOUND_TOOL_NAMES = frozenset({"rag_ask", "rag_web_search_answer"})


@dataclass(frozen=True, slots=True)
class ToolPrincipal:
    """Trusted, bounded identity used by tool authorization and limits."""

    principal_id: str = "anonymous"
    tenant_id: str | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        _validate_identity(self.principal_id, "principal")
        if self.tenant_id is not None:
            _validate_identity(self.tenant_id, "tenant")
        if self.request_id is not None:
            _validate_identity(self.request_id, "request")


_CURRENT_PRINCIPAL: ContextVar[ToolPrincipal | None] = ContextVar(
    "tool_policy_principal", default=None
)


def current_tool_principal() -> ToolPrincipal:
    """Return the trusted identity bound to this execution context."""

    return _CURRENT_PRINCIPAL.get() or ToolPrincipal()


def set_tool_principal(principal: ToolPrincipal) -> Token[ToolPrincipal | None]:
    """Set trusted tool identity for the current execution context."""

    if not isinstance(principal, ToolPrincipal):
        raise TypeError("Tool principal must be a ToolPrincipal")
    return _CURRENT_PRINCIPAL.set(principal)


def reset_tool_principal(token: Token[ToolPrincipal | None]) -> None:
    """Restore the preceding trusted tool identity."""

    _CURRENT_PRINCIPAL.reset(token)


@dataclass(frozen=True, slots=True)
class ToolPolicyRule:
    """Per-tool immutable authorization restriction."""

    enabled: bool = True
    allowed_principals: frozenset[str] = field(default_factory=frozenset)
    allowed_tenants: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("Tool policy enabled flag must be boolean")
        object.__setattr__(
            self,
            "allowed_principals",
            _freeze_identities(self.allowed_principals, "principal"),
        )
        object.__setattr__(
            self,
            "allowed_tenants",
            _freeze_identities(self.allowed_tenants, "tenant"),
        )


@dataclass(frozen=True, slots=True)
class ToolExecutionLimits:
    """Hard bounds applied to every policy-controlled invocation."""

    deadline_seconds: float = 30.0
    max_concurrency: int = 8
    max_concurrency_per_principal: int = 8
    max_concurrency_per_tenant: int = 8
    rate_limit_per_minute: int = 600
    tenant_rate_limit_per_minute: int = 600
    max_tracked_identities: int = 4_096
    max_output_bytes: int = 65_536
    max_output_chars: int = 32_768
    max_output_items: int = 256
    max_output_depth: int = 16

    def __post_init__(self) -> None:
        if (
            isinstance(self.deadline_seconds, bool)
            or not isinstance(self.deadline_seconds, (int, float))
            or not math.isfinite(self.deadline_seconds)
            or not 0 < self.deadline_seconds <= 300
        ):
            raise ValueError("Tool deadline must be between 0 and 300 seconds")
        _validate_integer_limit(self.max_concurrency, 1, 64, "concurrency")
        _validate_integer_limit(
            self.max_concurrency_per_principal,
            1,
            64,
            "per-principal concurrency",
        )
        _validate_integer_limit(
            self.max_concurrency_per_tenant,
            1,
            64,
            "per-tenant concurrency",
        )
        _validate_integer_limit(
            self.rate_limit_per_minute,
            1,
            100_000,
            "per-principal rate",
        )
        _validate_integer_limit(
            self.tenant_rate_limit_per_minute,
            1,
            100_000,
            "per-tenant rate",
        )
        _validate_integer_limit(
            self.max_tracked_identities,
            1,
            100_000,
            "tracked identity",
        )
        _validate_integer_limit(
            self.max_output_bytes,
            256,
            1_048_576,
            "output-byte",
        )
        _validate_integer_limit(
            self.max_output_chars,
            128,
            262_144,
            "output-character",
        )
        _validate_integer_limit(self.max_output_items, 1, 1_024, "output-item")
        _validate_integer_limit(self.max_output_depth, 1, 32, "output-depth")


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """Immutable allowlist policy evaluated before every tool invocation."""

    rules: Mapping[str, ToolPolicyRule] = field(default_factory=dict)
    allow_risk_levels: frozenset[str] = field(
        default_factory=lambda: frozenset({"read", "write", "execute"})
    )

    def __post_init__(self) -> None:
        copied_rules: dict[str, ToolPolicyRule] = {}
        for name, rule in self.rules.items():
            if not isinstance(name, str) or not name or len(name) > 128 or _has_control(name):
                raise ValueError("Tool policy rule name is invalid")
            if not isinstance(rule, ToolPolicyRule):
                raise ValueError("Tool policy rules must contain ToolPolicyRule values")
            copied_rules[name] = rule
        risks = frozenset(self.allow_risk_levels)
        if not risks.issubset({"read", "write", "execute", "admin"}):
            raise ValueError("Tool policy contains an invalid risk level")
        object.__setattr__(self, "rules", MappingProxyType(copied_rules))
        object.__setattr__(self, "allow_risk_levels", risks)

    def authorize(self, descriptor: ToolDescriptor, principal: ToolPrincipal) -> None:
        """Raise one non-disclosing denial when any policy restriction rejects a call."""

        rule = self.rules.get(descriptor.qualified_name, ToolPolicyRule())
        if not rule.enabled or descriptor.risk_level not in self.allow_risk_levels:
            _raise_denied()
        # Runtime policy may only narrow catalog metadata. It can never replace
        # or broaden a descriptor-level ownership/identity restriction.
        for principals in (descriptor.allowed_principals, rule.allowed_principals):
            if principals and principal.principal_id not in principals:
                _raise_denied()
        for tenants in (descriptor.allowed_tenants, rule.allowed_tenants):
            if tenants and principal.tenant_id not in tenants:
                _raise_denied()


class PolicyTool(BaseTool):
    """LangChain BaseTool that delegates through one sync/async policy pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    original: BaseTool = Field(exclude=True)
    descriptor: ToolDescriptor = Field(exclude=True)
    generation: int = Field(exclude=True, ge=1)
    pipeline: ToolExecutionPipeline = Field(exclude=True)

    def _parse_input(
        self,
        tool_input: str | dict[str, Any],
        tool_call_id: str | None,
    ) -> str | dict[str, Any]:
        # Authorization intentionally precedes validation in the central pipeline.
        return tool_input

    def _run(
        self,
        *args: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.pipeline.execute(
            self.original,
            self.descriptor,
            _tool_input(args, kwargs),
            config=config,
            generation=self.generation,
        )

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self.pipeline.aexecute(
            self.original,
            self.descriptor,
            _tool_input(args, kwargs),
            config=config,
            generation=self.generation,
        )


@dataclass(frozen=True, slots=True)
class _LimitLease:
    principal_id: str
    tenant_id: str | None


class _ToolInvocationLimiter:
    """One bounded process-local limiter shared by sync and async invocations."""

    def __init__(self, limits: ToolExecutionLimits) -> None:
        self._limits = limits
        self._lock = Lock()
        self._closed = False
        self._active = 0
        self._active_by_principal: dict[str, int] = {}
        self._active_by_tenant: dict[str, int] = {}
        self._calls_by_principal: dict[str, deque[float]] = {}
        self._calls_by_tenant: dict[str, deque[float]] = {}

    def acquire(self, principal: ToolPrincipal, now: float) -> _LimitLease:
        with self._lock:
            if self._closed:
                _raise_limited("Tool execution is unavailable")
            principal_calls = self._history(
                self._calls_by_principal,
                principal.principal_id,
                now,
            )
            tenant_calls: deque[float] | None = None
            if principal.tenant_id is not None:
                tenant_calls = self._history(
                    self._calls_by_tenant,
                    principal.tenant_id,
                    now,
                )
            if len(principal_calls) >= self._limits.rate_limit_per_minute:
                _raise_limited("Tool invocation rate limit exceeded")
            if (
                tenant_calls is not None
                and len(tenant_calls) >= self._limits.tenant_rate_limit_per_minute
            ):
                _raise_limited("Tool invocation rate limit exceeded")
            if self._active >= self._limits.max_concurrency:
                _raise_limited("Tool concurrency limit exceeded")
            if (
                self._active_by_principal.get(principal.principal_id, 0)
                >= self._limits.max_concurrency_per_principal
            ):
                _raise_limited("Tool concurrency limit exceeded")
            if (
                principal.tenant_id is not None
                and self._active_by_tenant.get(principal.tenant_id, 0)
                >= self._limits.max_concurrency_per_tenant
            ):
                _raise_limited("Tool concurrency limit exceeded")

            principal_calls.append(now)
            if tenant_calls is not None:
                tenant_calls.append(now)
            self._active += 1
            self._active_by_principal[principal.principal_id] = (
                self._active_by_principal.get(principal.principal_id, 0) + 1
            )
            if principal.tenant_id is not None:
                self._active_by_tenant[principal.tenant_id] = (
                    self._active_by_tenant.get(principal.tenant_id, 0) + 1
                )
            return _LimitLease(principal.principal_id, principal.tenant_id)

    def release(self, lease: _LimitLease) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
            _decrement_counter(self._active_by_principal, lease.principal_id)
            if lease.tenant_id is not None:
                _decrement_counter(self._active_by_tenant, lease.tenant_id)

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def _history(
        self,
        histories: dict[str, deque[float]],
        identity: str,
        now: float,
    ) -> deque[float]:
        history = histories.get(identity)
        if history is None:
            if len(histories) >= self._limits.max_tracked_identities:
                self._prune_histories(histories, now)
            if len(histories) >= self._limits.max_tracked_identities:
                _raise_limited("Tool identity limit exceeded")
            history = deque()
            histories[identity] = history
        _prune_history(history, now)
        return history

    @staticmethod
    def _prune_histories(histories: dict[str, deque[float]], now: float) -> None:
        for identity, history in tuple(histories.items()):
            _prune_history(history, now)
            if not history:
                del histories[identity]


class ToolExecutionPipeline:
    """Run authorize-to-outcome controls for synchronous and asynchronous tools."""

    def __init__(
        self,
        *,
        policy: ToolPolicy | None = None,
        limits: ToolExecutionLimits | None = None,
        audit_sink: ToolAuditSink | None = None,
        metrics: ToolMetricsRecorder | None = None,
        observability: MCPObservability | None = None,
        redaction_values: tuple[str, ...] = (),
    ) -> None:
        self.policy = policy or ToolPolicy()
        self.limits = limits or ToolExecutionLimits()
        self._audit_sink = audit_sink or NullToolAuditSink()
        self._metrics = metrics
        unique_secrets = dict.fromkeys(
            value for value in redaction_values if isinstance(value, str) and value
        )
        self._redaction_values = tuple(sorted(tuple(unique_secrets)[:128], key=len, reverse=True))
        if observability is not None and not isinstance(observability, MCPObservability):
            raise TypeError("Tool observability must be MCPObservability")
        self._observability = observability or MCPObservability(
            redaction_values=self._redaction_values
        )
        self._limiter = _ToolInvocationLimiter(self.limits)
        self._executor = ThreadPoolExecutor(
            max_workers=self.limits.max_concurrency,
            thread_name_prefix="tool-policy",
        )

    def wrap(
        self,
        tool: BaseTool,
        descriptor: ToolDescriptor,
        generation: int,
    ) -> BaseTool:
        """Return one policy-controlled BaseTool without trusting provider bypass markers."""

        if isinstance(tool, PolicyTool):
            if (
                tool.pipeline is self
                and tool.descriptor == descriptor
                and tool.generation == generation
            ):
                return tool
            tool = tool.original
        metadata = dict(tool.metadata or {})
        if (
            metadata.get("canonical_inbound_mcp_tool")
            and tool.name in _CANONICAL_INBOUND_TOOL_NAMES
        ):
            return tool
        metadata.update(
            {
                "catalog_generation": generation,
                "source_server": descriptor.server_name,
                "policy_pipeline_applied": True,
            }
        )
        return PolicyTool(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            return_direct=tool.return_direct,
            tags=tool.tags,
            metadata=metadata,
            response_format=tool.response_format,
            handle_tool_error=tool.handle_tool_error,
            handle_validation_error=tool.handle_validation_error,
            original=tool,
            descriptor=descriptor,
            generation=generation,
            pipeline=self,
        )

    def execute(
        self,
        tool: BaseTool,
        descriptor: ToolDescriptor,
        tool_input: str | dict[str, Any],
        *,
        config: RunnableConfig | None,
        generation: int,
    ) -> Any:
        """Execute one synchronous call through the complete ordered control sequence."""

        started = time.monotonic()
        outcome: ToolOutcome = "failed"
        principal = current_tool_principal()
        request_id = principal.request_id or uuid4().hex
        lease: _LimitLease | None = None
        try:
            self.policy.authorize(descriptor, principal)
            self._observe_authorization(descriptor, principal, started, generation, request_id)
            self._validate_input(tool, descriptor, tool_input)
            lease = self._limiter.acquire(principal, time.monotonic())
            remaining = self._remaining(started)
            context = copy_context()
            future = self._executor.submit(
                context.run,
                _invoke_original,
                tool,
                tool_input,
                config,
            )
            try:
                value = future.result(timeout=remaining)
            except FutureTimeoutError:
                # A running Python thread cannot be force-cancelled. Retain its
                # concurrency lease until it actually exits so timed-out work
                # cannot bypass the configured physical concurrency ceiling.
                if not future.cancel():
                    transferred = lease
                    lease = None
                    future.add_done_callback(
                        lambda _future: self._release_after_timeout(transferred)
                    )
                raise ToolPolicyError(
                    "Tool invocation exceeded its deadline",
                    code="TOOL_TIMEOUT",
                ) from None
            bounded = self._bound_and_redact(value, response_format=tool.response_format)
            outcome = "success"
            return bounded
        except ToolPolicyError as exc:
            outcome = _OUTCOME_BY_CODE.get(exc.code, "failed")
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            raise ToolPolicyError(
                "Tool invocation failed",
                code="TOOL_EXECUTION_FAILED",
            ) from exc
        finally:
            if lease is not None:
                self._limiter.release(lease)
            self._record(descriptor, principal, outcome, started, generation, request_id)

    async def aexecute(
        self,
        tool: BaseTool,
        descriptor: ToolDescriptor,
        tool_input: str | dict[str, Any],
        *,
        config: RunnableConfig | None,
        generation: int,
    ) -> Any:
        """Execute one asynchronous call through the same ordered controls."""

        started = time.monotonic()
        outcome: ToolOutcome = "failed"
        principal = current_tool_principal()
        request_id = principal.request_id or uuid4().hex
        lease: _LimitLease | None = None
        try:
            self.policy.authorize(descriptor, principal)
            self._observe_authorization(descriptor, principal, started, generation, request_id)
            self._validate_input(tool, descriptor, tool_input)
            lease = self._limiter.acquire(principal, time.monotonic())
            try:
                async with asyncio.timeout(self._remaining(started)):
                    value = await _ainvoke_original(tool, tool_input, config)
            except TimeoutError:
                raise ToolPolicyError(
                    "Tool invocation exceeded its deadline",
                    code="TOOL_TIMEOUT",
                ) from None
            bounded = self._bound_and_redact(value, response_format=tool.response_format)
            outcome = "success"
            return bounded
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except ToolPolicyError as exc:
            outcome = _OUTCOME_BY_CODE.get(exc.code, "failed")
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            raise ToolPolicyError(
                "Tool invocation failed",
                code="TOOL_EXECUTION_FAILED",
            ) from exc
        finally:
            if lease is not None:
                self._limiter.release(lease)
            self._record(descriptor, principal, outcome, started, generation, request_id)

    def close(self) -> None:
        """Reject new work and release policy workers without waiting past deadlines."""

        self._limiter.close()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _remaining(self, started: float) -> float:
        remaining = self.limits.deadline_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise ToolPolicyError(
                "Tool invocation exceeded its deadline",
                code="TOOL_TIMEOUT",
            )
        return remaining

    def _validate_input(
        self,
        tool: BaseTool,
        descriptor: ToolDescriptor,
        tool_input: str | dict[str, Any],
    ) -> None:
        schema = tool.get_input_schema()
        candidate: object = tool_input
        if isinstance(tool_input, str):
            fields = tuple(schema.model_fields)
            if len(fields) != 1:
                raise ToolPolicyError("Tool input validation failed", code="TOOL_INVALID")
            candidate = {fields[0]: tool_input}
        elif isinstance(tool_input, dict):
            allowed = set(schema.model_fields)
            properties = descriptor.input_schema.get("properties", {})
            if isinstance(properties, Mapping):
                allowed.update(str(name) for name in properties)
            for model_field in schema.model_fields.values():
                alias = model_field.alias
                if isinstance(alias, str):
                    allowed.add(alias)
            if any(not isinstance(name, str) or name not in allowed for name in tool_input):
                raise ToolPolicyError("Tool input validation failed", code="TOOL_INVALID")
        else:
            raise ToolPolicyError("Tool input validation failed", code="TOOL_INVALID")
        try:
            schema.model_validate(candidate)
        except ValidationError as exc:
            raise ToolPolicyError(
                "Tool input validation failed",
                code="TOOL_INVALID",
            ) from exc

    def _bound_and_redact(self, value: Any, *, response_format: str) -> Any:
        sanitizer = _OutputSanitizer(
            redact=self._redact,
            max_chars=self.limits.max_output_chars,
            max_items=self.limits.max_output_items,
            max_depth=self.limits.max_output_depth,
        )
        sanitized = sanitizer.sanitize(value)
        if _serialized_size(sanitized) <= self.limits.max_output_bytes:
            return sanitized

        if response_format == "content_and_artifact":
            content = sanitized[0] if isinstance(sanitized, tuple) and sanitized else sanitized
            bounded_content = _fit_text_output(
                self._redact(str(content)),
                max_chars=self.limits.max_output_chars,
                max_bytes=self.limits.max_output_bytes,
                envelope=lambda text: (text, None),
            )
            return bounded_content, None
        return _fit_text_output(
            self._redact(str(sanitized)),
            max_chars=self.limits.max_output_chars,
            max_bytes=self.limits.max_output_bytes,
            envelope=lambda text: text,
        )

    def _redact(self, value: str) -> str:
        redacted = _SECRET_PATTERN.sub(lambda match: f"{match.group(1)}={_REDACTED}", value)
        for secret in self._redaction_values:
            redacted = redacted.replace(secret, _REDACTED)
        return redacted

    def _release_after_timeout(self, lease: _LimitLease | None) -> None:
        if lease is not None:
            self._limiter.release(lease)

    def _record(
        self,
        descriptor: ToolDescriptor,
        principal: ToolPrincipal,
        outcome: ToolOutcome,
        started: float,
        generation: int,
        request_id: str,
    ) -> None:
        # Legacy callbacks remain best-effort. The shared exporter independently
        # enforces its explicit required-audit policy with a fixed safe error.
        try:
            event = ToolAuditEvent(
                tool=descriptor.qualified_name[:128],
                outcome=outcome,
                generation=generation,
                duration_ms=min(86_400_000, int((time.monotonic() - started) * 1000)),
                source_server=descriptor.server_name,
                principal_id=principal.principal_id,
                tenant_id=principal.tenant_id,
            )
        except Exception:
            return
        with suppress(Exception):
            self._audit_sink.emit(event)
        if self._metrics is not None:
            with suppress(Exception):
                self._metrics.record(event)
        if outcome == "denied":
            self._observability.emit(
                ObservationEvent(
                    signal="authorization",
                    outcome="denied",
                    request_id=request_id,
                    principal_id=principal.principal_id,
                    tenant_id=principal.tenant_id,
                    tool=descriptor.qualified_name[:128],
                    source_server=descriptor.server_name,
                    transport="internal",
                    duration_ms=event.duration_ms,
                    generation=generation,
                )
            )
        self._observability.emit(
            ObservationEvent(
                signal=_signal_for_outcome(outcome),
                outcome=outcome,
                request_id=request_id,
                principal_id=principal.principal_id,
                tenant_id=principal.tenant_id,
                tool=descriptor.qualified_name[:128],
                source_server=descriptor.server_name,
                transport="internal",
                duration_ms=event.duration_ms,
                generation=generation,
            )
        )

    def _observe_authorization(
        self,
        descriptor: ToolDescriptor,
        principal: ToolPrincipal,
        started: float,
        generation: int,
        request_id: str,
    ) -> None:
        self._observability.emit(
            ObservationEvent(
                signal="authorization",
                outcome="success",
                request_id=request_id,
                principal_id=principal.principal_id,
                tenant_id=principal.tenant_id,
                tool=descriptor.qualified_name[:128],
                source_server=descriptor.server_name,
                transport="internal",
                duration_ms=min(86_400_000, int((time.monotonic() - started) * 1000)),
                generation=generation,
            )
        )


def _signal_for_outcome(outcome: ToolOutcome) -> ObservationSignal:
    if outcome == "denied":
        return "policy_denial"
    if outcome == "limited":
        return "limit_rejection"
    return "tool_invocation"


class _OutputSanitizer:
    def __init__(
        self,
        *,
        redact: Callable[[str], str],
        max_chars: int,
        max_items: int,
        max_depth: int,
    ) -> None:
        self._redact = redact
        self._remaining_chars = max_chars
        self._max_items = max_items
        self._max_depth = max_depth
        self._active: set[int] = set()

    def sanitize(self, value: Any, *, depth: int = 0, sensitive: bool = False) -> Any:
        if sensitive:
            return self._text(_REDACTED)
        if isinstance(value, str):
            return self._text(self._redact(value))
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else self._text(str(value))
        if depth >= self._max_depth:
            return self._text(_TRUNCATION_MARKER)
        if isinstance(value, Mapping):
            return self._mapping(value, depth)
        if isinstance(value, (tuple, list)):
            return self._sequence(value, depth, preserve_tuple=isinstance(value, tuple))
        return self._text(self._redact(str(value)))

    def _mapping(self, value: Mapping[Any, Any], depth: int) -> dict[str, Any]:
        identity = id(value)
        if identity in self._active:
            return {"truncated": self._text(_TRUNCATION_MARKER)}
        self._active.add(identity)
        try:
            bounded: dict[str, Any] = {}
            entries = iter(value.items())
            for key, item in islice(entries, self._max_items):
                raw_key = str(key)
                safe_key = self._text(self._redact(raw_key), maximum=128)
                bounded[safe_key] = self.sanitize(
                    item,
                    depth=depth + 1,
                    sensitive=_is_sensitive_key(raw_key),
                )
            if next(entries, None) is not None:
                bounded["truncated"] = self._text(_TRUNCATION_MARKER)
            return bounded
        finally:
            self._active.remove(identity)

    def _sequence(
        self, value: tuple[Any, ...] | list[Any], depth: int, *, preserve_tuple: bool
    ) -> Any:
        identity = id(value)
        if identity in self._active:
            return (
                (self._text(_TRUNCATION_MARKER),)
                if preserve_tuple
                else [self._text(_TRUNCATION_MARKER)]
            )
        self._active.add(identity)
        try:
            items = [
                self.sanitize(item, depth=depth + 1)
                for item in islice(iter(value), self._max_items)
            ]
            if len(value) > self._max_items:
                items.append(self._text(_TRUNCATION_MARKER))
            return tuple(items) if preserve_tuple else items
        finally:
            self._active.remove(identity)

    def _text(self, value: str, *, maximum: int | None = None) -> str:
        limit = self._remaining_chars
        if maximum is not None:
            limit = min(limit, maximum)
        if limit <= 0:
            return ""
        bounded = _truncate_chars(value, limit)
        self._remaining_chars -= len(bounded)
        return bounded


def _invoke_original(
    tool: BaseTool,
    tool_input: str | dict[str, Any],
    config: RunnableConfig | None,
) -> Any:
    if tool.response_format != "content_and_artifact":
        return tool.invoke(tool_input, config=config)
    result = tool.invoke(_internal_tool_call(tool, tool_input), config=config)
    return getattr(result, "content", ""), getattr(result, "artifact", None)


async def _ainvoke_original(
    tool: BaseTool,
    tool_input: str | dict[str, Any],
    config: RunnableConfig | None,
) -> Any:
    if tool.response_format != "content_and_artifact":
        return await tool.ainvoke(tool_input, config=config)
    result = await tool.ainvoke(_internal_tool_call(tool, tool_input), config=config)
    return getattr(result, "content", ""), getattr(result, "artifact", None)


def _internal_tool_call(
    tool: BaseTool,
    tool_input: str | dict[str, Any],
) -> dict[str, Any]:
    arguments = tool_input if isinstance(tool_input, dict) else {"input": tool_input}
    return {
        "type": "tool_call",
        "id": "policy-internal-call",
        "name": tool.name,
        "args": arguments,
    }


def _tool_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | dict[str, Any]:
    if args and kwargs:
        return {"args": list(args), **kwargs}
    if len(args) == 1 and isinstance(args[0], str):
        return args[0]
    if args:
        return {"args": list(args)}
    return kwargs


def _fit_text_output(
    value: str,
    *,
    max_chars: int,
    max_bytes: int,
    envelope: Callable[[str], Any],
) -> str:
    bounded = _truncate_chars(value, max_chars)
    if _serialized_size(envelope(bounded)) <= max_bytes:
        return bounded
    marker = _TRUNCATION_MARKER
    low = 0
    high = min(len(bounded), max_chars - len(marker))
    best = marker if _serialized_size(envelope(marker)) <= max_bytes else ""
    while low <= high:
        middle = (low + high) // 2
        candidate = bounded[:middle] + marker
        if _serialized_size(envelope(candidate)) <= max_bytes:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def _serialized_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _truncate_chars(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    if maximum <= len(_TRUNCATION_MARKER):
        return value[:maximum]
    return value[: maximum - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def _is_sensitive_key(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    return normalized in _SENSITIVE_KEYS


def _validate_identity(value: str, kind: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 128 or _has_control(value):
        raise ValueError(f"Tool {kind} identity is invalid")


def _freeze_identities(values: object, kind: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(f"Tool {kind} allowlist must be a collection")
    frozen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"Tool {kind} allowlist must contain strings")
        _validate_identity(value, kind)
        frozen.add(value)
    if len(frozen) > 256:
        raise ValueError(f"Tool {kind} allowlist exceeds its bound")
    return frozenset(frozen)


def _validate_integer_limit(value: int, minimum: int, maximum: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"Tool {label} bound is invalid")


def _has_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _raise_denied() -> None:
    raise ToolPolicyError("Tool invocation is not authorized", code="TOOL_DENIED")


def _raise_limited(message: str) -> None:
    raise ToolPolicyError(message, code="TOOL_LIMITED")


def _prune_history(history: deque[float], now: float) -> None:
    while history and now - history[0] >= 60.0:
        history.popleft()


def _decrement_counter(counters: dict[str, int], key: str) -> None:
    remaining = counters.get(key, 0) - 1
    if remaining > 0:
        counters[key] = remaining
    else:
        counters.pop(key, None)


PolicyTool.model_rebuild()


__all__ = [
    "PolicyTool",
    "ToolExecutionLimits",
    "ToolExecutionPipeline",
    "ToolPolicy",
    "ToolPolicyRule",
    "ToolPrincipal",
    "current_tool_principal",
    "reset_tool_principal",
    "set_tool_principal",
]
