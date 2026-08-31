"""Central authorization, validation, limiting, redaction, and audit pipeline."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import suppress
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from threading import BoundedSemaphore
from types import MappingProxyType
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field, ValidationError

from .models import ToolDescriptor, ToolPolicyError
from .telemetry import (
    NullToolAuditSink,
    ToolAuditEvent,
    ToolAuditSink,
    ToolMetricsRecorder,
    ToolOutcome,
)

_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_ -]?key|authorization|bearer|access[_ -]?token|"
    r"refresh[_ -]?token|client[_ -]?secret)(\s*[:=]\s*|\s+)[^\s,;]+"
)


@dataclass(frozen=True, slots=True)
class ToolPrincipal:
    """Trusted identity used by tool authorization policy."""

    principal_id: str = "anonymous"
    tenant_id: str | None = None


_CURRENT_PRINCIPAL: ContextVar[ToolPrincipal | None] = ContextVar(
    "tool_policy_principal", default=None
)


def set_tool_principal(principal: ToolPrincipal) -> Token[ToolPrincipal | None]:
    """Set trusted tool identity for the current execution context."""

    return _CURRENT_PRINCIPAL.set(principal)


def reset_tool_principal(token: Token[ToolPrincipal | None]) -> None:
    """Restore the preceding trusted tool identity."""

    _CURRENT_PRINCIPAL.reset(token)


@dataclass(frozen=True, slots=True)
class ToolPolicyRule:
    """Per-tool immutable authorization override."""

    enabled: bool = True
    allowed_principals: frozenset[str] = field(default_factory=frozenset)
    allowed_tenants: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class ToolExecutionLimits:
    """Hard bounds applied to every policy-controlled invocation."""

    deadline_seconds: float = 30.0
    max_concurrency: int = 8
    max_output_bytes: int = 65_536
    max_output_chars: int = 32_768

    def __post_init__(self) -> None:
        if not 0 < self.deadline_seconds <= 300:
            raise ValueError("Tool deadline must be between 0 and 300 seconds")
        if not 1 <= self.max_concurrency <= 64:
            raise ValueError("Tool concurrency must be between 1 and 64")
        if not 256 <= self.max_output_bytes <= 1_048_576:
            raise ValueError("Tool output-byte bound is invalid")
        if not 128 <= self.max_output_chars <= 262_144:
            raise ValueError("Tool output-character bound is invalid")


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """Immutable allowlist policy evaluated before every tool invocation."""

    rules: Mapping[str, ToolPolicyRule] = field(default_factory=dict)
    allow_risk_levels: frozenset[str] = field(
        default_factory=lambda: frozenset({"read", "write", "execute"})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", MappingProxyType(dict(self.rules)))
        object.__setattr__(self, "allow_risk_levels", frozenset(self.allow_risk_levels))

    def authorize(self, descriptor: ToolDescriptor, principal: ToolPrincipal) -> None:
        """Raise a sanitized denial when identity or risk policy rejects a call."""

        rule = self.rules.get(descriptor.qualified_name, ToolPolicyRule())
        if not rule.enabled or descriptor.risk_level not in self.allow_risk_levels:
            raise ToolPolicyError("Tool invocation is not authorized", code="TOOL_DENIED")
        principals = rule.allowed_principals or descriptor.allowed_principals
        tenants = rule.allowed_tenants or descriptor.allowed_tenants
        if principals and principal.principal_id not in principals:
            raise ToolPolicyError("Tool invocation is not authorized", code="TOOL_DENIED")
        if tenants and principal.tenant_id not in tenants:
            raise ToolPolicyError("Tool invocation is not authorized", code="TOOL_DENIED")


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
        # Defer validation until after authorization inside the central pipeline.
        return tool_input

    def _run(
        self,
        *args: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_input = _tool_input(args, kwargs)
        return self.pipeline.execute(
            self.original,
            self.descriptor,
            tool_input,
            config=config,
            generation=self.generation,
        )

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_input = _tool_input(args, kwargs)
        return await self.pipeline.aexecute(
            self.original,
            self.descriptor,
            tool_input,
            config=config,
            generation=self.generation,
        )


class ToolExecutionPipeline:
    """Run authorize-to-outcome controls for synchronous and asynchronous tools."""

    def __init__(
        self,
        *,
        policy: ToolPolicy | None = None,
        limits: ToolExecutionLimits | None = None,
        audit_sink: ToolAuditSink | None = None,
        metrics: ToolMetricsRecorder | None = None,
        redaction_values: tuple[str, ...] = (),
    ) -> None:
        self.policy = policy or ToolPolicy()
        self.limits = limits or ToolExecutionLimits()
        self._audit_sink = audit_sink or NullToolAuditSink()
        self._metrics = metrics
        self._redaction_values = tuple(value for value in redaction_values if value)
        self._sync_slots = BoundedSemaphore(self.limits.max_concurrency)
        self._async_slots = asyncio.Semaphore(self.limits.max_concurrency)
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
        """Return one policy-controlled BaseTool without double wrapping canonical tools."""

        metadata = dict(tool.metadata or {})
        if isinstance(tool, PolicyTool) or metadata.get("policy_pipeline_applied"):
            return tool
        if metadata.get("canonical_inbound_mcp_tool"):
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
        """Execute one synchronous call through the complete control sequence."""

        started = time.monotonic()
        outcome: ToolOutcome = "failed"
        principal = _CURRENT_PRINCIPAL.get() or ToolPrincipal()
        try:
            self.policy.authorize(descriptor, principal)
            self._validate_input(tool, tool_input)
            if not self._sync_slots.acquire(timeout=self.limits.deadline_seconds):
                outcome = "limited"
                raise ToolPolicyError("Tool concurrency limit exceeded", code="TOOL_LIMITED")
            try:
                future = self._executor.submit(
                    _invoke_original, tool, tool_input, config
                )
                try:
                    value = future.result(timeout=self.limits.deadline_seconds)
                except FutureTimeoutError:
                    future.cancel()
                    outcome = "timeout"
                    raise ToolPolicyError(
                        "Tool invocation exceeded its deadline", code="TOOL_TIMEOUT"
                    ) from None
            finally:
                self._sync_slots.release()
            outcome = "success"
            return self._bound_and_redact(value)
        except ValidationError as exc:
            outcome = "invalid"
            raise ToolPolicyError("Tool input validation failed", code="TOOL_INVALID") from exc
        except ToolPolicyError as exc:
            if exc.code == "TOOL_DENIED":
                outcome = "denied"
            elif exc.code == "TOOL_INVALID":
                outcome = "invalid"
            elif exc.code == "TOOL_LIMITED":
                outcome = "limited"
            elif exc.code == "TOOL_TIMEOUT":
                outcome = "timeout"
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            raise ToolPolicyError("Tool invocation failed", code="TOOL_EXECUTION_FAILED") from exc
        finally:
            self._record(descriptor, principal, outcome, started, generation)

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
        principal = _CURRENT_PRINCIPAL.get() or ToolPrincipal()
        try:
            self.policy.authorize(descriptor, principal)
            self._validate_input(tool, tool_input)
            try:
                async with asyncio.timeout(self.limits.deadline_seconds):
                    async with self._async_slots:
                        value = await _ainvoke_original(tool, tool_input, config)
            except TimeoutError:
                outcome = "timeout"
                raise ToolPolicyError(
                    "Tool invocation exceeded its deadline", code="TOOL_TIMEOUT"
                ) from None
            outcome = "success"
            return self._bound_and_redact(value)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except ValidationError as exc:
            outcome = "invalid"
            raise ToolPolicyError("Tool input validation failed", code="TOOL_INVALID") from exc
        except ToolPolicyError as exc:
            if exc.code == "TOOL_DENIED":
                outcome = "denied"
            elif exc.code == "TOOL_INVALID":
                outcome = "invalid"
            elif exc.code == "TOOL_LIMITED":
                outcome = "limited"
            elif exc.code == "TOOL_TIMEOUT":
                outcome = "timeout"
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            raise ToolPolicyError("Tool invocation failed", code="TOOL_EXECUTION_FAILED") from exc
        finally:
            self._record(descriptor, principal, outcome, started, generation)

    def close(self) -> None:
        """Release policy worker resources without waiting on timed-out work."""

        self._executor.shutdown(wait=False, cancel_futures=True)

    def _validate_input(
        self,
        tool: BaseTool,
        tool_input: str | dict[str, Any],
    ) -> None:
        schema = tool.get_input_schema()
        if isinstance(tool_input, str):
            fields = list(schema.model_fields)
            if len(fields) == 1:
                schema.model_validate({fields[0]: tool_input})
                return
        schema.model_validate(tool_input)

    def _bound_and_redact(self, value: Any) -> Any:
        sanitized = _sanitize_value(
            value,
            redact=self._redact,
            max_chars=self.limits.max_output_chars,
        )
        try:
            size = len(json.dumps(sanitized, ensure_ascii=False, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            sanitized = self._redact(str(sanitized))
            size = len(sanitized.encode("utf-8"))
        if size <= self.limits.max_output_bytes:
            return sanitized
        marker = "...[tool output truncated]"
        text = self._redact(str(sanitized))
        return text[: max(0, self.limits.max_output_chars - len(marker))] + marker

    def _redact(self, value: str) -> str:
        redacted = _SECRET_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
        for secret in self._redaction_values:
            redacted = redacted.replace(secret, "[REDACTED]")
        return redacted

    def _record(
        self,
        descriptor: ToolDescriptor,
        principal: ToolPrincipal,
        outcome: ToolOutcome,
        started: float,
        generation: int,
    ) -> None:
        event = ToolAuditEvent(
            tool=descriptor.qualified_name[:128],
            outcome=outcome,
            generation=generation,
            duration_ms=min(86_400_000, int((time.monotonic() - started) * 1000)),
            source_server=descriptor.server_name,
            principal_id=principal.principal_id[:128] or "anonymous",
            tenant_id=principal.tenant_id[:128] if principal.tenant_id else None,
        )
        with suppress(Exception):
            self._audit_sink.emit(event)
        if self._metrics is not None:
            with suppress(Exception):
                self._metrics.record(event)


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


def _sanitize_value(
    value: Any,
    *,
    redact: Callable[[str], str],
    max_chars: int,
) -> Any:
    if isinstance(value, str):
        return redact(value)[:max_chars]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            redact(str(key))[:128]: _sanitize_value(item, redact=redact, max_chars=max_chars)
            for key, item in list(value.items())[:256]
        }
    if isinstance(value, tuple):
        return tuple(
            _sanitize_value(item, redact=redact, max_chars=max_chars) for item in value[:256]
        )
    if isinstance(value, list):
        return [_sanitize_value(item, redact=redact, max_chars=max_chars) for item in value[:256]]
    return redact(str(value))[:max_chars]


PolicyTool.model_rebuild()


__all__ = [
    "PolicyTool",
    "ToolExecutionLimits",
    "ToolExecutionPipeline",
    "ToolPolicy",
    "ToolPolicyRule",
    "ToolPrincipal",
    "reset_tool_principal",
    "set_tool_principal",
]
