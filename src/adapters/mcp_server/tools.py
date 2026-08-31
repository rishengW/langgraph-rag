"""Canonical inbound MCP tools and their bounded execution pipeline."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Annotated, Any, Protocol
from uuid import uuid4

from mcp.server import MCPServer
from mcp.server.context import ServerRequestContext
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.tools import Tool
from mcp_types import (
    CallToolRequestParams,
    CallToolResult,
    InputRequiredResult,
    TextContent,
)
from pydantic import BaseModel, ConfigDict, Field

from ...application import RagAnswer, RagApplicationError, RagRequest
from ...mcp import (
    MCPObservability,
    ToolPrincipal,
    reset_tool_principal,
    set_tool_principal,
)
from ...mcp.observability import ObservationOutcome, ObservationTransport
from .audit import AuditEvent, emit_audit
from .auth import current_principal_id
from .config import MCPSettings
from .url_policy import UnsafeSourceURLError, URLValidator

CANONICAL_TOOL_NAMES = ("rag_ask", "rag_web_search_answer")
_REDACTION = re.compile(
    r"(?i)\b(api[_ -]?key|authorization|bearer|access[_ -]?token|refresh[_ -]?token)"
    r"(\s*[:=]\s*|\s+)[^\s,;]+"
)


class RagService(Protocol):
    async def ask(self, request: RagRequest) -> RagAnswer: ...


class MCPSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    title: str | None = None
    citation_id: str | None = None


class MCPToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    sources: list[MCPSource]
    request_id: str
    grounded: bool
    warnings: list[str]


class PublicToolFailure(Exception):
    def __init__(self, code: str, message: str, request_id: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.request_id = request_id

    def wire_text(self) -> str:
        return json.dumps(
            {"code": self.code, "message": self.message, "request_id": self.request_id},
            sort_keys=True,
        )


class SafeMCPServer(MCPServer[Any]):
    """MCPServer that sanitizes SDK validation and unexpected tool failures."""

    _observability: MCPObservability | None = None
    _observation_settings: MCPSettings | None = None

    def configure_observability(
        self,
        observability: MCPObservability,
        settings: MCPSettings,
    ) -> None:
        """Attach the already-initialized exporter without altering SDK construction."""

        self._observability = observability
        self._observation_settings = settings

    async def _handle_call_tool(
        self,
        ctx: ServerRequestContext[Any],
        params: CallToolRequestParams,
    ) -> CallToolResult | InputRequiredResult:
        context = Context(
            request_context=ctx,
            mcp_server=self,
            input_params=params,
            subscriptions=self._subscriptions,
        )
        try:
            return await self.call_tool(params.name, params.arguments or {}, context)
        except Exception as exc:
            public = _find_public_failure(exc)
            adapter_recorded = public is not None
            if public is None:
                public = PublicToolFailure(
                    "INVALID_REQUEST" if isinstance(exc, ToolError) else "INTERNAL_ERROR",
                    "The tool request was rejected."
                    if isinstance(exc, ToolError)
                    else "The tool could not be completed.",
                    uuid4().hex,
                )
            if not adapter_recorded and self._observability is not None:
                settings = self._observation_settings
                principal_id = "unknown"
                transport: ObservationTransport = "internal"
                if settings is not None:
                    principal_id = current_principal_id(settings)
                    transport = settings.transport
                emit_audit(
                    AuditEvent(
                        request_id=public.request_id,
                        principal_id=_bounded_audit_field(principal_id, fallback="unknown"),
                        tool=_bounded_audit_field(params.name, fallback="unknown_tool"),
                        outcome="invalid" if isinstance(exc, ToolError) else "internal_error",
                        duration_ms=0,
                        signal="tool_invocation",
                        transport=transport,
                    ),
                    self._observability,
                )
            return CallToolResult(
                content=[TextContent(type="text", text=public.wire_text())],
                is_error=True,
            )


def _find_public_failure(exc: BaseException) -> PublicToolFailure | None:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, PublicToolFailure):
            return current
        current = current.__cause__ or current.__context__
    return None


def _bounded_audit_field(value: str, *, fallback: str) -> str:
    bounded = "".join(
        character if 32 <= ord(character) != 127 else "?" for character in str(value)
    )[:128]
    return bounded or fallback


class RequestGate:
    """Process-local rate and concurrency control for the supported single instance."""

    def __init__(self, settings: MCPSettings) -> None:
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._rate = settings.rate_limit_per_minute
        self._calls: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._accepting = True
        self._active = 0
        self._drained = asyncio.Event()
        self._drained.set()

    @property
    def accepting(self) -> bool:
        return self._accepting

    def stop_accepting(self) -> None:
        self._accepting = False

    async def wait_for_drain(self, timeout: float) -> bool:
        try:
            async with asyncio.timeout(timeout):
                await self._drained.wait()
            return True
        except TimeoutError:
            return False

    @asynccontextmanager
    async def slot(self, principal_id: str, request_id: str) -> AsyncIterator[None]:
        if not self._accepting:
            raise PublicToolFailure("NOT_READY", "The MCP server is shutting down.", request_id)
        now = time.monotonic()
        async with self._lock:
            calls = self._calls[principal_id]
            while calls and now - calls[0] >= 60.0:
                calls.popleft()
            if len(calls) >= self._rate:
                raise PublicToolFailure(
                    "RATE_LIMITED", "The request rate limit was exceeded.", request_id
                )
            calls.append(now)
        await self._semaphore.acquire()
        self._active += 1
        self._drained.clear()
        try:
            yield
        finally:
            self._active -= 1
            if self._active == 0:
                self._drained.set()
            self._semaphore.release()


class CanonicalToolAdapter:
    def __init__(
        self,
        *,
        service: RagService,
        settings: MCPSettings,
        url_validator: URLValidator,
        gate: RequestGate,
        observability: MCPObservability,
        redaction_values: Sequence[str] = (),
    ) -> None:
        self._service = service
        self._settings = settings
        self._url_validator = url_validator
        self._gate = gate
        self._observability = observability
        self._redaction_values = tuple(value for value in redaction_values if value)

    async def ask(
        self,
        question: str,
        sources: list[str] | None,
        ctx: Context[Any, Any],
    ) -> MCPToolResult:
        return await self._execute(
            tool="rag_ask",
            question=question,
            sources=sources,
            web_search=False,
            ctx=ctx,
        )

    async def web_search_answer(
        self,
        question: str,
        ctx: Context[Any, Any],
    ) -> MCPToolResult:
        return await self._execute(
            tool="rag_web_search_answer",
            question=question,
            sources=None,
            web_search=True,
            ctx=ctx,
        )

    async def _execute(
        self,
        *,
        tool: str,
        question: str,
        sources: list[str] | None,
        web_search: bool,
        ctx: Context[Any, Any],
    ) -> MCPToolResult:
        request_id = uuid4().hex
        started = time.monotonic()
        principal_id = current_principal_id(self._settings)
        principal_token = set_tool_principal(
            ToolPrincipal(principal_id=principal_id, request_id=request_id)
        )
        outcome: ObservationOutcome = "internal_error"
        try:
            if not question.strip() or len(question) > self._settings.max_question_chars:
                raise PublicToolFailure(
                    "INVALID_REQUEST",
                    f"Question must contain 1 to {self._settings.max_question_chars} characters.",
                    request_id,
                )
            async with asyncio.timeout(self._settings.deadline_seconds):
                validated_sources = None
                if sources is not None:
                    validated_sources = await self._url_validator.validate(
                        sources, maximum=self._settings.max_source_urls
                    )
                async with self._gate.slot(principal_id, request_id):
                    answer = await self._service.ask(
                        RagRequest(
                            question=question,
                            urls=validated_sources,
                            rebuild=False,
                            web_search=web_search,
                            debug=False,
                            request_id=request_id,
                        )
                    )
            if not answer.success or answer.error is not None or answer.answer is None:
                raise PublicToolFailure(
                    "UPSTREAM_FAILURE",
                    "The RAG service could not complete the request.",
                    request_id,
                )
            result = self._bounded_result(answer, require_web_search=web_search)
            outcome = "success"
            return result
        except TimeoutError:
            outcome = "timeout"
            raise PublicToolFailure(
                "DEADLINE_EXCEEDED", "The request exceeded the server deadline.", request_id
            ) from None
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except UnsafeSourceURLError:
            outcome = "rejected"
            raise PublicToolFailure(
                "UNSAFE_SOURCE_URL", "One or more source URLs were rejected.", request_id
            ) from None
        except PublicToolFailure as exc:
            outcome = "limited" if exc.code == "RATE_LIMITED" else "rejected"
            raise
        except RagApplicationError:
            outcome = "upstream_failure"
            raise PublicToolFailure(
                "UPSTREAM_FAILURE", "The RAG service could not complete the request.", request_id
            ) from None
        except Exception:
            outcome = "internal_error"
            raise PublicToolFailure(
                "INTERNAL_ERROR", "The tool could not be completed.", request_id
            ) from None
        finally:
            reset_tool_principal(principal_token)
            emit_audit(
                AuditEvent(
                    request_id=request_id,
                    principal_id=principal_id,
                    tool=tool,
                    outcome=outcome,
                    duration_ms=min(86_400_000, int((time.monotonic() - started) * 1000)),
                    signal=("limit_rejection" if outcome == "limited" else "tool_invocation"),
                    transport=self._settings.transport,
                ),
                self._observability,
            )

    def _bounded_result(self, answer: Any, *, require_web_search: bool) -> MCPToolResult:
        raw_answer = self._redact(str(answer.answer or ""))
        answer_text = raw_answer[: self._settings.max_answer_chars]
        warnings: list[str] = []
        if len(raw_answer) > len(answer_text):
            warnings.append("answer_truncated")
        note = str(answer.source_note or "")
        if note in {"web_search_failed", "web_search_no_results", "web_search_disabled"}:
            warnings.append(note)

        sources = [
            MCPSource(
                url=_safe_public_url(str(source.url)),
                title=self._redact(str(source.title))[:256] if source.title else None,
                citation_id=str(source.citation_id)[:64] if source.citation_id else None,
            )
            for source in answer.sources[: self._settings.max_returned_sources]
        ]
        if len(answer.sources) > len(sources):
            warnings.append("sources_truncated")
        grounded = bool(sources) and (not require_web_search or answer.source_mode == "web_search")
        result = MCPToolResult(
            answer=answer_text,
            sources=sources,
            request_id=str(answer.request_id)[:128],
            grounded=grounded,
            warnings=warnings[: self._settings.max_warnings],
        )
        return self._fit_output_bytes(result)

    def _fit_output_bytes(self, result: MCPToolResult) -> MCPToolResult:
        if _serialized_size(result) <= self._settings.max_output_bytes:
            return result
        warnings = list(result.warnings)
        if "output_truncated" not in warnings:
            warnings.append("output_truncated")
        candidate = result.model_copy(update={"warnings": warnings[: self._settings.max_warnings]})
        low, high = 0, len(candidate.answer)
        while low < high:
            middle = (low + high + 1) // 2
            probe = candidate.model_copy(update={"answer": candidate.answer[:middle]})
            if _serialized_size(probe) <= self._settings.max_output_bytes:
                low = middle
            else:
                high = middle - 1
        candidate = candidate.model_copy(update={"answer": candidate.answer[:low]})
        while _serialized_size(candidate) > self._settings.max_output_bytes and candidate.sources:
            candidate = candidate.model_copy(update={"sources": candidate.sources[:-1]})
        if _serialized_size(candidate) > self._settings.max_output_bytes:
            candidate = candidate.model_copy(update={"warnings": [], "answer": ""})
        return candidate

    def _redact(self, value: str) -> str:
        redacted = _REDACTION.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
        for secret in self._redaction_values:
            redacted = redacted.replace(secret, "[REDACTED]")
        return redacted


def _serialized_size(result: MCPToolResult) -> int:
    return len(json.dumps(result.model_dump(), ensure_ascii=False).encode("utf-8"))


def _safe_public_url(value: str) -> str:
    from urllib.parse import urlsplit, urlunsplit

    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))[:4096]


def _validate_schema_bounds(schema: dict[str, Any]) -> None:
    properties = 0
    enums = 0

    def visit(value: Any, depth: int) -> None:
        nonlocal properties, enums
        if depth > 8:
            raise ValueError("Generated tool schema exceeds the hard nesting-depth limit")
        if isinstance(value, dict):
            raw_properties = value.get("properties")
            if isinstance(raw_properties, dict):
                properties += len(raw_properties)
            raw_enum = value.get("enum")
            if isinstance(raw_enum, list):
                enums += len(raw_enum)
            for nested in value.values():
                visit(nested, depth + 1)
        elif isinstance(value, list):
            for nested in value:
                visit(nested, depth + 1)

    visit(schema, 0)
    if properties > 32:
        raise ValueError("Generated tool schema exceeds the hard property-count limit")
    if enums > 64:
        raise ValueError("Generated tool schema exceeds the hard enum-count limit")


def build_canonical_tools(adapter: CanonicalToolAdapter) -> tuple[Tool, Tool]:
    """Build and validate the complete generation before server publication."""

    async def rag_ask(
        question: Annotated[str, Field(min_length=1, max_length=16_000)],
        ctx: Context[Any, Any],
        sources: list[str] | None = None,
    ) -> MCPToolResult:
        """Answer a question using optional explicit HTTPS sources or configured defaults."""

        return await adapter.ask(question, sources, ctx)

    async def rag_web_search_answer(
        question: Annotated[str, Field(min_length=1, max_length=16_000)],
        ctx: Context[Any, Any],
    ) -> MCPToolResult:
        """Discover web sources and return a bounded grounded answer."""

        return await adapter.web_search_answer(question, ctx)

    tools = (
        Tool.from_function(rag_ask, name="rag_ask", structured_output=True),
        Tool.from_function(
            rag_web_search_answer,
            name="rag_web_search_answer",
            structured_output=True,
        ),
    )
    for tool in tools:
        argument_model = tool.fn_metadata.arg_model
        argument_model.model_config["extra"] = "forbid"
        argument_model.model_rebuild(force=True)
        tool.parameters = argument_model.model_json_schema(by_alias=True)
        tool.parameters["additionalProperties"] = False
        question_schema = tool.parameters.get("properties", {}).get("question", {})
        question_schema["maxLength"] = adapter._settings.max_question_chars
        if tool.name == "rag_ask":
            source_schema = tool.parameters.get("properties", {}).get("sources", {})
            for candidate in source_schema.get("anyOf", [source_schema]):
                if candidate.get("type") == "array":
                    candidate["maxItems"] = adapter._settings.max_source_urls
                    candidate["items"] = {
                        "type": "string",
                        "format": "uri",
                        "maxLength": 4096,
                    }
        _validate_schema_bounds(tool.parameters)
        if len(json.dumps(tool.parameters, separators=(",", ":")).encode("utf-8")) > 16_384:
            raise ValueError(f"Generated schema for {tool.name} exceeds the hard byte limit")
    if tuple(tool.name for tool in tools) != CANONICAL_TOOL_NAMES:
        raise ValueError("Canonical MCP tool generation is incomplete")
    return tools


def create_mcp_server(
    *,
    service: RagService,
    settings: MCPSettings,
    url_validator: URLValidator | None = None,
    gate: RequestGate | None = None,
    observability: MCPObservability | None = None,
    redaction_values: Sequence[str] = (),
    auth_settings: Any = None,
    token_verifier: Any = None,
) -> tuple[SafeMCPServer, RequestGate]:
    """Atomically construct a ready server containing exactly two tools."""

    resolved_gate = gate or RequestGate(settings)
    if observability is not None and not isinstance(observability, MCPObservability):
        raise TypeError("Inbound MCP observability must be MCPObservability")
    resolved_observability = observability or MCPObservability(redaction_values=redaction_values)
    adapter = CanonicalToolAdapter(
        service=service,
        settings=settings,
        url_validator=url_validator or URLValidator(),
        gate=resolved_gate,
        observability=resolved_observability,
        redaction_values=redaction_values,
    )
    tools = list(build_canonical_tools(adapter))
    server = SafeMCPServer(
        "langgraph-rag",
        title="LangGraph RAG",
        description="Stateless bounded RAG tools",
        version="0.1.0",
        tools=tools,
        warn_on_duplicate_tools=True,
        auth=auth_settings,
        token_verifier=token_verifier,
    )
    server.configure_observability(resolved_observability, settings)
    return server, resolved_gate


__all__ = [
    "CANONICAL_TOOL_NAMES",
    "MCPSource",
    "MCPToolResult",
    "PublicToolFailure",
    "RequestGate",
    "SafeMCPServer",
    "build_canonical_tools",
    "create_mcp_server",
]
