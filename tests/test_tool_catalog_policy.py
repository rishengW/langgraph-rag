"""Focused Phase 3 tests for immutable catalogs and tool policy execution."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, replace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool

from src.backend.graph.builder import GraphNodeOverrides, GraphProviders, build_graph
from src.backend.graph.events import ToolEndEvent, ToolStartEvent
from src.backend.graph.executor import GraphExecutor
from src.backend.graph.metrics import MetricsCollector
from src.backend.mcp import (
    CallbackToolAuditSink,
    DisabledOutboundMCPProvider,
    InjectedToolProvider,
    PolicyTool,
    ToolCatalog,
    ToolCatalogError,
    ToolCatalogSnapshot,
    ToolDescriptor,
    ToolExecutionLimits,
    ToolExecutionPipeline,
    ToolMetricsRecorder,
    ToolPolicy,
    ToolPolicyError,
    ToolPolicyRule,
    ToolPrincipal,
    compose_snapshot,
    descriptor_from_tool,
    reset_tool_principal,
    set_tool_principal,
    validate_descriptor,
)
from src.backend.mcp.catalog import (
    MAX_SCHEMA_BYTES,
    MAX_SCHEMA_DEPTH,
    MAX_SCHEMA_ENUM_VALUES,
    MAX_SCHEMA_PROPERTIES,
)


@tool
def echo_text(text: str) -> str:
    """Echo supplied text for policy tests."""

    return text


@tool
def second_echo(text: str) -> str:
    """Echo supplied text under a second Python function."""

    return text


def _descriptor(tool=echo_text, **updates):
    values = {
        "source": "builtin",
        "risk_level": "read",
    }
    values.update(updates)
    return descriptor_from_tool(tool, **values)


def test_descriptor_and_snapshot_are_deeply_immutable():
    descriptor = _descriptor()
    snapshot = ToolCatalogSnapshot(1, (echo_text,), (descriptor,))

    with pytest.raises(FrozenInstanceError):
        descriptor.qualified_name = "changed"
    with pytest.raises(TypeError):
        descriptor.input_schema["type"] = "array"
    with pytest.raises(TypeError):
        snapshot.tools_by_name["replacement"] = second_echo
    assert isinstance(snapshot.tools, tuple)
    assert isinstance(snapshot.descriptors, tuple)
    assert snapshot.tools_by_name == {echo_text.name: echo_text}
    assert snapshot.descriptors_by_name == {echo_text.name: descriptor}


def test_catalog_rejects_collision_before_publication():
    duplicate = second_echo.model_copy(update={"name": echo_text.name})
    entries = ((echo_text, _descriptor()), (duplicate, _descriptor(duplicate)))

    with pytest.raises(ToolCatalogError, match="Duplicate tool name"):
        compose_snapshot(entries, generation=1)


@pytest.mark.parametrize("name", ["1invalid", "bad name", "mcp__unscoped"])
def test_catalog_rejects_invalid_or_reserved_local_tool_names(name):
    invalid = echo_text.model_copy(update={"name": name})

    with pytest.raises(ToolCatalogError, match="Invalid tool name|MCP namespace"):
        _descriptor(invalid)


def test_catalog_validates_schema_bounds_and_unsupported_constructs():
    descriptor = ToolDescriptor(
        qualified_name="unsafe_schema",
        display_name="Unsafe Schema",
        source="builtin",
        server_name=None,
        description="Unsafe schema test.",
        input_schema={"type": "object", "not": {"type": "string"}},
        risk_level="read",
    )

    with pytest.raises(ToolCatalogError, match="unsupported constructs"):
        validate_descriptor(descriptor)


def test_catalog_accepts_property_names_that_match_schema_keywords():
    """Property and $defs names are user data, not schema keywords."""

    descriptor = ToolDescriptor(
        qualified_name="format_named_field",
        display_name="Format Named Field",
        source="builtin",
        server_name=None,
        description="Schema keywords used as property names.",
        input_schema={
            "type": "object",
            "properties": {
                "format": {"type": "string", "minLength": 1},
                "properties": {"type": "string"},
                "required": {"type": "string"},
            },
            "required": ["format"],
            "$defs": {"format": {"type": "string"}},
        },
        risk_level="read",
    )

    validate_descriptor(descriptor)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"source": "external"}, "source metadata"),
        ({"risk_level": "owner"}, "risk metadata"),
    ],
)
def test_catalog_rejects_invalid_runtime_source_and_risk_metadata(updates, message):
    descriptor = replace(_descriptor(), **updates)

    with pytest.raises(ToolCatalogError, match=message):
        validate_descriptor(descriptor)


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (
            {
                "type": "object",
                "properties": {
                    f"field_{index}": {"type": "string"}
                    for index in range(MAX_SCHEMA_PROPERTIES + 1)
                },
            },
            "property limit",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "choice": {
                        "type": "string",
                        "enum": [str(index) for index in range(MAX_SCHEMA_ENUM_VALUES + 1)],
                    }
                },
            },
            "enum limit",
        ),
    ],
)
def test_catalog_enforces_schema_property_and_enum_limits(schema, message):
    descriptor = replace(_descriptor(), input_schema=schema)

    with pytest.raises(ToolCatalogError, match=message):
        validate_descriptor(descriptor)


def test_catalog_enforces_schema_byte_and_depth_limits():
    oversized = replace(
        _descriptor(),
        input_schema={
            "type": "object",
            "properties": {},
            "description": "x" * MAX_SCHEMA_BYTES,
        },
    )
    nested = {"type": "string"}
    for _ in range(MAX_SCHEMA_DEPTH + 1):
        nested = {"anyOf": [nested]}
    too_deep = replace(
        _descriptor(),
        input_schema={"type": "object", "properties": {"value": nested}},
    )

    with pytest.raises(ToolCatalogError, match="byte limit"):
        validate_descriptor(oversized)
    with pytest.raises(ToolCatalogError, match="depth limit"):
        validate_descriptor(too_deep)


def test_catalog_rejects_non_json_schema_values():
    descriptor = replace(
        _descriptor(),
        input_schema={"type": "object", "properties": {}, "default": object()},
    )

    with pytest.raises(ToolCatalogError, match="not JSON serializable"):
        validate_descriptor(descriptor)


def test_catalog_rejects_descriptor_schema_that_differs_from_tool_dispatch_schema():
    descriptor = replace(
        _descriptor(),
        input_schema={
            "type": "object",
            "properties": {"other": {"type": "string"}},
            "required": ["other"],
        },
    )

    with pytest.raises(ToolCatalogError, match="does not match its descriptor"):
        compose_snapshot(((echo_text, descriptor),), generation=1)


def test_remote_namespace_requires_matching_server_name():
    valid = ToolDescriptor(
        qualified_name="mcp__github__read_issue",
        display_name="Read issue",
        source="mcp",
        server_name="github",
        description="Read one issue.",
        input_schema={"type": "object", "properties": {}},
        risk_level="read",
    )
    validate_descriptor(valid)

    invalid = ToolDescriptor(
        qualified_name="read_issue",
        display_name="Read issue",
        source="mcp",
        server_name="github",
        description="Read one issue.",
        input_schema={"type": "object", "properties": {}},
        risk_level="read",
    )
    with pytest.raises(ToolCatalogError, match="mcp__"):
        validate_descriptor(invalid)


async def _catalog_publication_is_atomic_and_generational():
    from src.backend.mcp.providers import InjectedToolProvider

    catalog = ToolCatalog()
    first = await catalog.publish((InjectedToolProvider((echo_text,)),))
    second = await catalog.publish((InjectedToolProvider((second_echo,)),))

    assert first.generation == 1
    assert second.generation == 2
    assert first.tools == (echo_text,)
    assert second.tools == (second_echo,)


async def _failed_catalog_publication_retains_prior_generation():
    from src.backend.mcp.providers import InjectedToolProvider

    catalog = ToolCatalog()
    active = await catalog.publish((InjectedToolProvider((echo_text,)),))

    with pytest.raises(ToolCatalogError, match="Duplicate tool name"):
        await catalog.publish(
            (
                InjectedToolProvider((echo_text,)),
                InjectedToolProvider((echo_text,)),
            )
        )

    assert catalog.current is active


class _SnapshotFailingProvider(InjectedToolProvider):
    def __init__(self):
        super().__init__((second_echo,))
        self.closed = False

    async def snapshot(self):
        raise ToolCatalogError("candidate snapshot failed")

    async def close(self):
        self.closed = True
        await super().close()


class _CloseFailingProvider(InjectedToolProvider):
    async def close(self):
        raise RuntimeError("retired provider cleanup failed")


async def _failed_candidate_is_closed_without_replacing_active_generation():
    catalog = ToolCatalog()
    active = await catalog.publish((InjectedToolProvider((echo_text,)),))
    failed = _SnapshotFailingProvider()

    with pytest.raises(ToolCatalogError, match="candidate snapshot failed"):
        await catalog.publish((failed,))

    assert catalog.current is active
    assert failed.closed is True


async def _retired_cleanup_failure_does_not_turn_committed_publication_into_failure():
    catalog = ToolCatalog()
    first = await catalog.publish((_CloseFailingProvider((echo_text,)),))
    second = await catalog.publish((InjectedToolProvider((second_echo,)),))

    assert catalog.current is second
    assert second.generation == first.generation + 1


async def _disabled_outbound_provider_has_no_transport_or_tools():
    provider = DisabledOutboundMCPProvider()
    await provider.start()

    assert (await provider.snapshot()).tools == ()
    assert (await provider.health()).status == "disabled"

    with pytest.raises(Exception, match="disabled"):
        DisabledOutboundMCPProvider(enabled=True)


def test_policy_sync_pipeline_preserves_basetool_redacts_bounds_and_audits():
    events = []
    metrics = ToolMetricsRecorder()
    pipeline = ToolExecutionPipeline(
        audit_sink=CallbackToolAuditSink(events.append),
        metrics=metrics,
        redaction_values=("top-secret",),
        limits=ToolExecutionLimits(max_output_chars=128, max_output_bytes=256),
    )
    wrapped = pipeline.wrap(echo_text, _descriptor(), 7)

    assert isinstance(wrapped, PolicyTool)
    result = wrapped.invoke({"text": "api_key=top-secret " + "x" * 300})
    pipeline.close()

    assert "top-secret" not in result
    assert len(result) <= 128
    assert events[-1].outcome == "success"
    assert events[-1].generation == 7
    assert metrics.snapshot().calls_by_outcome == {"success": 1}


def test_policy_authorizes_before_validation_and_sanitizes_denial():
    events = []
    policy = ToolPolicy(
        rules={echo_text.name: ToolPolicyRule(allowed_principals=frozenset({"allowed"}))}
    )
    pipeline = ToolExecutionPipeline(
        policy=policy,
        audit_sink=CallbackToolAuditSink(events.append),
    )
    wrapped = pipeline.wrap(echo_text, _descriptor(), 2)
    token = set_tool_principal(ToolPrincipal("blocked"))
    try:
        with pytest.raises(ToolPolicyError) as captured:
            wrapped.invoke({"unexpected": "secret-value"})
    finally:
        reset_tool_principal(token)
        pipeline.close()

    assert captured.value.code == "TOOL_DENIED"
    assert "secret-value" not in str(captured.value)
    assert events[-1].outcome == "denied"


async def _policy_async_deadline_records_sanitized_timeout():
    async def slow_echo(text: str) -> str:
        await asyncio.sleep(0.05)
        return text

    slow_tool = StructuredTool.from_function(
        coroutine=slow_echo,
        name="slow_echo",
        description="Slow asynchronous echo.",
    )
    descriptor = _descriptor(slow_tool)
    events = []
    pipeline = ToolExecutionPipeline(
        limits=ToolExecutionLimits(deadline_seconds=0.01),
        audit_sink=CallbackToolAuditSink(events.append),
    )
    wrapped = pipeline.wrap(slow_tool, descriptor, 4)

    with pytest.raises(ToolPolicyError) as captured:
        await wrapped.ainvoke({"text": "sensitive input"})
    pipeline.close()

    assert captured.value.code == "TOOL_TIMEOUT"
    assert "sensitive input" not in str(captured.value)
    assert events[-1].outcome == "timeout"


def test_policy_does_not_double_wrap_canonical_inbound_tool():
    canonical = echo_text.model_copy(
        update={
            "name": "rag_ask",
            "metadata": {"canonical_inbound_mcp_tool": True},
        }
    )
    untrusted_marker = echo_text.model_copy(
        update={"metadata": {"canonical_inbound_mcp_tool": True}}
    )
    pipeline = ToolExecutionPipeline()

    assert pipeline.wrap(canonical, _descriptor(canonical), 1) is canonical
    assert isinstance(
        pipeline.wrap(untrusted_marker, _descriptor(untrusted_marker), 1),
        PolicyTool,
    )
    pipeline.close()


def test_graph_rejects_catalog_collision_before_compile():
    duplicate = second_echo.model_copy(update={"name": echo_text.name})
    providers = GraphProviders(
        tools=(echo_text, duplicate),
        nodes=GraphNodeOverrides(
            agent=lambda _state: {},
            retrieve=lambda _state: {},
            grade_documents=lambda _state: "generate",
            rewrite=lambda _state: {},
            generate=lambda _state: {},
        ),
    )

    with pytest.raises(ToolCatalogError, match="Duplicate tool name"):
        build_graph(providers=providers)


def test_graph_attaches_same_immutable_snapshot_used_by_dispatcher():
    providers = GraphProviders(
        tools=(echo_text,),
        nodes=GraphNodeOverrides(
            condense=lambda _state: {},
            agent=lambda _state: {},
            retrieve=lambda _state: {},
            grade_documents=lambda _state: "generate",
            rewrite=lambda _state: {},
            generate=lambda _state: {},
        ),
        catalog_generation=9,
    )

    graph = build_graph(providers=providers)
    snapshot = graph.tool_catalog_snapshot

    assert snapshot.generation == 9
    assert graph.tool_catalog_generation == 9
    assert graph.tool_descriptors is snapshot.descriptors
    assert isinstance(snapshot.tools[0], PolicyTool)


def test_graph_revalidates_injected_snapshot_before_model_or_dispatcher_setup():
    invalid = replace(_descriptor(), risk_level="owner")
    snapshot = ToolCatalogSnapshot(1, (echo_text,), (invalid,))
    providers = GraphProviders(
        catalog_snapshot=snapshot,
        nodes=GraphNodeOverrides(
            agent=lambda _state: {},
            retrieve=lambda _state: {},
            grade_documents=lambda _state: "generate",
            rewrite=lambda _state: {},
            generate=lambda _state: {},
        ),
    )

    with pytest.raises(ToolCatalogError, match="risk metadata"):
        build_graph(providers=providers)


def test_graph_model_and_dispatcher_receive_same_immutable_catalog_tuple(
    monkeypatch,
    isolated_settings,
):
    import src.backend.graph.builder as builder_module

    captured = {}

    def capture_agent(_settings, tools, _resolver):
        captured["model"] = tools
        return lambda _state: {}

    def capture_dispatcher(tools):
        captured["dispatcher"] = tools
        return lambda _state: {}

    monkeypatch.setattr(builder_module, "agent_factory", capture_agent)
    monkeypatch.setattr(builder_module, "ToolNode", capture_dispatcher)
    providers = GraphProviders(
        tools=(echo_text,),
        nodes=GraphNodeOverrides(
            grade_documents=lambda _state: "generate",
            rewrite=lambda _state: {},
            generate=lambda _state: {},
        ),
    )

    graph = builder_module.build_graph(
        settings=isolated_settings(),
        providers=providers,
    )
    snapshot = graph.tool_catalog_snapshot

    assert captured["model"] is snapshot.tools
    assert captured["dispatcher"] is snapshot.tools


class _ToolEventGraph:
    def invoke(self, inputs, config=None):
        return inputs

    def stream(self, inputs, config=None):
        yield {
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "mcp__github__read_issue",
                                "args": {},
                                "id": "call-1",
                            }
                        ],
                    )
                ]
            }
        }
        yield {
            "tools": {
                "messages": [
                    ToolMessage(
                        content="safe result",
                        tool_call_id="call-1",
                        status="success",
                    )
                ]
            }
        }


def test_executor_tracks_catalog_tool_metadata_and_bounded_metrics():
    remote = echo_text.model_copy(
        update={
            "name": "mcp__github__read_issue",
            "description": "Read one issue.",
        }
    )
    descriptor = ToolDescriptor(
        qualified_name=remote.name,
        display_name="Read Issue",
        source="mcp",
        server_name="github",
        description="Read one issue.",
        input_schema=remote.get_input_schema().model_json_schema(),
        risk_level="read",
    )
    snapshot = compose_snapshot(((remote, descriptor),), generation=12)
    metrics = MetricsCollector()
    events = list(
        GraphExecutor(_ToolEventGraph(), metrics=metrics, catalog_snapshot=snapshot).stream({})
    )

    started = next(event for event in events if isinstance(event, ToolStartEvent))
    ended = next(event for event in events if isinstance(event, ToolEndEvent))
    assert started.source_server == "github"
    assert started.catalog_generation == 12
    assert ended.source_server == "github"
    assert ended.catalog_generation == 12
    assert ended.outcome == "success"
    assert 0 <= ended.duration_ms <= 86_400_000
    metric_snapshot = metrics.snapshot()
    assert metric_snapshot.tool_call_count == 1
    assert metric_snapshot.tool_outcomes == {"success": 1}
    assert metric_snapshot.catalog_generation == 12


def test_full_and_lightweight_catalog_composition_differs_only_at_required_path(
    monkeypatch,
    isolated_settings,
):
    import src.backend.core.retriever as retriever_module
    import src.backend.tools as tools_module
    from src.backend.graph.builder import _resolve_lightweight_tools, _resolve_tools

    settings = isolated_settings(
        web_search_enabled=False,
        math_enabled=True,
    )
    retriever = echo_text.model_copy(update={"name": "retrieve_source_documents"})
    web = echo_text.model_copy(update={"name": "live_web_search"})
    compute = echo_text.model_copy(update={"name": "solve_math"})
    monkeypatch.setattr(
        retriever_module,
        "build_retriever_tool",
        lambda _settings, rebuild=False: retriever,
    )
    monkeypatch.setattr(tools_module, "build_web_search_tool", lambda _settings: web)
    monkeypatch.setattr(tools_module, "build_math_tool", lambda _settings: compute)

    full = _resolve_tools(settings, GraphProviders(), rebuild_vectorstore=False)
    lightweight = _resolve_lightweight_tools(settings, GraphProviders())

    assert [item.name for item in full] == [retriever.name, compute.name]
    assert [item.name for item in lightweight] == [web.name, compute.name]


def test_catalog_publication_is_atomic_and_generational():
    asyncio.run(_catalog_publication_is_atomic_and_generational())


def test_failed_catalog_publication_retains_prior_generation():
    asyncio.run(_failed_catalog_publication_retains_prior_generation())


def test_disabled_outbound_provider_has_no_transport_or_tools():
    asyncio.run(_disabled_outbound_provider_has_no_transport_or_tools())


def test_policy_async_deadline_records_sanitized_timeout():
    asyncio.run(_policy_async_deadline_records_sanitized_timeout())


def test_failed_candidate_is_closed_without_replacing_active_generation():
    asyncio.run(_failed_candidate_is_closed_without_replacing_active_generation())


def test_retired_cleanup_failure_does_not_turn_committed_publication_into_failure():
    asyncio.run(_retired_cleanup_failure_does_not_turn_committed_publication_into_failure())


def test_policy_allowlists_intersect_and_denials_do_not_disclose_resource_existence():
    """Runtime policy can narrow, but never replace, catalog ownership restrictions."""

    descriptor = replace(
        _descriptor(),
        allowed_principals=frozenset({"shared-owner", "catalog-owner"}),
    )
    policy = ToolPolicy(
        rules={
            echo_text.name: ToolPolicyRule(
                allowed_principals=frozenset({"shared-owner", "runtime-owner"})
            )
        }
    )
    pipeline = ToolExecutionPipeline(policy=policy)
    try:
        token = set_tool_principal(ToolPrincipal("shared-owner"))
        try:
            assert pipeline.wrap(echo_text, descriptor, 1).invoke({"text": "ok"}) == "ok"
        finally:
            reset_tool_principal(token)

        denial_messages = []
        for principal_id in ("catalog-owner", "runtime-owner"):
            token = set_tool_principal(ToolPrincipal(principal_id))
            try:
                with pytest.raises(ToolPolicyError) as captured:
                    pipeline.execute(
                        echo_text,
                        descriptor,
                        {"text": "resource-does-not-exist"},
                        config=None,
                        generation=1,
                    )
                denial_messages.append(str(captured.value))
            finally:
                reset_tool_principal(token)
        assert denial_messages == ["Tool invocation is not authorized"] * 2
        assert all("resource-does-not-exist" not in message for message in denial_messages)
    finally:
        pipeline.close()


def test_policy_rejects_unknown_input_before_invocation_and_records_invalid_outcome():
    calls = []

    def capture(text: str) -> str:
        calls.append(text)
        return text

    capture_tool = StructuredTool.from_function(
        func=capture,
        name="capture_text",
        description="Capture text for validation ordering tests.",
    )
    events = []
    pipeline = ToolExecutionPipeline(audit_sink=CallbackToolAuditSink(events.append))
    wrapped = pipeline.wrap(capture_tool, _descriptor(capture_tool), 3)
    try:
        with pytest.raises(ToolPolicyError) as captured:
            wrapped.invoke({"text": "safe", "unknown": "secret-input"})
    finally:
        pipeline.close()

    assert captured.value.code == "TOOL_INVALID"
    assert "secret-input" not in str(captured.value)
    assert calls == []
    assert [event.outcome for event in events] == ["invalid"]


def test_policy_enforces_principal_and_tenant_rate_limits_with_bounded_outcomes():
    events = []
    pipeline = ToolExecutionPipeline(
        limits=ToolExecutionLimits(
            rate_limit_per_minute=1,
            tenant_rate_limit_per_minute=1,
        ),
        audit_sink=CallbackToolAuditSink(events.append),
    )
    wrapped = pipeline.wrap(echo_text, _descriptor(), 5)
    try:
        token = set_tool_principal(ToolPrincipal("principal-a", "tenant-a"))
        try:
            assert wrapped.invoke({"text": "first"}) == "first"
            with pytest.raises(ToolPolicyError) as principal_limited:
                wrapped.invoke({"text": "second"})
        finally:
            reset_tool_principal(token)

        token = set_tool_principal(ToolPrincipal("principal-b", "tenant-a"))
        try:
            with pytest.raises(ToolPolicyError) as tenant_limited:
                wrapped.invoke({"text": "third"})
        finally:
            reset_tool_principal(token)
    finally:
        pipeline.close()

    assert principal_limited.value.code == "TOOL_LIMITED"
    assert tenant_limited.value.code == "TOOL_LIMITED"
    assert [event.outcome for event in events] == ["success", "limited", "limited"]


def test_sync_timeout_retains_concurrency_slot_until_underlying_work_exits():
    import threading

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking(text: str) -> str:
        entered.set()
        try:
            release.wait(timeout=1)
            return text
        finally:
            finished.set()

    blocking_tool = StructuredTool.from_function(
        func=blocking,
        name="blocking_echo",
        description="Block until released for timeout limit tests.",
    )
    events = []
    pipeline = ToolExecutionPipeline(
        limits=ToolExecutionLimits(
            deadline_seconds=0.01,
            max_concurrency=2,
            max_concurrency_per_principal=1,
        ),
        audit_sink=CallbackToolAuditSink(events.append),
    )
    wrapped = pipeline.wrap(blocking_tool, _descriptor(blocking_tool), 6)
    token = set_tool_principal(ToolPrincipal("same-principal"))
    try:
        with pytest.raises(ToolPolicyError) as timed_out:
            wrapped.invoke({"text": "first"})
        assert entered.is_set()
        with pytest.raises(ToolPolicyError) as limited:
            wrapped.invoke({"text": "second"})
    finally:
        reset_tool_principal(token)
        release.set()
        assert finished.wait(timeout=1)
        pipeline.close()

    assert timed_out.value.code == "TOOL_TIMEOUT"
    assert limited.value.code == "TOOL_LIMITED"
    assert [event.outcome for event in events] == ["timeout", "limited"]


def test_policy_structurally_redacts_and_byte_bounds_output_when_audit_sink_fails():
    import json

    secret = "provider-secret-marker"

    def provider_payload(text: str) -> dict[str, object]:
        return {
            "api_key": secret,
            "authorization": f"Bearer {secret}",
            "data": [text, "😀" * 1_000],
        }

    payload_tool = StructuredTool.from_function(
        func=provider_payload,
        name="provider_payload",
        description="Return provider output for sanitization tests.",
    )

    def failing_audit(_event):
        raise RuntimeError("audit unavailable")

    pipeline = ToolExecutionPipeline(
        limits=ToolExecutionLimits(max_output_chars=128, max_output_bytes=256),
        audit_sink=CallbackToolAuditSink(failing_audit),
        redaction_values=(secret,),
    )
    wrapped = pipeline.wrap(payload_tool, _descriptor(payload_tool), 8)
    try:
        result = wrapped.invoke({"text": "visible"})
    finally:
        pipeline.close()

    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert len(encoded) <= 256
    assert secret.encode() not in encoded


def test_policy_output_non_disclosure_and_bounds_property():
    """**Validates: Requirements 5.5, 7.4**"""

    import json

    from hypothesis import given, settings
    from hypothesis import strategies as st

    @settings(max_examples=25, deadline=None)
    @given(
        payload=st.text(max_size=2_000),
        suffix=st.binary(min_size=4, max_size=12).map(bytes.hex),
    )
    def property_check(payload: str, suffix: str) -> None:
        secret = f"secret-{suffix}"
        events = []
        pipeline = ToolExecutionPipeline(
            limits=ToolExecutionLimits(max_output_chars=128, max_output_bytes=256),
            audit_sink=CallbackToolAuditSink(events.append),
            redaction_values=(secret,),
        )
        try:
            result = pipeline.execute(
                echo_text,
                _descriptor(),
                {"text": f"{payload}|{secret}"},
                config=None,
                generation=11,
            )
        finally:
            pipeline.close()

        encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        assert len(encoded) <= 256
        assert secret.encode() not in encoded
        assert [event.outcome for event in events] == ["success"]

    property_check()


def test_policy_async_cancellation_releases_limit_and_records_outcome():
    async def run() -> None:
        started = asyncio.Event()

        async def wait_forever(text: str) -> str:
            started.set()
            await asyncio.Event().wait()
            return text

        waiting_tool = StructuredTool.from_function(
            coroutine=wait_forever,
            name="waiting_echo",
            description="Wait for cancellation in policy tests.",
        )
        events = []
        pipeline = ToolExecutionPipeline(
            limits=ToolExecutionLimits(max_concurrency=1, max_concurrency_per_principal=1),
            audit_sink=CallbackToolAuditSink(events.append),
        )
        wrapped = pipeline.wrap(waiting_tool, _descriptor(waiting_tool), 9)
        task = asyncio.create_task(wrapped.ainvoke({"text": "cancel me"}))
        await started.wait()
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            pipeline.close()
        assert [event.outcome for event in events] == ["cancelled"]

    asyncio.run(run())


def test_graph_applies_policy_pipeline_to_an_injected_raw_catalog_snapshot():
    raw_snapshot = ToolCatalogSnapshot(13, (echo_text,), (_descriptor(),))
    providers = GraphProviders(
        catalog_snapshot=raw_snapshot,
        nodes=GraphNodeOverrides(
            condense=lambda _state: {},
            agent=lambda _state: {},
            retrieve=lambda _state: {},
            grade_documents=lambda _state: "generate",
            rewrite=lambda _state: {},
            generate=lambda _state: {},
        ),
    )

    graph = build_graph(providers=providers)

    assert graph.tool_catalog_generation == 13
    assert isinstance(graph.tool_catalog_snapshot.tools[0], PolicyTool)
    assert graph.tool_catalog_snapshot.descriptors is not raw_snapshot.descriptors
    assert graph.tool_catalog_snapshot.descriptors == raw_snapshot.descriptors
