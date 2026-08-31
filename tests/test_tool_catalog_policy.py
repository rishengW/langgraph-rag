"""Focused Phase 3 tests for immutable catalogs and tool policy execution."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool

from src.graph.builder import GraphNodeOverrides, GraphProviders, build_graph
from src.graph.events import ToolEndEvent, ToolStartEvent
from src.graph.executor import GraphExecutor
from src.graph.metrics import MetricsCollector
from src.mcp import (
    CallbackToolAuditSink,
    DisabledOutboundMCPProvider,
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
    assert isinstance(snapshot.tools, tuple)
    assert isinstance(snapshot.descriptors, tuple)


def test_catalog_rejects_collision_before_publication():
    duplicate = second_echo.model_copy(update={"name": echo_text.name})
    entries = ((echo_text, _descriptor()), (duplicate, _descriptor(duplicate)))

    with pytest.raises(ToolCatalogError, match="Duplicate tool name"):
        compose_snapshot(entries, generation=1)


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
    from src.mcp.providers import InjectedToolProvider

    catalog = ToolCatalog()
    first = await catalog.publish((InjectedToolProvider((echo_text,)),))
    second = await catalog.publish((InjectedToolProvider((second_echo,)),))

    assert first.generation == 1
    assert second.generation == 2
    assert first.tools == (echo_text,)
    assert second.tools == (second_echo,)


async def _failed_catalog_publication_retains_prior_generation():
    from src.mcp.providers import InjectedToolProvider

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
    canonical = echo_text.model_copy(update={"metadata": {"canonical_inbound_mcp_tool": True}})
    pipeline = ToolExecutionPipeline()

    assert pipeline.wrap(canonical, _descriptor(canonical), 1) is canonical
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
    remote = echo_text.model_copy(update={"name": "mcp__github__read_issue"})
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
    import src.core.retriever as retriever_module
    import src.tools as tools_module
    from src.graph.builder import _resolve_lightweight_tools, _resolve_tools

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
