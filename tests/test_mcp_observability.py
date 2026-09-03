"""Bounded observability coverage for MCP server spec task 7.1."""

from __future__ import annotations

import asyncio
import json
import logging
import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.tools import StructuredTool

from src.backend.mcp import (
    CallbackObservationSink,
    MCPObservability,
    NullObservationSink,
    RequiredAuditDeliveryError,
    ToolExecutionPipeline,
    ToolPolicy,
    ToolPolicyError,
    ToolPolicyRule,
    ToolPrincipal,
    descriptor_from_tool,
    reset_tool_principal,
    set_tool_principal,
)
from src.backend.mcp.observability import (
    MAX_OBSERVATION_RECORD_BYTES,
    ObservationEvent,
)


def _event(**updates: object) -> ObservationEvent:
    values: dict[str, object] = {
        "signal": "tool_invocation",
        "outcome": "success",
        "request_id": "request-1",
        "principal_id": "principal-1",
        "tenant_id": "tenant-1",
        "tool": "mcp__approved__read",
        "source_server": "approved",
        "transport": "http",
        "duration_ms": 12,
        "generation": 3,
    }
    values.update(updates)
    return ObservationEvent(**values)  # type: ignore[arg-type]


def _capture_observability(events: list[ObservationEvent]) -> MCPObservability:
    return MCPObservability(
        log_sink=CallbackObservationSink(events.append),
        metric_sink=NullObservationSink(),
        trace_sink=NullObservationSink(),
        audit_sink=NullObservationSink(),
    )


def test_json_exports_are_bounded_and_metric_records_have_fixed_dimensions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "credential-marker-123"
    caplog.set_level(logging.INFO)
    observability = MCPObservability(redaction_values=(secret,))

    sanitized = observability.emit(
        _event(
            request_id=f"request-{secret}",
            principal_id=f"Authorization=Bearer {secret}",
            tenant_id=f"tenant-{secret}",
            source_server=f"https://user:{secret}@approved.example/mcp?access_token={secret}",
        )
    )

    records = {
        json.loads(record.getMessage())["category"]: json.loads(record.getMessage())
        for record in caplog.records
        if record.name in {"mcp.events", "mcp.metrics", "mcp.traces", "mcp.audit"}
    }
    assert set(records) == {"log", "metric", "trace", "audit"}
    assert secret not in json.dumps(records, sort_keys=True)
    assert secret not in repr(sanitized)
    for record in records.values():
        assert len(json.dumps(record, separators=(",", ":")).encode("utf-8")) <= (
            MAX_OBSERVATION_RECORD_BYTES
        )

    metric = records["metric"]
    assert set(metric) == {
        "schema_version",
        "category",
        "signal",
        "outcome",
        "transport",
        "duration_ms",
        "generation",
    }
    for prohibited in ("request_id", "principal_id", "tenant_id", "tool", "source_server"):
        assert prohibited not in metric

    audit = records["audit"]
    assert audit["request_id"]
    assert audit["principal_id"]
    assert audit["tool"] == "mcp__approved__read"
    snapshot = observability.snapshot()
    assert snapshot.metrics.total_events == 1
    assert snapshot.metrics.events_by_signal == {"tool_invocation": 1}
    assert snapshot.metrics.events_by_outcome == {"success": 1}
    assert snapshot.metrics.events_by_transport == {"http": 1}


def test_optional_exporter_failures_are_isolated_and_required_audit_is_explicit() -> None:
    def fail(_event: ObservationEvent) -> None:
        raise RuntimeError("sink detail must remain internal")

    failing = CallbackObservationSink(fail)
    observability = MCPObservability(
        log_sink=failing,
        metric_sink=failing,
        trace_sink=failing,
        audit_sink=failing,
    )

    assert observability.emit(_event()).outcome == "success"
    assert observability.snapshot().export_failures == {
        "log": 1,
        "metric": 1,
        "trace": 1,
        "audit": 1,
    }

    required = MCPObservability(
        log_sink=NullObservationSink(),
        metric_sink=NullObservationSink(),
        trace_sink=NullObservationSink(),
        audit_sink=failing,
        require_audit_delivery=True,
    )
    with pytest.raises(RequiredAuditDeliveryError, match="Required audit delivery failed"):
        required.emit(_event())
    assert required.snapshot().metrics.total_events == 1
    assert required.snapshot().export_failures == {"audit": 1}


def test_observability_closes_shared_exporters_once() -> None:
    class ClosableSink:
        def __init__(self) -> None:
            self.close_calls = 0

        def emit(self, _event: ObservationEvent) -> None:
            return None

        async def aclose(self) -> None:
            self.close_calls += 1

    sink = ClosableSink()
    observability = MCPObservability(
        log_sink=sink,
        metric_sink=sink,
        trace_sink=sink,
        audit_sink=sink,
    )

    asyncio.run(observability.aclose())

    assert sink.close_calls == 1


def test_policy_denial_exports_authorization_and_policy_outcomes() -> None:
    def echo(text: str) -> str:
        return text

    tool = StructuredTool.from_function(
        func=echo,
        name="observed_echo",
        description="Echo text for observability testing.",
    )
    descriptor = descriptor_from_tool(tool, source="builtin", risk_level="read")
    events: list[ObservationEvent] = []
    policy = ToolPolicy(
        rules={tool.name: ToolPolicyRule(allowed_principals=frozenset({"allowed"}))}
    )
    pipeline = ToolExecutionPipeline(
        policy=policy,
        observability=_capture_observability(events),
    )
    wrapped = pipeline.wrap(tool, descriptor, 4)
    token = set_tool_principal(
        ToolPrincipal(principal_id="blocked", tenant_id="tenant-a", request_id="request-a")
    )
    try:
        with pytest.raises(ToolPolicyError) as captured:
            wrapped.invoke({"text": "must-not-appear"})
    finally:
        reset_tool_principal(token)
        pipeline.close()

    assert captured.value.code == "TOOL_DENIED"
    assert [(event.signal, event.outcome) for event in events] == [
        ("authorization", "denied"),
        ("policy_denial", "denied"),
    ]
    assert all(event.request_id == "request-a" for event in events)
    assert "must-not-appear" not in repr(events)


@settings(max_examples=40, deadline=None)
@given(
    secret_suffix=st.text(
        alphabet=string.ascii_lowercase + string.digits,
        min_size=12,
        max_size=24,
    ),
    identities=st.lists(
        st.text(
            alphabet=string.ascii_letters + string.digits + "-_",
            min_size=1,
            max_size=24,
        ),
        min_size=1,
        max_size=25,
        unique=True,
    ),
)
def test_error_and_telemetry_non_disclosure_property(
    secret_suffix: str,
    identities: list[str],
) -> None:
    """**Validates: Requirements 5.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 9.3**"""

    secret = f"credential-{secret_suffix}"
    exported: list[ObservationEvent] = []
    sink = CallbackObservationSink(exported.append)
    observability = MCPObservability(
        log_sink=sink,
        metric_sink=sink,
        trace_sink=sink,
        audit_sink=sink,
        redaction_values=(secret,),
    )

    for index, identity in enumerate(identities):
        observability.emit(
            _event(
                signal="authentication" if index % 2 else "tool_invocation",
                outcome="denied" if index % 2 else "success",
                request_id=f"request-{identity}-{secret}",
                principal_id=f"principal-{identity}-{secret}",
                tenant_id=f"tenant-{identity}-{secret}",
                tool=f"tool-{identity}",
                source_server=f"server-{identity}-{secret}",
                transport="outbound_http" if index % 2 else "stdio",
                generation=index,
            )
        )

    assert len(exported) == len(identities) * 4
    for event in exported:
        serialized = json.dumps(event.to_record("audit"), sort_keys=True)
        assert secret not in serialized
        assert len(serialized.encode("utf-8")) <= MAX_OBSERVATION_RECORD_BYTES
        metric_record = event.to_record("metric")
        assert not {
            "request_id",
            "principal_id",
            "tenant_id",
            "tool",
            "source_server",
        }.intersection(metric_record)

    snapshot = observability.snapshot().metrics
    assert snapshot.total_events == len(identities)
    assert set(snapshot.events_by_signal) <= {"authentication", "tool_invocation"}
    assert set(snapshot.events_by_outcome) <= {"denied", "success"}
    assert set(snapshot.events_by_transport) <= {"outbound_http", "stdio"}
    assert all(
        len(mapping) <= 2
        for mapping in (
            snapshot.events_by_signal,
            snapshot.events_by_outcome,
            snapshot.events_by_transport,
        )
    )
