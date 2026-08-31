from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from src.application import (
    SessionLifecycleDependencies,
    SessionLifecycleService,
    StartSessionRequest,
    TurnExecutionDependencies,
    TurnExecutionService,
    TurnRequest,
)
from src.graph.events import DoneEvent, ErrorEvent, NodeStartEvent
from src.sessions import ChatSession, ChatSessionRegistry


class RecordingCheckpointer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def snapshot_thread(self, thread_id: str) -> bytes:
        self.calls.append(("snapshot", thread_id))
        return b"exact-snapshot"

    def restore_thread(self, thread_id: str, snapshot: bytes) -> None:
        self.calls.append(("restore", (thread_id, snapshot)))


class InvokeGraph:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def get_state(self, _config: object) -> object:
        return SimpleNamespace(values={"messages": list(self.messages)})

    def invoke(self, inputs: dict[str, object], _config: object) -> dict[str, object]:
        self.messages.extend(inputs["messages"])  # type: ignore[arg-type]
        self.messages.append(AIMessage(content="completed answer"))
        return {"messages": list(self.messages)}


def make_session(mock_settings: object, graph: object) -> ChatSession:
    return ChatSession(thread_id="thread", graph=graph, settings=mock_settings)  # type: ignore[arg-type]


async def _direct_turn_service_completion_snapshots_and_runs_hook(mock_settings) -> None:
    graph = InvokeGraph()
    checkpointer = RecordingCheckpointer()
    hooks: list[str] = []
    service = TurnExecutionService(
        TurnExecutionDependencies(
            build_inputs=lambda _session, message: {"messages": [HumanMessage(content=message)]},
            checkpointer=checkpointer,
            after_turn=hooks.append,
        )
    )

    result = await service.execute(
        make_session(mock_settings, graph), TurnRequest("thread", "hello")
    )

    assert result.answer == "completed answer"
    assert result.error is None
    assert checkpointer.calls == [("snapshot", "thread")]
    assert hooks == ["thread"]


class EventExecutor:
    def __init__(self, events: list[object]) -> None:
        self.events = events

    def stream(self, *_args: object, **_kwargs: object):
        yield from self.events


async def _error_event_followed_by_done_rolls_back_exact_snapshot(mock_settings) -> None:
    checkpointer = RecordingCheckpointer()
    hooks: list[str] = []
    service = TurnExecutionService(
        TurnExecutionDependencies(
            build_inputs=lambda _session, _message: {"messages": []},
            checkpointer=checkpointer,
            after_turn=hooks.append,
            executor_factory=lambda _graph, _metrics: EventExecutor(
                [ErrorEvent("provider secret"), DoneEvent(answer="must not complete")]
            ),  # type: ignore[arg-type]
        )
    )

    events = [
        event
        async for event in service.stream(
            make_session(mock_settings, object()), TurnRequest("thread", "hello")
        )
    ]

    assert [type(event) for event in events] == [ErrorEvent, DoneEvent]
    assert "provider secret" not in events[0].message  # type: ignore[union-attr]
    assert checkpointer.calls == [
        ("snapshot", "thread"),
        ("restore", ("thread", b"exact-snapshot")),
    ]
    assert hooks == []


class BlockingEvents:
    def __init__(self, release: threading.Event, closed: threading.Event) -> None:
        self.release = release
        self.closed = closed

    def __iter__(self):
        try:
            yield NodeStartEvent("agent")
            self.release.wait(timeout=2)
            yield DoneEvent(answer="too late")
        finally:
            self.closed.set()

    def close(self) -> None:
        self.closed.set()


async def _stream_cancellation_closes_generator_waits_and_rolls_back(mock_settings) -> None:
    checkpointer = RecordingCheckpointer()
    release = threading.Event()
    closed = threading.Event()
    blocking = BlockingEvents(release, closed)
    service = TurnExecutionService(
        TurnExecutionDependencies(
            build_inputs=lambda _session, _message: {"messages": []},
            checkpointer=checkpointer,
            executor_factory=lambda _graph, _metrics: EventExecutor([]),
        )
    )
    service._dependencies = TurnExecutionDependencies(  # noqa: SLF001 - focused seam test
        build_inputs=lambda _session, _message: {"messages": []},
        checkpointer=checkpointer,
        executor_factory=lambda _graph, _metrics: SimpleNamespace(
            stream=lambda *_args, **_kwargs: iter(blocking)
        ),
    )
    stream = service.stream(make_session(mock_settings, object()), TurnRequest("thread", "hello"))
    assert isinstance(await anext(stream), NodeStartEvent)

    close_task = asyncio.create_task(stream.aclose())
    await asyncio.sleep(0)
    release.set()
    await close_task

    assert closed.is_set()
    assert checkpointer.calls == [
        ("snapshot", "thread"),
        ("restore", ("thread", b"exact-snapshot")),
    ]


def test_direct_turn_service_completion_snapshots_and_runs_hook(mock_settings) -> None:
    asyncio.run(_direct_turn_service_completion_snapshots_and_runs_hook(mock_settings))


def test_error_event_followed_by_done_rolls_back_exact_snapshot(mock_settings) -> None:
    asyncio.run(_error_event_followed_by_done_rolls_back_exact_snapshot(mock_settings))


def test_stream_cancellation_closes_generator_waits_and_rolls_back(mock_settings) -> None:
    asyncio.run(_stream_cancellation_closes_generator_waits_and_rolls_back(mock_settings))


def test_stream_preserves_event_order_and_artifacts(mock_settings) -> None:
    from src.graph.events import ArtifactEvent

    artifact = {"id": "file-1", "type": "file"}
    expected = [
        NodeStartEvent("tools"),
        ArtifactEvent(artifacts=[artifact], node="tools"),
        DoneEvent(answer="completed", artifacts=[artifact]),
    ]
    hooks: list[str] = []
    service = TurnExecutionService(
        TurnExecutionDependencies(
            build_inputs=lambda _session, _message: {"messages": []},
            after_turn=hooks.append,
            executor_factory=lambda _graph, _metrics: EventExecutor(expected),  # type: ignore[arg-type]
        )
    )

    async def collect() -> list[object]:
        return [
            event
            async for event in service.stream(
                make_session(mock_settings, object()),
                TurnRequest("thread", "hello"),
            )
        ]

    assert asyncio.run(collect()) == expected
    assert hooks == ["thread"]


def test_http_chat_adapters_preserve_legacy_contracts(monkeypatch, isolated_settings) -> None:
    from fastapi.testclient import TestClient

    from src.chat import api as chat_api

    settings = isolated_settings(
        source_urls=["https://default.test"],
        web_search_enabled=False,
    )

    class HttpGraph:
        def __init__(self) -> None:
            self.messages: list[object] = []

        def get_state(self, _config: object) -> object:
            return SimpleNamespace(values={"messages": list(self.messages)})

        def invoke(self, inputs: dict[str, object], _config: object) -> dict[str, object]:
            self.messages.extend(inputs["messages"])  # type: ignore[arg-type]
            self.messages.append(AIMessage(content="legacy answer"))
            return {"messages": list(self.messages)}

        def stream(self, inputs: dict[str, object], config: object = None, **_kwargs: object):
            del config
            self.messages.extend(inputs["messages"])  # type: ignore[arg-type]
            answer = AIMessage(content="stream answer")
            self.messages.append(answer)
            yield {"generate": {"messages": [answer]}}

    monkeypatch.setattr(chat_api, "load_settings", lambda: settings)
    monkeypatch.setattr(
        chat_api,
        "build_chat_graph",
        lambda *_args, **_kwargs: HttpGraph(),
    )

    with TestClient(chat_api.create_app()) as client:
        started = client.post("/chat", json={"web_search": False})
        thread_id = started.json()["thread_id"]
        completed = client.post(
            f"/chat/{thread_id}/message",
            json={"message": "hello"},
        )
        history = client.get(f"/chat/{thread_id}/history")
        streamed = client.post(
            f"/chat/{thread_id}/message/stream?tokens=false",
            json={"message": "again"},
        )
        deleted = client.delete(f"/chat/{thread_id}")

    assert started.json() == {
        "thread_id": thread_id,
        "source_urls": ["https://default.test"],
        "source_mode": "defaults",
        "source_note": None,
    }
    assert completed.json() == {
        "thread_id": thread_id,
        "answer": "legacy answer",
        "error": None,
        "artifacts": [],
    }
    assert history.json() == {
        "thread_id": thread_id,
        "turns": [
            {"role": "user", "content": "hello", "artifacts": []},
            {"role": "assistant", "content": "legacy answer", "artifacts": []},
        ],
        "source_urls": ["https://default.test"],
        "source_mode": "defaults",
    }
    assert streamed.status_code == 200
    assert streamed.text.index("event: node_start") < streamed.text.index("event: node_end")
    assert streamed.text.index("event: node_end") < streamed.text.index("event: done")
    assert "event: token" not in streamed.text
    assert deleted.json() == {"status": "deleted", "thread_id": thread_id}


class FailingInvokeGraph:
    def get_state(self, _config: object) -> object:
        return SimpleNamespace(values={"messages": []})

    def invoke(self, _inputs: object, _config: object) -> object:
        raise RuntimeError("provider-secret-detail")


def test_non_streaming_failure_rolls_back_and_sanitizes(mock_settings) -> None:
    checkpointer = RecordingCheckpointer()
    hooks: list[str] = []
    synchronized: list[object] = []
    service = TurnExecutionService(
        TurnExecutionDependencies(
            build_inputs=lambda _session, _message: {"messages": []},
            checkpointer=checkpointer,
            after_turn=hooks.append,
            sync_sources=lambda _session, values: synchronized.append(values),
        )
    )

    async def execute():
        return await service.execute(
            make_session(mock_settings, FailingInvokeGraph()),
            TurnRequest("thread", "hello", request_id="correlation-id"),
        )

    result = asyncio.run(execute())

    assert result.answer == ""
    assert "provider-secret-detail" not in str(result.error)
    assert result.error == "Internal server error. Request ID: correlation-id"
    assert checkpointer.calls == [
        ("snapshot", "thread"),
        ("restore", ("thread", b"exact-snapshot")),
    ]
    assert synchronized == []
    assert hooks == []


def test_success_synchronizes_sources_before_after_turn(mock_settings) -> None:
    graph = InvokeGraph()
    calls: list[tuple[str, object]] = []
    service = TurnExecutionService(
        TurnExecutionDependencies(
            build_inputs=lambda _session, message: {"messages": [HumanMessage(content=message)]},
            sync_sources=lambda _session, values: calls.append(("sync", values)),
            after_turn=lambda thread_id: calls.append(("hook", thread_id)),
        )
    )

    async def execute():
        return await service.execute(
            make_session(mock_settings, graph),
            TurnRequest("thread", "hello"),
        )

    result = asyncio.run(execute())

    assert result.answer == "completed answer"
    assert [name for name, _value in calls] == ["sync", "hook"]
    assert calls[0][1] == {"messages": graph.messages}
    assert calls[1] == ("hook", "thread")


def test_direct_session_lifecycle_start(mock_settings) -> None:
    settings = replace(mock_settings, web_search_enabled=False)
    registry = ChatSessionRegistry(cleanup=lambda _session: None)
    graph = InvokeGraph()
    builds: list[tuple[object, bool, object, str]] = []

    def build_graph(session_settings, rebuild, checkpointer, thread_id):
        builds.append((session_settings, rebuild, checkpointer, thread_id))
        return graph

    service = SessionLifecycleService(
        settings=settings,
        sessions=registry,
        graph_factory_lock=asyncio.Lock(),
        checkpointer="checkpointer",
        dependencies=SessionLifecycleDependencies(
            discover_urls=lambda _query, _settings: [],
            build_graph=build_graph,
            build_lightweight_graph=lambda *_args: graph,
            settings_for_session=lambda base, _urls, _thread_id, _isolated: base,
            condense_question=lambda _messages, question, _settings: question,
        ),
    )

    result = asyncio.run(service.start(StartSessionRequest()))

    assert result.source_urls == ("https://example.com/a",)
    assert result.source_mode == "defaults"
    assert service.require_session(result.thread_id).graph is graph
    assert builds == [(settings, False, "checkpointer", result.thread_id)]


def test_history_uses_existing_global_api_key_guard(monkeypatch, isolated_settings) -> None:
    from fastapi.testclient import TestClient

    from src.chat import api as chat_api

    settings = isolated_settings(
        api_key="history-key",
        source_urls=["https://default.test"],
        web_search_enabled=False,
    )
    graph = InvokeGraph()
    monkeypatch.setattr(chat_api, "load_settings", lambda: settings)
    monkeypatch.setattr(chat_api, "build_chat_graph", lambda *_args, **_kwargs: graph)
    headers = {"Authorization": "Bearer history-key"}

    with TestClient(chat_api.create_app()) as client:
        started = client.post(
            "/chat",
            json={"web_search": False},
            headers=headers,
        )
        thread_id = started.json()["thread_id"]
        unauthorized = client.get(f"/chat/{thread_id}/history")
        authorized = client.get(
            f"/chat/{thread_id}/history",
            headers=headers,
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["thread_id"] == thread_id
