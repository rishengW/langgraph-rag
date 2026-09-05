"""Integration tests for extraction triggers and chat entry points."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.backend.graph.events import DoneEvent
from src.backend.memory.scheduler import ExtractionScheduler
from src.backend.memory.watermark import WATERMARK_KEY, InMemoryWatermarkStore
from src.backend.sessions import ChatSessionRegistry, InMemoryStorage, SessionMetadata
from src.config import Settings
from src.frontend.chat import api as chat_api
from src.frontend.chat import main as chat_main
from src.frontend.chat.memory_hooks import (
    ExtractionRuntime,
    SessionWatermarkStore,
    after_turn,
    build_extraction_runtime,
    on_session_start,
)


class RecordingScheduler:
    def __init__(self):
        self.submissions = []
        self.shutdown_calls = 0

    def submit(self, trigger, thread_id):
        self.submissions.append((trigger, thread_id))
        return True

    def shutdown(self):
        self.shutdown_calls += 1


class ExtractorStub:
    def __init__(self, turns=None, decisions=None):
        self.turns = dict(turns or {})
        self.decisions = dict(decisions or {})

    def turn_count(self, thread_id):
        return self.turns.get(thread_id, 0)

    def should_extract(self, thread_id):
        return self.decisions.get(thread_id, False)


class RegistryStub:
    def __init__(self, ids):
        self.ids = list(ids)

    def list_ids(self):
        return list(self.ids)


class StorageStub:
    def __init__(self, metadata):
        self.metadata = {item.thread_id: item for item in metadata}

    def list_metadata(self):
        return list(self.metadata.values())

    def load(self, thread_id):
        return self.metadata.get(thread_id)

    def save(self, metadata):
        self.metadata[metadata.thread_id] = metadata


def extraction_settings(tmp_path, **overrides):
    values = {
        "dashscope_api_key": "test",
        "chroma_dir": tmp_path / "chroma",
        "memory_store_path": str(tmp_path / "memory.json"),
        "memory_enabled": True,
        "memory_extraction_enabled": True,
        "web_search_enabled": False,
    }
    values.update(overrides)
    return Settings(**values)


def metadata(thread_id, accessed, watermark=0):
    return SessionMetadata(
        thread_id=thread_id,
        created_at=accessed,
        last_accessed_at=accessed,
        config={WATERMARK_KEY: watermark},
    )


def runtime(tmp_path, metadata_rows, turns, *, registry_ids=None, **overrides):
    storage = StorageStub(metadata_rows)
    registry = RegistryStub(
        registry_ids if registry_ids is not None else [row.thread_id for row in metadata_rows]
    )
    scheduler = RecordingScheduler()
    watermarks = InMemoryWatermarkStore()
    value = ExtractionRuntime(
        settings=extraction_settings(tmp_path, **overrides),
        extractor=ExtractorStub(turns=turns),
        scheduler=scheduler,
        registry=registry,
        storage=storage,
        watermarks=watermarks,
    )
    return value, scheduler, watermarks


def test_disabled_runtime_is_inert(tmp_path):
    class CountingCheckpointer:
        calls = 0

        def get_tuple(self, _config):
            self.calls += 1
            return None

    checkpointer = CountingCheckpointer()
    for overrides in (
        {"memory_enabled": False},
        {"memory_extraction_enabled": False},
    ):
        assert (
            build_extraction_runtime(
                extraction_settings(tmp_path, **overrides),
                checkpointer=checkpointer,
            )
            is None
        )
    assert checkpointer.calls == 0
    after_turn(None, thread_id="thread")
    on_session_start(None, new_thread_id="new")


def test_session_watermark_prefers_persisted_and_writes_both(tmp_path):
    settings = extraction_settings(tmp_path)
    storage = InMemoryStorage()
    registry = ChatSessionRegistry(storage=storage)
    session = registry.create(object(), settings, [], "defaults", thread_id="thread")
    session.extraction_watermark = 2
    persisted = storage.load("thread")
    assert persisted is not None
    storage.save(replace(persisted, config={WATERMARK_KEY: 7}))

    watermarks = SessionWatermarkStore(registry, storage)
    assert watermarks.get("thread") == 7
    watermarks.set("thread", 9)

    assert session.extraction_watermark == 9
    assert storage.load("thread").config[WATERMARK_KEY] == 9


def test_session_start_selects_most_recent_with_greatest_id_tiebreak(tmp_path):
    now = time.time()
    rows = [metadata("a", now - 10), metadata("b", now - 2), metadata("c", now - 2)]
    value, scheduler, _watermarks = runtime(tmp_path, rows, {"a": 1, "b": 1, "c": 1})

    on_session_start(value, new_thread_id="new")

    assert scheduler.submissions == [("session_start", "c")]


def test_session_start_filters_deleted_zero_turn_and_current(tmp_path):
    now = time.time()
    rows = [
        metadata("deleted", now - 1),
        metadata("zero", now - 2),
        metadata("new", now - 3),
        metadata("eligible", now - 4),
    ]
    value, scheduler, _watermarks = runtime(
        tmp_path,
        rows,
        {"deleted": 3, "zero": 0, "new": 2, "eligible": 1},
        registry_ids=["zero", "new", "eligible"],
    )

    on_session_start(value, new_thread_id="new")
    assert scheduler.submissions == [("session_start", "eligible")]


def test_over_age_winner_is_marked_done_without_replacement(tmp_path, monkeypatch):
    now = 1_000_000.0
    rows = [metadata("old-winner", now - 7201), metadata("fresh", now - 1)]
    value, scheduler, watermarks = runtime(
        tmp_path,
        rows,
        {"old-winner": 3, "fresh": 2},
        memory_extraction_max_session_age_hours=1,
    )
    # The winner is the most recent eligible item, so make that item over-age
    # while the lower-ranked item is even older.
    value.storage.metadata["fresh"] = metadata("fresh", now - 7201)
    value.storage.metadata["old-winner"] = metadata("old-winner", now - 7202)
    monkeypatch.setattr("src.frontend.chat.memory_hooks.time.time", lambda: now)

    on_session_start(value, new_thread_id="new")

    assert scheduler.submissions == []
    assert watermarks.get("fresh") == 2


def test_adoption_schedules_only_one_of_one_hundred_sessions(tmp_path):
    now = time.time()
    rows = [metadata(f"thread-{index:03}", now - 100 + index) for index in range(100)]
    turns = {row.thread_id: 1 for row in rows}
    value, scheduler, _watermarks = runtime(tmp_path, rows, turns)

    on_session_start(value, new_thread_id="new")

    assert scheduler.submissions == [("session_start", "thread-099")]


def test_after_turn_submits_only_when_extractor_says_due(tmp_path):
    scheduler = RecordingScheduler()
    value = ExtractionRuntime(
        settings=extraction_settings(tmp_path),
        extractor=ExtractorStub(decisions={"due": True, "early": False}),
        scheduler=scheduler,
        registry=None,
        storage=None,
    )

    after_turn(value, thread_id="early")
    after_turn(value, thread_id="due")
    assert scheduler.submissions == [("round_complete", "due")]


def test_session_start_does_nothing_when_no_session_is_eligible(tmp_path):
    rows = [metadata("done", time.time(), watermark=3)]
    value, scheduler, _watermarks = runtime(tmp_path, rows, {"done": 3})

    on_session_start(value, new_thread_id="new")
    assert scheduler.submissions == []


class FakeGraph:
    def __init__(self):
        self.messages = []

    def get_state(self, _config):
        return SimpleNamespace(values={"messages": list(self.messages)})

    def invoke(self, inputs, _config=None):
        self.messages.extend(inputs["messages"])
        self.messages.append(AIMessage(content="answer"))
        return {"messages": list(self.messages)}

    def stream(self, inputs, config=None, *, stream_mode=None):
        del config
        self.messages.extend(inputs["messages"])
        self.messages.append(AIMessage(content="stream answer"))
        update = {"generate": {"messages": [AIMessage(content="stream answer")]}}
        if isinstance(stream_mode, list):
            yield "updates", update
        else:
            yield update


def test_http_entry_points_call_hooks_and_shutdown(monkeypatch, isolated_settings):
    settings = isolated_settings(web_search_enabled=False)
    calls = []
    scheduler = RecordingScheduler()
    fake_runtime = SimpleNamespace(scheduler=scheduler)
    monkeypatch.setattr(chat_api, "load_settings", lambda: settings)
    monkeypatch.setattr(chat_api, "build_chat_graph", lambda *_args, **_kwargs: FakeGraph())
    monkeypatch.setattr(
        chat_api, "build_extraction_runtime", lambda *_args, **_kwargs: fake_runtime
    )
    monkeypatch.setattr(
        chat_api,
        "on_session_start",
        lambda _runtime, *, new_thread_id: calls.append(("start", new_thread_id)),
    )
    monkeypatch.setattr(
        chat_api,
        "after_turn",
        lambda _runtime, *, thread_id: calls.append(("turn", thread_id)),
    )

    app = chat_api.create_app()
    with TestClient(app) as client:
        start = client.post("/chat", json={"web_search": False})
        thread_id = start.json()["thread_id"]
        reply = client.post(f"/chat/{thread_id}/message", json={"message": "hello"})
        stream = client.post(f"/chat/{thread_id}/message/stream", json={"message": "again"})
        assert reply.json()["answer"] == "answer"
        assert "event: done" in stream.text

    assert calls == [
        ("start", thread_id),
        ("turn", thread_id),
        ("turn", thread_id),
    ]
    assert scheduler.shutdown_calls == 1


def test_hook_failures_do_not_change_http_responses(monkeypatch, isolated_settings):
    settings = isolated_settings(web_search_enabled=False)
    monkeypatch.setattr(chat_api, "load_settings", lambda: settings)
    monkeypatch.setattr(chat_api, "build_chat_graph", lambda *_args, **_kwargs: FakeGraph())
    monkeypatch.setattr(chat_api, "on_session_start", lambda *_args, **_kwargs: 1 / 0)
    monkeypatch.setattr(chat_api, "after_turn", lambda *_args, **_kwargs: 1 / 0)

    app = chat_api.create_app()
    with TestClient(app) as client:
        start = client.post("/chat", json={"web_search": False})
        assert start.status_code == 200
        thread_id = start.json()["thread_id"]
        reply = client.post(f"/chat/{thread_id}/message", json={"message": "hello"})
        assert reply.status_code == 200
        assert reply.json()["answer"] == "answer"


def test_cli_calls_after_turn_and_shuts_down(monkeypatch, tmp_path):
    settings = extraction_settings(tmp_path, memory_extraction_enabled=False)
    scheduler = RecordingScheduler()
    runtime_value = SimpleNamespace(scheduler=scheduler)
    calls = []
    prompts = iter(["hello", "exit"])

    monkeypatch.setattr(chat_main, "load_settings", lambda **_kwargs: settings)
    monkeypatch.setattr("src.frontend.chat.graph._build_memory_saver", lambda: object())
    monkeypatch.setattr("src.frontend.chat.graph.build_chat_graph", lambda *_a, **_k: object())
    monkeypatch.setattr(chat_main, "build_extraction_runtime", lambda *_a, **_k: runtime_value)
    monkeypatch.setattr(
        chat_main,
        "after_turn",
        lambda _runtime, *, thread_id: calls.append(thread_id),
    )
    monkeypatch.setattr(
        chat_main,
        "GraphExecutor",
        lambda _graph: SimpleNamespace(
            stream=lambda *_args, **_kwargs: iter([DoneEvent(answer="answer")])
        ),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(prompts))
    args = SimpleNamespace(urls="", config=None, seed_question="")

    chat_main._repl(args)

    assert calls == ["cli"]
    assert scheduler.shutdown_calls == 1


def test_failed_http_turns_do_not_call_after_turn(monkeypatch, isolated_settings):
    settings = isolated_settings(web_search_enabled=False)
    calls = []

    class FailingGraph(FakeGraph):
        def invoke(self, inputs, _config=None):
            del inputs, _config
            raise RuntimeError("turn failed")

        def stream(self, inputs, config=None, *, stream_mode=None):
            del inputs, config, stream_mode
            raise RuntimeError("stream failed")
            yield

    monkeypatch.setattr(chat_api, "load_settings", lambda: settings)
    monkeypatch.setattr(
        chat_api,
        "build_chat_graph",
        lambda *_args, **_kwargs: FailingGraph(),
    )
    monkeypatch.setattr(
        chat_api,
        "after_turn",
        lambda _runtime, *, thread_id: calls.append(thread_id),
    )

    app = chat_api.create_app()
    with TestClient(app) as client:
        start = client.post("/chat", json={"web_search": False})
        thread_id = start.json()["thread_id"]
        reply = client.post(
            f"/chat/{thread_id}/message",
            json={"message": "fail"},
        )
        stream = client.post(
            f"/chat/{thread_id}/message/stream",
            json={"message": "fail"},
        )

    assert reply.status_code == 200
    assert "turn failed" not in reply.json()["error"]
    assert "Internal server error. Request ID:" in reply.json()["error"]
    assert "event: error" in stream.text
    assert calls == []


def test_http_responses_do_not_wait_for_extraction_worker(
    monkeypatch,
    isolated_settings,
):
    settings = isolated_settings(web_search_enabled=False)
    started = threading.Event()
    release = threading.Event()

    def runner(_trigger, _thread_id):
        started.set()
        release.wait()

    scheduler = ExtractionScheduler(max_concurrency=1, runner=runner)
    fake_runtime = SimpleNamespace(scheduler=scheduler)
    monkeypatch.setattr(chat_api, "load_settings", lambda: settings)
    monkeypatch.setattr(chat_api, "build_chat_graph", lambda *_a, **_k: FakeGraph())
    monkeypatch.setattr(
        chat_api,
        "build_extraction_runtime",
        lambda *_args, **_kwargs: fake_runtime,
    )
    monkeypatch.setattr(
        chat_api,
        "on_session_start",
        lambda runtime, *, new_thread_id: runtime.scheduler.submit(
            "session_start", "previous-thread"
        ),
    )
    monkeypatch.setattr(
        chat_api,
        "after_turn",
        lambda runtime, *, thread_id: runtime.scheduler.submit(
            "round_complete", thread_id
        ),
    )

    try:
        app = chat_api.create_app()
        with TestClient(app) as client:
            start = client.post("/chat", json={"web_search": False})
            assert start.status_code == 200
            assert started.wait(timeout=2)
            thread_id = start.json()["thread_id"]

            reply = client.post(
                f"/chat/{thread_id}/message",
                json={"message": "hello"},
            )
            assert reply.status_code == 200
            assert reply.json()["answer"] == "answer"
    finally:
        release.set()

    assert scheduler.wait_idle(timeout=2)


def test_cli_interrupt_shuts_down_runtime(monkeypatch, tmp_path):
    settings = extraction_settings(tmp_path, memory_extraction_enabled=False)
    scheduler = RecordingScheduler()
    runtime_value = SimpleNamespace(scheduler=scheduler)

    monkeypatch.setattr(chat_main, "load_settings", lambda **_kwargs: settings)
    monkeypatch.setattr("src.frontend.chat.graph._build_memory_saver", lambda: object())
    monkeypatch.setattr("src.frontend.chat.graph.build_chat_graph", lambda *_a, **_k: object())
    monkeypatch.setattr(
        chat_main,
        "build_extraction_runtime",
        lambda *_args, **_kwargs: runtime_value,
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    args = SimpleNamespace(urls="", config=None, seed_question="")

    chat_main._repl(args)

    assert scheduler.shutdown_calls == 1
