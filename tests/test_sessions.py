from __future__ import annotations

import sqlite3

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.backend.sessions import (
    ChatSession,
    ChatSessionRegistry,
    InMemoryStorage,
    SessionMetadata,
    SQLiteMemorySaver,
    SQLiteStorage,
    _settings_for_session,
    settings_for_session,
)
from src.frontend.chat.sessions import ChatSession as CompatChatSession
from src.frontend.chat.sessions import ChatSessionRegistry as CompatChatSessionRegistry


class CounterState(TypedDict):
    value: int


def test_chat_sessions_compatibility_reexports_new_types():
    assert CompatChatSession is ChatSession
    assert CompatChatSessionRegistry is ChatSessionRegistry
    assert _settings_for_session is settings_for_session


def test_registry_create_get_list_delete_with_cleanup(mock_settings):
    cleaned: list[str] = []

    def cleanup(session: ChatSession) -> None:
        cleaned.append(session.thread_id)

    registry = ChatSessionRegistry(cleanup=cleanup)
    session = registry.create(
        graph=object(),
        settings=mock_settings,
        source_urls=["https://example.com/session"],
        source_mode="explicit",
        thread_id="thread-1",
    )

    assert session.thread_id == "thread-1"
    assert session.source_urls == ["https://example.com/session"]
    assert session.source_mode == "explicit"
    assert session.isolated_chroma is True
    assert registry.get("thread-1") is session
    assert registry.list_ids() == ["thread-1"]
    assert len(registry) == 1

    assert registry.delete("thread-1") is True
    assert cleaned == ["thread-1"]
    assert registry.get("thread-1") is None
    assert registry.list_ids() == []
    assert len(registry) == 0
    assert registry.delete("thread-1") is False


def test_delete_removes_isolated_chroma_directory(mock_settings):
    session_settings = settings_for_session(
        mock_settings,
        ["https://example.com/isolated"],
        "isolated-thread",
        isolated=True,
    )
    session_settings.chroma_dir.mkdir(parents=True)
    (session_settings.chroma_dir / "index.sqlite3").write_text("placeholder", encoding="utf-8")

    registry = ChatSessionRegistry(ttl_seconds=None)
    registry.create(
        graph=object(),
        settings=session_settings,
        source_urls=session_settings.source_urls,
        source_mode="explicit",
        thread_id="isolated-thread",
    )

    assert session_settings.chroma_dir.exists()
    assert registry.delete("isolated-thread") is True
    assert not session_settings.chroma_dir.exists()


def test_cleanup_expired_removes_only_stale_sessions(mock_settings):
    now = 100.0

    def clock() -> float:
        return now

    registry = ChatSessionRegistry(ttl_seconds=10, time_func=clock)

    old_settings = settings_for_session(mock_settings, ["https://example.com/old"], "old", isolated=True)
    old_settings.chroma_dir.mkdir(parents=True)
    registry.create(
        graph=object(),
        settings=old_settings,
        source_urls=old_settings.source_urls,
        source_mode="explicit",
        thread_id="old",
    )

    now = 105.0
    fresh_settings = settings_for_session(mock_settings, ["https://example.com/fresh"], "fresh", isolated=True)
    fresh_settings.chroma_dir.mkdir(parents=True)
    registry.create(
        graph=object(),
        settings=fresh_settings,
        source_urls=fresh_settings.source_urls,
        source_mode="explicit",
        thread_id="fresh",
    )

    now = 111.0
    assert registry.cleanup_expired() == 1
    assert registry.get("old") is None
    assert registry.get("fresh", touch=False) is not None
    assert not old_settings.chroma_dir.exists()
    assert fresh_settings.chroma_dir.exists()


def test_get_touches_session_for_ttl(mock_settings):
    now = 100.0

    def clock() -> float:
        return now

    registry = ChatSessionRegistry(ttl_seconds=10, cleanup=lambda session: None, time_func=clock)
    registry.create(
        graph=object(),
        settings=mock_settings,
        source_urls=mock_settings.source_urls,
        source_mode="defaults",
        thread_id="active",
    )

    now = 105.0
    assert registry.get("active") is not None

    now = 111.0
    assert registry.cleanup_expired() == 0
    assert registry.get("active", touch=False) is not None


def test_settings_for_session_uses_shared_chroma_for_defaults(mock_settings):
    urls = ["https://example.com/default"]
    session_settings = settings_for_session(mock_settings, urls, "thread-default", isolated=False)
    urls.append("https://example.com/mutated")

    assert session_settings.source_urls == ["https://example.com/default"]
    assert session_settings.chroma_dir == mock_settings.chroma_dir
    assert session_settings.collection_name == mock_settings.collection_name


def test_settings_for_session_isolates_chroma_for_custom_sources(mock_settings):
    session_settings = settings_for_session(
        mock_settings,
        ["https://example.com/custom"],
        "thread-custom",
        isolated=True,
    )

    assert session_settings.source_urls == ["https://example.com/custom"]
    assert session_settings.chroma_dir == mock_settings.chroma_dir / "chat" / "thread-custom"
    assert session_settings.collection_name == f"{mock_settings.collection_name}-chat-thread-custom"


def test_in_memory_storage_copies_metadata():
    storage = InMemoryStorage()
    metadata = SessionMetadata(
        thread_id="thread-memory",
        source_urls=["https://example.com/a"],
        source_mode="explicit",
        created_at=1.0,
        last_accessed_at=2.0,
        chroma_dir=".chroma/chat/thread-memory",
        isolated_chroma=True,
        config={"collection_name": "rag-chat"},
    )

    storage.save(metadata)
    loaded = storage.load("thread-memory")
    assert loaded == metadata

    assert loaded is not None
    loaded.source_urls.append("https://example.com/mutated")
    loaded.config["collection_name"] = "mutated"

    loaded_again = storage.load("thread-memory")
    assert loaded_again == metadata
    assert storage.list_ids() == ["thread-memory"]
    assert storage.list_metadata() == [metadata]


def test_sqlite_storage_round_trips_metadata(tmp_path):
    storage = SQLiteStorage(tmp_path / "sessions.sqlite3")
    first = SessionMetadata(
        thread_id="thread-sqlite-a",
        source_urls=["https://example.com/a", "https://example.com/b"],
        source_mode="web_search",
        created_at=1.0,
        last_accessed_at=3.0,
        chroma_dir=".chroma/chat/thread-sqlite-a",
        isolated_chroma=True,
        config={"collection_name": "rag-chat-a", "top_k": 3},
    )
    second = SessionMetadata(
        thread_id="thread-sqlite-b",
        source_urls=[],
        source_mode="defaults",
        created_at=2.0,
        last_accessed_at=2.0,
        chroma_dir=".chroma",
        isolated_chroma=False,
    )

    storage.save(second)
    storage.save(first)

    assert storage.load("thread-sqlite-a") == first
    assert storage.load("missing") is None
    assert storage.list_ids() == ["thread-sqlite-a", "thread-sqlite-b"]
    assert storage.list_metadata() == [first, second]


def test_sqlite_storage_updates_and_deletes_metadata(tmp_path):
    storage = SQLiteStorage(tmp_path / "sessions.sqlite3")
    original = SessionMetadata(thread_id="thread-update", created_at=1.0, last_accessed_at=1.0)
    updated = SessionMetadata(
        thread_id="thread-update",
        source_urls=["https://example.com/updated"],
        source_mode="explicit",
        created_at=1.0,
        last_accessed_at=5.0,
        chroma_dir=".chroma/chat/thread-update",
        isolated_chroma=True,
    )

    storage.save(original)
    storage.save(updated)

    assert storage.load("thread-update") == updated
    assert storage.delete("thread-update") is True
    assert storage.delete("thread-update") is False
    assert storage.load("thread-update") is None


def test_registry_optionally_persists_session_metadata(mock_settings):
    now = 10.0

    def clock() -> float:
        return now

    storage = InMemoryStorage()
    registry = ChatSessionRegistry(
        cleanup=lambda session: None,
        storage=storage,
        time_func=clock,
    )

    registry.create(
        graph=object(),
        settings=mock_settings,
        source_urls=["https://example.com/session"],
        source_mode="explicit",
        thread_id="stored-thread",
    )
    saved = storage.load("stored-thread")
    assert saved is not None
    assert saved.source_urls == ["https://example.com/session"]
    assert saved.last_accessed_at == 10.0

    now = 12.0
    assert registry.get("stored-thread") is not None
    touched = storage.load("stored-thread")
    assert touched is not None
    assert touched.last_accessed_at == 12.0

    assert registry.delete("stored-thread") is True
    assert storage.load("stored-thread") is None


def test_registry_update_sources_replaces_graph_and_persists_metadata(mock_settings):
    storage = InMemoryStorage()
    registry = ChatSessionRegistry(
        cleanup=lambda session: None,
        storage=storage,
    )
    session = registry.create(
        graph="old-graph",
        settings=mock_settings,
        source_urls=["https://example.com/old"],
        source_mode="defaults",
        thread_id="refresh-thread",
    )
    refreshed_settings = settings_for_session(
        mock_settings,
        ["https://example.com/fresh"],
        "refresh-thread",
        isolated=True,
    )

    refreshed = registry.update_sources(
        "refresh-thread",
        graph="new-graph",
        settings=refreshed_settings,
        source_urls=refreshed_settings.source_urls,
        source_mode="web_search",
        isolated_chroma=True,
    )

    assert refreshed is session
    assert session.graph == "new-graph"
    assert session.settings is refreshed_settings
    assert session.source_urls == ["https://example.com/fresh"]
    assert session.source_mode == "web_search"
    assert session.isolated_chroma is True

    saved = storage.load("refresh-thread")
    assert saved is not None
    assert saved.source_urls == ["https://example.com/fresh"]
    assert saved.source_mode == "web_search"
    assert saved.isolated_chroma is True


def test_registry_restores_session_from_metadata(mock_settings):
    registry = ChatSessionRegistry(cleanup=lambda session: None)
    metadata = SessionMetadata(
        thread_id="restored-thread",
        source_urls=["https://example.com/restored"],
        source_mode="explicit",
        created_at=1.0,
        last_accessed_at=2.0,
        chroma_dir=str(mock_settings.chroma_dir),
        isolated_chroma=True,
    )

    session = registry.restore(
        graph="graph",
        settings=mock_settings,
        metadata=metadata,
    )

    assert session.thread_id == "restored-thread"
    assert session.graph == "graph"
    assert session.source_urls == ["https://example.com/restored"]
    assert session.created_at == 1.0
    assert session.last_accessed_at == 2.0
    assert session.isolated_chroma is True
    assert registry.get("restored-thread", touch=False) is session


def test_extraction_watermark_round_trips_and_survives_registry_touch(
    mock_settings,
    tmp_path,
):
    session = ChatSession(
        thread_id="watermark-thread",
        graph="graph",
        settings=mock_settings,
        source_urls=["https://example.com/watermark"],
        source_mode="explicit",
        created_at=1.0,
        last_accessed_at=2.0,
        isolated_chroma=True,
        extraction_watermark=17,
    )
    metadata = SessionMetadata.from_session(session)
    assert metadata.config["extraction_watermark"] == 17

    storage = SQLiteStorage(tmp_path / "watermark-sessions.sqlite3")
    storage.save(metadata)
    persisted = storage.load(session.thread_id)
    assert persisted == metadata

    registry = ChatSessionRegistry(
        cleanup=lambda restored: None,
        storage=storage,
        time_func=lambda: 10.0,
    )
    assert persisted is not None
    restored = registry.restore(graph="graph", settings=mock_settings, metadata=persisted)
    assert restored.extraction_watermark == 17

    assert registry.get(session.thread_id) is restored
    touched = storage.load(session.thread_id)
    assert touched is not None
    assert touched.config["extraction_watermark"] == 17


def test_legacy_session_metadata_defaults_watermark_and_schema_stays_stable(
    mock_settings,
    tmp_path,
):
    storage = SQLiteStorage(tmp_path / "legacy-sessions.sqlite3")
    legacy = SessionMetadata(
        thread_id="legacy-thread",
        source_urls=["https://example.com/legacy"],
        source_mode="explicit",
        created_at=1.0,
        last_accessed_at=2.0,
        chroma_dir=str(mock_settings.chroma_dir),
        isolated_chroma=True,
        config={"collection_name": mock_settings.collection_name},
    )
    storage.save(legacy)

    loaded = storage.load(legacy.thread_id)
    assert loaded == legacy
    assert loaded is not None
    assert "extraction_watermark" not in loaded.config

    registry = ChatSessionRegistry(
        cleanup=lambda session: None,
        storage=storage,
        time_func=lambda: 2.0,
    )
    restored = registry.restore(graph="graph", settings=mock_settings, metadata=loaded)
    assert restored.extraction_watermark == 0
    assert registry.get(legacy.thread_id) is restored
    resaved = storage.load(legacy.thread_id)
    assert resaved == SessionMetadata(
        thread_id=legacy.thread_id,
        source_urls=legacy.source_urls,
        source_mode=legacy.source_mode,
        created_at=legacy.created_at,
        last_accessed_at=legacy.last_accessed_at,
        chroma_dir=legacy.chroma_dir,
        isolated_chroma=legacy.isolated_chroma,
        config={
            "collection_name": mock_settings.collection_name,
            "extraction_watermark": 0,
        },
    )

    constructed_without_watermark = ChatSession(
        thread_id="old-constructor",
        graph="graph",
        settings=mock_settings,
    )
    assert constructed_without_watermark.extraction_watermark == 0
    assert SQLiteStorage.SCHEMA_VERSION == 2


def test_sqlite_storage_writes_schema_version(tmp_path):
    db_path = tmp_path / "sessions.sqlite3"
    SQLiteStorage(db_path)

    with sqlite3.connect(db_path) as connection:
        version = connection.execute(
            "SELECT version FROM schema_version WHERE id = 1",
        ).fetchone()[0]

    assert version == SQLiteStorage.SCHEMA_VERSION


def test_sqlite_memory_saver_restores_graph_state_after_reopen(tmp_path):
    builder = StateGraph(CounterState)
    builder.add_node("increment", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)

    db_path = tmp_path / "checkpoints.sqlite3"
    config = {"configurable": {"thread_id": "checkpoint-thread"}}

    first_saver = SQLiteMemorySaver(db_path)
    first_graph = builder.compile(checkpointer=first_saver)
    assert first_graph.invoke({"value": 1}, config) == {"value": 2}

    second_saver = SQLiteMemorySaver(db_path)
    second_graph = builder.compile(checkpointer=second_saver)

    assert second_graph.get_state(config).values == {"value": 2}


def test_sqlite_memory_saver_serializes_writes_under_lock(tmp_path, monkeypatch):
    # Structural regression guard for the original crash surface: when
    # two concurrent ``/chat`` requests hit the same session, the
    # inherited ``MemorySaver.put`` / ``put_writes`` mutations and the
    # ``_plain_writes`` snapshot in ``_persist_state`` must run under
    # the same critical section. Otherwise the dict comprehension can
    # race with a concurrent key insertion and raise
    # ``RuntimeError: dictionary changed size during iteration``.
    #
    # The race window in production is sub-millisecond and hard to hit
    # deterministically, so this test asserts the structural invariant
    # directly: while the inherited ``super().put`` /
    # ``super().put_writes`` is running, ``self._lock`` must be held.
    # If a future change reverts the lock scoping, this test fires
    # immediately without depending on timing.
    from langgraph.checkpoint.memory import MemorySaver

    db_path = tmp_path / "checkpoints.sqlite3"
    saver = SQLiteMemorySaver(db_path)

    held_during_super: list[bool] = []
    real_put = MemorySaver.put
    real_put_writes = MemorySaver.put_writes

    def _is_held(lock: object) -> bool:
        # ``threading.RLock`` exposes ``_is_owned()`` (current thread holds
        # the lock) on every supported Python version. ``Lock.locked()``
        # is only on the non-reentrant ``Lock`` and only added to
        # ``RLock`` in 3.13. ``_is_owned`` works on both.
        is_owned = getattr(lock, "_is_owned", None)
        if callable(is_owned):
            return bool(is_owned())
        return bool(getattr(lock, "locked", lambda: False)())

    def tracking_put(self, config, checkpoint, metadata, new_versions):
        held_during_super.append(_is_held(self._lock) if hasattr(self, "_lock") else False)
        return real_put(self, config, checkpoint, metadata, new_versions)

    def tracking_put_writes(self, config, writes, task_id, task_path=""):
        held_during_super.append(_is_held(self._lock) if hasattr(self, "_lock") else False)
        return real_put_writes(self, config, writes, task_id, task_path)

    monkeypatch.setattr(MemorySaver, "put", tracking_put)
    monkeypatch.setattr(MemorySaver, "put_writes", tracking_put_writes)

    checkpoint = {
        "v": 1,
        "id": "cp-1",
        "ts": "2026-01-01T00:00:00+00:00",
        "channel_values": {"value": 1},
        "channel_versions": {"value": 1},
        "versions_seen": {},
        "pending_sends": [],
    }
    full_config = {
        "configurable": {
            "thread_id": "lock-test",
            "checkpoint_ns": "ns-1",
            "checkpoint_id": "cp-1",
        }
    }

    saver.put(full_config, checkpoint, {}, {})
    saver.put_writes(full_config, [(("value",), 1)], task_id="t-1")

    assert held_during_super == [True, True], (
        "self._lock must be held during the inherited super().put and "
        "super().put_writes calls; otherwise concurrent writers can race "
        "with the _plain_writes snapshot and crash with "
        "'dictionary changed size during iteration'."
    )


def test_sqlite_memory_saver_concurrent_writers_do_not_corrupt_state(tmp_path):
    # Behavioral smoke test for the production crash: spin up several
    # threads that each call ``put`` and ``put_writes`` in a tight loop
    # against the same saver. The exact ``RuntimeError`` is timing-
    # dependent and hard to reproduce in a unit test, but this test
    # catches the broader class of "concurrent access corrupts the
    # saver" failures and ensures the fix is not silently regressed.
    import threading

    db_path = tmp_path / "checkpoints.sqlite3"
    saver = SQLiteMemorySaver(db_path)
    thread_id = "concurrent-thread"

    errors: list[BaseException] = []
    stop = threading.Event()

    def runner(index: int) -> None:
        try:
            iteration = 0
            while not stop.is_set():
                config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": f"ns-{index}",
                        "checkpoint_id": f"cp-{index}-{iteration}",
                    }
                }
                saver.put_writes(
                    config,
                    [(("value",), iteration)],
                    task_id=f"task-{index}-{iteration}",
                )
                iteration += 1
        except BaseException as exc:  # pragma: no cover - assertion below
            errors.append(exc)

    threads = [threading.Thread(target=runner, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()

    import time

    time.sleep(0.3)
    stop.set()
    for thread in threads:
        thread.join()

    assert errors == [], f"concurrent put_writes raised: {errors!r}"
