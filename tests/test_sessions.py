from __future__ import annotations

import sqlite3

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.chat.sessions import ChatSession as CompatChatSession
from src.chat.sessions import ChatSessionRegistry as CompatChatSessionRegistry
from src.sessions import (
    ChatSession,
    ChatSessionRegistry,
    InMemoryStorage,
    SQLiteMemorySaver,
    SQLiteStorage,
    SessionMetadata,
    _settings_for_session,
    settings_for_session,
)


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
