from __future__ import annotations

from src.chat.sessions import ChatSession as CompatChatSession
from src.chat.sessions import ChatSessionRegistry as CompatChatSessionRegistry
from src.sessions import (
    ChatSession,
    ChatSessionRegistry,
    _settings_for_session,
    settings_for_session,
)


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
