"""Cross-layer tests: graph wiring, prompt, chat turn assembly, history.

Every test points the store at ``tmp_path``, so none of them can touch the real
``memory/long_term_memory.json``.
"""

from __future__ import annotations

import types

import pytest

from src.chat.api import _graph_inputs_for_turn, _serialize_messages
from src.config.settings import Settings
from src.graph.builder import GraphProviders, _resolve_lightweight_tools, _resolve_tools
from src.llm.prompts import AGENT_SYSTEM_PROMPT
from src.memory.recall import MEMORY_NOTE_LABEL
from src.memory.store import MemoryStore, reset_store_cache

MEMORY_TOOL_NAMES = {"save_memory", "recall_memory", "forget_memory"}
TS = "2026-07-01T00:00:00+00:00"


@pytest.fixture(autouse=True)
def clean_store_cache():
    reset_store_cache()
    yield
    reset_store_cache()


def make_settings(tmp_path, **overrides) -> Settings:
    values = {
        "memory_enabled": True,
        "memory_auto_recall_enabled": True,
        "memory_store_path": str(tmp_path / "long_term_memory.json"),
        "web_search_enabled": False,
    }
    values.update(overrides)
    return Settings(dashscope_api_key="x", **values)


def seeded_store(tmp_path, *records: str) -> MemoryStore:
    store = MemoryStore(
        tmp_path / "long_term_memory.json",
        max_records=500,
        max_record_chars=1000,
        clock=lambda: TS,
    )
    for content in records:
        assert store.save(content=content).ok
    return store


def fake_session(thread_id: str = "thread-42"):
    """A session stub with only what the turn builder reads."""

    return types.SimpleNamespace(
        thread_id=thread_id,
        source_urls=[],
        source_mode="defaults",
        announced_uploads=set(),
    )


# ---- graph wiring ----------------------------------------------------------


@pytest.fixture
def stub_retriever(monkeypatch):
    """Replace the Chroma retriever so the heavy resolver needs no vector store."""

    from langchain_core.tools import StructuredTool

    def fake_build_retriever_tool(settings, rebuild=False):
        return StructuredTool.from_function(
            func=lambda query: "stub",
            name="retrieve_source_documents",
            description="stub retriever",
        )

    monkeypatch.setattr(
        "src.core.retriever.build_retriever_tool", fake_build_retriever_tool
    )


def heavy_tools(settings):
    return _resolve_tools(settings, GraphProviders(), False)


def lightweight_tools(settings):
    return _resolve_lightweight_tools(settings, GraphProviders())


@pytest.mark.parametrize("resolver", [heavy_tools, lightweight_tools])
def test_both_graphs_expose_the_memory_tools_when_enabled(
    tmp_path, stub_retriever, resolver
):
    settings = make_settings(tmp_path, web_search_enabled=True)

    names = {tool.name for tool in resolver(settings)}

    assert names >= MEMORY_TOOL_NAMES


@pytest.mark.parametrize("resolver", [heavy_tools, lightweight_tools])
def test_neither_graph_exposes_the_memory_tools_when_disabled(
    tmp_path, stub_retriever, resolver
):
    settings = make_settings(tmp_path, memory_enabled=False, web_search_enabled=True)

    names = {tool.name for tool in resolver(settings)}

    assert not MEMORY_TOOL_NAMES & names


def test_both_graphs_expose_identical_memory_schemas(tmp_path, stub_retriever):
    """The two wiring sites must not drift apart."""

    settings = make_settings(tmp_path, web_search_enabled=True)

    def memory_schemas(tools):
        return {
            tool.name: sorted(tool.args)
            for tool in tools
            if tool.name in MEMORY_TOOL_NAMES
        }

    heavy = memory_schemas(heavy_tools(settings))
    lightweight = memory_schemas(lightweight_tools(settings))

    assert set(heavy) == MEMORY_TOOL_NAMES
    assert heavy == lightweight, "the two wiring sites have drifted apart"


def test_disabled_memory_touches_no_file(tmp_path, stub_retriever):
    settings = make_settings(tmp_path, memory_enabled=False, web_search_enabled=True)

    heavy_tools(settings)
    lightweight_tools(settings)
    _graph_inputs_for_turn(fake_session(), "remember my name is Ada", settings)

    assert not (tmp_path / "long_term_memory.json").exists()


@pytest.mark.parametrize("resolver", [heavy_tools, lightweight_tools])
def test_enabling_memory_leaves_other_tools_unchanged(
    tmp_path, stub_retriever, resolver
):
    settings_off = make_settings(tmp_path, memory_enabled=False, web_search_enabled=True)
    settings_on = make_settings(tmp_path, web_search_enabled=True)

    off = {tool.name for tool in resolver(settings_off)}
    on = {tool.name for tool in resolver(settings_on)}

    assert on - off == MEMORY_TOOL_NAMES
    assert off - on == set()


def test_provider_supplied_tools_still_win(tmp_path):
    settings = make_settings(tmp_path)
    providers = GraphProviders(tools=[])

    assert _resolve_tools(settings, providers, False) == []
    assert _resolve_lightweight_tools(settings, providers) == []


# ---- prompt ----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(MEMORY_TOOL_NAMES))
def test_prompt_names_every_memory_tool(name):
    assert name in AGENT_SYSTEM_PROMPT


def test_prompt_explains_when_to_remember_and_recall():
    assert "REMEMBER and RECALL" in AGENT_SYSTEM_PROMPT


# ---- chat turn assembly ----------------------------------------------------


def test_memory_note_precedes_the_upload_note(tmp_path, monkeypatch):
    seeded_store(tmp_path, "Prefers metric units")
    settings = make_settings(tmp_path)

    monkeypatch.setattr(
        "src.chat.api._new_upload_context", lambda session, settings: "Uploaded: a.txt"
    )

    inputs = _graph_inputs_for_turn(
        fake_session(), "which units do I prefer?", settings
    )
    messages = inputs["messages"]

    assert [type(m).__name__ for m in messages] == [
        "SystemMessage",
        "SystemMessage",
        "HumanMessage",
    ]
    assert messages[0].content.startswith(MEMORY_NOTE_LABEL)
    assert "Uploaded" in messages[1].content
    assert messages[2].content == "which units do I prefer?"


def test_no_memory_note_when_nothing_matches(tmp_path, monkeypatch):
    seeded_store(tmp_path, "Prefers metric units")
    settings = make_settings(tmp_path)
    monkeypatch.setattr(
        "src.chat.api._new_upload_context", lambda session, settings: None
    )

    inputs = _graph_inputs_for_turn(fake_session(), "bicycles", settings)

    assert [type(m).__name__ for m in inputs["messages"]] == ["HumanMessage"]


def test_no_memory_note_when_auto_recall_is_off(tmp_path, monkeypatch):
    seeded_store(tmp_path, "Prefers metric units")
    settings = make_settings(tmp_path, memory_auto_recall_enabled=False)
    monkeypatch.setattr(
        "src.chat.api._new_upload_context", lambda session, settings: None
    )

    inputs = _graph_inputs_for_turn(
        fake_session(), "which units do I prefer?", settings
    )

    assert [type(m).__name__ for m in inputs["messages"]] == ["HumanMessage"]


def test_turn_completes_when_the_store_raises(tmp_path, monkeypatch, caplog):
    seeded_store(tmp_path, "Prefers metric units")
    settings = make_settings(tmp_path)
    monkeypatch.setattr(
        "src.chat.api._new_upload_context", lambda session, settings: None
    )

    def boom(self, thread_id):
        raise RuntimeError("store is on fire")

    monkeypatch.setattr(MemoryStore, "in_scope", boom)

    with caplog.at_level("WARNING"):
        inputs = _graph_inputs_for_turn(
            fake_session(), "which units do I prefer?", settings
        )

    monkeypatch.undo()

    assert [type(m).__name__ for m in inputs["messages"]] == ["HumanMessage"]
    assert any("skipping long-term memory" in m for m in caplog.messages)


def test_injection_is_read_only(tmp_path, monkeypatch):
    store = seeded_store(tmp_path, "Prefers metric units")
    settings = make_settings(tmp_path)
    monkeypatch.setattr(
        "src.chat.api._new_upload_context", lambda session, settings: None
    )
    before = store.path.read_text(encoding="utf-8")

    _graph_inputs_for_turn(fake_session(), "which units do I prefer?", settings)

    assert store.path.read_text(encoding="utf-8") == before
    assert store.read()[0].last_recalled_at is None


def test_session_scoped_memory_is_only_injected_for_its_thread(tmp_path, monkeypatch):
    store = MemoryStore(
        tmp_path / "long_term_memory.json",
        max_records=500,
        max_record_chars=1000,
        clock=lambda: TS,
    )
    assert store.save(
        content="metric units only here", scope="session", thread_id="thread-42"
    ).ok
    settings = make_settings(tmp_path)
    monkeypatch.setattr(
        "src.chat.api._new_upload_context", lambda session, settings: None
    )

    mine = _graph_inputs_for_turn(fake_session("thread-42"), "metric", settings)
    theirs = _graph_inputs_for_turn(fake_session("other"), "metric", settings)

    assert [type(m).__name__ for m in mine["messages"]][0] == "SystemMessage"
    assert [type(m).__name__ for m in theirs["messages"]] == ["HumanMessage"]


def test_turn_without_settings_still_builds_a_question():
    inputs = _graph_inputs_for_turn(fake_session(), "hello", None)

    assert [type(m).__name__ for m in inputs["messages"]] == ["HumanMessage"]


# ---- history excludes the note --------------------------------------------


def test_history_excludes_the_memory_note(tmp_path, monkeypatch):
    seeded_store(tmp_path, "Prefers metric units")
    settings = make_settings(tmp_path)
    monkeypatch.setattr(
        "src.chat.api._new_upload_context", lambda session, settings: None
    )

    from langchain_core.messages import AIMessage

    inputs = _graph_inputs_for_turn(
        fake_session(), "which units do I prefer?", settings
    )
    transcript = [*inputs["messages"], AIMessage(content="You prefer metric.")]

    turns = _serialize_messages(transcript)

    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert all(MEMORY_NOTE_LABEL not in turn.content for turn in turns)
