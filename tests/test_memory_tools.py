"""Tests for the agent-facing memory tools.

The load-bearing guarantees here: nothing raises, failures are marked and
distinguishable from empty results, and neither the store path nor the thread id
is reachable from the model's argument schema.
"""

from __future__ import annotations

import logging

import pydantic
import pytest

from src.config.settings import Settings
from src.backend.memory.recall import MemoryCallBudget
from src.backend.memory.store import MemoryStore, SaveOutcome
from src.backend.tools.memory_tool import (
    FAILURE_MARKER,
    ForgetMemoryInput,
    RecallMemoryInput,
    SaveMemoryInput,
    build_forget_memory_tool,
    build_memory_tools,
    build_recall_memory_tool,
    build_save_memory_tool,
    thread_id_from_config,
)

TS = "2026-07-01T00:00:00+00:00"
CONFIG = {"configurable": {"thread_id": "thread-42"}}


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        dashscope_api_key="x",
        memory_enabled=True,
        memory_store_path=str(tmp_path / "long_term_memory.json"),
    )


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    return MemoryStore(
        tmp_path / "long_term_memory.json",
        max_records=500,
        max_record_chars=1000,
        clock=lambda: TS,
    )


@pytest.fixture
def tools(settings, store):
    """The three tools wired to an isolated store and a fresh budget."""

    budget = MemoryCallBudget()
    budget.reset("thread-42")
    save, recall, forget = build_memory_tools(settings, store=store, budget=budget)
    return save, recall, forget


# ---- input schemas ---------------------------------------------------------


@pytest.mark.parametrize("payload", [
    {"content": ""},
    {"content": "x" * 10_001},
    {},
])
def test_save_schema_rejects_bad_content(payload):
    with pytest.raises(pydantic.ValidationError):
        SaveMemoryInput(**payload)


def test_save_schema_rejects_too_many_tags():
    with pytest.raises(pydantic.ValidationError):
        SaveMemoryInput(content="x", tags=[f"t{i}" for i in range(11)])


def test_save_schema_defaults():
    parsed = SaveMemoryInput(content="x")

    assert parsed.category is None
    assert parsed.tags == []
    assert parsed.scope is None


@pytest.mark.parametrize("query", ["", "x" * 501])
def test_recall_schema_rejects_bad_query(query):
    with pytest.raises(pydantic.ValidationError):
        RecallMemoryInput(query=query)


def test_forget_schema_allows_either_parameter():
    assert ForgetMemoryInput().memory_id is None
    assert ForgetMemoryInput(memory_id="a" * 32).query is None
    assert ForgetMemoryInput(query="units").memory_id is None


def test_forget_schema_rejects_an_over_long_id():
    with pytest.raises(pydantic.ValidationError):
        ForgetMemoryInput(memory_id="a" * 33)


# ---- tool surface ----------------------------------------------------------


def test_three_tools_with_expected_names(tools):
    assert [tool.name for tool in tools] == [
        "save_memory",
        "recall_memory",
        "forget_memory",
    ]


def test_descriptions_are_present_and_bounded(tools):
    for tool in tools:
        assert tool.description
        assert len(tool.description) <= 300, tool.name


def test_no_tool_exposes_a_path_or_thread_parameter(tools):
    forbidden = {"config", "thread_id", "path", "file", "filename", "directory", "dir"}

    for tool in tools:
        exposed = set(tool.args)
        assert not exposed & forbidden, (tool.name, exposed)


def test_expected_argument_sets(tools):
    save, recall, forget = tools

    assert set(save.args) == {"content", "category", "tags", "scope"}
    assert set(recall.args) == {"query"}
    assert set(forget.args) == {"memory_id", "query"}


# ---- thread id injection ---------------------------------------------------


@pytest.mark.parametrize("config,expected", [
    (None, None),
    ({}, None),
    ({"configurable": {}}, None),
    ({"configurable": {"thread_id": None}}, None),
    ({"configurable": {"thread_id": "  "}}, None),
    ({"configurable": {"thread_id": " thread-9 "}}, "thread-9"),
    ({"configurable": "not-a-dict"}, None),
])
def test_thread_id_from_config(config, expected):
    assert thread_id_from_config(config) == expected


def test_injected_thread_id_reaches_the_store(settings):
    seen: list[str | None] = []

    class Recording:
        def save(self, **kwargs):
            seen.append(kwargs["thread_id"])
            return SaveOutcome(ok=True, message="ok", record_count=1)

    tool = build_save_memory_tool(
        settings, store=Recording(), budget=MemoryCallBudget()
    )

    tool.invoke({"content": "x"}, config=CONFIG)

    assert seen == ["thread-42"]


def test_session_scope_uses_the_injected_thread(tools, store):
    save, _, _ = tools

    save.invoke({"content": "local note", "scope": "session"}, config=CONFIG)

    (record,) = store.read()
    assert record.scope == "session"
    assert record.scope_id == "thread-42"


def test_session_records_are_invisible_to_another_thread(tools, store):
    save, recall, _ = tools
    save.invoke({"content": "local note", "scope": "session"}, config=CONFIG)

    mine = recall.invoke({"query": "local"}, config=CONFIG)
    theirs = recall.invoke(
        {"query": "local"}, config={"configurable": {"thread_id": "other"}}
    )

    assert "local note" in mine
    assert "No matching memory" in theirs


# ---- happy paths -----------------------------------------------------------


def test_save_then_recall_round_trip(tools):
    save, recall, _ = tools

    saved = save.invoke({"content": "Name is Ada", "category": "fact"}, config=CONFIG)
    found = recall.invoke({"query": "name"}, config=CONFIG)

    assert not saved.startswith(FAILURE_MARKER)
    assert "Saved memory" in saved
    assert "Name is Ada" in found


def test_forget_by_query_reports_the_deletion(tools):
    save, _, forget = tools
    save.invoke({"content": "Name is Ada"}, config=CONFIG)

    result = forget.invoke({"query": "name"}, config=CONFIG)

    assert not result.startswith(FAILURE_MARKER)
    assert "Deleted 1 memory" in result


# ---- failures are marked, and nothing raises -------------------------------


def test_validation_failure_returns_a_marked_string(tools):
    save, _, _ = tools

    result = save.invoke({"content": "x", "category": "opinion"}, config=CONFIG)

    assert result.startswith(FAILURE_MARKER)
    assert "category" in result


def test_secret_refusal_omits_the_secret(tools):
    save, _, _ = tools
    secret = "sk-abcdefghijklmnopqrstuv"

    result = save.invoke({"content": f"my key is {secret}"}, config=CONFIG)

    assert result.startswith(FAILURE_MARKER)
    assert secret not in result


def test_unexpected_exception_type_is_converted_to_a_string(settings):
    class Exploding:
        def save(self, **kwargs):
            raise ZeroDivisionError("nonsense from deep inside")

    tool = build_save_memory_tool(
        settings, store=Exploding(), budget=MemoryCallBudget()
    )

    result = tool.invoke({"content": "x"}, config=CONFIG)

    assert result.startswith(FAILURE_MARKER)
    assert "nonsense from deep inside" in result


def test_base_exception_is_also_converted(settings):
    class Exploding:
        def recall(self, **kwargs):
            raise MemoryError("out of memory, ironically")

    tool = build_recall_memory_tool(
        settings, store=Exploding(), budget=MemoryCallBudget()
    )

    result = tool.invoke({"query": "x"}, config=CONFIG)

    assert result.startswith(FAILURE_MARKER)


def test_rejected_store_path_returns_a_string_not_an_exception():
    settings = Settings(
        dashscope_api_key="x",
        memory_enabled=True,
        memory_store_path="../escape.json",
    )
    tool = build_save_memory_tool(settings, budget=MemoryCallBudget())

    result = tool.invoke({"content": "x"}, config=CONFIG)

    assert result.startswith(FAILURE_MARKER)
    assert ".." in result


def test_failure_string_is_bounded(settings):
    class Exploding:
        def save(self, **kwargs):
            raise RuntimeError("y" * 2000)

    tool = build_save_memory_tool(
        settings, store=Exploding(), budget=MemoryCallBudget()
    )

    result = tool.invoke({"content": "x"}, config=CONFIG)

    assert len(result) <= 500
    assert result.startswith(FAILURE_MARKER)


def test_broken_logger_does_not_change_the_outcome(tools, monkeypatch):
    save, _, _ = tools

    def boom(*args, **kwargs):
        raise RuntimeError("logging subsystem down")

    monkeypatch.setattr(logging.Logger, "log", boom)

    result = save.invoke({"content": "Name is Ada"}, config=CONFIG)

    monkeypatch.undo()

    assert not result.startswith(FAILURE_MARKER)
    assert "Saved memory" in result


# ---- empty results are successes, not failures -----------------------------


def test_recall_with_no_match_is_not_a_failure(tools):
    _, recall, _ = tools

    result = recall.invoke({"query": "bicycles"}, config=CONFIG)

    assert not result.startswith(FAILURE_MARKER)
    assert "No matching memory" in result


def test_forget_not_found_is_not_a_failure(tools):
    _, _, forget = tools

    result = forget.invoke({"memory_id": "a" * 32}, config=CONFIG)

    assert not result.startswith(FAILURE_MARKER)
    assert "No memory with id" in result


def test_forget_with_neither_parameter_is_a_failure(tools):
    _, _, forget = tools

    result = forget.invoke({}, config=CONFIG)

    assert result.startswith(FAILURE_MARKER)
    assert "memory_id" in result and "query" in result


# ---- truncation ------------------------------------------------------------


def test_long_success_payload_is_truncated_with_a_marker(tmp_path, store):
    settings = Settings(
        dashscope_api_key="x",
        memory_enabled=True,
        memory_store_path=str(tmp_path / "long_term_memory.json"),
        memory_context_max_chars=120,
        memory_recall_top_k=50,
    )
    budget = MemoryCallBudget()
    save, recall, _ = build_memory_tools(settings, store=store, budget=budget)
    for index in range(8):
        save.invoke(
            {"content": f"metric memory number {index} with padding"}, config=CONFIG
        )

    result = recall.invoke({"query": "metric"}, config=CONFIG)

    assert len(result) <= 120


# ---- per-turn budget -------------------------------------------------------


def test_eleventh_call_in_a_turn_is_refused(settings, store):
    budget = MemoryCallBudget()
    save, _, _ = build_memory_tools(settings, store=store, budget=budget)
    budget.reset("thread-42")

    for index in range(10):
        result = save.invoke({"content": f"memory {index}"}, config=CONFIG)
        assert not result.startswith(FAILURE_MARKER), index

    refused = save.invoke({"content": "one too many"}, config=CONFIG)

    assert refused.startswith(FAILURE_MARKER)
    assert "limit" in refused
    assert len(store.read()) == 10, "the refused call must not be persisted"


def test_budget_is_shared_across_the_three_tools(settings, store):
    budget = MemoryCallBudget(limit=2)
    save, recall, forget = build_memory_tools(settings, store=store, budget=budget)
    budget.reset("thread-42")

    assert not save.invoke({"content": "x"}, config=CONFIG).startswith(FAILURE_MARKER)
    assert not recall.invoke({"query": "x"}, config=CONFIG).startswith(FAILURE_MARKER)

    assert forget.invoke({"query": "x"}, config=CONFIG).startswith(FAILURE_MARKER)


def test_factories_accept_no_budget_and_use_the_shared_one(settings, store):
    tool = build_forget_memory_tool(settings, store=store)

    result = tool.invoke({"query": "nothing here"}, config=CONFIG)

    assert isinstance(result, str)
