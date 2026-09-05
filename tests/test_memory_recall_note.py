"""Tests for automatic recall injection and the per-turn tool-call budget."""

from __future__ import annotations

import time

import pytest

from src.backend.memory.models import MAX_TOOL_CALLS_PER_TURN
from src.backend.memory.recall import (
    MEMORY_NOTE_LABEL,
    MEMORY_NOTE_MARKER,
    MemoryCallBudget,
    build_memory_note,
    build_turn_messages,
    format_records,
)
from src.backend.memory.store import MemoryStore, reset_store_cache
from src.backend.memory.transcript import is_memory_note
from src.config.settings import Settings

TS = "2026-07-01T00:00:00+00:00"


@pytest.fixture
def store(tmp_path):
    return MemoryStore(
        tmp_path / "long_term_memory.json",
        max_records=500,
        max_record_chars=1000,
        clock=lambda: TS,
    )


# ---- note rendering --------------------------------------------------------


def test_note_carries_the_label_ids_and_content(store):
    saved = store.save(content="Prefers metric units", category="preference")

    note = build_memory_note(
        store, message="what units do I prefer?", thread_id=None, top_k=5, max_chars=2000
    )

    assert note is not None
    assert note.startswith(MEMORY_NOTE_LABEL)
    assert note.startswith(f"{MEMORY_NOTE_LABEL}{MEMORY_NOTE_MARKER}")
    assert note.replace(MEMORY_NOTE_MARKER, "") == (
        f"{MEMORY_NOTE_LABEL}\n{format_records(store.read(), max_chars=2000)}"
    )
    assert saved.record_id in note
    assert "Prefers metric units" in note


def test_note_respects_the_character_budget(store):
    for index in range(10):
        store.save(content=f"metric memory number {index} with some extra text")

    note = build_memory_note(
        store, message="metric", thread_id=None, top_k=10, max_chars=200
    )

    assert note is not None
    assert len(note) <= 200
    for line in note.splitlines()[1:]:
        assert line.startswith("- ["), "only whole records are included"


def test_note_is_none_when_nothing_matches(store):
    store.save(content="Prefers metric units")

    assert (
        build_memory_note(
            store, message="bicycles and trains", thread_id=None, top_k=5, max_chars=2000
        )
        is None
    )


def test_note_is_none_when_the_store_is_empty(store):
    assert (
        build_memory_note(
            store, message="anything at all", thread_id=None, top_k=5, max_chars=2000
        )
        is None
    )


def test_note_is_none_for_a_message_with_no_usable_terms(store):
    store.save(content="Prefers metric units")

    assert (
        build_memory_note(store, message="a b c", thread_id=None, top_k=5, max_chars=2000)
        is None
    )


def test_note_only_includes_in_scope_records(store):
    store.save(content="global metric note")
    store.save(content="session metric note", scope="session", thread_id="mine")

    mine = build_memory_note(
        store, message="metric", thread_id="mine", top_k=5, max_chars=2000
    )
    theirs = build_memory_note(
        store, message="metric", thread_id="other", top_k=5, max_chars=2000
    )

    assert mine is not None and theirs is not None
    assert "session metric note" in mine
    assert "session metric note" not in theirs


def test_note_scores_against_the_first_500_characters(store):
    store.save(content="Prefers metric units")

    padded = ("x" * 600) + " metric"
    assert (
        build_memory_note(
            store, message=padded, thread_id=None, top_k=5, max_chars=2000
        )
        is None
    ), "terms beyond the 500-character window are not considered"


# ---- injection is read-only ------------------------------------------------


def test_note_building_does_not_write_to_the_store(store):
    store.save(content="Prefers metric units")
    before_text = store.path.read_text(encoding="utf-8")
    before_mtime = store.path.stat().st_mtime_ns
    time.sleep(0.01)

    note = build_memory_note(
        store, message="metric", thread_id=None, top_k=5, max_chars=2000
    )

    assert note is not None
    assert store.path.read_text(encoding="utf-8") == before_text
    assert store.path.stat().st_mtime_ns == before_mtime


def test_note_building_leaves_last_recalled_at_untouched(store):
    store.save(content="Prefers metric units")
    assert store.read()[0].last_recalled_at is None

    build_memory_note(
        store, message="metric", thread_id=None, top_k=5, max_chars=2000
    )

    assert store.read()[0].last_recalled_at is None, (
        "only an explicit recall_memory call may update recency"
    )


def test_note_selection_is_stable_across_turns(store):
    for index in range(4):
        store.save(content=f"metric memory {index}")

    first = build_memory_note(
        store, message="metric", thread_id=None, top_k=3, max_chars=2000
    )
    second = build_memory_note(
        store, message="metric", thread_id=None, top_k=3, max_chars=2000
    )

    assert first == second


# ---- failure degrades the turn --------------------------------------------


def test_note_building_survives_a_failing_store(caplog):
    class Exploding:
        def in_scope(self, thread_id):
            raise RuntimeError("store is on fire")

    with caplog.at_level("WARNING"):
        note = build_memory_note(
            Exploding(), message="metric", thread_id=None, top_k=5, max_chars=2000
        )

    assert note is None
    assert any("skipping long-term memory" in message for message in caplog.messages)


def test_format_records_always_emits_at_least_one_whole_entry(store):
    store.save(content="a fairly long memory that exceeds a tiny budget on its own")
    records = store.read()

    rendered = format_records(records, max_chars=5)

    assert rendered.startswith("- [")
    assert "\n" not in rendered


# ---- call budget -----------------------------------------------------------


def test_budget_allows_the_limit_then_refuses():
    budget = MemoryCallBudget()
    budget.reset("thread-a")

    allowed = [budget.consume("thread-a") for _ in range(MAX_TOOL_CALLS_PER_TURN)]

    assert all(allowed)
    assert budget.consume("thread-a") is False, "the 11th call in a turn is refused"


def test_budget_resets_per_turn():
    budget = MemoryCallBudget(limit=2)
    budget.reset("thread-a")
    assert budget.consume("thread-a")
    assert budget.consume("thread-a")
    assert budget.consume("thread-a") is False

    budget.reset("thread-a")

    assert budget.consume("thread-a") is True


def test_budget_is_tracked_per_thread():
    budget = MemoryCallBudget(limit=1)
    budget.reset("thread-a")
    budget.reset("thread-b")

    assert budget.consume("thread-a")
    assert budget.consume("thread-a") is False
    assert budget.consume("thread-b") is True


def test_budget_handles_a_missing_thread_id():
    budget = MemoryCallBudget(limit=1)
    budget.reset(None)

    assert budget.consume(None) is True
    assert budget.consume(None) is False


def test_budget_without_a_reset_still_bounds_calls():
    budget = MemoryCallBudget(limit=1)

    assert budget.consume("fresh-thread") is True
    assert budget.consume("fresh-thread") is False


# ---- turn assembly ---------------------------------------------------------


def make_settings(tmp_path, **overrides) -> Settings:
    values = {
        "memory_enabled": True,
        "memory_auto_recall_enabled": True,
        "memory_store_path": str(tmp_path / "long_term_memory.json"),
    }
    values.update(overrides)
    return Settings(dashscope_api_key="x", **values)


def roles(messages) -> list[str]:
    return [type(message).__name__ for message in messages]


def test_turn_messages_place_memory_before_upload_and_question(tmp_path, store):
    reset_store_cache()
    store.save(content="Prefers metric units")
    settings = make_settings(tmp_path)

    messages = build_turn_messages(
        settings,
        thread_id=None,
        message="which units do I prefer?",
        upload_note="Uploaded files: a.txt",
    )

    assert roles(messages) == ["SystemMessage", "SystemMessage", "HumanMessage"]
    assert messages[0].content.startswith(MEMORY_NOTE_LABEL)
    assert is_memory_note(messages[0])
    assert "Uploaded files" in messages[1].content
    assert messages[2].content == "which units do I prefer?"
    reset_store_cache()


def test_turn_messages_omit_the_note_when_nothing_matches(tmp_path, store):
    reset_store_cache()
    store.save(content="Prefers metric units")

    messages = build_turn_messages(
        make_settings(tmp_path), thread_id=None, message="bicycles"
    )

    assert roles(messages) == ["HumanMessage"]
    reset_store_cache()


def test_turn_messages_omit_the_note_when_auto_recall_is_off(tmp_path, store):
    reset_store_cache()
    store.save(content="Prefers metric units")

    messages = build_turn_messages(
        make_settings(tmp_path, memory_auto_recall_enabled=False),
        thread_id=None,
        message="which units do I prefer?",
    )

    assert roles(messages) == ["HumanMessage"]
    reset_store_cache()


def test_turn_messages_omit_the_note_when_memory_is_disabled(tmp_path, store):
    reset_store_cache()
    store.save(content="Prefers metric units")

    messages = build_turn_messages(
        make_settings(tmp_path, memory_enabled=False),
        thread_id=None,
        message="which units do I prefer?",
    )

    assert roles(messages) == ["HumanMessage"]
    reset_store_cache()


def test_turn_messages_survive_a_rejected_store_path(tmp_path, caplog):
    reset_store_cache()
    settings = Settings(
        dashscope_api_key="x",
        memory_enabled=True,
        memory_auto_recall_enabled=True,
        memory_store_path="../escape.json",
    )

    with caplog.at_level("WARNING"):
        messages = build_turn_messages(
            settings, thread_id=None, message="which units do I prefer?"
        )

    assert roles(messages) == ["HumanMessage"]
    assert any("unavailable" in message for message in caplog.messages)
    reset_store_cache()


def test_turn_messages_reset_the_budget(tmp_path):
    budget = MemoryCallBudget(limit=1)
    budget.reset("thread-a")
    assert budget.consume("thread-a")
    assert budget.consume("thread-a") is False

    build_turn_messages(
        make_settings(tmp_path, memory_enabled=False),
        thread_id="thread-a",
        message="hello",
        budget=budget,
    )

    assert budget.consume("thread-a") is True, "each turn starts with a fresh budget"
