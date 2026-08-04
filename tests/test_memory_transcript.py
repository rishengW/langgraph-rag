"""Tests for pure transcript handling used by automatic memory extraction."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.memory.recall import MEMORY_NOTE_LABEL, MEMORY_NOTE_MARKER
from src.memory.transcript import (
    ASSISTANT_LABEL,
    MAX_SLICE_MESSAGES,
    TRUNCATION_MARKER,
    USER_LABEL,
    count_turns,
    is_memory_note,
    message_role,
    normalize_message_content,
    render_slice,
    select_slice,
)


def memory_note(body: str = "- [abc] Prefers metric units") -> SystemMessage:
    return SystemMessage(content=f"{MEMORY_NOTE_LABEL}{MEMORY_NOTE_MARKER}\n{body}")


def tool_carrier() -> AIMessage:
    """An assistant message that only selects a tool: empty content."""

    return AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])


def sample_messages() -> list:
    return [
        memory_note(),
        SystemMessage(content="Uploaded files: a.txt"),
        HumanMessage(content="first question"),
        tool_carrier(),
        ToolMessage(content="tool output", tool_call_id="1"),
        AIMessage(content="first answer"),
        HumanMessage(content="second question"),
        AIMessage(content="second answer"),
    ]


# ---- role classification ---------------------------------------------------


@pytest.mark.parametrize("message,expected", [
    (HumanMessage(content="q"), "user"),
    (AIMessage(content="a"), "assistant"),
    (SystemMessage(content="s"), "system"),
    (ToolMessage(content="t", tool_call_id="1"), "tool"),
])
def test_message_role_matches_history_serializer_classes(message, expected):
    assert message_role(message) == expected


def test_message_role_of_an_unknown_object_falls_back_to_its_kind():
    class Weird:
        content = "x"

    assert message_role(Weird()) == "weird"


# ---- content normalization -------------------------------------------------


def test_normalize_string_content_is_trimmed():
    assert normalize_message_content(HumanMessage(content="  hi  ")) == "hi"


def test_normalize_flattens_content_blocks_and_drops_non_text():
    message = AIMessage(
        content=[
            {"type": "text", "text": "first"},
            {"type": "tool_use", "id": "1", "name": "x"},
            {"type": "text", "text": "second"},
        ]
    )

    assert normalize_message_content(message) == "first second"


def test_normalize_blocks_with_no_text_yields_empty():
    message = AIMessage(content=[{"type": "tool_use", "id": "1"}])

    assert normalize_message_content(message) == ""


def test_normalize_handles_missing_and_odd_content():
    class NoContent:
        pass

    assert normalize_message_content(NoContent()) == ""
    assert normalize_message_content(AIMessage(content=[])) == ""


# ---- turn counting ---------------------------------------------------------


def test_count_turns_counts_only_user_messages():
    assert count_turns(sample_messages()) == 2


def test_count_turns_of_an_empty_list_is_zero():
    assert count_turns([]) == 0


def test_count_turns_ignores_empty_assistant_carriers():
    assert count_turns([tool_carrier(), HumanMessage(content="q")]) == 1


# ---- memory-note detection -------------------------------------------------


def test_memory_note_is_detected_by_marker():
    assert is_memory_note(memory_note()) is True


def test_ordinary_messages_are_not_memory_notes():
    assert is_memory_note(HumanMessage(content="first question")) is False
    assert is_memory_note(SystemMessage(content="Uploaded files: a.txt")) is False


def test_memory_note_detection_tolerates_non_string_content():
    assert is_memory_note(AIMessage(content=[{"type": "text", "text": "x"}])) is False


# ---- slice selection -------------------------------------------------------


def test_select_slice_at_watermark_zero_keeps_all_conversational_messages():
    kept = select_slice(sample_messages(), watermark=0)

    assert [normalize_message_content(m) for m in kept] == [
        "first question",
        "first answer",
        "second question",
        "second answer",
    ]


def test_select_slice_excludes_system_tool_note_and_blank_messages():
    kept = select_slice(sample_messages(), watermark=0)

    assert all(message_role(m) in ("user", "assistant") for m in kept)
    assert not any(is_memory_note(m) for m in kept)
    assert all(normalize_message_content(m) for m in kept)
    assert "tool output" not in [normalize_message_content(m) for m in kept]
    assert "Uploaded files: a.txt" not in [normalize_message_content(m) for m in kept]


def test_select_slice_starts_after_the_watermark_th_user_turn():
    kept = select_slice(sample_messages(), watermark=1)

    assert [normalize_message_content(m) for m in kept] == [
        "second question",
        "second answer",
    ]


def test_select_slice_at_watermark_equal_to_turn_count_is_empty():
    assert select_slice(sample_messages(), watermark=2) == []


def test_select_slice_beyond_turn_count_is_empty():
    assert select_slice(sample_messages(), watermark=99) == []


def test_select_slice_is_identical_for_both_triggers():
    messages = sample_messages()

    assert select_slice(messages, watermark=1, trigger="round_complete") == select_slice(
        messages, watermark=1, trigger="session_start"
    )


def test_select_slice_of_an_empty_transcript_is_empty():
    assert select_slice([], watermark=0) == []


def test_select_slice_excludes_a_previously_injected_note_mid_transcript():
    messages = [
        HumanMessage(content="q1"),
        AIMessage(content="a1"),
        memory_note("- [xyz] Name is Ada"),
        HumanMessage(content="q2"),
        AIMessage(content="a2"),
    ]

    kept = select_slice(messages, watermark=1)

    assert [normalize_message_content(m) for m in kept] == ["q2", "a2"]


# ---- rendering -------------------------------------------------------------


def test_render_slice_labels_roles_and_separates_blocks():
    kept = select_slice(sample_messages(), watermark=1)

    rendered = render_slice(kept, max_chars=10_000)

    assert rendered == (
        f"{USER_LABEL} second question\n\n{ASSISTANT_LABEL} second answer"
    )


def test_render_slice_is_deterministic():
    kept = select_slice(sample_messages(), watermark=0)

    assert render_slice(kept, max_chars=500) == render_slice(kept, max_chars=500)


def test_render_slice_drops_whole_messages_from_the_start():
    kept = select_slice(sample_messages(), watermark=0)

    rendered = render_slice(kept, max_chars=60)

    assert len(rendered) <= 60
    assert "second answer" in rendered, "the most recent turns survive"
    assert "first question" not in rendered
    for line in rendered.split("\n\n"):
        assert line.startswith((USER_LABEL, ASSISTANT_LABEL))


@pytest.mark.parametrize("budget", [5, 12, 16, 30, 200, 10_000])
def test_render_slice_never_exceeds_its_budget(budget):
    kept = select_slice(sample_messages(), watermark=0)

    assert len(render_slice(kept, max_chars=budget)) <= budget


def test_render_slice_truncates_a_lone_oversized_message():
    message = HumanMessage(content="x" * 500)

    rendered = render_slice([message], max_chars=100)

    assert len(rendered) == 100
    assert rendered.endswith(TRUNCATION_MARKER)
    assert rendered.startswith(USER_LABEL)


def test_render_slice_of_nothing_is_empty():
    assert render_slice([], max_chars=100) == ""


def test_render_slice_applies_the_message_cap_before_the_character_cap():
    messages = [HumanMessage(content=f"q{i}") for i in range(MAX_SLICE_MESSAGES + 50)]

    rendered = render_slice(messages, max_chars=100_000)

    assert rendered.count(USER_LABEL) == MAX_SLICE_MESSAGES
    assert f"{USER_LABEL} q{MAX_SLICE_MESSAGES + 49}" in rendered, "newest retained"
    assert f"{USER_LABEL} q0\n" not in rendered, "oldest dropped"
