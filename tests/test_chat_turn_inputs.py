from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from src.frontend.chat.api import _graph_inputs_for_turn


def _session(source_urls, source_mode="web_search"):
    return SimpleNamespace(source_urls=list(source_urls), source_mode=source_mode)


def test_graph_inputs_seed_current_session_source_urls():
    """Each turn must seed the session's current URLs into graph state.

    This overwrites any stale ``source_urls`` left in the per-thread
    checkpoint by a previous turn, so a new question fetches its own freshly
    discovered pages instead of the first turn's pages.
    """

    session = _session(
        ["https://fifa.com/top-scorers", "https://espn.com/worldcup"],
        source_mode="web_search",
    )

    inputs = _graph_inputs_for_turn(session, "Who is the top scorer?")

    assert isinstance(inputs["messages"][0], HumanMessage)
    assert inputs["messages"][0].content == "Who is the top scorer?"
    assert inputs["source_urls"] == [
        "https://fifa.com/top-scorers",
        "https://espn.com/worldcup",
    ]
    assert inputs["source_mode"] == "web_search"


def test_graph_inputs_overwrite_previous_turn_urls():
    """A fresh turn's seeded URLs differ from a prior turn's URLs."""

    japan_turn = _graph_inputs_for_turn(
        _session(["https://japan.travel/en/"]), "Tell me about Japan"
    )
    worldcup_turn = _graph_inputs_for_turn(
        _session(["https://fifa.com/top-scorers"]), "Top scorer?"
    )

    assert japan_turn["source_urls"] != worldcup_turn["source_urls"]
    assert worldcup_turn["source_urls"] == ["https://fifa.com/top-scorers"]


def test_graph_inputs_omit_source_urls_when_session_has_none():
    """With no session URLs, only messages are seeded (no empty override)."""

    inputs = _graph_inputs_for_turn(_session([], source_mode="defaults"), "hi")

    assert "source_urls" not in inputs
    assert "source_mode" not in inputs
    assert inputs["messages"][0].content == "hi"


def test_lightweight_web_turn_resets_checkpointed_search_state():
    settings = SimpleNamespace(
        file_read_enabled=False,
        web_search_lightweight=True,
    )
    inputs = _graph_inputs_for_turn(
        _session(["https://previous.test/page"], source_mode="web_search"),
        "A new question",
        settings,
    )

    assert inputs["source_urls"] == []
    assert inputs["current_question"] == "A new question"
    assert inputs["sub_questions"] == []
    assert inputs["expanded_queries"] == []
    assert inputs["search_queries"] == []
    assert inputs["web_search_results"] == []
    assert inputs["web_search_result_metadata"] == []
    assert inputs["web_answer_attempts"] == 0
    assert inputs["web_answer_no_readable_content"] is False
    assert inputs["expansion_attempted"] is False
