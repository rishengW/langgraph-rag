from __future__ import annotations

import importlib
import sys
from threading import Lock
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.graph.builder import (
    GraphNodeOverrides,
    GraphProviders,
    build_graph,
    build_lightweight_graph,
)
from src.graph.edges import (
    route_after_agent_with_critique,
    route_after_lightweight_agent_with_critique,
    route_after_web_answer_with_fallback,
)
from src.graph.nodes import planning as planning_module
from src.graph.nodes.planning import (
    normalize_plan,
    planner_node,
    reflection_revise_node,
    route_after_self_critique,
    route_subgoals,
)
from src.tools.web_search import build_web_search_tool


@tool
def noop_tool(value: str) -> str:
    """Return a deterministic test value."""

    return value


def _plan_state() -> dict[str, object]:
    return {
        "plan": [
            {"id": "a", "description": "A", "dependencies": [], "status": "pending"},
            {"id": "b", "description": "B", "dependencies": [], "status": "pending"},
            {
                "id": "c",
                "description": "C",
                "dependencies": ["a", "b"],
                "status": "pending",
            },
        ],
        "planning_question": "Original question",
        "planning_input_context": "Concise source context",
        "planning_run_id": 1,
    }


def test_normalize_plan_clamps_ids_dependencies_and_cycles():
    raw = [
        {"id": "x", "description": "first", "dependencies": ["y", "missing"]},
        {"id": "x", "description": "second", "dependencies": ["x"]},
        {"id": "z", "description": "third", "dependencies": ["x"]},
    ]

    result = normalize_plan(raw, max_subgoals=2)

    assert len(result) == 2
    assert len({item["id"] for item in result}) == 2
    known = {item["id"] for item in result}
    assert all(set(item["dependencies"]) <= known - {item["id"]} for item in result)
    assert all(item["status"] == "pending" for item in result)


def test_planner_generates_normalized_bounded_plan(monkeypatch, isolated_settings):
    captured: dict[str, object] = {}
    monkeypatch.setattr(planning_module, "new_structured_chat_model", lambda *_args: "model")

    def fake_invoke(model, messages, **kwargs):
        captured.update(model=model, prompt=messages[0].content, kwargs=kwargs)
        return SimpleNamespace(
            subgoals=[
                {"id": "sg-1", "description": "Collect evidence", "dependencies": []},
                {"id": "sg-2", "description": "Synthesize", "dependencies": ["sg-1"]},
                {"id": "sg-3", "description": "Overflow", "dependencies": []},
            ],
            reasoning_scratchpad="Two concise execution stages.",
        )

    monkeypatch.setattr(planning_module, "invoke_with_retry", fake_invoke)
    node = planner_node(isolated_settings(), max_subgoals=2)

    result = node({"messages": [HumanMessage(content="Compare A and B")]})

    assert [item["id"] for item in result["plan"]] == ["sg-1", "sg-2"]
    assert result["planning_question"] == "Compare A and B"
    assert result["planning_input_context"] == "Compare A and B"
    assert result["planning_run_id"] == 1
    assert "hidden chain-of-thought" in captured["prompt"]


def test_send_payload_preserves_original_question_and_context():
    state = _plan_state()
    state["dispatched_subgoals"] = [state["plan"][0]]

    sends = route_subgoals(state)

    assert len(sends) == 1
    assert sends[0].arg["current_question"] == "Original question"
    assert sends[0].arg["planning_input_context"] == "Concise source context"
    assert sends[0].arg["planning_run_id"] == 1


def test_dependent_subgoals_execute_all_ready_waves(isolated_settings):
    seen: list[str] = []
    lock = Lock()

    def planner(_state):
        return _plan_state()

    def worker(state):
        goal = state["subgoal"]
        assert state["current_question"] == "Original question"
        assert state["planning_input_context"] == "Concise source context"
        with lock:
            if goal["id"] == "c":
                assert {"a", "b"} <= set(seen)
            seen.append(goal["id"])
        return {
            "subgoal_results": [
                {
                    "id": goal["id"],
                    "result": f"result-{goal['id']}",
                    "status": "completed",
                    "planning_run_id": state["planning_run_id"],
                }
            ]
        }

    def agent(state):
        assert all(item["status"] == "completed" for item in state["plan"])
        return {"messages": [AIMessage(content="planned answer")]}

    graph = build_graph(
        settings=isolated_settings(planning_enabled=True, planning_max_subgoals=3),
        providers=GraphProviders(
            tools=[noop_tool],
            nodes=GraphNodeOverrides(
                planner=planner,
                subgoal_worker=worker,
                agent=agent,
                retrieve=lambda _state: {},
                grade_documents=lambda _state: "generate",
                rewrite=lambda _state: {},
                generate=lambda _state: {"messages": [AIMessage(content="unused")]},
                answer_self_critique=lambda _state: {
                    "answer_critique": {
                        "correctness_score": 1.0,
                        "groundedness_score": 1.0,
                        "completeness_score": 1.0,
                    }
                },
                reflection_revise=lambda _state: {},
            ),
        ),
    )

    state = graph.invoke({"messages": [HumanMessage(content="Original question")]})

    assert set(seen[:2]) == {"a", "b"}
    assert seen[2:] == ["c"]
    assert state["messages"][-1].content == "planned answer"
    assert "3 completed" in state["global_scratchpad"]


def test_direct_answer_and_web_answer_critique_routing():
    direct = {"messages": [AIMessage(content="final answer")]}
    tool_call = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "noop_tool", "args": {"value": "x"}, "id": "1"}],
            )
        ]
    }

    assert route_after_agent_with_critique(direct, critique_enabled=True) == "answer_self_critique"
    assert route_after_agent_with_critique(tool_call, critique_enabled=True) == "tools"
    assert (
        route_after_lightweight_agent_with_critique(direct, critique_enabled=True)
        == "answer_self_critique"
    )
    assert route_after_web_answer_with_fallback(
        {"web_answer_no_readable_content": False}, planning_enabled=True
    ) == "answer_self_critique"


def test_reflection_revision_is_bounded(monkeypatch, isolated_settings):
    calls = 0
    monkeypatch.setattr(planning_module, "new_chat_model", lambda _settings: "model")

    def fake_invoke(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return AIMessage(content="revised answer")

    monkeypatch.setattr(planning_module, "invoke_with_retry", fake_invoke)
    node = reflection_revise_node(isolated_settings(), max_retries=1)
    state = {
        "messages": [HumanMessage(content="Question"), AIMessage(content="draft")],
        "answer_critique": {
            "correctness_score": 0.2,
            "groundedness_score": 0.2,
            "completeness_score": 0.2,
            "critique_notes": "Needs evidence.",
            "revision_suggestions": "Use supplied evidence.",
        },
        "reflection_retry_count": 0,
    }

    first = node(state)
    second = node({**state, **first, "reflection_retry_count": 1})

    assert first["messages"][0].content == "revised answer"
    assert first["reflection_retry_count"] == 1
    assert second == {"reflection_retry_count": 1}
    assert calls == 1
    assert route_after_self_critique(state, max_retries=1) == "reflection_revise"
    assert route_after_self_critique({**state, "reflection_retry_count": 1}, max_retries=1) == "__end__"


def test_lightweight_graph_runs_planning_and_critiques_web_answer(isolated_settings):
    seen: list[str] = []
    web_tool = build_web_search_tool(
        isolated_settings(),
        discovery=lambda *_args: ["https://example.test/source"],
    )

    def planner(_state):
        seen.append("planner")
        return {
            "plan": [
                {"id": "a", "description": "prepare", "dependencies": [], "status": "pending"}
            ],
            "planning_question": "Current question",
            "planning_input_context": "Current question",
            "planning_run_id": 1,
        }

    def worker(state):
        seen.append("worker")
        return {
            "subgoal_results": [
                {
                    "id": state["subgoal"]["id"],
                    "result": "prepared evidence",
                    "status": "completed",
                    "planning_run_id": 1,
                }
            ]
        }

    def agent(_state):
        seen.append("agent")
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "live_web_search",
                            "args": {"query": "current question"},
                            "id": "search-1",
                        }
                    ],
                )
            ]
        }

    graph = build_lightweight_graph(
        settings=isolated_settings(planning_enabled=True),
        providers=GraphProviders(
            tools=[web_tool],
            nodes=GraphNodeOverrides(
                planner=planner,
                subgoal_worker=worker,
                agent=agent,
                decompose=lambda _state: {"sub_questions": ["current question"]},
                merge=lambda _state: {"source_urls": ["https://example.test/source"]},
                web_answer=lambda _state: (
                    seen.append("web_answer")
                    or {
                        "messages": [AIMessage(content="grounded answer")],
                        "web_answer_no_readable_content": False,
                    }
                ),
                fallback_answer=lambda _state: pytest.fail("fallback must not run"),
                answer_self_critique=lambda _state: (
                    seen.append("critique")
                    or {
                        "answer_critique": {
                            "correctness_score": 1.0,
                            "groundedness_score": 1.0,
                            "completeness_score": 1.0,
                        }
                    }
                ),
                reflection_revise=lambda _state: pytest.fail("revision must not run"),
            ),
        ),
    )

    state = graph.invoke({"messages": [HumanMessage(content="Current question")]})

    assert seen == ["planner", "worker", "agent", "web_answer", "critique"]
    assert state["messages"][-1].content == "grounded answer"


def test_lightweight_graph_uses_fallback_after_exhausted_expansion(isolated_settings):
    calls = {"web_answer": 0, "fallback": 0}
    web_tool = build_web_search_tool(
        isolated_settings(),
        discovery=lambda *_args: ["https://example.test/unreadable"],
    )

    def agent(_state):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "live_web_search",
                            "args": {"query": "question"},
                            "id": "search-1",
                        }
                    ],
                )
            ]
        }

    def web_answer(_state):
        calls["web_answer"] += 1
        return {
            "messages": [AIMessage(content="grounded refusal")],
            "web_answer_no_readable_content": True,
            "web_answer_attempts": calls["web_answer"],
        }

    def fallback(_state):
        calls["fallback"] += 1
        return {"messages": [AIMessage(content="unverified fallback answer")]}

    graph = build_lightweight_graph(
        settings=isolated_settings(planning_enabled=False),
        providers=GraphProviders(
            tools=[web_tool],
            nodes=GraphNodeOverrides(
                agent=agent,
                decompose=lambda _state: {"sub_questions": ["question"]},
                expand=lambda _state: {
                    "expanded_queries": ["question expanded"],
                    "expansion_attempted": True,
                },
                merge=lambda _state: {"source_urls": ["https://example.test/unreadable"]},
                web_answer=web_answer,
                fallback_answer=fallback,
            ),
        ),
    )

    state = graph.invoke({"messages": [HumanMessage(content="Question")]})

    assert calls == {"web_answer": 2, "fallback": 1}
    assert state["messages"][-1].content == "unverified fallback answer"


def test_planning_remains_default_off_in_both_graphs(isolated_settings):
    providers = GraphProviders(
        tools=[noop_tool],
        nodes=GraphNodeOverrides(
            agent=lambda _state: {"messages": [AIMessage(content="answer")]},
            retrieve=lambda _state: {},
            grade_documents=lambda _state: "generate",
            rewrite=lambda _state: {},
            generate=lambda _state: {"messages": [AIMessage(content="generated")]},
            fallback_answer=lambda _state: {"messages": [AIMessage(content="fallback")]},
        ),
    )

    standard_nodes = build_graph(settings=isolated_settings(), providers=providers).get_graph().nodes
    lightweight_providers = GraphProviders(
        tools=[
            build_web_search_tool(
                isolated_settings(),
                discovery=lambda *_args: ["https://example.test/source"],
            )
        ],
        nodes=providers.nodes,
    )
    lightweight_nodes = build_lightweight_graph(
        settings=isolated_settings(), providers=lightweight_providers
    ).get_graph().nodes

    assert "planner" not in standard_nodes
    assert "answer_self_critique" not in standard_nodes
    assert "planner" not in lightweight_nodes
    assert "answer_self_critique" not in lightweight_nodes
    assert "fallback_answer" in lightweight_nodes


def test_legacy_web_search_tool_import_is_compatible():
    canonical = importlib.import_module("src.tools.web_search")
    sys.modules.pop("src.web_search.tool", None)

    with pytest.warns(DeprecationWarning, match="src.web_search.tool is deprecated"):
        legacy = importlib.import_module("src.web_search.tool")

    assert legacy.build_web_search_tool is canonical.build_web_search_tool
    assert legacy.WebSearchInput is canonical.WebSearchInput


def test_failed_dependency_plan_exits_without_dispatch_deadlock():
    state = {
        "plan": [
            {"id": "a", "description": "A", "dependencies": [], "status": "failed"},
            {"id": "b", "description": "B", "dependencies": ["a"], "status": "pending"},
        ]
    }

    dispatched = planning_module.subgoal_dispatcher_node(state, max_dispatch=2)

    assert all(item["status"] == "failed" for item in dispatched["plan"])
    assert route_subgoals(dispatched) == "subgoal_aggregator"
