from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.graph.builder import GraphNodeOverrides, GraphProviders, build_graph


@tool
def retrieve_source_documents(query: str) -> str:
    """Return fake source context for graph builder tests."""

    return f"retrieved context for {query}"


def test_qa_builder_accepts_injected_nodes_and_tools():
    def agent(_state):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "retrieve_source_documents",
                            "args": {"query": "PAI"},
                            "id": "call_fake_retrieve",
                        }
                    ],
                )
            ]
        }

    def grade_documents(state):
        assert state["messages"][-1].content == "retrieved context for PAI"
        return "generate"

    def generate(state):
        assert state["messages"][-1].content == "retrieved context for PAI"
        return {"messages": [AIMessage(content="final answer")]}

    providers = GraphProviders(
        tools=[retrieve_source_documents],
        nodes=GraphNodeOverrides(
            agent=agent,
            grade_documents=grade_documents,
            rewrite=lambda _state: {"messages": [AIMessage(content="unused")]},
            generate=generate,
        ),
    )

    graph = build_graph(mode="qa", providers=providers)
    state = graph.invoke({"messages": [HumanMessage(content="What is PAI?")]})

    assert state["messages"][-1].content == "final answer"


def test_chat_builder_adds_condense_and_allows_checkpointer_injection():
    seen: list[str] = []

    def condense(state):
        seen.append("condense")
        assert state["messages"][-1].content == "raw follow-up"
        return {
            "current_question": "standalone follow-up",
            "current_question_index": 0,
            "rewrite_count": 0,
        }

    def agent(state):
        seen.append("agent")
        assert state["current_question"] == "standalone follow-up"
        return {"messages": [AIMessage(content="direct answer")]}

    providers = GraphProviders(
        checkpointer=None,
        nodes=GraphNodeOverrides(
            condense=condense,
            agent=agent,
            retrieve=lambda _state: {"messages": [AIMessage(content="unused")]},
            grade_documents=lambda _state: "generate",
            rewrite=lambda _state: {"messages": [AIMessage(content="unused")]},
            generate=lambda _state: {"messages": [AIMessage(content="unused")]},
        ),
    )

    graph = build_graph(mode="chat", providers=providers)
    state = graph.invoke({"messages": [HumanMessage(content="raw follow-up")]})

    assert seen == ["condense", "agent"]
    assert state["messages"][-1].content == "direct answer"


def test_builder_requires_settings_for_default_nodes():
    with pytest.raises(ValueError, match="settings are required"):
        build_graph(mode="qa")


def test_legacy_graph_wrappers_delegate_to_shared_builder(monkeypatch, mock_settings):
    import src.chat.graph as chat_graph
    import src.core.graph as core_graph

    qa_sentinel = object()
    chat_sentinel = object()
    captured = []

    def fake_core_builder(**kwargs):
        captured.append(kwargs)
        return qa_sentinel

    def fake_chat_builder(**kwargs):
        captured.append(kwargs)
        return chat_sentinel

    monkeypatch.setattr(core_graph, "_build_graph", fake_core_builder)
    monkeypatch.setattr(chat_graph, "_build_graph", fake_chat_builder)

    assert core_graph.build_graph(mock_settings, rebuild_vectorstore=True) is qa_sentinel
    assert chat_graph.build_chat_graph(mock_settings, rebuild_vectorstore=False) is chat_sentinel
    assert captured == [
        {
            "mode": "qa",
            "settings": mock_settings,
            "rebuild_vectorstore": True,
        },
        {
            "mode": "chat",
            "settings": mock_settings,
            "rebuild_vectorstore": False,
        },
    ]
