\
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .config import Settings
from .nodes import (
    agent_factory,
    generate_factory,
    grade_documents_factory,
    rewrite_factory,
)
from .retriever import build_retriever_tool
from .state import AgentState


def build_graph(settings: Settings, rebuild_vectorstore: bool = False):
    """Compile and return the LangGraph workflow."""

    retriever_tool = build_retriever_tool(settings, rebuild=rebuild_vectorstore)
    tools = [retriever_tool]

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_factory(settings, tools))
    workflow.add_node("retrieve", ToolNode(tools))
    workflow.add_node("rewrite", rewrite_factory(settings))
    workflow.add_node("generate", generate_factory(settings))

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "retrieve",
        grade_documents_factory(settings),
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )

    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    return workflow.compile()
