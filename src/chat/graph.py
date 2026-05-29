"""Chat-flavored LangGraph workflow.

Differences from ``src.core.graph.build_graph``:

1. A ``condense`` node runs first and rewrites the latest user turn into a
   standalone question using the prior conversation history. This is the
   standard fix for RAG-chat: the retriever's vector lookup has no
   conversational context, so a vague follow-up turns into noisy retrieval
   unless the question is condensed first.
2. A LangGraph ``MemorySaver`` checkpointer is attached, so each
   ``thread_id`` accumulates a persistent message history across turns.
3. The agent/grader/rewriter/generator nodes all read the current question
   from ``state["current_question"]`` rather than ``messages[0]``.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..core.config import Settings
from ..core.retriever import build_retriever_tool
from .nodes import (
    agent_factory,
    condense_question_factory,
    generate_factory,
    grade_documents_factory,
    rewrite_factory,
)
from .state import ChatState


def _build_memory_saver():
    """Locate ``MemorySaver`` across LangGraph versions.

    The class moved between releases; the current location is
    ``langgraph.checkpoint.memory``. We fall back to the older path for
    older installations rather than hard-fail."""

    try:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    except ImportError:  # pragma: no cover - covered indirectly by import tests
        from langgraph.checkpoint.memory import InMemorySaver  # type: ignore

        return InMemorySaver()


def build_chat_graph(settings: Settings, rebuild_vectorstore: bool = False):
    """Compile and return the chat workflow.

    The compiled graph is bound to the supplied ``settings.source_urls``
    via the retriever tool. Each chat session that has a different source
    set must call ``build_chat_graph`` again with the appropriate
    settings."""

    retriever_tool = build_retriever_tool(settings, rebuild=rebuild_vectorstore)
    tools = [retriever_tool]

    workflow = StateGraph(ChatState)

    workflow.add_node("condense", condense_question_factory(settings))
    workflow.add_node("agent", agent_factory(settings, tools))
    workflow.add_node("retrieve", ToolNode(tools))
    workflow.add_node("rewrite", rewrite_factory(settings))
    workflow.add_node("generate", generate_factory(settings))

    workflow.add_edge(START, "condense")
    workflow.add_edge("condense", "agent")

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

    return workflow.compile(checkpointer=_build_memory_saver())
