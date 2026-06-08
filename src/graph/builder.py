from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from ..config import Settings
from .edges import AGENT_EDGE_MAP, GRADE_EDGE_MAP, route_after_agent
from .nodes import (
    agent_factory,
    chat_question_resolver,
    condense_question_factory,
    generate_factory,
    grade_documents_factory,
    qa_question_resolver,
    rewrite_factory,
)
from .state import AgentState, ChatState

GraphMode = Literal["qa", "chat"]
NodeCallable = Callable[[dict[str, Any]], dict[str, Any]]
GradeEdgeCallable = Callable[[dict[str, Any]], Literal["generate", "rewrite"]]

_DEFAULT_CHECKPOINTER = object()


@dataclass(frozen=True)
class GraphNodeOverrides:
    """Optional node hooks for tests or provider-specific graph assembly."""

    agent: NodeCallable | None = None
    retrieve: NodeCallable | None = None
    grade_documents: GradeEdgeCallable | None = None
    rewrite: NodeCallable | None = None
    generate: NodeCallable | None = None
    condense: NodeCallable | None = None


@dataclass(frozen=True)
class GraphProviders:
    """Dependency injection hooks for graph construction.

    Supplying ``tools`` skips the Settings-based Chroma retriever tool build.
    Supplying node overrides skips the Settings-based node factories for those
    nodes. Existing application entry points can continue passing only Settings.
    """

    tools: Sequence[Any] | None = None
    nodes: GraphNodeOverrides = field(default_factory=GraphNodeOverrides)
    checkpointer: Any = _DEFAULT_CHECKPOINTER


def build_memory_saver():
    """Create the default in-memory LangGraph checkpointer for chat mode."""

    try:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    except ImportError:  # pragma: no cover - compatibility with older LangGraph
        from langgraph.checkpoint.memory import InMemorySaver  # type: ignore

        return InMemorySaver()


def build_graph(
    mode: GraphMode = "qa",
    *,
    settings: Settings | None = None,
    providers: GraphProviders | None = None,
    rebuild_vectorstore: bool = False,
    checkpointer: Any = _DEFAULT_CHECKPOINTER,
):
    """Compile the QA or chat LangGraph workflow.

    Defaults preserve the legacy Settings-based builders. ``providers`` is an
    additive DI surface used by tests and future API wiring to avoid live
    provider construction.
    """

    if mode not in ("qa", "chat"):
        raise ValueError("mode must be 'qa' or 'chat'")

    providers = providers or GraphProviders()
    nodes = providers.nodes
    tools = _resolve_tools(settings, providers, rebuild_vectorstore)
    question_resolver = qa_question_resolver if mode == "qa" else chat_question_resolver
    state_type = AgentState if mode == "qa" else ChatState

    workflow = StateGraph(state_type)

    if mode == "chat":
        workflow.add_node(
            "condense",
            nodes.condense or condense_question_factory(_require_settings(settings, "condense")),
        )

    workflow.add_node(
        "agent",
        nodes.agent
        or agent_factory(_require_settings(settings, "agent"), tools, question_resolver),
    )

    if nodes.retrieve is not None:
        workflow.add_node("retrieve", nodes.retrieve)
    else:
        if not tools:
            raise ValueError("providers.tools or settings are required for the retrieve node")
        workflow.add_node("retrieve", ToolNode(list(tools)))

    workflow.add_node(
        "rewrite",
        nodes.rewrite
        or rewrite_factory(
            _require_settings(settings, "rewrite"),
            question_resolver,
            update_current_question=(mode == "chat"),
        ),
    )
    workflow.add_node(
        "generate",
        nodes.generate
        or generate_factory(_require_settings(settings, "generate"), question_resolver),
    )

    if mode == "chat":
        workflow.add_edge(START, "condense")
        workflow.add_edge("condense", "agent")
    else:
        workflow.add_edge(START, "agent")

    workflow.add_conditional_edges("agent", route_after_agent, AGENT_EDGE_MAP)
    workflow.add_conditional_edges(
        "retrieve",
        nodes.grade_documents
        or grade_documents_factory(
            _require_settings(settings, "grade_documents"),
            question_resolver,
        ),
        GRADE_EDGE_MAP,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    resolved_checkpointer = _resolve_checkpointer(mode, providers, checkpointer)
    if resolved_checkpointer is None:
        return workflow.compile()
    return workflow.compile(checkpointer=resolved_checkpointer)


def _resolve_tools(
    settings: Settings | None,
    providers: GraphProviders,
    rebuild_vectorstore: bool,
) -> list[Any]:
    if providers.tools is not None:
        return list(providers.tools)

    if settings is None:
        return []

    from ..core.retriever import build_retriever_tool

    return [build_retriever_tool(settings, rebuild=rebuild_vectorstore)]


def _resolve_checkpointer(
    mode: GraphMode,
    providers: GraphProviders,
    checkpointer: Any,
) -> Any:
    if checkpointer is not _DEFAULT_CHECKPOINTER:
        return checkpointer
    if providers.checkpointer is not _DEFAULT_CHECKPOINTER:
        return providers.checkpointer
    if mode == "chat":
        return build_memory_saver()
    return None


def _require_settings(settings: Settings | None, dependency: str) -> Settings:
    if settings is None:
        raise ValueError(
            f"settings are required for the default {dependency} node; "
            "provide settings or a GraphNodeOverrides replacement"
        )
    return settings


__all__ = [
    "GraphMode",
    "GraphNodeOverrides",
    "GraphProviders",
    "build_graph",
    "build_memory_saver",
]
