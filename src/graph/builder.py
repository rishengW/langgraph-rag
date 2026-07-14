from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from ..config import Settings
from .edges import (
    AGENT_EDGE_MAP,
    GRADE_EDGE_MAP,
    LIGHTWEIGHT_TOOL_EDGE_MAP,
    WEB_ANSWER_EDGE_MAP,
    route_after_agent,
    route_after_lightweight_agent,
    route_after_lightweight_tool,
    route_after_web_answer,
)
from .nodes import (
    agent_factory,
    chat_question_resolver,
    condense_question_factory,
    decompose_factory,
    expand_factory,
    fallback_answer_factory,
    generate_factory,
    grade_documents_factory,
    merge_factory,
    qa_question_resolver,
    rewrite_factory,
    search_queries_factory,
    web_answer_factory,
)
from .state import AgentState, ChatState

GraphMode = Literal["qa", "chat"]
NodeCallable = Callable[[dict[str, Any]], dict[str, Any]]
GradeEdgeCallable = Callable[[dict[str, Any]], Literal["generate", "rewrite"]]

# REFACTOR: Conditional-expansion routing. The agent's tool call now
# routes to ``decompose`` instead of ``web_search`` so compound questions
# can be split into sub-questions before the search fan-out. When the
# agent answers directly (e.g. for arithmetic, common knowledge, or
# chitchat where the system prompt steers it away from tools), the
# graph terminates immediately so the direct answer is preserved
# instead of being overridden by web_answer re-prompting against
# irrelevant fetched pages.
LIGHTWEIGHT_AGENT_EDGE_MAP: dict[Hashable, str] = {
    "decompose": "decompose",
    "web_search": "web_search",
    END: END,
}

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
    web_answer: NodeCallable | None = None
    fallback_answer: NodeCallable | None = None
    # REFACTOR: Conditional-expansion nodes. Overridable so tests can
    # inject deterministic fakes for ``decompose``, ``expand`` and
    # ``merge`` without running the real LLM-backed implementations.
    decompose: NodeCallable | None = None
    expand: NodeCallable | None = None
    merge: NodeCallable | None = None


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


def build_memory_saver() -> Any:
    """Create the default in-memory LangGraph checkpointer for chat mode."""

    try:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    except ImportError:  # pragma: no cover - compatibility with older LangGraph
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()


def build_graph(
    mode: GraphMode = "qa",
    *,
    settings: Settings | None = None,
    providers: GraphProviders | None = None,
    rebuild_vectorstore: bool = False,
    checkpointer: Any = _DEFAULT_CHECKPOINTER,
) -> Any:
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

    workflow = cast(Any, StateGraph(state_type))

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


def build_lightweight_graph(
    settings: Settings | None = None,
    *,
    mode: GraphMode = "qa",
    providers: GraphProviders | None = None,
    checkpointer: Any = _DEFAULT_CHECKPOINTER,
) -> Any:
    """Compile the lightweight graph for one-shot web-search sources.

    This graph deliberately skips the Chroma/retriever/grade/rewrite path. It
    only gives the agent the live web-search tool, then sends either the tool
    output or the existing state source URLs to ``web_answer``.
    """

    if mode not in ("qa", "chat"):
        raise ValueError("mode must be 'qa' or 'chat'")

    providers = providers or GraphProviders()
    nodes = providers.nodes
    tools = _resolve_lightweight_tools(settings, providers)
    question_resolver = qa_question_resolver if mode == "qa" else chat_question_resolver
    state_type = AgentState if mode == "qa" else ChatState

    workflow = cast(Any, StateGraph(state_type))
    workflow.add_node(
        "agent",
        nodes.agent
        or agent_factory(
            _require_settings(settings, "lightweight agent"),
            tools,
            question_resolver,
        ),
    )

    if not tools:
        raise ValueError("providers.tools or settings are required for web_search")
    workflow.add_node("web_search", ToolNode(list(tools)))
    workflow.add_node("search_queries", search_queries_factory(tools))
    # Conditional-expansion nodes. The decompose -> search_queries -> merge
    # -> web_answer chain is the first-attempt path; the expand node re-enters
    # search_queries when the first web_answer run produced
    # no readable content (post-web_answer edge -> "expand"). After
    # expansion, search_queries -> merge -> web_answer runs a second time
    # with the combined first-attempt + expanded URLs. State field
    # ``expansion_attempted`` is set to True by ``expand`` so the
    # post-web_answer edge does not loop back to expand a second time.
    workflow.add_node(
        "decompose",
        nodes.decompose
        or decompose_factory(
            _require_settings(settings, "decompose"),
            question_resolver,
        ),
    )
    workflow.add_node(
        "expand",
        nodes.expand
        or expand_factory(
            _require_settings(settings, "expand"),
            question_resolver,
        ),
    )
    workflow.add_node(
        "merge",
        nodes.merge
        or merge_factory(
            _require_settings(settings, "merge"),
            question_resolver,
        ),
    )
    workflow.add_node(
        "web_answer",
        nodes.web_answer
        or web_answer_factory(
            _require_settings(settings, "web_answer"),
            question_resolver,
        ),
    )
    workflow.add_node(
        "fallback_answer",
        nodes.fallback_answer
        or fallback_answer_factory(
            _require_settings(settings, "fallback answer"),
            question_resolver,
        ),
    )

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        route_after_lightweight_agent,
        LIGHTWEIGHT_AGENT_EDGE_MAP,
    )
    # REFACTOR: Deterministic edges for the conditional-expansion path.
    # ``decompose -> search_queries`` executes the bounded first query batch.
    # ``search_queries -> merge -> web_answer`` collapses the first-attempt
    # URLs into ``source_urls`` and grounds the answer. ``expand ->
    # search_queries`` re-enters the search fan-out after a first-attempt
    # failure (the post-web_answer edge routes to "expand" when no
    # readable content was found and expansion has not yet been
    # attempted). The non-web-search tool case is still handled by the
    # ``web_search`` conditional edge (route_after_lightweight_tool
    # returns "agent" for non-live_web_search tools, preserving the
    # existing regression test for weather/stock/currency/wikipedia).
    workflow.add_edge("decompose", "search_queries")
    workflow.add_edge("expand", "search_queries")
    workflow.add_edge("search_queries", "merge")
    workflow.add_edge("merge", "web_answer")
    workflow.add_conditional_edges(
        "web_search",
        route_after_lightweight_tool,
        LIGHTWEIGHT_TOOL_EDGE_MAP,
    )
    workflow.add_conditional_edges(
        "web_answer",
        route_after_web_answer,
        WEB_ANSWER_EDGE_MAP,
    )
    workflow.add_edge("fallback_answer", END)

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

    from ..core.retriever import build_retriever_tool as build_retriever_tool
    from ..web_search import build_web_search_tool as build_web_search_tool

    # REFACTOR: Default settings-based graph tools now include live web search.
    tool_module = cast(Any, import_module("..tools", package=__package__))
    tools = [build_retriever_tool(settings, rebuild=rebuild_vectorstore)]
    if settings.web_search_enabled:
        tools.append(build_web_search_tool(settings))
    if settings.weather_enabled:
        tools.append(tool_module.build_weather_tool(settings))
    if settings.stock_enabled:
        tools.append(tool_module.build_stock_tool(settings))
    if settings.currency_enabled:
        tools.append(tool_module.build_currency_tool(settings))
    if settings.wikipedia_enabled:
        tools.append(tool_module.build_wikipedia_tool(settings))
    if settings.directions_enabled:
        tools.append(tool_module.build_directions_tool(settings))
    if settings.map_enabled:
        tools.append(tool_module.build_map_tool(settings))
    if settings.math_enabled:
        tools.append(tool_module.build_math_tool(settings))
    if settings.statistics_enabled:
        tools.append(tool_module.build_statistics_tool(settings))
    if settings.linalg_enabled:
        tools.append(tool_module.build_linalg_tool(settings))
    if settings.number_theory_enabled:
        tools.append(tool_module.build_number_theory_tool(settings))
    if settings.datetime_enabled:
        tools.append(tool_module.build_datetime_tool(settings))
    if settings.summarize_url_enabled:
        tools.append(tool_module.build_summarize_url_tool(settings))
    if settings.file_read_enabled:
        tools.append(tool_module.build_text_file_tool(settings))
        tools.append(tool_module.build_word_tool(settings))
        tools.append(tool_module.build_excel_tool(settings))
        tools.append(tool_module.build_pdf_tool(settings))
    return tools


def _resolve_lightweight_tools(
    settings: Settings | None,
    providers: GraphProviders,
) -> list[Any]:
    if providers.tools is not None:
        return list(providers.tools)

    if settings is None:
        return []

    from ..web_search import build_web_search_tool as build_web_search_tool

    tool_module = cast(Any, import_module("..tools", package=__package__))
    tools: list[Any] = [build_web_search_tool(settings)]
    if settings.weather_enabled:
        tools.append(tool_module.build_weather_tool(settings))
    if settings.stock_enabled:
        tools.append(tool_module.build_stock_tool(settings))
    if settings.currency_enabled:
        tools.append(tool_module.build_currency_tool(settings))
    if settings.wikipedia_enabled:
        tools.append(tool_module.build_wikipedia_tool(settings))
    if settings.directions_enabled:
        tools.append(tool_module.build_directions_tool(settings))
    if settings.map_enabled:
        tools.append(tool_module.build_map_tool(settings))
    if settings.math_enabled:
        tools.append(tool_module.build_math_tool(settings))
    if settings.statistics_enabled:
        tools.append(tool_module.build_statistics_tool(settings))
    if settings.linalg_enabled:
        tools.append(tool_module.build_linalg_tool(settings))
    if settings.number_theory_enabled:
        tools.append(tool_module.build_number_theory_tool(settings))
    if settings.datetime_enabled:
        tools.append(tool_module.build_datetime_tool(settings))
    if settings.summarize_url_enabled:
        tools.append(tool_module.build_summarize_url_tool(settings))
    if settings.file_read_enabled:
        tools.append(tool_module.build_text_file_tool(settings))
        tools.append(tool_module.build_word_tool(settings))
        tools.append(tool_module.build_excel_tool(settings))
        tools.append(tool_module.build_pdf_tool(settings))
    return tools


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
    "build_lightweight_graph",
    "build_memory_saver",
]
