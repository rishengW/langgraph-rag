from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.config import Settings

from ..mcp import (
    InjectedToolProvider,
    ToolCatalogSnapshot,
    ToolExecutionPipeline,
    ToolPolicy,
    compose_snapshot,
    default_provider_entries,
    default_provider_tools,
    validate_snapshot,
)
from .edges import (
    AGENT_EDGE_MAP,
    GRADE_EDGE_MAP,
    LIGHTWEIGHT_TOOL_EDGE_MAP,
    WEB_ANSWER_EDGE_MAP,
    route_after_agent_with_critique,
    route_after_lightweight_agent_with_critique,
    route_after_lightweight_tool,
    route_after_web_answer_with_fallback,
)
from .nodes import (
    agent_factory,
    answer_self_critique_node,
    chat_question_resolver,
    condense_question_factory,
    decompose_factory,
    expand_factory,
    fallback_answer_factory,
    generate_factory,
    grade_documents_factory,
    merge_factory,
    planner_node,
    reflection_revise_node,
    rewrite_factory,
    route_after_self_critique,
    route_after_subgoal_aggregation,
    route_subgoals,
    search_queries_factory,
    subgoal_aggregator_node,
    subgoal_dispatcher_node,
    subgoal_worker_node,
    web_answer_factory,
)
from .state import ChatState

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
    planner: NodeCallable | None = None
    subgoal_dispatcher: NodeCallable | None = None
    subgoal_worker: NodeCallable | None = None
    subgoal_aggregator: NodeCallable | None = None
    answer_self_critique: NodeCallable | None = None
    reflection_revise: NodeCallable | None = None


@dataclass(frozen=True)
class GraphProviders:
    """Dependency injection hooks for graph construction.

    Supplying ``tools`` skips the Settings-based Chroma retriever tool build.
    Supplying node overrides skips the Settings-based node factories for those
    nodes. Existing application entry points can continue passing only Settings.
    """

    tools: Sequence[Any] | None = None
    catalog_snapshot: ToolCatalogSnapshot | None = None
    tool_policy: ToolPolicy | None = None
    tool_pipeline: ToolExecutionPipeline | None = None
    catalog_generation: int = 1
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
    *,
    settings: Settings | None = None,
    providers: GraphProviders | None = None,
    rebuild_vectorstore: bool = False,
    checkpointer: Any = _DEFAULT_CHECKPOINTER,
    session_root: Path | None = None,
    thread_id: str = "",
) -> Any:
    """Compile the chat LangGraph workflow.

    Defaults preserve the legacy Settings-based builders. ``providers`` is an
    additive DI surface used by tests and future API wiring to avoid live
    provider construction.

    ``session_root`` and ``thread_id`` scope session-bound document editing
    tools to one chat session's upload directory. Omit them for a stateless
    single-shot graph (no session editors, no checkpointer).
    """

    providers = providers or GraphProviders()
    nodes = providers.nodes
    edit_root = session_root
    edit_thread_id = thread_id
    snapshot = _resolve_catalog_snapshot(
        settings,
        providers,
        lightweight=False,
        rebuild_vectorstore=rebuild_vectorstore,
        session_root=edit_root,
        thread_id=edit_thread_id,
    )
    tools = snapshot.tools
    question_resolver = chat_question_resolver
    state_type = ChatState

    workflow = cast(Any, StateGraph(state_type))

    planning_enabled = bool(getattr(settings, "planning_enabled", False))

    workflow.add_node(
        "condense",
        nodes.condense or condense_question_factory(_require_settings(settings, "condense")),
    )

    if planning_enabled:
        workflow.add_node(
            "planner",
            nodes.planner
            or planner_node(
                _require_settings(settings, "planner"),
                question_resolver,
                max_subgoals=getattr(settings, "planning_max_subgoals", 4),
            ),
        )
        workflow.add_node(
            "subgoal_dispatcher",
            nodes.subgoal_dispatcher
            or (
                lambda state: subgoal_dispatcher_node(
                    state,
                    max_dispatch=getattr(settings, "planning_max_subgoals", 4),
                )
            ),
        )
        workflow.add_node(
            "execute_subgoal",
            nodes.subgoal_worker
            or subgoal_worker_node(
                _require_settings(settings, "sub-goal worker"), question_resolver
            ),
        )
        workflow.add_node(
            "subgoal_aggregator",
            nodes.subgoal_aggregator or subgoal_aggregator_node,
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
        workflow.add_node("retrieve", ToolNode(tools))

    workflow.add_node(
        "rewrite",
        nodes.rewrite
        or rewrite_factory(
            _require_settings(settings, "rewrite"),
            question_resolver,
            update_current_question=True,
        ),
    )
    workflow.add_node(
        "generate",
        nodes.generate
        or generate_factory(_require_settings(settings, "generate"), question_resolver),
    )

    workflow.add_edge(START, "condense")
    workflow.add_edge("condense", "planner" if planning_enabled else "agent")

    if planning_enabled:
        workflow.add_edge("planner", "subgoal_dispatcher")
        workflow.add_conditional_edges(
            "subgoal_dispatcher",
            lambda state: route_subgoals(state, target="execute_subgoal"),
        )
        workflow.add_edge("execute_subgoal", "subgoal_aggregator")
        workflow.add_conditional_edges(
            "subgoal_aggregator",
            route_after_subgoal_aggregation,
            {
                "subgoal_dispatcher": "subgoal_dispatcher",
                "agent": "agent",
            },
        )

    workflow.add_conditional_edges(
        "agent",
        lambda state: route_after_agent_with_critique(state, critique_enabled=planning_enabled),
        cast(
            dict[Hashable, str],
            {
                **AGENT_EDGE_MAP,
                **({"answer_self_critique": "answer_self_critique"} if planning_enabled else {}),
            },
        ),
    )
    workflow.add_conditional_edges(
        "retrieve",
        nodes.grade_documents
        or grade_documents_factory(
            _require_settings(settings, "grade_documents"),
            question_resolver,
        ),
        GRADE_EDGE_MAP,
    )
    if planning_enabled:
        workflow.add_node(
            "answer_self_critique",
            nodes.answer_self_critique
            or answer_self_critique_node(
                _require_settings(settings, "answer self-critique"), question_resolver
            ),
        )
        workflow.add_node(
            "reflection_revise",
            nodes.reflection_revise
            or reflection_revise_node(
                _require_settings(settings, "reflection revision"),
                question_resolver,
                max_retries=getattr(settings, "planning_max_reflection_retries", 1),
            ),
        )
        workflow.add_edge("generate", "answer_self_critique")
        workflow.add_conditional_edges(
            "answer_self_critique",
            lambda state: route_after_self_critique(
                state,
                max_retries=getattr(settings, "planning_max_reflection_retries", 1),
                threshold=getattr(settings, "planning_critic_threshold", 0.7),
            ),
            {"reflection_revise": "reflection_revise", END: END},
        )
        workflow.add_edge("reflection_revise", "answer_self_critique")
    else:
        workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    resolved_checkpointer = _resolve_checkpointer(providers, checkpointer)
    return _compile_with_catalog(workflow, resolved_checkpointer, snapshot)


def build_lightweight_graph(
    settings: Settings | None = None,
    *,
    providers: GraphProviders | None = None,
    checkpointer: Any = _DEFAULT_CHECKPOINTER,
    session_root: Path | None = None,
    thread_id: str = "",
) -> Any:
    """Compile the lightweight graph for one-shot web-search sources.

    This graph deliberately skips the Chroma/retriever/grade/rewrite path. It
    only gives the agent the live web-search tool, then sends either the tool
    output or the existing state source URLs to ``web_answer``.

    ``session_root`` and ``thread_id`` scope session-bound document editing
    tools to one chat session's upload directory. Omit them for a stateless
    single-shot graph.
    """

    providers = providers or GraphProviders()
    nodes = providers.nodes
    snapshot = _resolve_catalog_snapshot(
        settings,
        providers,
        lightweight=True,
        rebuild_vectorstore=False,
        session_root=session_root,
        thread_id=thread_id,
    )
    tools = snapshot.tools
    question_resolver = chat_question_resolver
    state_type = ChatState

    workflow = cast(Any, StateGraph(state_type))
    planning_enabled = bool(getattr(settings, "planning_enabled", False))

    if planning_enabled:
        workflow.add_node(
            "planner",
            nodes.planner
            or planner_node(
                _require_settings(settings, "lightweight planner"),
                question_resolver,
                max_subgoals=getattr(settings, "planning_max_subgoals", 4),
            ),
        )
        workflow.add_node(
            "subgoal_dispatcher",
            nodes.subgoal_dispatcher
            or (
                lambda state: subgoal_dispatcher_node(
                    state,
                    max_dispatch=getattr(settings, "planning_max_subgoals", 4),
                )
            ),
        )
        workflow.add_node(
            "execute_subgoal",
            nodes.subgoal_worker
            or subgoal_worker_node(
                _require_settings(settings, "lightweight sub-goal worker"),
                question_resolver,
            ),
        )
        workflow.add_node(
            "subgoal_aggregator",
            nodes.subgoal_aggregator or subgoal_aggregator_node,
        )

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
    workflow.add_node("web_search", ToolNode(tools))
    # Newer LangGraph releases reject a node name that is also a state key.
    # Keep ``search_queries`` as the public state field and give the internal
    # execution node a distinct name.
    workflow.add_node("execute_search_queries", search_queries_factory(tools))
    # Conditional-expansion nodes. The decompose -> execute_search_queries -> merge
    # -> web_answer chain is the first-attempt path; the expand node re-enters
    # execute_search_queries when the first web_answer run produced
    # no readable content (post-web_answer edge -> "expand"). After
    # expansion, execute_search_queries -> merge -> web_answer runs a second time
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
            _require_settings(settings, "fallback_answer"),
            question_resolver,
        ),
    )
    if planning_enabled:
        workflow.add_node(
            "answer_self_critique",
            nodes.answer_self_critique
            or answer_self_critique_node(
                _require_settings(settings, "lightweight answer self-critique"),
                question_resolver,
            ),
        )
        workflow.add_node(
            "reflection_revise",
            nodes.reflection_revise
            or reflection_revise_node(
                _require_settings(settings, "lightweight reflection revision"),
                question_resolver,
                max_retries=getattr(settings, "planning_max_reflection_retries", 1),
            ),
        )

    workflow.add_edge(START, "planner" if planning_enabled else "agent")
    if planning_enabled:
        workflow.add_edge("planner", "subgoal_dispatcher")
        workflow.add_conditional_edges(
            "subgoal_dispatcher",
            lambda state: route_subgoals(state, target="execute_subgoal"),
        )
        workflow.add_edge("execute_subgoal", "subgoal_aggregator")
        workflow.add_conditional_edges(
            "subgoal_aggregator",
            route_after_subgoal_aggregation,
            {
                "subgoal_dispatcher": "subgoal_dispatcher",
                "agent": "agent",
            },
        )

    workflow.add_conditional_edges(
        "agent",
        lambda state: route_after_lightweight_agent_with_critique(
            state, critique_enabled=planning_enabled
        ),
        cast(
            dict[Hashable, str],
            {
                **LIGHTWEIGHT_AGENT_EDGE_MAP,
                **({"answer_self_critique": "answer_self_critique"} if planning_enabled else {}),
            },
        ),
    )
    # REFACTOR: Deterministic edges for the conditional-expansion path.
    # ``decompose -> execute_search_queries`` executes the bounded first query batch.
    # ``execute_search_queries -> merge -> web_answer`` collapses the first-attempt
    # URLs into ``source_urls`` and grounds the answer. ``expand ->
    # execute_search_queries`` re-enters the search fan-out after a first-attempt
    # failure (the post-web_answer edge routes to "expand" when no
    # readable content was found and expansion has not yet been
    # attempted). The non-web-search tool case is still handled by the
    # ``web_search`` conditional edge (route_after_lightweight_tool
    # returns "agent" for non-live_web_search tools, preserving the
    # existing regression test for weather/stock/currency/wikipedia).
    workflow.add_edge("decompose", "execute_search_queries")
    workflow.add_edge("expand", "execute_search_queries")
    workflow.add_edge("execute_search_queries", "merge")
    workflow.add_edge("merge", "web_answer")
    workflow.add_conditional_edges(
        "web_search",
        route_after_lightweight_tool,
        LIGHTWEIGHT_TOOL_EDGE_MAP,
    )
    workflow.add_conditional_edges(
        "web_answer",
        lambda state: route_after_web_answer_with_fallback(
            state, planning_enabled=planning_enabled
        ),
        {
            key: target
            for key, target in WEB_ANSWER_EDGE_MAP.items()
            if planning_enabled or key != "answer_self_critique"
        },
    )
    workflow.add_edge("fallback_answer", END)
    if planning_enabled:
        workflow.add_conditional_edges(
            "answer_self_critique",
            lambda state: route_after_self_critique(
                state,
                max_retries=getattr(settings, "planning_max_reflection_retries", 1),
                threshold=getattr(settings, "planning_critic_threshold", 0.7),
            ),
            {"reflection_revise": "reflection_revise", END: END},
        )
        workflow.add_edge("reflection_revise", "answer_self_critique")

    resolved_checkpointer = _resolve_checkpointer(providers, checkpointer)
    return _compile_with_catalog(workflow, resolved_checkpointer, snapshot)


def _resolve_tools(
    settings: Settings | None,
    providers: GraphProviders,
    rebuild_vectorstore: bool,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> list[Any]:
    """Return raw heavy-path tools through the shared provider composition."""

    return _resolve_raw_tools(
        settings,
        providers,
        lightweight=False,
        rebuild_vectorstore=rebuild_vectorstore,
        session_root=session_root,
        thread_id=thread_id,
    )


def _resolve_lightweight_tools(
    settings: Settings | None,
    providers: GraphProviders,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> list[Any]:
    """Return raw lightweight tools through the shared provider composition."""

    return _resolve_raw_tools(
        settings,
        providers,
        lightweight=True,
        rebuild_vectorstore=False,
        session_root=session_root,
        thread_id=thread_id,
    )


def _resolve_raw_tools(
    settings: Settings | None,
    providers: GraphProviders,
    *,
    lightweight: bool,
    rebuild_vectorstore: bool,
    session_root: Path | None,
    thread_id: str,
) -> list[Any]:
    if providers.catalog_snapshot is not None:
        return list(providers.catalog_snapshot.tools)
    if providers.tools is not None:
        return list(providers.tools)
    if settings is None:
        return []
    return list(
        default_provider_tools(
            settings,
            lightweight=lightweight,
            rebuild_vectorstore=rebuild_vectorstore,
            session_root=session_root,
            thread_id=thread_id,
        )
    )


def _resolve_catalog_snapshot(
    settings: Settings | None,
    providers: GraphProviders,
    *,
    lightweight: bool,
    rebuild_vectorstore: bool,
    session_root: Path | None,
    thread_id: str,
) -> ToolCatalogSnapshot:
    pipeline = providers.tool_pipeline or ToolExecutionPipeline(policy=providers.tool_policy)
    if providers.catalog_snapshot is not None:
        validate_snapshot(providers.catalog_snapshot)
        entries = tuple(
            zip(
                providers.catalog_snapshot.tools,
                providers.catalog_snapshot.descriptors,
                strict=True,
            )
        )
        return compose_snapshot(
            entries,
            generation=providers.catalog_snapshot.generation,
            transform=pipeline.wrap,
        )
    if providers.tools is not None:
        injected = tuple(providers.tools)
        if not all(isinstance(tool, BaseTool) for tool in injected):
            raise ValueError("GraphProviders.tools must contain BaseTool instances")
        entries = InjectedToolProvider(cast(Sequence[BaseTool], injected)).entries()
    elif settings is not None:
        entries = default_provider_entries(
            settings,
            lightweight=lightweight,
            rebuild_vectorstore=rebuild_vectorstore,
            session_root=session_root,
            thread_id=thread_id,
        )
    else:
        entries = ()
    return compose_snapshot(
        entries,
        generation=providers.catalog_generation,
        transform=pipeline.wrap,
    )


def _compile_with_catalog(
    workflow: Any,
    checkpointer: Any,
    snapshot: ToolCatalogSnapshot,
) -> Any:
    compiled = (
        workflow.compile() if checkpointer is None else workflow.compile(checkpointer=checkpointer)
    )
    compiled.tool_catalog_snapshot = snapshot
    compiled.tool_catalog_generation = snapshot.generation
    compiled.tool_descriptors = snapshot.descriptors
    return compiled


def _resolve_checkpointer(
    providers: GraphProviders,
    checkpointer: Any,
) -> Any:
    if checkpointer is not _DEFAULT_CHECKPOINTER:
        return checkpointer
    if providers.checkpointer is not _DEFAULT_CHECKPOINTER:
        return providers.checkpointer
    return None


def _require_settings(settings: Settings | None, dependency: str) -> Settings:
    if settings is None:
        raise ValueError(
            f"settings are required for the default {dependency} node; "
            "provide settings or a GraphNodeOverrides replacement"
        )
    return settings


__all__ = [
    "GraphNodeOverrides",
    "GraphProviders",
    "build_graph",
    "build_lightweight_graph",
    "build_memory_saver",
]
