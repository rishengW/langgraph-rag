from __future__ import annotations

from collections.abc import Hashable
from typing import Any

from langgraph.graph import END
from langgraph.prebuilt import tools_condition

AGENT_EDGE_MAP: dict[Hashable, str] = {
    "tools": "retrieve",
    END: END,
}

GRADE_EDGE_MAP: dict[Hashable, str] = {
    "generate": "generate",
    "rewrite": "rewrite",
}

# REFACTOR: After the lightweight graph's tool node executes, route
# based on which tool the agent actually called. ``live_web_search``
# results feed into the merge -> web_answer chain (the merge node
# deduplicates and ranks the search URLs before grounding the answer).
# Every other tool (weather, stock quotes, currency conversion,
# Wikipedia, etc.) returns a structured textual result that the agent
# should see and synthesize from -- routing those through merge /
# web_answer would discard the tool output and re-prompt the LLM
# against the session's curated source URLs.
LIGHTWEIGHT_TOOL_EDGE_MAP: dict[Hashable, str] = {
    "merge": "merge",
    "agent": "agent",
}

WEB_SEARCH_TOOL_NAME = "live_web_search"

# REFACTOR: After ``web_answer`` runs, decide whether to terminate the
# lightweight graph (the fetched pages grounded a real answer), retry via
# conditional expansion (one-shot decompose -> N x k search -> merge ->
# ``web_answer``). After that bounded retry, zero grounded sources terminate
# with the web-answer refusal instead of falling back to model knowledge.
WEB_ANSWER_EDGE_MAP: dict[Hashable, str] = {
    "expand": "expand",
    "fallback_answer": "fallback_answer",
    "answer_self_critique": "answer_self_critique",
    END: END,
}

# Defensive ceiling for externally supplied or stale graph state. Normal
# execution runs web_answer at most twice: the initial search and one expanded
# retry, followed by the fixed tool-free fallback edge.
WEB_ANSWER_FALLBACK_MAX_ATTEMPTS = 2


def route_after_agent(state: Any) -> str:
    """Route after the agent node using LangGraph's built-in tool condition."""

    return tools_condition(state)


def route_after_agent_with_critique(
    state: Any, *, critique_enabled: bool = False
) -> str:
    """Critique a direct agent answer without intercepting tool calls."""

    result = route_after_agent(state)
    if critique_enabled and result == END:
        return "answer_self_critique"
    return result


def route_after_lightweight_agent(state: Any) -> str:
    """Send pure web-search calls through fan-out and other tools to ToolNode.

    Agents can technically emit multiple tool calls in one message. Fan-out is
    only selected for one ``live_web_search`` call; multi-call batches stay on
    the generic ToolNode path so every call receives its required response.
    """

    route = tools_condition(state)
    if route != "tools":
        return route
    names = _last_ai_tool_call_names(_state_messages(state))
    if names == [WEB_SEARCH_TOOL_NAME]:
        return "decompose"
    return "web_search"


def route_after_lightweight_agent_with_critique(
    state: Any, *, critique_enabled: bool = False
) -> str:
    """Preserve lightweight tool routing and critique only direct answers."""

    result = route_after_lightweight_agent(state)
    if critique_enabled and result == END:
        return "answer_self_critique"
    return result


def route_after_lightweight_tool(state: Any) -> str:
    """Route after the lightweight graph's tool node based on the called tool.

    Returns ``"merge"`` when the most recent tool message came from
    ``live_web_search`` (URLs to deduplicate, rank, and ground against
    in the merge -> web_answer chain), and ``"agent"`` for every other
    tool so the agent can synthesize a final answer from the tool's
    structured output.
    """

    messages = _state_messages(state)
    last_tool_name = _last_tool_message_name(messages)
    called_tool_names = _last_ai_tool_call_names(messages)
    if (
        last_tool_name == WEB_SEARCH_TOOL_NAME
        and called_tool_names
        and all(name == WEB_SEARCH_TOOL_NAME for name in called_tool_names)
    ):
        return "merge"
    return "agent"


def route_after_web_answer(state: Any) -> str:
    """Route after ``web_answer``: one expansion retry, then end.

    Three outcomes drive the post-``web_answer`` conditional edge:

    - ``web_answer`` produced readable page content -> ``END`` (the
      grounded answer is the final user-visible reply).
    - ``web_answer`` produced no readable content AND
      ``expansion_attempted`` is False -> ``"expand"``: the graph will
      go through ``expand -> search_queries -> merge -> web_answer`` to
      retry the grounded answer with N x k rewritten queries.
    - ``web_answer`` produced no readable content AND
      ``expansion_attempted`` is True -> ``END``. Web-search mode must never
      replace missing evidence with an answer from model training data.
    """

    if not _state_bool(state, "web_answer_no_readable_content"):
        return END
    if not _state_bool(state, "expansion_attempted"):
        return "expand"
    return END


def route_after_web_answer_with_fallback(
    state: Any,
    *,
    planning_enabled: bool = False,
) -> str:
    """Route web answers with the optional fallback/critique stages.

    ``route_after_web_answer`` keeps its historical return values for callers
    and tests.  The builder uses this opt-in wrapper so a failed expanded
    search invokes ``fallback_answer`` instead of silently terminating.
    """

    result = route_after_web_answer(state)
    if result == "expand":
        return result
    if _state_bool(state, "web_answer_no_readable_content") and _state_bool(
        state, "expansion_attempted"
    ):
        return "fallback_answer"
    if planning_enabled:
        return "answer_self_critique"
    return result


def _state_messages(state: Any) -> list[Any]:
    if isinstance(state, dict):
        messages = state.get("messages")
    else:
        messages = getattr(state, "messages", None)
    if isinstance(messages, list):
        return messages
    if messages is None:
        return []
    try:
        return list(messages)
    except TypeError:
        return []


def _state_bool(state: Any, key: str) -> bool:
    """Read a boolean from graph state, tolerating dict or attribute shapes."""

    value = state.get(key) if isinstance(state, dict) else getattr(state, key, None)
    return bool(value)


def _state_int(state: Any, key: str) -> int:
    """Read an int from graph state, tolerating dict or attribute shapes."""

    value = state.get(key) if isinstance(state, dict) else getattr(state, key, None)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _last_tool_message_name(messages: list[Any]) -> str | None:
    for message in reversed(messages):
        if _message_role(message) != "tool":
            continue
        name = getattr(message, "name", None)
        return name if isinstance(name, str) else None
    return None


def _last_ai_tool_call_names(messages: list[Any]) -> list[str]:
    for message in reversed(messages):
        calls = getattr(message, "tool_calls", None)
        if not calls:
            continue
        return [
            str(call.get("name")) for call in calls if isinstance(call, dict) and call.get("name")
        ]
    return []


def _message_role(message: Any) -> str:
    role = getattr(message, "type", None)
    if role:
        return str(role)
    return str(message.__class__.__name__).lower()


__all__ = [
    "AGENT_EDGE_MAP",
    "GRADE_EDGE_MAP",
    "LIGHTWEIGHT_TOOL_EDGE_MAP",
    "WEB_ANSWER_EDGE_MAP",
    "WEB_ANSWER_FALLBACK_MAX_ATTEMPTS",
    "WEB_SEARCH_TOOL_NAME",
    "route_after_agent",
    "route_after_agent_with_critique",
    "route_after_lightweight_agent",
    "route_after_lightweight_agent_with_critique",
    "route_after_lightweight_tool",
    "route_after_web_answer",
    "route_after_web_answer_with_fallback",
]
