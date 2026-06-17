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
# ``web_answer``), or fall back to the agent (synthesise from training data
# with a "I couldn't verify" caveat) when no readable page content was
# found AND expansion has already been attempted. The agent retry is
# bounded by ``WEB_ANSWER_FALLBACK_MAX_ATTEMPTS`` so a second failure
# terminates the graph with the grounded refusal rather than spinning
# forever.
WEB_ANSWER_EDGE_MAP: dict[Hashable, str] = {
    "agent": "agent",
    "expand": "expand",
    END: END,
}

# REFACTOR: Cap the lightweight web-search fallback loop at this many
# ``web_answer`` runs. A second failure routes to the agent fallback
# (when expansion has already been attempted) which can either
# synthesise from training data or call the tool once more; the third
# ``web_answer`` run terminates the graph with the grounded refusal
# rather than spinning forever. Bumped from 2 to 3 so the conditional
# expansion's agent fallback (third outcome of route_after_web_answer)
# is reachable after 2 web_answer failures -- with max=2 the check
# ``attempts < max`` would be ``2 < 2 == False`` and the agent would
# never get its one-shot synthesis opportunity.
WEB_ANSWER_FALLBACK_MAX_ATTEMPTS = 3


def route_after_agent(state: Any) -> str:
    """Route after the agent node using LangGraph's built-in tool condition."""

    return tools_condition(state)


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
    if last_tool_name == WEB_SEARCH_TOOL_NAME:
        return "merge"
    return "agent"


def route_after_web_answer(state: Any) -> str:
    """Route after ``web_answer``: expansion, agent retry, then end.

    Three outcomes drive the post-``web_answer`` conditional edge:

    - ``web_answer`` produced readable page content -> ``END`` (the
      grounded answer is the final user-visible reply).
    - ``web_answer`` produced no readable content AND
      ``expansion_attempted`` is False -> ``"expand"``: the graph will
      go through ``decompose -> web_search -> merge -> web_answer`` to
      retry the grounded answer with N x k rewritten queries.
    - ``web_answer`` produced no readable content AND
      ``expansion_attempted`` is True -> ``"agent"`` (if the bounded
      retry budget has not been exhausted) so the LLM can synthesise
      a final answer from its own knowledge with a "couldn't verify
      against the live web" caveat, or ``END`` once the bound is
      reached.
    """

    if not _state_bool(state, "web_answer_no_readable_content"):
        return END
    if not _state_bool(state, "expansion_attempted"):
        return "expand"
    attempts = _state_int(state, "web_answer_attempts")
    if attempts < WEB_ANSWER_FALLBACK_MAX_ATTEMPTS:
        return "agent"
    return END


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

    if isinstance(state, dict):
        value = state.get(key)
    else:
        value = getattr(state, key, None)
    return bool(value)


def _state_int(state: Any, key: str) -> int:
    """Read an int from graph state, tolerating dict or attribute shapes."""

    if isinstance(state, dict):
        value = state.get(key)
    else:
        value = getattr(state, key, None)
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
    "route_after_lightweight_tool",
    "route_after_web_answer",
]
