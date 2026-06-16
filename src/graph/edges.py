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

# After the lightweight graph's tool node executes, route based on which tool
# the agent actually called. Only ``live_web_search`` produces a list of URLs
# meant to ground a final answer through ``web_answer``. Every other tool
# (weather, stock quotes, currency conversion, Wikipedia, etc.) returns a
# structured textual result that the agent should see and synthesize from --
# routing those through ``web_answer`` would discard the tool output and
# re-prompt the LLM against the session's curated source URLs.
LIGHTWEIGHT_TOOL_EDGE_MAP: dict[Hashable, str] = {
    "web_answer": "web_answer",
    "agent": "agent",
}

WEB_SEARCH_TOOL_NAME = "live_web_search"

# REFACTOR: After ``web_answer`` runs, decide whether to terminate the
# lightweight graph (the fetched pages grounded a real answer) or to fall
# back to the agent for one retry when no readable page content was found.
# This prevents the previous one-way-trip failure mode where the user saw a
# hard "I couldn't retrieve readable content" refusal even for stable
# historical questions whose answer the LLM already knew.
WEB_ANSWER_EDGE_MAP: dict[Hashable, str] = {
    "agent": "agent",
    END: END,
}

# REFACTOR: Cap the lightweight web-search fallback loop at this many
# ``web_answer`` runs. A second failure terminates the graph with the
# grounded refusal rather than spinning forever.
WEB_ANSWER_FALLBACK_MAX_ATTEMPTS = 2


def route_after_agent(state: Any) -> str:
    """Route after the agent node using LangGraph's built-in tool condition."""

    return tools_condition(state)


def route_after_lightweight_tool(state: Any) -> str:
    """Route after the lightweight graph's tool node based on the called tool.

    Returns ``"web_answer"`` when the most recent tool message came from
    ``live_web_search`` (URLs to fetch and ground against), and ``"agent"``
    for every other tool so the agent can synthesize a final answer from the
    tool's structured output.
    """

    messages = _state_messages(state)
    last_tool_name = _last_tool_message_name(messages)
    if last_tool_name == WEB_SEARCH_TOOL_NAME:
        return "web_answer"
    return "agent"


def route_after_web_answer(state: Any) -> str:
    """Route after ``web_answer``: retry the agent once, then terminate.

    When ``web_answer`` produced no readable page content, the LLM
    unnecessarily called ``live_web_search`` (e.g. for a stable historical
    fact it already knew). Looping back to the agent lets it synthesize a
    final answer from its own knowledge with a "I couldn't verify against
    the live web" caveat, instead of the user seeing the hard refusal.
    The retry is bounded to ``WEB_ANSWER_FALLBACK_MAX_ATTEMPTS`` to prevent
    an infinite loop when the second web_answer run also fails.
    """

    if not _state_bool(state, "web_answer_no_readable_content"):
        return END
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
