from __future__ import annotations

from typing import Any

from langgraph.graph import END
from langgraph.prebuilt import tools_condition

AGENT_EDGE_MAP = {
    "tools": "retrieve",
    END: END,
}

GRADE_EDGE_MAP = {
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
LIGHTWEIGHT_TOOL_EDGE_MAP = {
    "web_answer": "web_answer",
    "agent": "agent",
}

WEB_SEARCH_TOOL_NAME = "live_web_search"


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
    return message.__class__.__name__.lower()


__all__ = [
    "AGENT_EDGE_MAP",
    "GRADE_EDGE_MAP",
    "LIGHTWEIGHT_TOOL_EDGE_MAP",
    "WEB_SEARCH_TOOL_NAME",
    "route_after_agent",
    "route_after_lightweight_tool",
]
