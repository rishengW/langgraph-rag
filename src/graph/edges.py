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


def route_after_agent(state: Any) -> str:
    """Route after the agent node using LangGraph's built-in tool condition."""

    return tools_condition(state)


__all__ = ["AGENT_EDGE_MAP", "GRADE_EDGE_MAP", "route_after_agent"]
