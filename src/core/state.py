from __future__ import annotations

from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Shared graph state.

    `add_messages` appends new messages instead of replacing the existing list.

    `rewrite_count` tracks how many times the query-rewriting node has run.
    The grader uses it to stop the agent → retrieve → grade → rewrite loop
    after a configurable number of attempts so a stubborn grader or a flaky
    API cannot trap the graph in an infinite cycle.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    rewrite_count: int
