\
from __future__ import annotations

from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared graph state.

    `add_messages` appends new messages instead of replacing the existing list.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
