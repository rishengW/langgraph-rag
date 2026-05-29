"""Chat-flavored graph state.

The single-shot ``AgentState`` in ``src.core.state`` is keyed only on
``messages`` and ``rewrite_count`` because the user's question is always
``messages[0]``. In a multi-turn chat that assumption breaks: by the third
turn ``messages[0]`` is still the *first* user turn, not the current one.

We add two fields:

* ``current_question`` — the standalone form of the user's latest turn,
  produced by the condense node. The agent and retriever read this so
  they don't have to re-derive it from the message history.
* ``current_question_index`` — the index in ``messages`` of the user
  message that started this turn. Lets nodes that need the *raw* user
  text (vs. the condensed form) find it without scanning back through
  AI messages and tool calls.
"""

from __future__ import annotations

from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ChatState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rewrite_count: int
    current_question: str
    current_question_index: int
