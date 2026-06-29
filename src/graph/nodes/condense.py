from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser

from ...config import Settings
from ...llm.prompts import CONDENSE_PROMPT
from ...utils.retry import invoke_with_retry
from .common import new_chat_model

logger = logging.getLogger(__name__)


def format_history(messages: Sequence[BaseMessage]) -> str:
    """Render prior human/assistant turns as a compact transcript."""

    lines: list[str] = []
    for msg in messages:
        role = getattr(msg, "type", None) or msg.__class__.__name__.lower()
        content = getattr(msg, "content", None)
        if not content:
            continue
        if role in ("tool", "function"):
            continue
        if role.startswith("human") or role == "user":
            lines.append(f"User: {content}")
        elif role.startswith("ai") or role == "assistant":
            lines.append(f"Assistant: {content}")
    return "\n".join(lines) if lines else "(no prior turns)"


def latest_user_index(messages: Sequence[BaseMessage]) -> int:
    """Return the index of the most recent human/user message."""

    for index in range(len(messages) - 1, -1, -1):
        msg = messages[index]
        role = getattr(msg, "type", None) or msg.__class__.__name__.lower()
        if role.startswith("human") or role == "user":
            return index
    return 0


def condense_followup_question(
    messages: Sequence[BaseMessage],
    latest_text: str,
    settings: Settings,
) -> str:
    """Rewrite a follow-up turn into a standalone question using prior turns.

    Shared by the in-graph ``condense`` node and the chat web-search refresh
    step so both contextualize a vague follow-up (e.g. "Argentina and Jordan",
    "group stage not knockout") against the conversation history before it
    drives a search. Returns ``latest_text`` unchanged when there is no prior
    history or the LLM call fails (passthrough fallback).
    """

    history_messages = [
        msg
        for msg in messages
        if (getattr(msg, "content", None) or "").strip()
        and (getattr(msg, "type", "") or "").lower() not in ("tool", "function")
    ]
    if not history_messages:
        return latest_text

    history_text = format_history(history_messages)
    if history_text == "(no prior turns)":
        return latest_text

    dated_prompt = CONDENSE_PROMPT.partial(current_date=date.today().isoformat())
    chain = dated_prompt | new_chat_model(settings) | StrOutputParser()
    try:
        standalone = invoke_with_retry(
            chain,
            {"history": history_text, "question": latest_text},
            max_retries=settings.dashscope_max_retries,
        )
        return (standalone or "").strip() or latest_text
    except Exception as exc:
        logger.error("Follow-up condense error; using raw question: %s", exc)
        return latest_text


def condense_question_factory(settings: Settings) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Condense the latest chat turn into a standalone question."""

    def condense_question(state: dict[str, Any]) -> dict[str, Any]:
        logger.info("CONDENSE QUESTION")
        messages = list(state["messages"])
        if not messages:
            return {"current_question": "", "current_question_index": 0}

        latest_idx = latest_user_index(messages)
        latest_msg = messages[latest_idx]
        latest_text = getattr(latest_msg, "content", str(latest_msg))

        history = messages[:latest_idx]
        if not history:
            return {
                "current_question": latest_text,
                "current_question_index": latest_idx,
                "rewrite_count": 0,
            }

        standalone = condense_followup_question(history, latest_text, settings)
        logger.info("Condensed question: %r", standalone)
        return {
            "current_question": standalone,
            "current_question_index": latest_idx,
            "rewrite_count": 0,
        }

    return condense_question

