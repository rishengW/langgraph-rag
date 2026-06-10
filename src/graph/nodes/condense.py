from __future__ import annotations

import logging
from datetime import date

from langchain_core.output_parsers import StrOutputParser

from ...config import Settings
from ...llm.prompts import CONDENSE_PROMPT
from ...utils.retry import invoke_with_retry
from .common import new_chat_model

logger = logging.getLogger(__name__)


def format_history(messages) -> str:
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


def latest_user_index(messages) -> int:
    """Return the index of the most recent human/user message."""

    for index in range(len(messages) - 1, -1, -1):
        msg = messages[index]
        role = getattr(msg, "type", None) or msg.__class__.__name__.lower()
        if role.startswith("human") or role == "user":
            return index
    return 0


def condense_question_factory(settings: Settings):
    """Condense the latest chat turn into a standalone question."""

    def condense_question(state):
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

        history_text = format_history(history)
        # Bind today's date so the condenser doesn't rewrite "the latest" into
        # "the latest as of 2024" or otherwise inject a stale temporal anchor.
        dated_prompt = CONDENSE_PROMPT.partial(current_date=date.today().isoformat())
        chain = dated_prompt | new_chat_model(settings) | StrOutputParser()

        try:
            standalone = invoke_with_retry(
                chain,
                {"history": history_text, "question": latest_text},
                max_retries=settings.dashscope_max_retries,
            )
            standalone = (standalone or "").strip() or latest_text
        except Exception as exc:
            logger.error("Condense error; using raw question: %s", exc)
            standalone = latest_text

        logger.info("Condensed question: %r", standalone)
        return {
            "current_question": standalone,
            "current_question_index": latest_idx,
            "rewrite_count": 0,
        }

    return condense_question

