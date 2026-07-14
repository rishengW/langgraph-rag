"""Tool-free fallback for web questions whose sources cannot be read."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from ...config import Settings
from ...utils.retry import invoke_with_retry
from .common import QuestionResolver, message_text, new_chat_model

logger = logging.getLogger(__name__)

_FALLBACK_REFUSAL = (
    "I couldn't retrieve readable, relevant web content for this question, "
    "and I couldn't produce a reliable answer from my internal knowledge."
)


def fallback_answer_factory(
    settings: Settings,
    question_resolver: QuestionResolver,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a one-shot, tool-free training-data fallback node.

    Only the preserved user question is sent to an unbound chat model. Prior
    web-search refusals and tool messages are deliberately excluded, so they
    cannot become a new search query or contaminate the fallback answer.
    """

    def fallback_answer(state: dict[str, Any]) -> dict[str, Any]:
        question = _original_question(state, question_resolver)
        if not question:
            return {"messages": [AIMessage(content=_FALLBACK_REFUSAL)]}

        prompt = (
            "Live web search did not yield readable, relevant evidence. "
            "Answer the original user question from your internal knowledge "
            "only if you are reasonably confident. Start by clearly saying "
            "that you could not verify the answer against live web sources. "
            "Do not cite or imply that you read any web source. If you are not "
            "confident, say that you do not know.\n\n"
            f"Original user question:\n{question}"
        )
        try:
            response = invoke_with_retry(
                new_chat_model(settings),
                [HumanMessage(content=prompt)],
                max_retries=settings.dashscope_max_retries,
            )
        except Exception as exc:  # noqa: BLE001 - final fallback must terminate
            logger.warning("Tool-free fallback answer failed: %s", exc)
            response = AIMessage(content=_FALLBACK_REFUSAL)
        return {"messages": [response]}

    return fallback_answer


def _original_question(
    state: dict[str, Any], question_resolver: QuestionResolver
) -> str:
    """Resolve the user question without ever selecting an AI refusal."""

    standalone = state.get("current_question")
    if isinstance(standalone, str) and standalone.strip():
        return standalone.strip()

    messages = state.get("messages") or []
    for message in reversed(list(messages)):
        if getattr(message, "type", None) == "human":
            text = message_text(message).strip()
            if text:
                return text

    try:
        return question_resolver(state).strip()
    except (IndexError, KeyError, TypeError):
        return ""


__all__ = ["fallback_answer_factory"]
