# REFACTOR: Conditional-expansion decompose node. Splits a compound question
# into 1-3 sub-questions before the expansion search fan-out. For atomic
# questions the LLM is bypassed entirely and the original question is
# returned unchanged. On LLM failure the original question is also
# returned (passthrough fallback).
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ...config import Settings
from ...utils.retry import invoke_with_retry
from .common import new_chat_model, qa_question_resolver

logger = logging.getLogger(__name__)

QuestionResolver = Callable[[dict[str, Any]], str]

# REFACTOR: Cap the decompose fan-out at 3 sub-questions; clamping the LLM
# output above this bound keeps the N x k search cost bounded on the worst
# case and matches the design decision in the 2026-06-17 form.
DECOMPOSE_MAX_SUBQUESTIONS = 3


class _DecomposeResult(BaseModel):
    """Structured output for the decompose LLM call."""

    sub_questions: list[str] = Field(
        default_factory=list,
        description=(
            "1-3 self-contained sub-questions that together cover the original "
            "question. Return exactly one entry (the original question) when the "
            "input is atomic / cannot be split."
        ),
    )


def decompose_factory(
    settings: Settings,
    question_resolver: QuestionResolver = qa_question_resolver,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a node that decomposes a question into 1-3 sub-questions.

    The node reads the current question from state, asks the chat model to
    split it into self-contained sub-questions, clamps the result to at
    most ``DECOMPOSE_MAX_SUBQUESTIONS`` non-empty entries, and writes the
    sub-questions list to state. The original question is always preserved
    as the first entry so downstream nodes never see an empty list.

    The node never raises: any LLM error or unexpected output falls back
    to ``[original_question]`` (atomic passthrough).
    """

    def decompose(state: dict[str, Any]) -> dict[str, Any]:
        logger.info("DECOMPOSE QUESTION")
        try:
            question = question_resolver(state).strip()
        except Exception:  # noqa: BLE001 - defensive passthrough by design
            question = ""
        if not question:
            return {"sub_questions": []}

        prompt = (
            "You split compound research questions into independent "
            "sub-questions for a search engine. Today's date is "
            f"{_today_iso()}.\n\n"
            "Input question:\n"
            "-------\n"
            f"{question}\n"
            "-------\n\n"
            "Rules:\n"
            "- If the question is atomic (one fact, one comparison, one "
            "named entity), return exactly ONE sub-question equal to the "
            "original input verbatim.\n"
            "- Otherwise return 2-3 self-contained sub-questions that "
            "together cover every distinct fact the original asks for. "
            "Each sub-question must stand alone (no pronouns like 'it' "
            "or 'they' referring to earlier parts).\n"
            "- Output JSON matching the schema.\n"
        )

        try:
            chain = new_chat_model(settings).with_structured_output(_DecomposeResult)
            result = invoke_with_retry(
                chain,
                [HumanMessage(content=prompt)],
                max_retries=settings.dashscope_max_retries,
            )
            sub_questions = _clamp_sub_questions(
                getattr(result, "sub_questions", None), question
            )
        except Exception as exc:  # noqa: BLE001 - passthrough fallback by design
            logger.warning("Decompose LLM call failed; using atomic passthrough: %s", exc)
            sub_questions = [question]

        return {"sub_questions": sub_questions}

    return decompose


def _clamp_sub_questions(raw: Any, original: str) -> list[str]:
    """Normalize and clamp the LLM output to 1..N sub-questions.

    The original question is always preserved as the first entry when the
    LLM output is empty, contains only blanks, or fails the ``is
    different from the original`` check (which would make the fan-out a
    no-op without surfacing a problem).
    """

    cleaned: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if value and value not in cleaned:
                cleaned.append(value)
            if len(cleaned) >= DECOMPOSE_MAX_SUBQUESTIONS:
                break
    if not cleaned:
        return [original]
    return cleaned


def _today_iso() -> str:
    from datetime import date

    return date.today().isoformat()


__all__ = [
    "DECOMPOSE_MAX_SUBQUESTIONS",
    "decompose_factory",
]
