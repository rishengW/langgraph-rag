# REFACTOR: Conditional-expansion decompose node. Splits a compound question
# into 1-3 sub-questions before the expansion search fan-out. For atomic
# questions the LLM is bypassed entirely and the original question is
# returned unchanged. On LLM failure the original question is also
# returned (passthrough fallback).
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ...config import Settings
from ...utils.retry import invoke_with_retry
from ...web_search.query_constraints import validate_query_candidate
from .common import new_structured_chat_model, qa_question_resolver

logger = logging.getLogger(__name__)

QuestionResolver = Callable[[dict[str, Any]], str]

# REFACTOR: Cap the decompose fan-out at 3 sub-questions; clamping the LLM
# output above this bound keeps the N x k search cost bounded on the worst
# case and matches the design decision in the 2026-06-17 form.
DECOMPOSE_MAX_SUBQUESTIONS = 3

_COMPARISON_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?|difference|differences|differ)\b"
    r"|\bpros\s+and\s+cons\b"
    r"|\badvantages\s+and\s+disadvantages\b"
    r"|(?:比较|对比|相比|区别|差异|不同(?!意)|优缺点|利弊|哪个好|孰优孰劣)",
    re.IGNORECASE,
)
_INTERROGATIVE_RE = re.compile(
    r"\b(?:what|when|where|who|whom|whose|why|how|which)\b"
    r"|(?:什么时候|什么|何时|哪里|哪儿|谁|为何|为什么|怎么|如何|"
    r"哪(?:个|些|种)?|多少|几(?:个|条|次|年|岁)?)",
    re.IGNORECASE,
)
_REQUEST_VERB_RE = re.compile(
    r"\b(?:analyze|compare|describe|evaluate|explain|identify|list|summarize)\b"
    r"|(?:分析|比较|对比|说明|描述|评估|解释|列出|总结|介绍|查询|查找)",
    re.IGNORECASE,
)
_CLAUSE_CONNECTOR_RE = re.compile(
    r"\b(?:also|and|then)\b|(?:以及|并且|而且|然后|同时|另外|还|并|和|与|及|分别)|[?？;；]"
)


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
            resolved_question = question_resolver(state).strip()
        except Exception:  # noqa: BLE001 - defensive passthrough by design
            resolved_question = ""
        preserved_question = str(state.get("current_question") or "").strip()
        question = preserved_question or _search_tool_query(state) or resolved_question
        if not question:
            return {
                "current_question": "",
                "sub_questions": [],
                "expanded_queries": [],
                "search_queries": [],
                "web_search_results": [],
                "web_search_result_metadata": [],
            }
        if not _is_likely_compound(question):
            return {
                "current_question": question,
                "sub_questions": [question],
                "expanded_queries": [],
                "search_queries": [question],
                "web_search_results": [],
                "web_search_result_metadata": [],
            }

        prompt = (
            "You split compound research questions into independent "
            "sub-questions for a search engine. Today's date is "
            f"{_today_iso()}.\n\n"
            "Input question:\n"
            "-------\n"
            f"{question}\n"
            "-------\n\n"
            "Rules:\n"
            "- If the question is atomic (one fact or one named entity), return "
            "exactly ONE sub-question equal to the "
            "original input verbatim.\n"
            "- Otherwise return 2-3 self-contained sub-questions that "
            "together cover every distinct fact the original asks for. "
            "Each sub-question must stand alone (no pronouns like 'it' "
            "or 'they' referring to earlier parts).\n"
            "- Preserve the input language. Never translate a Chinese question "
            "into English or an English question into Chinese.\n"
            "- Preserve explicit years, quoted names, model identifiers, product "
            "codes, and numbered lines exactly. Never invent a year.\n"
            "- Return only one JSON object with this exact shape: "
            '{"sub_questions":["question 1","question 2"]}.\n'
        )

        try:
            chain = new_structured_chat_model(settings, _DecomposeResult)
            result = invoke_with_retry(
                chain,
                [HumanMessage(content=prompt)],
                max_retries=settings.dashscope_max_retries,
            )
            sub_questions = _clamp_sub_questions(getattr(result, "sub_questions", None), question)
        except Exception as exc:  # noqa: BLE001 - passthrough fallback by design
            logger.warning("Decompose LLM call failed; using atomic passthrough: %s", exc)
            sub_questions = [question]

        return {
            # Preserve the contextualized live-search query across both
            # web-answer attempts. Intermediate refusal messages are appended
            # to chat history and must never become the grounding question.
            "current_question": question,
            "sub_questions": sub_questions,
            "expanded_queries": [],
            "search_queries": list(sub_questions),
            "web_search_results": [],
            "web_search_result_metadata": [],
        }

    return decompose


def _search_tool_query(state: dict[str, Any]) -> str:
    """Return the latest live-search query selected by the agent."""

    messages = state.get("messages") or []
    for message in reversed(list(messages)):
        calls = getattr(message, "tool_calls", None) or []
        for call in reversed(list(calls)):
            if not isinstance(call, dict) or call.get("name") != "live_web_search":
                continue
            args = call.get("args")
            query = args.get("query") if isinstance(args, dict) else None
            if isinstance(query, str) and query.strip():
                return query.strip()
    return ""


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
            if (
                value
                and value not in cleaned
                and validate_query_candidate(original, value, allow_partial=True)
            ):
                cleaned.append(value)
            if len(cleaned) >= DECOMPOSE_MAX_SUBQUESTIONS:
                break
    if not cleaned:
        return [original]
    return cleaned


def _is_likely_compound(question: str) -> bool:
    """Return True only for strong, cheap signals of independent search tasks."""

    normalized = " ".join(question.split())
    if not normalized:
        return False
    if _COMPARISON_RE.search(normalized):
        return True
    if normalized.count("?") + normalized.count("？") >= 2:
        return True

    interrogatives = list(_INTERROGATIVE_RE.finditer(normalized.lower()))
    for first, second in zip(interrogatives, interrogatives[1:], strict=False):
        between = normalized[first.end() : second.start()]
        if _CLAUSE_CONNECTOR_RE.search(between):
            return True

    request_verbs = list(_REQUEST_VERB_RE.finditer(normalized))
    for first, second in zip(request_verbs, request_verbs[1:], strict=False):
        between = normalized[first.end() : second.start()]
        if _CLAUSE_CONNECTOR_RE.search(between):
            return True
    return False


def _today_iso() -> str:
    from datetime import date

    return date.today().isoformat()


__all__ = [
    "DECOMPOSE_MAX_SUBQUESTIONS",
    "decompose_factory",
]
