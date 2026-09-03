# REFACTOR: Conditional-expansion expand node. Produces k=1-3 paraphrases
# per sub-question so the search fan-out can run multiple rewrites in
# parallel. For a single input query the LLM is bypassed and the input is
# returned as-is. On LLM failure the input query is also returned (k=1
# passthrough fallback).
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ...config import Settings
from ...utils.retry import invoke_with_retry
from .common import chat_question_resolver, new_structured_chat_model
from .search_queries import WEB_SEARCH_MAX_QUERIES

logger = logging.getLogger(__name__)

QuestionResolver = Callable[[dict[str, Any]], str]

# REFACTOR: Cap the per-sub-question expansion fan-out at 3 paraphrases
# including the input. Clamping keeps the N x k search cost bounded on
# the worst case and matches the design decision in the 2026-06-17 form.
EXPAND_MAX_PARAPHRASES = 3


class _ExpandResult(BaseModel):
    """Structured output for the expand LLM call."""

    paraphrases: list[str] = Field(
        default_factory=list,
        description=(
            "1-3 keyword-form search queries that reformulate the input "
            "from different angles (synonyms, related terms, alternate "
            "phrasings). The first entry must equal the input query "
            "verbatim so the original search is preserved."
        ),
    )


def expand_factory(
    settings: Settings,
    question_resolver: QuestionResolver = chat_question_resolver,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a node that produces k paraphrases per sub-question.

    The node reads ``sub_questions`` from state (produced by
    ``decompose``) and emits ``expanded_queries``: the flat list of
    paraphrases across all sub-questions, in order. The first entry of
    each sub-question's paraphrase list is always the input itself so
    the original search never gets lost in the fan-out.

    The node never raises: any LLM error or unexpected output falls
    back to the input sub-questions (k=1 passthrough per sub-question).
    """

    def expand(state: dict[str, Any]) -> dict[str, Any]:
        logger.info("EXPAND QUERIES")
        sub_questions = _resolve_sub_questions(state, question_resolver)
        if not sub_questions:
            # REFACTOR: Mark expansion as attempted even on the empty-input
            # short circuit so the post-web_answer edge cannot route back to
            # ``expand`` again. Without this the graph loops
            # web_answer -> expand -> web_search -> merge -> web_answer
            # forever until the recursion limit.
            return {
                "expanded_queries": [],
                "search_queries": [],
                "expansion_attempted": True,
            }

        expanded: list[str] = []
        for sub_question in sub_questions:
            for query in _paraphrases_for(sub_question, settings):
                if query not in expanded:
                    expanded.append(query)
                if len(expanded) >= WEB_SEARCH_MAX_QUERIES:
                    break
            if len(expanded) >= WEB_SEARCH_MAX_QUERIES:
                break

        # REFACTOR: One-shot switch. ``route_after_web_answer`` routes to
        # ``expand`` only while ``expansion_attempted`` is False; setting it
        # here bounds the conditional-expansion retry to a single pass so a
        # persistently unreadable source set falls through to the agent
        # fallback / grounded refusal instead of looping forever.
        return {
            "expanded_queries": expanded,
            "search_queries": list(expanded),
            "expansion_attempted": True,
        }

    return expand


def _paraphrases_for(sub_question: str, settings: Settings) -> list[str]:
    """Return 1-3 paraphrases for a single sub-question.

    Always returns the input as the first entry. On LLM failure returns
    ``[sub_question]`` (k=1 passthrough). On partial LLM output the
    surviving entries are appended after the input and the list is
    clamped to ``EXPAND_MAX_PARAPHRASES``.
    """

    sub_question = sub_question.strip()
    if not sub_question:
        return []

    prompt = (
        "You reformulate a research question into keyword-form search "
        "queries for a web search engine. Today's date is "
        f"{_today_iso()}.\n\n"
        "Input question:\n"
        "-------\n"
        f"{sub_question}\n"
        "-------\n\n"
        "Rules:\n"
        "- Return 1-3 distinct search queries, each on its own JSON "
        "array entry, that cover different angles of the input "
        "(synonyms, related entities, alternate phrasings).\n"
        "- The first entry must equal the input question verbatim so "
        "the original search is preserved.\n"
        "- Queries should be keyword-form, not full natural-language "
        "questions. Drop filler words (what, is, the, of, does, etc.).\n"
        '- Return only one JSON object with this exact shape: '
        '{"paraphrases":["query 1","query 2"]}.\n'
    )

    try:
        chain = new_structured_chat_model(settings, _ExpandResult)
        result = invoke_with_retry(
            chain,
            [HumanMessage(content=prompt)],
            max_retries=settings.dashscope_max_retries,
        )
        paraphrases = _clamp_paraphrases(
            getattr(result, "paraphrases", None), sub_question
        )
    except Exception as exc:  # noqa: BLE001 - passthrough fallback by design
        logger.warning("Expand LLM call failed; using atomic passthrough: %s", exc)
        paraphrases = [sub_question]
    return paraphrases


def _clamp_paraphrases(raw: Any, original: str) -> list[str]:
    """Normalize and clamp the LLM output to 1..N paraphrases.

    The original sub-question is always the first entry. Duplicates are
    removed. Output is clamped to ``EXPAND_MAX_PARAPHRASES`` entries.
    """

    cleaned: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, str):
                continue
            value = item.strip()
            if value and value not in cleaned:
                cleaned.append(value)
            if len(cleaned) >= EXPAND_MAX_PARAPHRASES:
                break
    if not cleaned or cleaned[0] != original:
        cleaned.insert(0, original)
        cleaned = cleaned[:EXPAND_MAX_PARAPHRASES]
    return cleaned


def _resolve_sub_questions(
    state: dict[str, Any], question_resolver: QuestionResolver
) -> list[str]:
    raw = state.get("sub_questions")
    if isinstance(raw, list) and raw:
        return [str(item).strip() for item in raw if str(item).strip()]
    # REFACTOR: Fall back to the original question when the state is
    # missing the sub_questions field (e.g. when ``expand`` is invoked
    # outside the standard decompose -> expand flow). Guard against
    # empty-state edge cases where a resolver might raise on ``{"messages": []}``.
    try:
        fallback = question_resolver(state).strip()
    except (IndexError, KeyError, TypeError):
        fallback = ""
    return [fallback] if fallback else []


def _today_iso() -> str:
    from datetime import date

    return date.today().isoformat()


__all__ = [
    "EXPAND_MAX_PARAPHRASES",
    "expand_factory",
]
