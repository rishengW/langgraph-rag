# Purpose: preprocess search queries for better search-engine results.
from __future__ import annotations

import logging
import re
from datetime import date
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from ..config import Settings

from langchain_core.messages import BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

QUESTION_FILLER_WORDS = frozenset({
    "a", "about", "am", "an", "any", "are", "be", "been", "being",
    "can", "could", "describe", "did", "do", "does", "doing",
    "explain", "find", "give", "has", "have", "having", "how",
    "i", "if", "in", "is", "it", "its", "just", "know", "let",
    "list", "me", "my", "of", "please", "should", "show", "tell",
    "that", "the", "there", "they", "this", "to", "us", "want",
    "was", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you",
})

CURRENT_YEAR = str(date.today().year)
TIME_SENSITIVE_PATTERNS = (
    re.compile(r"\b(?:latest|newest|current|recent|new|now|today|up.?to.?date)\b", re.I),
    re.compile(r"\b20\d{2}\b"),
)

_REWRITE_PROMPT = (
    "Rewrite the following user question into a short, keyword-focused search "
    "engine query. Extract only the core concepts, named entities, and key "
    "terms. Drop all filler words, question words, and conversational phrasing. "
    "If the question implies recency (latest, newest, current, recent, new), "
    f"include the year {CURRENT_YEAR}. Return ONLY the rewritten query string — "
    "no explanation, no punctuation, no quotes.\n\n"
    "Examples:\n"
    "  User: what is the latest model of deepseek\n"
    "  Query: DeepSeek latest model 2026\n\n"
    "  User: tell me about langgraph rag architecture\n"
    "  Query: LangGraph RAG architecture\n\n"
    "User question: {question}\n"
    "Query:"
)


class QueryRewriteLLM(Protocol):
    """Minimal LLM surface used by query rewrite tests and providers."""

    def invoke(self, input_data: list[BaseMessage]) -> Any:
        """Invoke the chat model with one prompt message."""


def prepare_search_query(question: str) -> str:
    """Return a search-engine-optimized query string using mechanical rules.

    Strips common question filler words while preserving named entities and
    key concepts. Appends the current year when the question appears
    time-sensitive and doesn't already contain a year reference.
    """

    original = (question or "").strip()
    if not original:
        return original

    key_terms = _extract_key_terms(original)
    if not key_terms:
        return original

    result = " ".join(key_terms)

    if _is_time_sensitive(original) and not _contains_year(original):
        result = f"{result} {CURRENT_YEAR}"

    return result.strip()


def rewrite_search_query_llm(
    question: str,
    settings: Settings,
    *,
    _llm: QueryRewriteLLM | None = None,
) -> str | None:
    """Use a lightweight LLM to rewrite the question into search keywords.

    Calls DeepSeek V4 Flash for fast, low-cost query reformulation. Returns
    ``None`` when the LLM is unavailable, the API key is missing, or the
    call fails — callers should fall back to :func:`prepare_search_query`.
    """

    if not settings.deepseek_api_key:
        logger.debug("Skipping LLM query rewrite: DEEPSEEK_API_KEY is not set.")
        return None

    prompt = _REWRITE_PROMPT.format(question=question.strip())

    if _llm is not None:
        llm = _llm
    else:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            logger.warning("Skipping LLM query rewrite: langchain-openai not installed.")
            return None

        chat_openai = cast(Any, ChatOpenAI)
        llm = chat_openai(
            model="deepseek-v4-flash",
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0,
            max_tokens=60,
            request_timeout=10,
            max_retries=1,
        )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        rewritten = str(response.content).strip().strip("\"'")
    except Exception as exc:
        logger.warning("LLM query rewrite failed; using mechanical fallback: %s", exc)
        return None

    if not rewritten or rewritten == question.strip():
        return None

    logger.info(
        "Query rewrite: original=%r → llm_rewritten=%r",
        question,
        rewritten,
    )
    return rewritten


def build_search_query(question: str, settings: Settings) -> str:
    """Produce the best available search query using LLM rewrite + mechanical fallback.

    Tries an LLM-based rewrite first (DeepSeek V4 Flash), then applies
    mechanical preprocessing as a safety net on the result.
    """

    llm_rewritten = rewrite_search_query_llm(question, settings)
    base = llm_rewritten if llm_rewritten else question
    return prepare_search_query(base)


def _extract_key_terms(text: str) -> list[str]:
    words = text.split()
    kept: list[str] = []
    for word in words:
        stripped = word.strip(".,;:!?\"'()[]{}")
        if not stripped:
            continue
        if stripped.lower() in QUESTION_FILLER_WORDS:
            continue
        kept.append(stripped)
    return kept


def _is_time_sensitive(text: str) -> bool:
    return any(pattern.search(text) for pattern in TIME_SENSITIVE_PATTERNS)


def _contains_year(text: str) -> bool:
    return bool(re.search(r"\b20\d{2}\b", text))


__all__ = ["build_search_query", "prepare_search_query", "rewrite_search_query_llm"]
