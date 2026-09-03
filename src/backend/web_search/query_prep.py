# Purpose: preprocess search queries for better search-engine results.
from __future__ import annotations

import logging
import re
from datetime import date
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from src.config import Settings

from langchain_core.messages import BaseMessage, HumanMessage

from .query_constraints import (
    COMPARISON_INTENT,
    DATE_INTENT,
    POLICY_INTENT,
    extract_query_constraints,
    validate_query_candidate,
)

logger = logging.getLogger(__name__)

QUESTION_FILLER_WORDS = frozenset(
    {
        "a",
        "about",
        "am",
        "an",
        "any",
        "are",
        "be",
        "been",
        "being",
        "can",
        "could",
        "describe",
        "did",
        "do",
        "does",
        "doing",
        "explain",
        "find",
        "give",
        "has",
        "have",
        "having",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "just",
        "know",
        "let",
        "list",
        "me",
        "my",
        "of",
        "please",
        "should",
        "show",
        "tell",
        "that",
        "the",
        "there",
        "they",
        "this",
        "to",
        "us",
        "want",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
    }
)

CURRENT_YEAR = str(date.today().year)
TIME_SENSITIVE_PATTERNS = (
    re.compile(r"\b(?:latest|newest|current|currently|recent|new|now|today|up.?to.?date)\b", re.I),
    # Relative wording is as time-sensitive as "latest". Without these, "who
    # wins the world cup this year" produced an undated query and providers
    # answered with evergreen all-time list pages.
    re.compile(
        r"\b(?:this|last|next|past|coming)\s+(?:year|month|week|season)\b|"
        r"\b(?:so\s+far|thus\s+far|to\s+date|right\s+now|at\s+the\s+moment|"
        r"this\s+time)\b",
        re.I,
    ),
    re.compile(r"(?<!\d)20\d{2}(?!\d)"),
    re.compile(
        r"(?:最新|当前|现在|近期|最近|截至(?:目前|现在)?|"
        r"今年|去年|明年|本年|本月|今日|今天|目前|当下)"
    ),
)

# Relative time wording resolved to a concrete year offset from today. Leaving
# the wording in the query hurts twice: providers cannot use it, and the terms
# "this"/"year" are then treated as the query's distinctive entities.
_RELATIVE_YEAR_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\b(?:this|the\s+current)\s+(?:year|season)\b", re.I), 0),
    (re.compile(r"\b(?:last|previous|past)\s+(?:year|season)\b", re.I), -1),
    (re.compile(r"\b(?:next|coming|following)\s+(?:year|season)\b", re.I), 1),
    (re.compile(r"今年|本年(?:度)?"), 0),
    (re.compile(r"去年|上一?年"), -1),
    (re.compile(r"明年|下一?年"), 1),
)
MANDARIN_SEARCH_MAX_VARIANTS = 2

_MANDARIN_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_MANDARIN_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?:\s*年)?(?!\d)")
_MANDARIN_PUNCTUATION_RE = re.compile(r"[，。！？；：、,!?;“”‘’（）()\[\]【】{}《》<>]")
_MANDARIN_LEADING_FILLER_RES = (
    re.compile(r"^(?:请帮我|帮我|麻烦(?:帮我)?)(?:查询|搜索|了解|查)(?:一下)?"),
    re.compile(r"^(?:我想(?:知道|了解|问)|你能(?:否)?告诉我|能否告诉我|可以告诉我)"),
    re.compile(r"^(?:请问|麻烦问一下)"),
)
_MANDARIN_TRAILING_PARTICLE_RE = re.compile(r"(?:可以吗|行吗|好吗|吗|呢|吧|呀|啊)$")
_MANDARIN_YEAR_SPLIT_FILLER_RE = re.compile(r"(?:分别|各自|每年)")
_MANDARIN_COUNT_INTENT_RE = re.compile(
    r"(?:数量|数目|总数|多少(?!钱)(?:个|条|家|项|种|人|座|所|名|次)?|"
    r"几(?:个|条|家|项|种|人|座|所|名|次))"
)
_MANDARIN_DATE_INTENT_RE = re.compile(
    r"(?:发布日期|发布时间|生效日期|什么时候|何时|哪天|日期|时间)"
)
_MANDARIN_PRICE_INTENT_RE = re.compile(r"(?:多少钱|价格|费用|票价|收费|售价)")
_MANDARIN_POLICY_INTENT_RE = re.compile(r"(?:政策|规定|办法|条例|通知|标准|规范|实施细则)")
_MANDARIN_COMPARISON_INTENT_RE = re.compile(
    r"(?:比较|对比|相比|区别|差异|不同|优缺点|利弊|哪个好|孰优孰劣)"
)
_METRO_LINE_COUNT_RE = re.compile(
    r"(?:地铁|轨道交通).*(?:线路\s*数量|线路数|条数|几条(?:线|线路)?)"
)
_METRO_PLANNING_RE = re.compile(r"(?:规划|计划|新线|新增|建设|将开通)")

_REWRITE_PROMPT = (
    "Rewrite the following user question into a short, keyword-focused search "
    "engine query. Extract only the core concepts, named entities, and key "
    "terms. Drop all filler words, question words, and conversational phrasing. "
    "If the question implies recency (latest, newest, current, recent, new), "
    f"include the year {CURRENT_YEAR}. Return ONLY the rewritten query string — "
    "no explanation, no punctuation, no quotes. Preserve the input language, "
    "all model/product identifiers, quoted names, and explicit years exactly. "
    "Never invent a year.\n\n"
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

    # Resolve relative wording before term extraction: dropping "this year" as
    # a phrase avoids leaving the orphaned, meaningless token "year" behind.
    relative_year = None if _contains_year(original) else _relative_year(original)
    source = _strip_relative_time(original) if relative_year is not None else original

    if _is_mandarin_query(source):
        result = _prepare_mandarin_query(source)
    else:
        key_terms = _extract_key_terms(source)
        if not key_terms:
            return original
        result = " ".join(key_terms)

    if not _contains_year(original):
        if relative_year is not None:
            result = f"{result} {relative_year}".strip()
        elif _is_time_sensitive(original):
            result = f"{result} {CURRENT_YEAR}"

    if _is_mandarin_query(original):
        result = _add_mandarin_intent_hints(original, result)

    if (
        _METRO_LINE_COUNT_RE.search(original)
        and not _METRO_PLANNING_RE.search(original)
        and "运营线路总数" not in result
    ):
        result = f"{result} 运营线路总数"

    return result.strip()


def plan_search_queries(question: str) -> list[str]:
    """Plan at most two deterministic search variants for one input.

    Non-Mandarin input is returned unchanged so this helper can be applied at
    the graph boundary without altering the existing English query path. A
    Mandarin query gets a concise exact variant plus an official-source
    variant. When exactly two years are requested, the two slots are used for
    one official-evidence query per year instead, preventing a page that only
    mentions one year from satisfying the combined search.
    """

    original = (question or "").strip()
    if not original:
        return []
    if not _is_mandarin_query(original):
        return [original]

    exact_query = prepare_search_query(original)
    years = _unique_mandarin_years(original)
    if len(years) == 2:
        official_terms = _mandarin_official_terms(original)
        return [
            _append_missing_terms(_query_for_year(exact_query, year), official_terms)
            for year in years
        ][:MANDARIN_SEARCH_MAX_VARIANTS]

    official_query = _mandarin_official_query(original, exact_query)
    queries = [exact_query]
    if official_query != exact_query:
        queries.append(official_query)
    return queries[:MANDARIN_SEARCH_MAX_VARIANTS]


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
    if not validate_query_candidate(question, rewritten):
        logger.warning(
            "Rejected LLM query rewrite that changed language or hard constraints: %r",
            rewritten,
        )
        return None

    logger.info(
        "Query rewrite: original=%r → llm_rewritten=%r",
        question,
        rewritten,
    )
    return rewritten


def build_search_query(question: str, settings: Settings) -> str:
    """Produce a search query using deterministic cleanup by default.

    LLM rewriting is retained as an explicit compatibility option. Keeping it
    disabled avoids an additional model call when the graph has already
    formulated a search query.
    """

    llm_rewritten = None
    if settings.web_search_llm_query_rewrite_enabled:
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


def _is_mandarin_query(text: str) -> bool:
    return bool(_MANDARIN_CHAR_RE.search(text))


def _prepare_mandarin_query(text: str) -> str:
    cleaned = text.strip()
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        for pattern in _MANDARIN_LEADING_FILLER_RES:
            cleaned = pattern.sub("", cleaned, count=1).strip()
    cleaned = _MANDARIN_PUNCTUATION_RE.sub(" ", cleaned)
    cleaned = " ".join(cleaned.split())
    cleaned = _MANDARIN_TRAILING_PARTICLE_RE.sub("", cleaned).strip()
    return " ".join(cleaned.split()) or text.strip()


def _add_mandarin_intent_hints(original: str, query: str) -> str:
    terms: list[str] = []
    if _has_mandarin_count_intent(original) and not re.search(r"(?:数量|数目|总数)", query):
        terms.append("总数")
    if _MANDARIN_DATE_INTENT_RE.search(original) and not re.search(
        r"(?:发布日期|发布时间|生效日期)", query
    ):
        terms.append("发布时间")
    if _MANDARIN_PRICE_INTENT_RE.search(original) and "价格" not in query:
        terms.append("价格")
    if _MANDARIN_COMPARISON_INTENT_RE.search(original) and "对比" not in query:
        terms.append("对比")
    return _append_missing_terms(query, terms)


def _mandarin_official_terms(original: str) -> list[str]:
    terms = ["官方"]
    if _MANDARIN_POLICY_INTENT_RE.search(original):
        terms.extend(["原文", "site:gov.cn"])
    elif _has_mandarin_count_intent(original):
        terms.append("数据")
    elif _MANDARIN_DATE_INTENT_RE.search(original):
        terms.append("公告")
    elif _MANDARIN_PRICE_INTENT_RE.search(original):
        terms.append("价格")
    return terms


def _mandarin_official_query(original: str, exact_query: str) -> str:
    constraints = extract_query_constraints(original)
    terms = _mandarin_official_terms(original)
    if POLICY_INTENT not in constraints.intents:
        return _append_missing_terms(exact_query, terms)

    if constraints.quoted_phrases:
        exact_titles = " ".join(f'"{title}"' for title in constraints.quoted_phrases)
        years = " ".join(sorted(constraints.years))
        intent_term = (
            "生效日期"
            if DATE_INTENT in constraints.intents
            else "对比"
            if COMPARISON_INTENT in constraints.intents
            else "原文"
        )
        return " ".join(part for part in (exact_titles, years, intent_term, "site:gov.cn") if part)
    return _append_missing_terms(exact_query, [*terms, "site:gov.cn"])


def _has_mandarin_count_intent(text: str) -> bool:
    return bool(_MANDARIN_COUNT_INTENT_RE.search(text)) and not bool(
        _MANDARIN_PRICE_INTENT_RE.search(text)
    )


def _unique_mandarin_years(text: str) -> list[str]:
    years: list[str] = []
    for year in _MANDARIN_YEAR_RE.findall(text):
        if year not in years:
            years.append(year)
    return years


def _query_for_year(query: str, year: str) -> str:
    without_years = _MANDARIN_YEAR_RE.sub(" ", query)
    without_split_fillers = _MANDARIN_YEAR_SPLIT_FILLER_RE.sub("", without_years)
    normalized = " ".join(without_split_fillers.split())
    return f"{normalized} {year}".strip()


def _append_missing_terms(query: str, terms: list[str]) -> str:
    result = query.strip()
    for term in terms:
        if term not in result:
            result = f"{result} {term}".strip()
    return result


def _is_time_sensitive(text: str) -> bool:
    return any(pattern.search(text) for pattern in TIME_SENSITIVE_PATTERNS)


def _relative_year(text: str) -> str | None:
    """Return the concrete year a relative expression refers to, if any."""

    for pattern, offset in _RELATIVE_YEAR_PATTERNS:
        if pattern.search(text or ""):
            return str(date.today().year + offset)
    return None


def _strip_relative_time(text: str) -> str:
    """Remove resolved relative time wording from a prepared query."""

    cleaned = text or ""
    for pattern, _offset in _RELATIVE_YEAR_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return " ".join(cleaned.split())


def _contains_year(text: str) -> bool:
    return bool(re.search(r"(?<!\d)20\d{2}(?!\d)", text))


__all__ = [
    "MANDARIN_SEARCH_MAX_VARIANTS",
    "build_search_query",
    "plan_search_queries",
    "prepare_search_query",
    "rewrite_search_query_llm",
]
