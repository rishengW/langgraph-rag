from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

DATE_INTENT = "date"
PRICE_INTENT = "price"
POLICY_INTENT = "policy"
COMPARISON_INTENT = "comparison"
STATUS_INTENT = "status"
COUNT_INTENT = "count"

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{2,}")
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_QUOTED_PHRASE_RES = (
    re.compile(r"《([^》]{2,100})》"),
    re.compile(r"“([^”]{2,100})”"),
    re.compile(r'"([^"\r\n]{2,100})"'),
)
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[._-][A-Za-z0-9]+)*")
_CJK_IDENTIFIER_RE = re.compile(
    r"(?:第?[一二三四五六七八九十百千万\d]+(?:\.\d+)?号(?:线|线路|文件|公告)?|"
    r"[一二三四五六七八九十百千万\d]+(?:\.\d+)?号线)"
)

_INTENT_PATTERNS = {
    DATE_INTENT: re.compile(
        r"(?:什么时候|何时|哪天|日期|几月|哪一年|何年|发布日期|发布时间|生效日期|"
        r"\bwhen\b|\bwhat\s+date\b|\brelease\s+date\b)",
        re.I,
    ),
    PRICE_INTENT: re.compile(
        r"(?:多少钱|价格|费用|收费|票价|售价|成本|"
        r"\bprice\b|\bcost\b|\bfee\b|\bfare\b)",
        re.I,
    ),
    POLICY_INTENT: re.compile(
        r"(?:政策|规定|办法|条例|规范|标准|要求|条件|通知|实施细则|"
        r"\bpolicy\b|\bregulation\b|\brule\b|\brequirement\b)",
        re.I,
    ),
    COMPARISON_INTENT: re.compile(
        r"(?:区别|差异|对比|比较|相比|优缺点|哪个好|"
        r"\bcompare\b|\bcomparison\b|\bversus\b|\bvs\.?\b|"
        r"\bdifference\b)",
        re.I,
    ),
    STATUS_INTENT: re.compile(
        r"(?:最新|目前|现在|现状|进展|当前|是否已经|是否已|开通了吗|"
        r"\blatest\b|\bcurrent\b|\bstatus\b|\bprogress\b)",
        re.I,
    ),
    COUNT_INTENT: re.compile(
        r"(?:数量|数目|总数|多少(?!钱)|几(?:个|条|家|项|种|人|座|所|名|次)|"
        r"\bhow\s+many\b|\bnumber\s+of\b|\bcount\b)",
        re.I,
    ),
}

_CJK_ENTITY_STOP_RE = re.compile(
    r"(?:请问|请帮我|帮我|告诉我|查询|查找|搜索|官方|原文|数据|公告|"
    r"什么时候|何时|哪天|日期|时间|多少|几条|价格|费用|票价|售价|"
    r"比较|对比|相比|区别|差异|目前|现在|当前|最新|进展|是否|"
    r"发布|生效|施行|实施|开通|了吗|是什么|怎么样|如何|有何|有什么)"
)
_GENERIC_CJK_ENTITIES = frozenset(
    {
        "规定",
        "政策",
        "办法",
        "条例",
        "通知",
        "标准",
        "方案",
        "产品",
        "价格",
        "日期",
        "时间",
        "状态",
        "新规定",
        "新政策",
    }
)
_GENERIC_LATIN_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "compare",
        "current",
        "date",
        "difference",
        "features",
        "how",
        "is",
        "latest",
        "official",
        "price",
        "release",
        "status",
        "the",
        "versus",
        "what",
        "when",
        "which",
        "with",
    }
)


@dataclass(frozen=True)
class QueryConstraints:
    language: str
    entities: tuple[str, ...]
    identifiers: tuple[str, ...]
    quoted_phrases: tuple[str, ...]
    years: frozenset[str]
    intents: frozenset[str]

    @property
    def anchor_terms(self) -> tuple[str, ...]:
        return _dedupe((*self.quoted_phrases, *self.identifiers, *self.entities))


def extract_query_constraints(query: str) -> QueryConstraints:
    """Extract language and hard search constraints without an LLM call."""

    original = query or ""
    normalized = normalize_constraint_text(original)
    has_cjk = bool(_CJK_RE.search(original))
    has_latin = bool(re.search(r"[A-Za-z]", original))
    language = "mixed" if has_cjk and has_latin else "zh" if has_cjk else "en"

    quoted = _dedupe(
        normalize_constraint_text(match.group(1))
        for pattern in _QUOTED_PHRASE_RES
        for match in pattern.finditer(original)
    )
    identifiers = _extract_identifiers(original, has_cjk=has_cjk)
    entities = _extract_entities(original, identifiers=identifiers, quoted=quoted)
    intents = frozenset(
        intent for intent, pattern in _INTENT_PATTERNS.items() if pattern.search(normalized)
    )
    return QueryConstraints(
        language=language,
        entities=entities,
        identifiers=identifiers,
        quoted_phrases=quoted,
        years=frozenset(_YEAR_RE.findall(normalized)),
        intents=intents,
    )


def detect_query_intents(query: str) -> frozenset[str]:
    return extract_query_constraints(query).intents


def validate_query_candidate(
    original: str,
    candidate: str,
    *,
    allow_partial: bool = False,
) -> bool:
    """Reject translated or constraint-dropping LLM search queries."""

    source = extract_query_constraints(original)
    proposed = extract_query_constraints(candidate)
    if not candidate.strip():
        return False
    if source.language in {"zh", "mixed"} and proposed.language == "en":
        return False

    if proposed.years - source.years:
        return False
    if source.years:
        if allow_partial and not (proposed.years & source.years):
            return False
        if not allow_partial and not source.years.issubset(proposed.years):
            return False

    if not _constraints_preserved(
        source.identifiers,
        proposed.identifiers,
        allow_partial=allow_partial,
    ):
        return False
    if not _constraints_preserved(
        source.quoted_phrases,
        proposed.quoted_phrases,
        allow_partial=allow_partial,
    ):
        return False

    source_entities = set(source.entities)
    if source_entities:
        proposed_text = normalize_constraint_text(candidate)
        matches = {entity for entity in source_entities if entity in proposed_text}
        if not matches and not set(source.identifiers) & set(proposed.identifiers):
            return False
        if not allow_partial and len(source_entities) == 1 and not matches:
            return False
    return True


def normalize_constraint_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text or "").casefold().split())


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _extract_identifiers(text: str, *, has_cjk: bool) -> tuple[str, ...]:
    values: list[str] = []
    for match in _LATIN_TOKEN_RE.finditer(text):
        raw = match.group(0)
        normalized = normalize_constraint_text(raw)
        if normalized in _GENERIC_LATIN_TOKENS:
            continue
        has_number = bool(re.search(r"\d", raw))
        has_mixed_case = raw.lower() != raw and raw.upper() != raw and raw[1:] != raw[1:].lower()
        is_acronym = len(raw) >= 2 and raw.isupper()
        if has_cjk or has_number or has_mixed_case or is_acronym:
            values.append(normalized)
    values.extend(normalize_constraint_text(value) for value in _CJK_IDENTIFIER_RE.findall(text))
    return _dedupe(values)


def _extract_entities(
    text: str,
    *,
    identifiers: tuple[str, ...],
    quoted: tuple[str, ...],
) -> tuple[str, ...]:
    values: list[str] = []
    quoted_set = set(quoted)
    for run in _CJK_RUN_RE.findall(text):
        cleaned = _CJK_ENTITY_STOP_RE.split(run, maxsplit=1)[0]
        cleaned = normalize_constraint_text(cleaned).strip()
        if (
            2 <= len(cleaned) <= 20
            and cleaned not in _GENERIC_CJK_ENTITIES
            and cleaned not in quoted_set
        ):
            values.append(cleaned)

    for raw in _LATIN_TOKEN_RE.findall(text):
        normalized = normalize_constraint_text(raw)
        if normalized in identifiers or normalized in _GENERIC_LATIN_TOKENS:
            continue
        if raw[:1].isupper() and len(raw) >= 2:
            values.append(normalized)
    return _dedupe(values)


def _constraints_preserved(
    required: tuple[str, ...],
    observed: tuple[str, ...],
    *,
    allow_partial: bool,
) -> bool:
    if not required:
        return True
    required_set = set(required)
    observed_set = set(observed)
    return bool(required_set & observed_set) if allow_partial else required_set <= observed_set


def _dedupe(values: Iterable[object]) -> tuple[str, ...]:
    ordered: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if value and value not in ordered:
            ordered.append(value)
    return tuple(ordered)


__all__ = [
    "COMPARISON_INTENT",
    "COUNT_INTENT",
    "DATE_INTENT",
    "POLICY_INTENT",
    "PRICE_INTENT",
    "QueryConstraints",
    "STATUS_INTENT",
    "contains_cjk",
    "detect_query_intents",
    "extract_query_constraints",
    "normalize_constraint_text",
    "validate_query_candidate",
]
