from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .query_constraints import (
    COMPARISON_INTENT,
    DATE_INTENT,
    POLICY_INTENT,
    PRICE_INTENT,
    STATUS_INTENT,
    QueryConstraints,
    detect_query_intents,
    extract_query_constraints,
    normalize_constraint_text,
)

_DATE_EVIDENCE_RE = re.compile(
    r"(?:\b(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])(?:[-/.](?:0?[1-9]|[12]\d|3[01]))?\b|"
    r"(?:19|20)\d{2}年(?:\d{1,2}月(?:\d{1,2}日)?)?|"
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2},?\s+(?:19|20)\d{2}\b)",
    re.I,
)
_PRICE_EVIDENCE_RE = re.compile(
    r"(?:[¥￥$€£]\s*\d[\d,.]*|(?:rmb|cny|usd|hkd|eur|gbp)\s*\d[\d,.]*|"
    r"\d[\d,.]*\s*(?:元|万元|亿元|人民币|美元|港元|欧元|英镑|"
    r"rmb|cny|usd|hkd|eur|gbp))",
    re.I,
)
_POLICY_EVIDENCE_RE = re.compile(
    r"(?:发布|施行|实施|生效|有效|废止|通知|公告|文件|第[一二三四五六七八九十百\d]+条|"
    r"\bissued\b|\beffective\b|\benacted\b|\bsection\s+\d+\b)",
    re.I,
)
_COMPARISON_EVIDENCE_RE = re.compile(
    r"(?:相比|区别|差异|优于|劣于|高于|低于|分别|而|"
    r"\bwhereas\b|\bcompared\s+with\b|\bunlike\b|\bbetter\b|\bworse\b)",
    re.I,
)
_STATUS_EVIDENCE_RE = re.compile(
    r"(?:截至|目前|现已|最新|已完成|在建|运营|发布|上线|"
    r"\bas\s+of\b|\bcurrently\b|\bnow\b|\bcompleted\b|\boperational\b)",
    re.I,
)

_EVIDENCE_PATTERNS = {
    DATE_INTENT: _DATE_EVIDENCE_RE,
    PRICE_INTENT: _PRICE_EVIDENCE_RE,
    POLICY_INTENT: _POLICY_EVIDENCE_RE,
    COMPARISON_INTENT: _COMPARISON_EVIDENCE_RE,
    STATUS_INTENT: _STATUS_EVIDENCE_RE,
}
_HARD_REQUIRED_INTENTS = frozenset({DATE_INTENT, PRICE_INTENT})
EVIDENCE_WINDOW_CHARS = 360


@dataclass(frozen=True)
class EvidenceAssessment:
    intents: frozenset[str]
    matched_intents: frozenset[str]
    missing_required_intents: frozenset[str]

    @property
    def has_required_evidence(self) -> bool:
        return not self.missing_required_intents


def assess_answer_evidence(text: str, query: str) -> EvidenceAssessment:
    """Assess typed facts only when they co-occur with query constraints."""

    constraints = extract_query_constraints(query)
    normalized = _normalize(text)
    evidence_intents = frozenset(
        intent for intent in constraints.intents if intent in _EVIDENCE_PATTERNS
    )
    matched = frozenset(
        intent
        for intent in evidence_intents
        if _has_scoped_match(normalized, _EVIDENCE_PATTERNS[intent], constraints)
    )
    return EvidenceAssessment(
        intents=evidence_intents,
        matched_intents=matched,
        missing_required_intents=frozenset(
            intent for intent in evidence_intents & _HARD_REQUIRED_INTENTS if intent not in matched
        ),
    )


def answer_evidence_delta(text: str, query: str) -> int:
    """Return a small intent-evidence adjustment for provider/page ranking."""

    assessment = assess_answer_evidence(text, query)
    if not assessment.intents:
        return 0

    score = 0
    for intent in assessment.intents:
        if intent in assessment.matched_intents:
            score += 10 if intent in _HARD_REQUIRED_INTENTS else 6
        else:
            score -= 18 if intent in _HARD_REQUIRED_INTENTS else 5
    return score


def has_required_answer_evidence(text: str, query: str) -> bool:
    """Require concrete evidence only for intents with reliable surface forms."""

    return assess_answer_evidence(text, query).has_required_evidence


def _has_scoped_match(
    text: str,
    pattern: re.Pattern[str],
    constraints: QueryConstraints,
) -> bool:
    for match in pattern.finditer(text):
        window = text[
            max(0, match.start() - EVIDENCE_WINDOW_CHARS) : min(
                len(text), match.end() + EVIDENCE_WINDOW_CHARS
            )
        ]
        if _window_matches_constraints(text, window, constraints):
            return True
    return False


def _window_matches_constraints(
    full_text: str,
    window: str,
    constraints: QueryConstraints,
) -> bool:
    if constraints.quoted_phrases and not all(
        phrase in full_text for phrase in constraints.quoted_phrases
    ):
        return False
    if constraints.identifiers and not all(
        identifier in window for identifier in constraints.identifiers
    ):
        return False
    if constraints.years and not constraints.years & set(
        re.findall(r"(?<!\d)20\d{2}(?!\d)", window)
    ):
        return False
    return not (
        not constraints.quoted_phrases
        and constraints.entities
        and not all(normalize_constraint_text(entity) in window for entity in constraints.entities)
    )


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").casefold()


__all__ = [
    "COMPARISON_INTENT",
    "DATE_INTENT",
    "EVIDENCE_WINDOW_CHARS",
    "EvidenceAssessment",
    "POLICY_INTENT",
    "PRICE_INTENT",
    "STATUS_INTENT",
    "assess_answer_evidence",
    "answer_evidence_delta",
    "detect_query_intents",
    "has_required_answer_evidence",
]
