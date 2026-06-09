# Purpose: deterministic pre-index quality checks for loaded source documents.
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import logging
import re

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

BOILERPLATE_TERMS = {
    "advertisement",
    "ads",
    "captcha",
    "cookie",
    "cookies",
    "copyright",
    "javascript",
    "login",
    "menu",
    "navigation",
    "privacy",
    "reserved",
    "signin",
    "subscribe",
    "terms",
}
STOP_WORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "for",
    "from",
    "how",
    "into",
    "that",
    "the",
    "this",
    "what",
    "when",
    "where",
    "with",
    "your",
}
WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class DocumentQualityConfig:
    """Configuration for deterministic loaded-document quality checks."""

    enabled: bool = True
    min_text_length: int = 80
    min_text_signal_ratio: float = 0.35
    min_unique_terms: int = 8
    max_repeated_line_ratio: float = 0.60
    max_boilerplate_term_ratio: float = 0.45
    relevance_query: str = ""
    min_query_term_overlap: int = 1


def filter_quality_documents(
    documents: Sequence[Document],
    config: DocumentQualityConfig | None = None,
) -> list[Document]:
    """Return loaded documents that pass pre-index quality checks.

    Args:
        documents: Loaded source documents to evaluate before splitting.
        config: Quality thresholds and optional relevance query.
    """

    quality_config = config or DocumentQualityConfig()
    if not quality_config.enabled:
        return list(documents)

    kept: list[Document] = []
    rejected: Counter[str] = Counter()
    for document in documents:
        reason = rejection_reason(document, quality_config)
        if reason is None:
            kept.append(document)
        else:
            rejected[reason] += 1

    if rejected:
        logger.info(
            "Document quality filter kept %d/%d document(s); dropped %s",
            len(kept),
            len(documents),
            dict(rejected),
        )
    return kept


def rejection_reason(
    document: Document,
    config: DocumentQualityConfig,
) -> str | None:
    """Return a deterministic rejection reason, or None when the document is kept."""

    text = _normalized_text(document.page_content)
    if len(text) < max(0, config.min_text_length):
        return "short_text"
    if _text_signal_ratio(text) < config.min_text_signal_ratio:
        return "low_text_signal"
    if _repeated_line_ratio(document.page_content) > config.max_repeated_line_ratio:
        return "repeated_boilerplate"
    if _boilerplate_ratio(text) > config.max_boilerplate_term_ratio:
        return "boilerplate"
    if not _has_enough_terms(text, max(0, config.min_unique_terms)):
        return "low_unique_terms"
    if not _matches_relevance_query(text, config):
        return "query_mismatch"
    return None


def _normalized_text(text: str) -> str:
    return " ".join((text or "").split())


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def _text_signal_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    signal = sum(1 for char in visible if char.isalnum())
    return signal / len(visible)


def _repeated_line_ratio(text: str) -> float:
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return 0.0
    counts = Counter(lines)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(lines)


def _boilerplate_ratio(text: str) -> float:
    terms = _tokens(text)
    if not terms:
        return 1.0
    boilerplate_count = sum(1 for term in terms if term in BOILERPLATE_TERMS)
    return boilerplate_count / len(terms)


def _has_enough_terms(text: str, min_unique_terms: int) -> bool:
    if min_unique_terms <= 0:
        return True
    if _cjk_character_count(text) >= min_unique_terms * 4:
        return True
    terms = {term for term in _tokens(text) if len(term) >= 3}
    return len(terms) >= min_unique_terms


def _matches_relevance_query(text: str, config: DocumentQualityConfig) -> bool:
    query_terms = _query_terms(config.relevance_query)
    required = max(0, config.min_query_term_overlap)
    if required <= 0 or not query_terms:
        return True
    document_terms = set(_tokens(text))
    return len(query_terms.intersection(document_terms)) >= required


def _query_terms(query: str) -> set[str]:
    return {
        token
        for token in _tokens(query)
        if len(token) >= 3 and token not in STOP_WORDS
    }


def _cjk_character_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
