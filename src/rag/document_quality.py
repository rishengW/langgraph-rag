# Purpose: deterministic pre-index quality checks for loaded source documents.
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
import hashlib
import logging
import math
import re

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

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
SEMANTIC_SAMPLE_CHARS = 800
DEFAULT_RECENCY_BIAS_DAYS = 365


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
    min_similarity: float = 0.5
    recency_bias_days: int = DEFAULT_RECENCY_BIAS_DAYS


def filter_quality_documents(
    documents: Sequence[Document],
    config: DocumentQualityConfig | None = None,
    *,
    embeddings: Embeddings | None = None,
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
    seen_hashes: set[str] = set()
    query_embedding = _query_embedding(quality_config, embeddings)
    for document in documents:
        text_hash = _content_hash(document.page_content)
        if text_hash in seen_hashes:
            rejected["duplicate_content"] += 1
            continue

        reason = rejection_reason(
            document,
            quality_config,
            embeddings=embeddings,
            query_embedding=query_embedding,
        )
        if reason is None:
            seen_hashes.add(text_hash)
            _annotate_recency(document, quality_config)
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
    return _bias_recent_documents(kept)


def rejection_reason(
    document: Document,
    config: DocumentQualityConfig,
    *,
    embeddings: Embeddings | None = None,
    query_embedding: list[float] | None = None,
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
    if not _matches_relevance_query(
        text,
        config,
        embeddings=embeddings,
        query_embedding=query_embedding,
    ):
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


def _matches_semantic_relevance(
    text: str,
    config: DocumentQualityConfig,
    embeddings: Embeddings | None,
    query_embedding: list[float] | None,
) -> bool | None:
    if embeddings is None or query_embedding is None or not config.relevance_query.strip():
        return None
    try:
        document_embedding = embeddings.embed_query(text[:SEMANTIC_SAMPLE_CHARS])
    except Exception as exc:
        logger.warning("Document semantic quality check failed; using keyword fallback: %s", exc)
        return None
    return _cosine_similarity(query_embedding, document_embedding) >= config.min_similarity


def _matches_relevance_query(
    text: str,
    config: DocumentQualityConfig,
    *,
    embeddings: Embeddings | None = None,
    query_embedding: list[float] | None = None,
) -> bool:
    semantic_match = _matches_semantic_relevance(
        text,
        config,
        embeddings,
        query_embedding,
    )
    if semantic_match is not None:
        return semantic_match

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


def _query_embedding(
    config: DocumentQualityConfig,
    embeddings: Embeddings | None,
) -> list[float] | None:
    if embeddings is None or not config.relevance_query.strip():
        return None
    try:
        return embeddings.embed_query(config.relevance_query)
    except Exception as exc:
        logger.warning("Query semantic quality check failed; using keyword fallback: %s", exc)
        return None


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _content_hash(text: str) -> str:
    normalized = _normalized_text(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _publication_date(document: Document) -> date | None:
    parsed = _parse_publication_date(document.metadata.get("publication_date"))
    if parsed is not None:
        return parsed
    return _extract_publication_date(document.page_content, document.metadata)


def _parse_publication_date(value: object) -> date | None:
    from ..web_search.date_extractor import parse_publication_date

    return parse_publication_date(value)


def _extract_publication_date(html: str, metadata: dict[str, object]) -> date | None:
    from ..web_search.date_extractor import extract_publication_date

    return extract_publication_date(html, metadata)


def _annotate_recency(document: Document, config: DocumentQualityConfig) -> None:
    published_date = _publication_date(document)
    if published_date is None:
        return
    age_days = max(0, (date.today() - published_date).days)
    document.metadata["publication_date"] = published_date.isoformat()
    document.metadata["document_quality_recency_score"] = _recency_score(
        age_days,
        config.recency_bias_days,
    )


def _recency_score(age_days: int, bias_days: int) -> float:
    window = max(0, bias_days)
    if window <= 0 or age_days > window:
        return 0.0
    return 1.0 - (age_days / window)


def _bias_recent_documents(documents: list[Document]) -> list[Document]:
    return sorted(
        documents,
        key=lambda document: float(
            document.metadata.get("document_quality_recency_score", 0.0)
        ),
        reverse=True,
    )
