"""Deterministic lexical relevance scoring for memory recall.

Pure functions, no ``Settings`` and no I/O. ``rank_records`` is the single
ordering authority: explicit recall, keyword-based forget, and automatic recall
injection all call it, so the three cannot drift apart.

Matching is substring-based against the normalized content and tags rather than
token-equality. This project handles Mandarin queries heavily, and CJK text has
no whitespace token boundaries, so a token-equality test would score almost
every Chinese memory at zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

from .models import MAX_QUERY_TERMS, MIN_TERM_CHARS, MemoryRecord


def normalize_content(text: str) -> str:
    """Case-fold, collapse whitespace runs to one space, and trim."""

    return " ".join(str(text).casefold().split())


def derive_query_terms(query: str) -> tuple[str, ...]:
    """Return the first ``MAX_QUERY_TERMS`` distinct terms of usable length."""

    terms: list[str] = []
    seen: set[str] = set()
    for token in normalize_content(query).split(" "):
        if len(token) < MIN_TERM_CHARS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= MAX_QUERY_TERMS:
            break
    return tuple(terms)


def searchable_text(record: MemoryRecord) -> str:
    """Return the normalized haystack for one record: content plus tags."""

    parts = [normalize_content(record.content)]
    parts.extend(normalize_content(tag) for tag in record.tags)
    return " ".join(part for part in parts if part)


def relevance_score(record: MemoryRecord, terms: Sequence[str]) -> int:
    """Count distinct terms appearing in the record's content or tags."""

    if not terms:
        return 0
    haystack = searchable_text(record)
    if not haystack:
        return 0
    return sum(1 for term in dict.fromkeys(terms) if term in haystack)


def rank_records(
    records: Iterable[MemoryRecord],
    terms: Sequence[str],
    *,
    top_k: int | None = None,
) -> list[MemoryRecord]:
    """Rank matching records by score, then recency, then id.

    Keeps records scoring 1 or more, ordered by descending score, then
    descending ``updated_at``, then ascending ``id``. Ids are unique, so the
    order is total and two calls over an unchanged set agree exactly.
    """

    scored: list[tuple[MemoryRecord, int]] = []
    for record in records:
        score = relevance_score(record, terms)
        if score >= 1:
            scored.append((record, score))

    scored.sort(
        key=lambda pair: (-pair[1], -timestamp_epoch(pair[0].updated_at), pair[0].id)
    )
    ranked = [record for record, _ in scored]
    if top_k is not None:
        ranked = ranked[: max(0, top_k)]
    return ranked


def timestamp_epoch(timestamp: str) -> float:
    """Convert an ISO-8601 timestamp to a sortable float.

    An unparseable value sorts oldest rather than raising, so one malformed
    record cannot break a recall or make eviction non-deterministic.
    """

    try:
        return datetime.fromisoformat(timestamp).timestamp()
    except (TypeError, ValueError):
        return float("-inf")


__all__ = [
    "derive_query_terms",
    "normalize_content",
    "rank_records",
    "relevance_score",
    "searchable_text",
    "timestamp_epoch",
]
