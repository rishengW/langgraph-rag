from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .query_constraints import STATUS_INTENT, detect_query_intents

_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_RECENCY_RE = re.compile(
    r"(?:\b(?:latest|newest|current|recent|now|today|up[ -]?to[ -]?date)\b|"
    r"\u6700\u65b0|\u5f53\u524d|\u73b0\u5728|\u8fd1\u671f|\u6700\u8fd1|"
    r"\u622a\u81f3(?:\u76ee\u524d|\u73b0\u5728)?|\u4eca\u5e74|\u4eca\u65e5|"
    r"\u4eca\u5929)",
    re.I,
)


@dataclass(frozen=True)
class PublicationDateAssessment:
    """Query-aware interpretation of one page's publication date."""

    applies: bool
    score: int = 0
    conflicts_with_required_year: bool = False


def assess_publication_date(
    published_on: date | None,
    query: str,
    *,
    today: date | None = None,
) -> PublicationDateAssessment:
    """Score publication freshness without penalizing missing dates.

    A single explicit year is a hard publication-year constraint. Multi-year
    questions are left to the existing page-text coverage checks because one
    comparison page cannot have multiple publication years. For current-status
    queries, recent pages rank ahead of old pages while undated pages remain
    neutral and eligible.
    """

    requested_years = frozenset(int(year) for year in _YEAR_RE.findall(query or ""))
    recency_requested = bool(
        requested_years
        or STATUS_INTENT in detect_query_intents(query)
        or _RECENCY_RE.search(query or "")
    )
    if not recency_requested or published_on is None:
        return PublicationDateAssessment(applies=recency_requested)

    if len(requested_years) == 1:
        required_year = next(iter(requested_years))
        conflicts = published_on.year != required_year
        return PublicationDateAssessment(
            applies=True,
            score=-60 if conflicts else 20,
            conflicts_with_required_year=conflicts,
        )

    if requested_years:
        return PublicationDateAssessment(applies=True)

    age_days = ((today or date.today()) - published_on).days
    if age_days < -31:
        return PublicationDateAssessment(applies=True, score=-12)
    if age_days <= 90:
        return PublicationDateAssessment(applies=True, score=18)
    if age_days <= 365:
        return PublicationDateAssessment(applies=True, score=12)
    if age_days <= 730:
        return PublicationDateAssessment(applies=True, score=4)

    years_old = max(0, (age_days - 730) // 365)
    return PublicationDateAssessment(
        applies=True,
        score=-min(18, 4 + years_old * 3),
    )


__all__ = ["PublicationDateAssessment", "assess_publication_date"]
