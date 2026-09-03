from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

from src.backend.graph.nodes.web_answer import _rank_pages_by_publication_date
from src.backend.web_search.date_extractor import extract_publication_date
from src.backend.web_search.factory import SUPPORTED_PROVIDER_NAMES
from src.backend.web_search.recency import assess_publication_date


def test_publication_date_extraction_prefers_published_over_modified():
    html = """
    <html><head>
      <meta property="article:modified_time" content="2026-07-18T09:00:00+08:00">
      <script type="application/ld+json">
        {
          "@type": "NewsArticle",
          "datePublished": "2026-06-12",
          "dateModified": "2026-07-18"
        }
      </script>
    </head><body>Article</body></html>
    """

    assert extract_publication_date(html) == date(2026, 6, 12)


def test_publication_date_extraction_uses_modified_date_as_fallback():
    html = """
    <html><head>
      <script type="application/ld+json">
        {"@type": "Article", "dateModified": "2026-07-18"}
      </script>
    </head><body>Article</body></html>
    """

    assert extract_publication_date(html) == date(2026, 7, 18)


def test_recency_assessment_is_neutral_for_ordinary_or_undated_pages():
    published = date(2020, 1, 1)

    ordinary = assess_publication_date(published, "Explain the TCP handshake")
    undated_current = assess_publication_date(None, "What is the current release?")

    assert ordinary.applies is False
    assert ordinary.score == 0
    assert undated_current.applies is True
    assert undated_current.score == 0
    assert undated_current.conflicts_with_required_year is False


def test_recency_assessment_prefers_newer_pages_for_current_queries():
    today = date(2026, 7, 20)
    recent = assess_publication_date(date(2026, 7, 1), "latest release", today=today)
    old = assess_publication_date(date(2022, 7, 1), "latest release", today=today)

    assert recent.score > 0
    assert old.score < 0
    assert recent.score > old.score


def test_single_explicit_year_rejects_only_known_conflicts():
    matching = assess_publication_date(date(2025, 5, 1), "2025 policy changes")
    conflicting = assess_publication_date(date(2026, 1, 1), "2025 policy changes")
    undated = assess_publication_date(None, "2025 policy changes")

    assert matching.conflicts_with_required_year is False
    assert matching.score > 0
    assert conflicting.conflicts_with_required_year is True
    assert undated.conflicts_with_required_year is False


def test_multi_year_queries_leave_publication_year_neutral():
    assessment = assess_publication_date(
        date(2026, 1, 1),
        "Compare the 2024 and 2025 policies",
    )

    assert assessment.applies is True
    assert assessment.score == 0
    assert assessment.conflicts_with_required_year is False


def test_web_answer_date_ranking_is_stable_and_filters_year_conflicts():
    today = date.today()
    old = SimpleNamespace(url="https://old.test", publication_date=today - timedelta(days=1200))
    undated = SimpleNamespace(url="https://undated.test", publication_date=None)
    recent = SimpleNamespace(url="https://recent.test", publication_date=today - timedelta(days=10))

    ranked, conflicts = _rank_pages_by_publication_date(
        [old, undated, recent],
        "What is the latest status?",
    )

    assert [page.url for page in ranked] == [recent.url, undated.url, old.url]
    assert conflicts == []

    matching = SimpleNamespace(url="https://matching.test", publication_date=date(2025, 6, 1))
    conflicting = SimpleNamespace(
        url="https://conflicting.test",
        publication_date=date(2026, 1, 1),
    )
    undated_2025 = SimpleNamespace(url="https://undated-2025.test", publication_date=None)
    ranked, conflicts = _rank_pages_by_publication_date(
        [conflicting, undated_2025, matching],
        "2025 policy changes",
    )

    assert [page.url for page in ranked] == [matching.url, undated_2025.url]
    assert conflicts == [conflicting.url]


def test_web_search_skill_provider_table_matches_factory_registry():
    skill_path = Path(__file__).resolve().parents[1] / "src" / "backend" / "web_search" / "SKILL.md"
    documented = tuple(
        re.findall(
            r"^\| `([a-z_]+)` \| (?:JSON API|HTML) \|",
            skill_path.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    )

    assert documented == SUPPORTED_PROVIDER_NAMES
