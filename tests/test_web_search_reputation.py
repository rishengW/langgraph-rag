from __future__ import annotations

from src.backend.graph.nodes.merge import merge_factory
from src.backend.web_search.reputation import (
    MAX_REPUTATION_BONUS,
    MAX_REPUTATION_PENALTY,
    OUTCOME_GROUNDED,
    OUTCOME_REJECTED,
    OUTCOME_UNREACHABLE,
    DomainReputationStore,
    DomainStats,
    build_reputation_store,
    reputation_delta,
)


def _store(tmp_path) -> DomainReputationStore:
    return DomainReputationStore(tmp_path / "reputation.sqlite3")


def test_outcomes_accumulate_per_registrable_domain(tmp_path):
    store = _store(tmp_path)

    store.record("https://news.example.com/a", OUTCOME_GROUNDED)
    store.record("https://www.example.com/b", OUTCOME_GROUNDED)
    store.record("https://example.com/c", OUTCOME_REJECTED)
    store.record("https://other.example.org/d", OUTCOME_UNREACHABLE)

    stats = store.stats("https://example.com/anything")
    assert stats is not None
    assert (stats.domain, stats.attempts, stats.grounded, stats.rejected) == (
        "example.com",
        3,
        2,
        1,
    )
    assert store.stats("https://unseen.example.net/x") is None
    assert {item.domain for item in store.all_stats()} == {"example.com", "example.org"}


def test_reputation_is_neutral_until_enough_samples_exist():
    thin = DomainStats(domain="example.com", attempts=3, grounded=0, rejected=3)

    assert reputation_delta(thin, min_samples=5) == 0
    assert reputation_delta(None, min_samples=5) == 0


def test_reputation_rewards_grounding_domains_and_penalizes_junk_domains():
    good = DomainStats(domain="good.com", attempts=10, grounded=10)
    bad = DomainStats(domain="bad.com", attempts=10, grounded=0, rejected=10)
    mixed = DomainStats(domain="mixed.com", attempts=10, grounded=5, rejected=5)

    assert reputation_delta(good, min_samples=5) == MAX_REPUTATION_BONUS
    assert reputation_delta(bad, min_samples=5) == -MAX_REPUTATION_PENALTY
    assert reputation_delta(mixed, min_samples=5) == 0


def test_store_survives_reopening_the_database(tmp_path):
    first = _store(tmp_path)
    for _ in range(6):
        first.record("https://example.com/a", OUTCOME_GROUNDED)

    reopened = DomainReputationStore(tmp_path / "reputation.sqlite3")
    stats = reopened.stats("example.com")

    assert stats is not None
    assert stats.attempts == 6
    assert reputation_delta(stats, min_samples=5) == MAX_REPUTATION_BONUS


def test_build_reputation_store_respects_the_flag(tmp_path, isolated_settings):
    disabled = isolated_settings(web_search_domain_reputation_enabled=False)
    enabled = isolated_settings(web_search_domain_reputation_enabled=True)

    assert build_reputation_store(disabled) is None
    store = build_reputation_store(enabled)
    assert store is not None
    assert store.database_path.parent.name == "web-search"


def test_merge_prefers_a_domain_with_a_better_learned_reputation(
    tmp_path,
    isolated_settings,
):
    settings = isolated_settings(
        web_search_domain_reputation_enabled=True,
        web_search_domain_reputation_min_samples=5,
        web_search_top_k=2,
    )
    store = build_reputation_store(settings)
    assert store is not None
    for _ in range(6):
        store.record("https://reliable.example.com/x", OUTCOME_GROUNDED)
        store.record("https://junk.example.net/x", OUTCOME_REJECTED)

    metadata = [
        {
            "url": "https://junk.example.net/news/metro",
            "title": "Nanjing metro lines 2026",
            "snippet": "Nanjing metro operates 14 lines in 2026.",
            "provider": "bing",
            "provider_rank": 0,
            "relevance_score": 20,
            "quality_score": 80,
        },
        {
            "url": "https://reliable.example.com/news/metro",
            "title": "Nanjing metro lines 2026",
            "snippet": "Nanjing metro operates 14 lines in 2026.",
            "provider": "bing",
            "provider_rank": 1,
            "relevance_score": 20,
            "quality_score": 80,
        },
    ]
    merge = merge_factory(settings, question_resolver=lambda _state: "Nanjing metro lines 2026")

    result = merge(
        {
            "current_question": "Nanjing metro lines 2026",
            "source_urls": [],
            "web_search_results": [[item["url"] for item in metadata]],
            "web_search_result_metadata": [metadata],
            "messages": [],
        }
    )

    assert result["source_urls"][0] == "https://reliable.example.com/news/metro"


def test_merge_ranking_is_unchanged_when_reputation_is_disabled(isolated_settings):
    settings = isolated_settings(
        web_search_domain_reputation_enabled=False,
        web_search_top_k=2,
    )
    metadata = [
        {
            "url": "https://first.example.com/news/metro",
            "title": "Nanjing metro lines 2026",
            "snippet": "Nanjing metro operates 14 lines in 2026.",
            "provider": "bing",
            "provider_rank": 0,
            "relevance_score": 20,
            "quality_score": 80,
        },
        {
            "url": "https://second.example.com/news/metro",
            "title": "Nanjing metro lines 2026",
            "snippet": "Nanjing metro operates 14 lines in 2026.",
            "provider": "bing",
            "provider_rank": 1,
            "relevance_score": 20,
            "quality_score": 80,
        },
    ]
    merge = merge_factory(settings, question_resolver=lambda _state: "Nanjing metro lines 2026")

    result = merge(
        {
            "current_question": "Nanjing metro lines 2026",
            "source_urls": [],
            "web_search_results": [[item["url"] for item in metadata]],
            "web_search_result_metadata": [metadata],
            "messages": [],
        }
    )

    assert result["source_urls"] == [item["url"] for item in metadata]
