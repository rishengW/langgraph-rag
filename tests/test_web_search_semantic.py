from __future__ import annotations

from src.web_search import SearchResult, discover_search_results_from_web
from src.web_search import discovery as discovery_module
from src.web_search.semantic import (
    SEMANTIC_MAX_BONUS,
    build_semantic_scorer,
    cosine_similarity,
    semantic_bonus,
)


class StubScorer:
    """Deterministic scorer so tests never load an embedding model."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def similarities(self, query: str, texts) -> list[float]:
        self.calls.append((query, tuple(texts)))
        return [
            next(
                (score for token, score in self.scores.items() if token in text),
                0.0,
            )
            for text in texts
        ]


class ResultProvider:
    provider_name = "bing"

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search_results(self, _query: str, max_results: int = 20) -> list[SearchResult]:
        return self.results[:max_results]

    def search(self, query: str, max_results: int = 20) -> list[str]:
        return [result.url for result in self.search_results(query, max_results)]


def test_cosine_similarity_handles_degenerate_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_semantic_bonus_only_rewards_similarity_above_the_threshold():
    assert semantic_bonus(0.2, 0.35) == 0
    assert semantic_bonus(0.35, 0.35) == 0
    assert semantic_bonus(1.0, 0.35) == SEMANTIC_MAX_BONUS
    assert 0 < semantic_bonus(0.6, 0.35) < SEMANTIC_MAX_BONUS


def test_build_semantic_scorer_is_disabled_by_default(isolated_settings):
    assert build_semantic_scorer(isolated_settings()) is None
    scorer = build_semantic_scorer(
        isolated_settings(web_search_semantic_filter_enabled=True),
    )
    assert scorer is not None


def test_semantic_bonus_rescues_a_cross_language_result(monkeypatch, isolated_settings):
    """An English query with a Mandarin snippet has near-zero lexical overlap."""

    result = SearchResult(
        url="https://www.njmetro.com.cn/gsgk",
        title="\u5357\u4eac\u5730\u94c1\u8fd0\u8425\u60c5\u51b5",
        snippet="\u5357\u4eac\u5730\u94c1\u5171\u8fd0\u8425 14 \u6761\u7ebf\u8def",
    )
    provider = ResultProvider([result])
    monkeypatch.setattr(
        discovery_module,
        "get_search_provider",
        lambda _name, _settings: provider,
    )
    question = "how many metro lines does Nanjing operate"

    without_semantics = discover_search_results_from_web(
        question,
        isolated_settings(web_search_provider="bing", web_search_top_k=3),
        provider=provider,
    )

    scorer = StubScorer({"\u5357\u4eac\u5730\u94c1": 0.82})
    monkeypatch.setattr(
        discovery_module,
        "build_semantic_scorer",
        lambda _settings: scorer,
    )
    with_semantics = discover_search_results_from_web(
        question,
        isolated_settings(
            web_search_provider="bing",
            web_search_top_k=3,
            web_search_semantic_filter_enabled=True,
        ),
        provider=provider,
    )

    assert without_semantics == []
    assert [item.url for item in with_semantics] == [result.url]
    assert scorer.calls


def test_semantic_scoring_is_skipped_when_the_flag_is_off(monkeypatch, isolated_settings):
    calls: list[str] = []

    def scorer_factory(_settings):
        calls.append("built")
        return None

    monkeypatch.setattr(discovery_module, "build_semantic_scorer", scorer_factory)
    provider = ResultProvider(
        [
            SearchResult(
                url="https://example.com/news/2026/metro",
                title="Nanjing metro lines 2026",
                snippet="Nanjing metro operates 14 lines in 2026.",
            )
        ]
    )

    results = discover_search_results_from_web(
        "Nanjing metro lines 2026",
        isolated_settings(web_search_provider="bing", web_search_top_k=3),
        provider=provider,
    )

    assert [item.url for item in results] == ["https://example.com/news/2026/metro"]
    assert calls == ["built"]
