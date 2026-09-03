from __future__ import annotations

from threading import Event
from time import monotonic

from src.backend.web_search import SearchResult, discover_search_results_from_web
from src.backend.web_search import discovery as discovery_module


class ResultProvider:
    def __init__(self, name: str, results: list[SearchResult]) -> None:
        self.provider_name = name
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int = 20) -> list[str]:
        return [result.url for result in self.search_results(query, max_results)]

    def search_results(
        self,
        query: str,
        max_results: int = 20,
    ) -> list[SearchResult]:
        self.calls.append((query, max_results))
        return self.results[:max_results]


def test_mandarin_prefers_configured_api_provider_pair(
    monkeypatch,
    isolated_settings,
):
    query = "\u5357\u4eac\u5730\u94c16\u53f7\u7ebf\u5f00\u901a\u72b6\u6001"
    providers = {
        name: ResultProvider(
            name,
            [
                SearchResult(
                    url=f"https://{name}.example.com/news/metro-6",
                    title=query,
                    snippet=f"{query} \u5b98\u65b9\u516c\u544a",
                )
            ],
        )
        for name in ("serper", "brave", "baidu", "bing", "duckduckgo")
    }
    monkeypatch.setattr(
        discovery_module,
        "get_search_provider",
        lambda name, _settings: providers[name],
    )

    results = discover_search_results_from_web(
        query,
        isolated_settings(
            serper_api_key="serper-key",
            brave_search_api_key="brave-key",
            web_search_top_k=4,
            web_search_min_url_score=0,
        ),
    )

    assert {result.provider for result in results} == {"serper", "brave"}
    assert len(providers["serper"].calls) == 1
    assert len(providers["brave"].calls) == 1
    assert providers["baidu"].calls == []
    assert providers["bing"].calls == []


def test_provider_circuit_opens_after_consecutive_failures(isolated_settings):
    class FailingProvider(ResultProvider):
        def search_results(self, query: str, max_results: int = 20) -> list[SearchResult]:
            raise TimeoutError("planned timeout")

    provider = FailingProvider("serper", [])
    discovery_module._provider_cooldowns.clear()
    discovery_module._provider_failure_counts.clear()
    try:
        discovery_module._discover_results_with_provider(
            "query",
            isolated_settings(),
            provider,
        )
        discovery_module._discover_results_with_provider(
            "query",
            isolated_settings(),
            provider,
        )

        assert discovery_module._provider_cooldown_remaining("serper") > 0
    finally:
        discovery_module._provider_cooldowns.clear()
        discovery_module._provider_failure_counts.clear()


def test_chinese_search_blends_baidu_and_bing_before_duckduckgo(
    monkeypatch,
    isolated_settings,
):
    providers = {
        "baidu": ResultProvider(
            "baidu",
            [
                SearchResult(
                    url="https://metro.example.cn/news/2026/lines",
                    title="2026年南京地铁线路数量",
                    snippet="南京地铁线网运营线路共14条",
                )
            ],
        ),
        "bing": ResultProvider(
            "bing",
            [
                SearchResult(
                    url="https://transit.example.com/reports/2025/nanjing",
                    title="2025南京地铁线路统计",
                    snippet="南京地铁共有13条运营线路",
                )
            ],
        ),
        "duckduckgo": ResultProvider(
            "duckduckgo",
            [SearchResult(url="https://fallback.example.com/metro")],
        ),
    }
    monkeypatch.setattr(
        discovery_module,
        "get_search_provider",
        lambda name, _settings: providers[name],
    )

    results = discover_search_results_from_web(
        "南京地铁线路数量 2025 2026 几条线",
        isolated_settings(web_search_top_k=4, web_search_min_url_score=45),
    )

    assert {result.provider for result in results} == {"baidu", "bing"}
    expected_calls = [
        (
            "南京地铁线路数量 2025 2026 几条线 运营线路总数",
            20,
        )
    ]
    assert providers["baidu"].calls == expected_calls
    assert providers["bing"].calls == expected_calls
    assert providers["duckduckgo"].calls == []


def test_chinese_search_uses_duckduckgo_only_after_primary_pair_misses(
    monkeypatch,
    isolated_settings,
):
    providers = {
        "baidu": ResultProvider("baidu", []),
        "bing": ResultProvider("bing", []),
        "duckduckgo": ResultProvider(
            "duckduckgo",
            [SearchResult(url="https://example.com/news/2026/metro")],
        ),
    }
    monkeypatch.setattr(
        discovery_module,
        "get_search_provider",
        lambda name, _settings: providers[name],
    )

    results = discover_search_results_from_web(
        "南京地铁 2026",
        isolated_settings(web_search_top_k=1),
    )

    assert [result.provider for result in results] == ["duckduckgo"]
    assert len(providers["baidu"].calls) == 1
    assert len(providers["bing"].calls) == 1
    assert len(providers["duckduckgo"].calls) == 1


def test_provider_stage_timeout_allows_final_fallback_without_waiting_for_workers(
    monkeypatch,
    isolated_settings,
):
    release = Event()

    class BlockingProvider:
        def __init__(self, name: str) -> None:
            self.provider_name = name

        def search(self, _query: str, _max_results: int = 20) -> list[str]:
            release.wait(timeout=2)
            return []

    fallback = ResultProvider(
        "duckduckgo",
        [SearchResult(url="https://example.com/news/2026/metro")],
    )
    providers = {
        "baidu": BlockingProvider("baidu"),
        "bing": BlockingProvider("bing"),
        "duckduckgo": fallback,
    }
    monkeypatch.setattr(
        discovery_module,
        "get_search_provider",
        lambda name, _settings: providers[name],
    )

    started_at = monotonic()
    try:
        results = discover_search_results_from_web(
            "南京地铁 2026",
            isolated_settings(
                web_search_provider_timeout_seconds=0.05,
                web_search_deadline_seconds=0.5,
                web_search_top_k=1,
            ),
        )
    finally:
        release.set()
    elapsed = monotonic() - started_at

    assert [result.provider for result in results] == ["duckduckgo"]
    assert elapsed < 0.4


def test_search_query_is_prepared_once_for_a_chinese_provider_blend(
    monkeypatch,
    isolated_settings,
):
    prepared_questions: list[str] = []
    providers = {
        "baidu": ResultProvider(
            "baidu",
            [SearchResult(url="https://example.cn/news/2026/metro")],
        ),
        "bing": ResultProvider(
            "bing",
            [SearchResult(url="https://example.com/news/2026/metro")],
        ),
        "duckduckgo": ResultProvider("duckduckgo", []),
    }

    def prepare(question, _settings):
        prepared_questions.append(question)
        return "prepared metro query 2026"

    monkeypatch.setattr(discovery_module, "build_search_query", prepare)
    monkeypatch.setattr(
        discovery_module,
        "get_search_provider",
        lambda name, _settings: providers[name],
    )

    discover_search_results_from_web(
        "南京地铁 2026",
        isolated_settings(web_search_top_k=2),
    )

    assert prepared_questions == ["南京地铁 2026"]
    assert providers["baidu"].calls[0][0] == "prepared metro query 2026"
    assert providers["bing"].calls[0][0] == "prepared metro query 2026"


def test_duckduckgo_configuration_still_keeps_it_as_final_fallback():
    assert discovery_module._fallback_provider_names("duckduckgo", "English query") == [
        "bing",
        "baidu",
        "duckduckgo",
    ]


def test_single_provider_stage_does_not_create_an_orphan_executor(
    monkeypatch,
    isolated_settings,
):
    provider = ResultProvider(
        "bing",
        [
            SearchResult(
                url="https://example.com/news/2026/deepseek",
                title="DeepSeek release 2026",
            )
        ],
    )
    monkeypatch.setattr(
        discovery_module,
        "ThreadPoolExecutor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("single-provider stages must run directly")
        ),
    )
    monkeypatch.setattr(
        discovery_module,
        "get_search_provider",
        lambda _name, _settings: provider,
    )

    results = discover_search_results_from_web(
        "DeepSeek release 2026",
        isolated_settings(web_search_provider="bing", web_search_top_k=1),
        provider=provider,
    )

    assert [result.provider for result in results] == ["bing"]


def test_thin_first_stage_still_runs_fallback_providers_and_merges_results(
    monkeypatch,
    isolated_settings,
):
    """One usable URL is thin recall, not a reason to skip every fallback."""

    providers = {
        "tavily": ResultProvider(
            "tavily",
            [
                SearchResult(
                    url="https://example.com/news/2026/deepseek-release",
                    title="DeepSeek release 2026",
                    snippet="DeepSeek released a new model in 2026.",
                )
            ],
        ),
        "bing": ResultProvider(
            "bing",
            [
                SearchResult(
                    url="https://api-docs.deepseek.com/news/2026/release",
                    title="DeepSeek release notes 2026",
                    snippet="Official DeepSeek 2026 release notes.",
                )
            ],
        ),
        "baidu": ResultProvider(
            "baidu",
            [
                SearchResult(
                    url="https://other.example.com/news/2026/deepseek",
                    title="DeepSeek release 2026",
                    snippet="DeepSeek 2026 model coverage.",
                )
            ],
        ),
        "duckduckgo": ResultProvider("duckduckgo", []),
    }
    monkeypatch.setattr(
        discovery_module,
        "get_search_provider",
        lambda name, _settings: providers[name],
    )

    results = discover_search_results_from_web(
        "DeepSeek release 2026",
        isolated_settings(
            web_search_provider="tavily",
            tavily_api_key="tavily-key",
            web_search_provider_fanout=1,
            web_search_top_k=6,
        ),
    )

    assert {result.provider for result in results} == {"tavily", "bing"}
    assert len(providers["tavily"].calls) == 1
    assert len(providers["bing"].calls) == 1
    assert providers["baidu"].calls == []


def test_stage_threshold_scales_down_with_a_small_top_k():
    assert discovery_module._stage_result_threshold(0) == 2
    assert discovery_module._stage_result_threshold(1) == 1
    assert discovery_module._stage_result_threshold(6) == 2
