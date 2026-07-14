from __future__ import annotations

import base64
import logging
from urllib.parse import parse_qs, urlparse

import pytest

from src.core.web_search import discover_urls_from_web as core_discover_urls_from_web
from src.web_search import (
    BaiduWebSearch,
    BingWebSearch,
    DuckDuckGoWebSearch,
    RankedSearchResult,
    SearchResult,
    WebSearchProvider,
    build_web_search_tool,
    discover_search_results_from_web,
    discover_urls_from_web,
    format_web_search_results,
    get_search_provider,
)
from src.web_search import baidu as baidu_module
from src.web_search import bing as bing_module
from src.web_search import discovery as discovery_module
from src.web_search.common import is_noise_url, select_top_urls, url_quality_score


class StaticSearchProvider:
    provider_name = "static"

    def __init__(self, urls: list[str]) -> None:
        self.urls = urls
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int = 20) -> list[str]:
        self.calls.append((query, max_results))
        return self.urls


@pytest.fixture(autouse=True)
def clear_web_search_provider_cooldowns():
    discovery_module._provider_cooldowns.clear()
    yield
    discovery_module._provider_cooldowns.clear()


def test_provider_implementations_satisfy_protocol():
    assert isinstance(BaiduWebSearch(), WebSearchProvider)
    assert isinstance(BingWebSearch(), WebSearchProvider)
    assert isinstance(DuckDuckGoWebSearch(), WebSearchProvider)


def test_factory_selects_bing_default_and_provider_aliases(isolated_settings):
    settings = isolated_settings(
        web_search_region="us-en",
        web_search_timelimit="w",
        web_search_verify_ssl=False,
    )

    bing = get_search_provider("bing", settings)
    baidu = get_search_provider("baidu", settings)
    duckduckgo = get_search_provider("ddg", settings)
    default_provider = get_search_provider(config=isolated_settings())
    settings_only_provider = get_search_provider(
        isolated_settings(web_search_provider="baidu")
    )

    assert isinstance(bing, BingWebSearch)
    assert bing.market == "en-US"
    assert bing.timelimit == "w"
    assert bing.verify_ssl is False
    assert isinstance(baidu, BaiduWebSearch)
    assert baidu.verify_ssl is False
    assert isinstance(duckduckgo, DuckDuckGoWebSearch)
    assert duckduckgo.region == "us-en"
    assert duckduckgo.timelimit == "w"
    assert duckduckgo.verify_ssl is False
    assert isinstance(default_provider, BingWebSearch)
    assert isinstance(settings_only_provider, BaiduWebSearch)


def test_factory_rejects_unsupported_provider(isolated_settings):
    settings = isolated_settings(web_search_provider="yahoo")

    with pytest.raises(ValueError, match="Unsupported WEB_SEARCH_PROVIDER 'yahoo'"):
        get_search_provider(config=settings)


# REFACTOR: Bing timelimit mapping and URL construction coverage.
def test_bing_timelimit_filter_maps_shared_values_conservatively():
    assert bing_module.bing_timelimit_filter("d") == 'ex1:"ez1"'
    assert bing_module.bing_timelimit_filter("day") == 'ex1:"ez1"'
    assert bing_module.bing_timelimit_filter(" w ") == 'ex1:"ez2"'
    assert bing_module.bing_timelimit_filter("week") == 'ex1:"ez2"'
    assert bing_module.bing_timelimit_filter("M") == 'ex1:"ez3"'
    assert bing_module.bing_timelimit_filter("month") == 'ex1:"ez3"'
    assert bing_module.bing_timelimit_filter("y") is None
    assert bing_module.bing_timelimit_filter("latest") is None
    assert bing_module.bing_timelimit_filter(None) is None


def test_bing_search_url_includes_recency_filter_when_supported():
    url = bing_module.build_bing_search_url(
        query="MiniMax latest model",
        max_results=0,
        market="en-US",
        timelimit="w",
    )
    params = parse_qs(urlparse(url).query)

    assert params["q"] == ["MiniMax latest model"]
    assert params["count"] == ["1"]
    assert params["mkt"] == ["en-US"]
    assert params["setlang"] == ["en"]
    assert params["filters"] == ['ex1:"ez2"']


def test_bing_search_url_preserves_existing_params_without_timelimit():
    url = bing_module.build_bing_search_url(
        query="MiniMax latest model",
        max_results=10,
        market="zh-CN",
        timelimit=None,
    )
    params = parse_qs(urlparse(url).query)

    assert params == {
        "q": ["MiniMax latest model"],
        "count": ["10"],
        "mkt": ["zh-CN"],
        "setlang": ["zh"],
    }


def test_bing_search_url_omits_unsupported_timelimit_values():
    url = bing_module.build_bing_search_url(
        query="MiniMax latest model",
        max_results=10,
        market="zh-CN",
        timelimit="y",
    )
    params = parse_qs(urlparse(url).query)

    assert "filters" not in params


def test_url_quality_gate_filters_scores_and_deduplicates_candidates():
    urls = [
        "https://example.com/search?q=rag",
        "https://example.com/tag/rag",
        "https://example.com/file/report.pdf",
        "https://www.example.com/news/2026/launch?utm_source=test",
        "http://example.com/news/2026/launch?ref=duplicate",
        "https://example.com/privacy",
        "https://docs.example.com/guide/rag",
        "https://example.com/",
        "https://example.com/blog/rag",
    ]

    selected = select_top_urls(urls, top_k=3)

    assert is_noise_url("https://example.com/search?q=rag")
    assert is_noise_url("https://example.com/tag/rag")
    assert is_noise_url("https://example.com/file/report.pdf")
    assert url_quality_score("https://example.com/") < url_quality_score(
        "https://example.com/blog/rag"
    )
    assert selected == [
        "https://www.example.com/news/2026/launch?utm_source=test",
        "https://docs.example.com/guide/rag",
        "https://example.com/blog/rag",
    ]


def test_url_quality_gate_uses_configurable_threshold_and_debug_logs(caplog):
    caplog.set_level(logging.DEBUG, logger="src.web_search.common")
    urls = [
        "https://example.com/a",
        "https://example.com/blog/2026/rag",
        "not-a-url",
    ]

    selected = select_top_urls(urls, top_k=10, min_score=80)

    assert selected == ["https://example.com/blog/2026/rag"]
    score_records = [
        record
        for record in caplog.records
        if record.message.startswith("Scored web search URL candidate")
    ]
    assert [
        (record.url, record.score, record.min_score, record.usable)
        for record in score_records
    ] == [
        ("https://example.com/a", url_quality_score("https://example.com/a"), 80, False),
        (
            "https://example.com/blog/2026/rag",
            url_quality_score("https://example.com/blog/2026/rag"),
            80,
            True,
        ),
    ]


def test_baidu_provider_reports_verification_page(monkeypatch):
    class FakeHeaders:
        def get_content_charset(self):
            return "utf-8"

    class FakeResponse:
        headers = FakeHeaders()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def geturl(self):
            return "https://wappass.baidu.com/static/captcha/tuxing_v2.html"

        def read(self):
            return "百度安全验证".encode()

    monkeypatch.setattr(baidu_module, "urlopen", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(baidu_module.BaiduVerificationError, match="verification/captcha"):
        BaiduWebSearch().search("MiniMax latest model", 10)


def test_bing_provider_extracts_html_results_and_unwraps_redirect(monkeypatch):
    class FakeHeaders:
        def get_content_charset(self):
            return "utf-8"

    redirected = base64.urlsafe_b64encode(b"https://example.com/redirected").decode(
        "ascii"
    ).rstrip("=")

    class FakeResponse:
        headers = FakeHeaders()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def geturl(self):
            return "https://cn.bing.com/search?q=test"

        def read(self):
            return f"""
            <html>
              <body>
                <ol id="b_results">
                  <li class="b_algo">
                    <h2><a href="https://example.com/a">A</a></h2>
                  </li>
                  <li class="b_algo">
                    <h2><a href="/ck/a?u=a1{redirected}">Redirected</a></h2>
                  </li>
                </ol>
              </body>
            </html>
            """.encode()

    monkeypatch.setattr(bing_module, "urlopen", lambda *args, **kwargs: FakeResponse())

    assert BingWebSearch().search("MiniMax latest model", 10) == [
        "https://example.com/a",
        "https://example.com/redirected",
    ]


def test_bing_provider_reports_verification_page(monkeypatch):
    assert bing_module.is_bing_verification_page(
        "",
        "https://www.bing.com/?q=OpenAI+GPT-5&count=5",
    )

    class FakeHeaders:
        def get_content_charset(self):
            return "utf-8"

    class FakeResponse:
        headers = FakeHeaders()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def geturl(self):
            return "https://www.bing.com/search?q=test"

        def read(self):
            return b"<html><body>captcha challenge</body></html>"

    monkeypatch.setattr(bing_module, "urlopen", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(bing_module.BingVerificationError, match="verification/captcha"):
        BingWebSearch().search("MiniMax latest model", 10)


def test_discover_urls_accepts_injected_provider_and_filters_top_k(isolated_settings):
    settings = isolated_settings(web_search_max_results=4, web_search_top_k=2)
    provider = StaticSearchProvider(
        [
            "https://tieba.baidu.com/p/123",
            "https://example.com/a",
            "not-a-url",
            "https://example.com/b",
            "https://example.com/a",
            "https://example.com/c",
        ]
    )

    urls = discover_urls_from_web("phase two", settings, provider=provider)

    assert urls == ["https://example.com/a", "https://example.com/b"]
    assert provider.calls == [("phase two", 4)]


def test_discover_urls_uses_configured_min_url_score(isolated_settings):
    settings = isolated_settings(
        web_search_max_results=4,
        web_search_min_url_score=80,
        web_search_top_k=2,
    )
    provider = StaticSearchProvider(
        [
            "https://example.com/a",
            "https://example.com/blog/2026/rag",
        ]
    )

    urls = discover_urls_from_web("phase two", settings, provider=provider)

    assert urls == ["https://example.com/blog/2026/rag"]


def test_discover_urls_falls_back_to_alternate_provider(monkeypatch, isolated_settings):
    settings = isolated_settings(
        web_search_provider="baidu",
        web_search_max_results=4,
        web_search_top_k=2,
    )
    providers = {
        "baidu": StaticSearchProvider([]),
        "bing": StaticSearchProvider(
            ["https://example.com/fallback", "https://example.com/second"]
        ),
        "duckduckgo": StaticSearchProvider(["https://example.com/duck"]),
    }

    def fake_get_search_provider(name, config):
        return providers[name]

    monkeypatch.setattr(discovery_module, "get_search_provider", fake_get_search_provider)

    urls = discover_urls_from_web("minimax latest model", settings)

    assert urls == ["https://example.com/fallback", "https://example.com/second"]
    assert providers["baidu"].calls == [("minimax latest model 2026", 4)]
    assert providers["bing"].calls == [("minimax latest model 2026", 4)]
    assert providers["duckduckgo"].calls == []


def test_discover_urls_prefers_baidu_for_predominantly_chinese_query(
    monkeypatch,
    isolated_settings,
):
    settings = isolated_settings(web_search_provider="bing", web_search_top_k=1)
    providers = {
        "baidu": StaticSearchProvider(["https://example.cn/nanjing/metro"]),
        "bing": StaticSearchProvider(["https://example.com/bing"]),
        "duckduckgo": StaticSearchProvider(["https://example.com/duck"]),
    }

    monkeypatch.setattr(
        discovery_module,
        "get_search_provider",
        lambda name, _config: providers[name],
    )

    urls = discover_urls_from_web("南京地铁线路数量 2026", settings)

    assert urls == ["https://example.cn/nanjing/metro"]
    assert providers["baidu"].calls == [("南京地铁线路数量 2026", 20)]
    assert providers["bing"].calls == []
    assert providers["duckduckgo"].calls == []


def test_discover_urls_respects_injected_provider_for_chinese_query(isolated_settings):
    provider = StaticSearchProvider(["https://example.com/pinned"])

    urls = discover_urls_from_web(
        "南京地铁线路数量 2026",
        isolated_settings(web_search_provider="baidu"),
        provider=provider,
    )

    assert urls == ["https://example.com/pinned"]
    assert provider.calls == [("南京地铁线路数量 2026", 20)]


def test_discovery_retains_provider_relevance_metadata(isolated_settings):
    class MetadataProvider:
        provider_name = "bing"

        @staticmethod
        def search(_query: str, _max_results: int = 20) -> list[str]:
            return []

        @staticmethod
        def search_results(_query: str, _max_results: int = 20) -> list[SearchResult]:
            return [
                SearchResult(
                    url="https://example.com/news/deepseek-v4",
                    title="DeepSeek V4 release",
                    snippet="DeepSeek V4 model release details for 2026",
                )
            ]

    results = discover_search_results_from_web(
        "DeepSeek V4 release 2026",
        isolated_settings(web_search_top_k=1),
        provider=MetadataProvider(),
    )

    assert len(results) == 1
    assert isinstance(results[0], RankedSearchResult)
    assert results[0].provider == "bing"
    assert results[0].provider_rank == 0
    assert results[0].relevance_score > 0
    assert results[0].quality_score >= 45


def test_discover_urls_falls_back_when_provider_urls_fail_quality_gates(
    monkeypatch,
    isolated_settings,
):
    settings = isolated_settings(
        web_search_provider="baidu",
        web_search_max_results=4,
        web_search_top_k=1,
    )
    providers = {
        "baidu": StaticSearchProvider(
            [
                "https://example.com/search?q=minimax",
                "https://example.com/tag/models",
            ]
        ),
        "bing": StaticSearchProvider(["https://example.com/news/2026/minimax-model"]),
        "duckduckgo": StaticSearchProvider(["https://example.com/duck"]),
    }

    def fake_get_search_provider(name, config):
        return providers[name]

    monkeypatch.setattr(discovery_module, "get_search_provider", fake_get_search_provider)

    urls = discover_urls_from_web("minimax latest model", settings)

    assert urls == ["https://example.com/news/2026/minimax-model"]
    assert providers["baidu"].calls == [("minimax latest model 2026", 4)]
    assert providers["bing"].calls == [("minimax latest model 2026", 4)]
    assert providers["duckduckgo"].calls == []


def test_discover_urls_skips_baidu_during_verification_cooldown(
    monkeypatch,
    isolated_settings,
):
    settings = isolated_settings(
        web_search_provider="baidu",
        web_search_max_results=4,
        web_search_top_k=1,
    )

    class VerificationBlockedProvider:
        provider_name = "baidu"

        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def search(self, query: str, max_results: int = 20) -> list[str]:
            self.calls.append((query, max_results))
            raise baidu_module.BaiduVerificationError("verification/captcha")

    providers = {
        "baidu": VerificationBlockedProvider(),
        "bing": StaticSearchProvider(["https://example.com/fallback"]),
        "duckduckgo": StaticSearchProvider(["https://example.com/duck"]),
    }

    def fake_get_search_provider(name, config):
        return providers[name]

    monkeypatch.setattr(discovery_module, "get_search_provider", fake_get_search_provider)

    first_urls = discover_urls_from_web("minimax latest model", settings)
    second_urls = discover_urls_from_web("deepseek latest model", settings)

    assert first_urls == ["https://example.com/fallback"]
    assert second_urls == ["https://example.com/fallback"]
    assert providers["baidu"].calls == [("minimax latest model 2026", 4)]
    assert providers["bing"].calls == [
        ("minimax latest model 2026", 4),
        ("deepseek latest model 2026", 4),
    ]
    assert providers["duckduckgo"].calls == []


def test_discover_urls_skips_bing_during_verification_cooldown(
    monkeypatch,
    isolated_settings,
):
    settings = isolated_settings(
        web_search_provider="bing",
        web_search_max_results=4,
        web_search_top_k=1,
    )

    class VerificationBlockedProvider:
        provider_name = "bing"

        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def search(self, query: str, max_results: int = 20) -> list[str]:
            self.calls.append((query, max_results))
            raise bing_module.BingVerificationError("verification/captcha")

    providers = {
        "bing": VerificationBlockedProvider(),
        "baidu": StaticSearchProvider(["https://example.com/fallback"]),
        "duckduckgo": StaticSearchProvider(["https://example.com/duck"]),
    }

    def fake_get_search_provider(name, config):
        return providers[name]

    monkeypatch.setattr(discovery_module, "get_search_provider", fake_get_search_provider)

    first_urls = discover_urls_from_web("minimax latest model", settings)
    second_urls = discover_urls_from_web("deepseek latest model", settings)

    assert first_urls == ["https://example.com/fallback"]
    assert second_urls == ["https://example.com/fallback"]
    assert providers["bing"].calls == [("minimax latest model 2026", 4)]
    assert providers["baidu"].calls == [
        ("minimax latest model 2026", 4),
        ("deepseek latest model 2026", 4),
    ]
    assert providers["duckduckgo"].calls == []


def test_discover_urls_disabled_does_not_call_provider(isolated_settings):
    settings = isolated_settings(web_search_enabled=False)
    provider = StaticSearchProvider(["https://example.com/a"])

    assert discover_urls_from_web("phase two", settings, provider=provider) == []
    assert provider.calls == []


def test_build_web_search_tool_formats_injected_provider_results(isolated_settings):
    settings = isolated_settings(web_search_max_results=5, web_search_top_k=2)
    provider = StaticSearchProvider(
        [
            "https://example.com/a",
            "not-a-url",
            "https://example.com/b",
            "https://example.com/c",
        ]
    )

    tool = build_web_search_tool(settings, provider=provider)
    result = tool.invoke({"query": "phase two", "max_results": 3})

    assert tool.name == "live_web_search"
    assert result == "\n".join(
        [
            "Live web search results for: phase two",
            "1. https://example.com/a",
            "2. https://example.com/b",
        ]
    )
    assert provider.calls == [("phase two", 3)]


def test_build_web_search_tool_reports_empty_results(isolated_settings):
    provider = StaticSearchProvider([])
    tool = build_web_search_tool(isolated_settings(), provider=provider)

    assert tool.invoke({"query": "missing"}) == (
        "No live web search results found for: missing"
    )


def test_format_web_search_results_keeps_ranked_urls():
    assert format_web_search_results("query", ["https://example.com/a"]) == (
        "Live web search results for: query\n1. https://example.com/a"
    )


def test_core_import_path_keeps_provider_injection(isolated_settings):
    settings = isolated_settings(web_search_max_results=2, web_search_top_k=1)
    provider = StaticSearchProvider(["https://example.com/a", "https://example.com/b"])

    assert core_discover_urls_from_web("compat", settings, provider=provider) == [
        "https://example.com/a"
    ]


# ── query_prep ──────────────────────────────────────────────────────────────


def test_prepare_search_query_strips_filler_words():
    from src.web_search.query_prep import prepare_search_query

    result = prepare_search_query("what is the latest model of deepseek")
    assert "what" not in result.lower().split()
    assert "is" not in result.lower().split()
    assert "the" not in result.lower().split()
    assert "of" not in result.lower().split()
    assert "deepseek" in result.lower()
    assert "model" in result.lower()


def test_prepare_search_query_appends_year_for_time_sensitive():
    from src.web_search.query_prep import prepare_search_query

    result = prepare_search_query("what is the latest model of deepseek")
    assert "2026" in result


def test_prepare_search_query_preserves_non_time_sensitive():
    from src.web_search.query_prep import prepare_search_query

    result = prepare_search_query("what is deepseek")
    # Not time-sensitive: "what" and "is" stripped, but no year appended
    assert "what" not in result.lower().split()
    assert "is" not in result.lower().split()
    assert "2026" not in result


def test_prepare_search_query_does_not_double_year():
    from src.web_search.query_prep import prepare_search_query

    result = prepare_search_query("deepseek v4 pro 2026 release")
    assert result.count("2026") == 1


def test_prepare_search_query_preserves_named_entities():
    from src.web_search.query_prep import prepare_search_query

    result = prepare_search_query("tell me about DeepSeek V4 Pro")
    assert "DeepSeek" in result
    assert "V4" in result
    assert "Pro" in result


def test_rewrite_search_query_llm_skips_when_no_api_key(isolated_settings):
    from src.web_search.query_prep import rewrite_search_query_llm

    settings = isolated_settings(deepseek_api_key="")
    result = rewrite_search_query_llm("what is the latest model of deepseek", settings)
    assert result is None


def test_rewrite_search_query_llm_returns_rewritten_query(isolated_settings):
    from src.web_search.query_prep import rewrite_search_query_llm

    settings = isolated_settings(
        deepseek_api_key="sk-test",
        deepseek_base_url="https://api.deepseek.com",
    )

    class FakeLLM:
        def invoke(self, messages):
            class FakeResponse:
                content = "DeepSeek V4 Pro latest model"
            return FakeResponse()

    result = rewrite_search_query_llm(
        "what is the latest model of deepseek", settings, _llm=FakeLLM()
    )
    assert result == "DeepSeek V4 Pro latest model"


def test_build_search_query_skips_llm_by_default(isolated_settings, monkeypatch):
    from src.web_search import query_prep as qp

    settings = isolated_settings(
        deepseek_api_key="sk-test",
        deepseek_base_url="https://api.deepseek.com",
    )

    def fake_rewrite(question, settings_obj, *, _llm=None):
        raise AssertionError("LLM rewrite must be opt-in")

    monkeypatch.setattr(qp, "rewrite_search_query_llm", fake_rewrite)
    result = qp.build_search_query("what is the latest model of deepseek", settings)
    assert "deepseek" in result.lower()
    assert "what" not in result.lower()


def test_build_search_query_uses_llm_when_explicitly_enabled(
    isolated_settings, monkeypatch
):
    from src.web_search import query_prep as qp

    settings = isolated_settings(
        deepseek_api_key="sk-test",
        web_search_llm_query_rewrite_enabled=True,
    )
    calls = []

    def fake_rewrite(question, settings_obj, *, _llm=None):
        calls.append((question, settings_obj))
        return "DeepSeek V4 Pro latest model"

    monkeypatch.setattr(qp, "rewrite_search_query_llm", fake_rewrite)

    result = qp.build_search_query("what is the latest model of deepseek", settings)

    assert calls == [("what is the latest model of deepseek", settings)]
    assert result == "DeepSeek V4 Pro latest model 2026"


# ── url_quality_score query relevance ───────────────────────────────────────


def test_url_quality_score_relevance_bonus():
    from src.web_search.common import url_quality_score

    base = url_quality_score("https://example.com/blog/deepseek-v4-pro")
    with_query = url_quality_score(
        "https://example.com/blog/deepseek-v4-pro",
        query="deepseek v4 pro",
    )
    assert with_query > base

    no_match = url_quality_score(
        "https://other.com/about",
        query="deepseek v4 pro",
    )
    assert no_match < with_query


def test_select_top_urls_passes_query_through(isolated_settings):
    from src.web_search.common import select_top_urls

    urls = [
        "https://deepseek.net/docs/v4",
        "https://other.com/about",
        "https://deepseek.net/blog",
    ]
    selected = select_top_urls(urls, top_k=3, query="deepseek v4")
    # URLs matching the query should rank higher
    assert "deepseek.net" in selected[0]


# ── snippet-aware relevance ranking ─────────────────────────────────────────


class SnippetSearchProvider:
    """Provider that returns SearchResult objects with title/snippet text."""

    provider_name = "snippet"

    def __init__(self, results):
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search_results(self, query: str, max_results: int = 20):
        self.calls.append((query, max_results))
        return self.results

    def search(self, query: str, max_results: int = 20):
        return [result.url for result in self.results]


def test_text_relevance_delta_rewards_matches_and_penalizes_misses():
    from src.web_search.common import text_relevance_delta

    full = text_relevance_delta("DeepSeek V4 model release", "deepseek v4 model")
    partial = text_relevance_delta("DeepSeek announcement", "deepseek v4 model")
    miss = text_relevance_delta("unrelated cooking recipe", "deepseek v4 model")
    empty = text_relevance_delta("", "deepseek v4 model")

    assert full > partial > 0
    assert miss < 0
    assert empty == 0


def test_result_quality_score_uses_snippet_text():
    from src.web_search.common import SearchResult, result_quality_score

    on_topic = SearchResult(
        url="https://news.example.com/2026/launch",
        title="DeepSeek V4 launch",
        snippet="DeepSeek released its V4 model in 2026.",
    )
    off_topic = SearchResult(
        url="https://news.example.com/2026/launch",
        title="Best pizza recipes",
        snippet="A roundup of pizza recipes for dinner.",
    )
    query = "deepseek v4 model"

    assert result_quality_score(on_topic, query=query) > result_quality_score(
        off_topic, query=query
    )


def test_select_top_results_ranks_relevant_snippet_first():
    from src.web_search.common import SearchResult, select_top_results

    results = [
        SearchResult(
            url="https://example.com/blog/cooking",
            title="Weeknight cooking",
            snippet="Recipes and meal prep ideas.",
        ),
        SearchResult(
            url="https://example.com/news/announcement",
            title="DeepSeek V4 model released",
            snippet="The new DeepSeek V4 model launched in 2026.",
        ),
    ]

    selected = select_top_results(results, top_k=2, query="deepseek v4 model")

    assert selected[0] == "https://example.com/news/announcement"


def test_discover_urls_prefers_relevant_snippet(isolated_settings):
    from src.web_search.common import SearchResult

    settings = isolated_settings(web_search_max_results=5, web_search_top_k=1)
    provider = SnippetSearchProvider(
        [
            SearchResult(
                url="https://example.com/blog/random",
                title="Unrelated gardening tips",
                snippet="How to grow tomatoes at home.",
            ),
            SearchResult(
                url="https://example.com/news/deepseek",
                title="DeepSeek V4 model release",
                snippet="DeepSeek announced the V4 model in 2026.",
            ),
        ]
    )

    urls = discover_urls_from_web("what is the latest deepseek model", settings, provider=provider)

    assert urls == ["https://example.com/news/deepseek"]
    assert provider.calls and provider.calls[0][0]


def test_discover_urls_drops_off_topic_snippet_below_gate(isolated_settings):
    from src.web_search.common import SearchResult

    settings = isolated_settings(web_search_max_results=5, web_search_top_k=3)
    provider = SnippetSearchProvider(
        [
            SearchResult(
                url="https://example.com/blog/pizza",
                title="Pizza recipes",
                snippet="The best homemade pizza recipes for any night.",
            ),
        ]
    )

    urls = discover_urls_from_web("deepseek v4 model architecture", settings, provider=provider)

    assert urls == []


# ── generic-term hardening (MCP vs "model" regression) ───────────────────────


def test_text_relevance_delta_ignores_lone_generic_term_match():
    """A page that only shares the generic word 'model' is not on-topic."""
    from src.web_search.common import text_relevance_delta

    query = "model context protocol mcp anthropic"
    # Cambridge "model" definition / Tesla "Model S" news share only "model".
    generic_only = text_relevance_delta("model definition meaning", query)
    on_topic = text_relevance_delta(
        "Model Context Protocol (MCP) by Anthropic explained", query
    )

    assert generic_only < 0
    assert on_topic > 0


def test_result_quality_score_drops_generic_only_off_topic_page():
    """The exact failure from the MCP run: 'model' pages must fall below gate."""
    from src.web_search.common import (
        DEFAULT_MIN_USABLE_URL_SCORE,
        SearchResult,
        result_quality_score,
    )

    query = "model context protocol mcp anthropic"
    dictionary = SearchResult(
        url="https://dictionary.cambridge.org/dictionary/english/model",
        title="MODEL | English meaning",
        snippet="model definition: 1. a representation of something; 2. a person who...",
    )
    tesla_news = SearchResult(
        url="https://news.qq.com/rain/a/20260511A07GD300",
        title="Tesla Model S and Model X discontinued",
        snippet="Tesla announced it will stop producing the Model S and Model X.",
    )
    on_topic = SearchResult(
        url="https://mcpcn.com/docs/intro",
        title="Model Context Protocol (MCP) Introduction",
        snippet="MCP is an open protocol from Anthropic standardizing tool access.",
    )

    assert result_quality_score(dictionary, query=query) < DEFAULT_MIN_USABLE_URL_SCORE
    assert result_quality_score(tesla_news, query=query) < DEFAULT_MIN_USABLE_URL_SCORE
    assert result_quality_score(on_topic, query=query) >= DEFAULT_MIN_USABLE_URL_SCORE


# ── multi-entity / brand-collision hardening (Jordan vs Argentina) ───────────


def test_term_weight_does_not_boost_ordinary_short_words():
    """Short common words ('cup') must not get the acronym distinctiveness boost."""
    from src.web_search.common import _acronym_terms, _term_weight

    query = "Jordan Argentina World Cup result"
    acronyms = _acronym_terms(query)
    # No genuine uppercase acronyms in this query.
    assert acronyms == set()
    # "cup" is short but ordinary filler -> generic weight, never the 1.3
    # acronym boost that previously let brand pages clear the gate.
    assert _term_weight("cup", acronyms) == 0.2
    # A distinctive named entity keeps full weight.
    assert _term_weight("argentina", acronyms) == 1.0


def test_acronym_terms_detects_genuine_uppercase_acronyms():
    from src.web_search.common import _acronym_terms

    assert "mcp" in _acronym_terms("What is MCP by Anthropic")
    # Lowercase short words are not acronyms.
    assert "cup" not in _acronym_terms("world cup result")


def test_text_relevance_delta_requires_both_entities_for_match_query():
    """A page mentioning only one of two named entities is off-topic."""
    from src.web_search.common import text_relevance_delta

    query = "Jordan Argentina World Cup result"
    only_jordan = text_relevance_delta(
        "Air Jordan sneakers official store - Nike", query
    )
    only_argentina = text_relevance_delta(
        "Argentina | History, Geography, and Culture - Britannica", query
    )
    both = text_relevance_delta(
        "Argentina vs Jordan World Cup 2026 match result and score", query
    )

    assert only_jordan < 0
    assert only_argentina < 0
    assert both > 0


def test_result_quality_score_drops_brand_pages_below_gate():
    """The Jordan/Argentina run: brand + country pages must fall below gate."""
    from src.web_search.common import (
        DEFAULT_MIN_USABLE_URL_SCORE,
        SearchResult,
        result_quality_score,
    )

    query = "Jordan Argentina World Cup result"
    shoe_store = SearchResult(
        url="https://www.nike.com/w/jordan-37eef/",
        title="Jordan. Nike.com",
        snippet="Shop the latest Air Jordan shoes and apparel at Nike.com.",
    )
    country_page = SearchResult(
        url="https://www.britannica.com/place/Jordan",
        title="Jordan | History, Map, Flag, Population - Britannica",
        snippet="Jordan, Arab country of Southwest Asia, in the rocky desert...",
    )
    on_topic = SearchResult(
        url="https://www.espn.com/soccer/report/argentina-jordan-2026",
        title="Argentina vs Jordan - World Cup 2026 Result",
        snippet="Full time score and report from the Argentina vs Jordan match.",
    )

    assert result_quality_score(shoe_store, query=query) < DEFAULT_MIN_USABLE_URL_SCORE
    assert result_quality_score(country_page, query=query) < DEFAULT_MIN_USABLE_URL_SCORE
    assert result_quality_score(on_topic, query=query) >= DEFAULT_MIN_USABLE_URL_SCORE


def test_discover_urls_drops_generic_only_matches_end_to_end(isolated_settings):
    from src.web_search.common import SearchResult

    settings = isolated_settings(web_search_max_results=10, web_search_top_k=3)
    provider = SnippetSearchProvider(
        [
            SearchResult(
                url="https://dictionary.cambridge.org/dictionary/english/model",
                title="MODEL | English meaning",
                snippet="model definition: a representation of something.",
            ),
            SearchResult(
                url="https://news.qq.com/rain/a/20260511A07GD300",
                title="Tesla Model S and Model X discontinued",
                snippet="Tesla will stop producing the Model S and Model X.",
            ),
            SearchResult(
                url="https://mcpcn.com/docs/intro",
                title="Model Context Protocol MCP Introduction",
                snippet="MCP is an open protocol from Anthropic for tool access.",
            ),
        ]
    )

    urls = discover_urls_from_web(
        "what is the model context protocol mcp by anthropic", settings, provider=provider
    )

    assert urls == ["https://mcpcn.com/docs/intro"]


def test_chinese_query_terms_use_cjk_bigrams():
    from src.web_search.common import _search_query_terms

    terms = _search_query_terms("南京地铁线路数量 2025 2026 几条线")

    assert {"南京", "地铁", "线路", "数量"} <= set(terms)
    assert _search_query_terms("龙") == ["龙"]


def test_chinese_text_relevance_penalizes_topical_miss():
    from src.web_search.common import text_relevance_delta

    query = "南京地铁线路数量 2025 2026 几条线"

    assert text_relevance_delta("南京地铁目前共运营14条线路", query) > 0
    assert text_relevance_delta("南京市政府门户网站，提供旅游和政务信息", query) < 0


def test_chinese_off_topic_snippet_falls_below_url_gate():
    from src.web_search.common import (
        DEFAULT_MIN_USABLE_URL_SCORE,
        SearchResult,
        result_quality_score,
    )

    query = "南京地铁线路数量 2025 2026 几条线"
    off_topic = SearchResult(
        url="https://example.com/news/2026/nanjing-overview",
        title="南京市人民政府门户网站",
        snippet="南京旅游、政务服务和城市介绍。",
    )
    on_topic = SearchResult(
        url="https://example.com/news/2026/nanjing-metro",
        title="南京地铁运营线路数量",
        snippet="南京地铁目前共运营14条线路。",
    )

    assert result_quality_score(off_topic, query=query) < DEFAULT_MIN_USABLE_URL_SCORE
    assert result_quality_score(on_topic, query=query) >= DEFAULT_MIN_USABLE_URL_SCORE


def test_page_relevance_keeps_short_chinese_fact_and_rejects_unrelated_page():
    from src.web_search.common import is_page_text_relevant

    query = "南京地铁线路数量 2025 2026 几条线"

    assert is_page_text_relevant(
        "目前共运营14条线路。",
        query,
        title="南京地铁线路",
    )
    assert not is_page_text_relevant(
        "Slack brings team communication, channels, integrations, and workflow tools together."
        * 20,
        query,
        title="Slack product updates",
    )
