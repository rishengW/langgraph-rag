from __future__ import annotations

import base64
from urllib.parse import parse_qs, urlparse

import pytest

from src.core.web_search import discover_urls_from_web as core_discover_urls_from_web
from src.web_search import baidu as baidu_module
from src.web_search import bing as bing_module
from src.web_search.common import is_noise_url, select_top_urls, url_quality_score
from src.web_search import discovery as discovery_module
from src.web_search import (
    BaiduWebSearch,
    BingWebSearch,
    DuckDuckGoWebSearch,
    WebSearchProvider,
    build_web_search_tool,
    discover_urls_from_web,
    format_web_search_results,
    get_search_provider,
)


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
            return "百度安全验证".encode("utf-8")

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
            """.encode("utf-8")

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
            return "<html><body>captcha challenge</body></html>".encode("utf-8")

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
    assert providers["baidu"].calls == [("minimax latest model", 4)]
    assert providers["bing"].calls == [("minimax latest model", 4)]
    assert providers["duckduckgo"].calls == []


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
    assert providers["baidu"].calls == [("minimax latest model", 4)]
    assert providers["bing"].calls == [("minimax latest model", 4)]
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
    assert providers["baidu"].calls == [("minimax latest model", 4)]
    assert providers["bing"].calls == [
        ("minimax latest model", 4),
        ("deepseek latest model", 4),
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
    assert providers["bing"].calls == [("minimax latest model", 4)]
    assert providers["baidu"].calls == [
        ("minimax latest model", 4),
        ("deepseek latest model", 4),
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
