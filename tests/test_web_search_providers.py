from __future__ import annotations

import pytest

from src.core.web_search import discover_urls_from_web as core_discover_urls_from_web
from src.web_search import (
    BaiduWebSearch,
    DuckDuckGoWebSearch,
    WebSearchProvider,
    discover_urls_from_web,
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


def test_provider_implementations_satisfy_protocol():
    assert isinstance(BaiduWebSearch(), WebSearchProvider)
    assert isinstance(DuckDuckGoWebSearch(), WebSearchProvider)


def test_factory_selects_baidu_and_duckduckgo_aliases(isolated_settings):
    settings = isolated_settings(
        web_search_region="us-en",
        web_search_timelimit="w",
        web_search_verify_ssl=False,
    )

    baidu = get_search_provider("baidu", settings)
    duckduckgo = get_search_provider("ddg", settings)
    default_provider = get_search_provider(
        config=isolated_settings(web_search_provider="duckduckgo")
    )
    settings_only_provider = get_search_provider(
        isolated_settings(web_search_provider="baidu")
    )

    assert isinstance(baidu, BaiduWebSearch)
    assert baidu.verify_ssl is False
    assert isinstance(duckduckgo, DuckDuckGoWebSearch)
    assert duckduckgo.region == "us-en"
    assert duckduckgo.timelimit == "w"
    assert duckduckgo.verify_ssl is False
    assert isinstance(default_provider, DuckDuckGoWebSearch)
    assert isinstance(settings_only_provider, BaiduWebSearch)


def test_factory_rejects_unsupported_provider(isolated_settings):
    settings = isolated_settings(web_search_provider="bing")

    with pytest.raises(ValueError, match="Unsupported WEB_SEARCH_PROVIDER 'bing'"):
        get_search_provider(config=settings)


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


def test_discover_urls_disabled_does_not_call_provider(isolated_settings):
    settings = isolated_settings(web_search_enabled=False)
    provider = StaticSearchProvider(["https://example.com/a"])

    assert discover_urls_from_web("phase two", settings, provider=provider) == []
    assert provider.calls == []


def test_core_import_path_keeps_provider_injection(isolated_settings):
    settings = isolated_settings(web_search_max_results=2, web_search_top_k=1)
    provider = StaticSearchProvider(["https://example.com/a", "https://example.com/b"])

    assert core_discover_urls_from_web("compat", settings, provider=provider) == [
        "https://example.com/a"
    ]
