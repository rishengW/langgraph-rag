from __future__ import annotations

from typing import Any

import pytest

from src.backend.web_search import (
    BingApiWebSearch,
    BraveWebSearch,
    SerperWebSearch,
    TavilyWebSearch,
    get_search_provider,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeSession:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("GET", url, kwargs))
        return FakeResponse(self.payload)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("POST", url, kwargs))
        return FakeResponse(self.payload)


@pytest.mark.parametrize(
    ("provider", "expected_url", "method"),
    [
        (
            SerperWebSearch(
                api_key="key",
                session=FakeSession(
                    {"organic": [{"link": "https://a.test", "title": "A", "snippet": "SA"}]}
                ),
            ),
            "https://a.test",
            "POST",
        ),
        (
            BraveWebSearch(
                api_key="key",
                session=FakeSession(
                    {
                        "web": {
                            "results": [
                                {"url": "https://b.test", "title": "B", "description": "SB"}
                            ]
                        }
                    }
                ),
            ),
            "https://b.test",
            "GET",
        ),
        (
            TavilyWebSearch(
                api_key="key",
                session=FakeSession(
                    {"results": [{"url": "https://c.test", "title": "C", "content": "SC"}]}
                ),
            ),
            "https://c.test",
            "POST",
        ),
        (
            BingApiWebSearch(
                api_key="key",
                session=FakeSession(
                    {
                        "webPages": {
                            "value": [{"url": "https://d.test", "name": "D", "snippet": "SD"}]
                        }
                    }
                ),
            ),
            "https://d.test",
            "GET",
        ),
    ],
)
def test_api_provider_normalizes_title_snippet_results(
    provider: Any,
    expected_url: str,
    method: str,
) -> None:
    results = provider.search_results("Mandarin search", 5)

    assert [result.url for result in results] == [expected_url]
    assert results[0].title
    assert results[0].snippet
    assert provider._session.calls[0][0] == method
    assert provider._session.calls[0][2]["timeout"] == 8.0


def test_api_provider_requires_key() -> None:
    with pytest.raises(ValueError, match="requires an API key"):
        SerperWebSearch(api_key="")


def test_factory_builds_configured_api_providers(isolated_settings) -> None:
    settings = isolated_settings(
        serper_api_key="serper-key",
        brave_search_api_key="brave-key",
        tavily_api_key="tavily-key",
        bing_search_api_key="bing-key",
    )

    assert isinstance(get_search_provider("serper", settings), SerperWebSearch)
    assert isinstance(get_search_provider("brave", settings), BraveWebSearch)
    assert isinstance(get_search_provider("tavily", settings), TavilyWebSearch)
    assert isinstance(get_search_provider("bing_api", settings), BingApiWebSearch)
    assert get_search_provider("tavily", settings).timeout == 20.0
