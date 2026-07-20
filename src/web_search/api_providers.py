from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import requests

from .common import SearchResult

SERPER_ENDPOINT = "https://google.serper.dev/search"
BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
TAVILY_ENDPOINT = "https://api.tavily.com/search"
DEFAULT_BING_SEARCH_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"


class _ApiSearchProvider:
    provider_name = "api"

    def __init__(
        self,
        *,
        api_key: str,
        timeout: int | float = 8,
        verify_ssl: bool = True,
        session: Any | None = None,
    ) -> None:
        key = (api_key or "").strip()
        if not key:
            raise ValueError(f"{self.provider_name} search requires an API key")
        self.api_key = key
        self.timeout = max(1.0, float(timeout))
        self.verify_ssl = verify_ssl
        self._session = session or requests.Session()

    def search(self, query: str, max_results: int = 20) -> list[str]:
        return [result.url for result in self.search_results(query, max_results)]

    def search_results(self, query: str, max_results: int = 20) -> list[SearchResult]:
        raise NotImplementedError


class SerperWebSearch(_ApiSearchProvider):
    """Google results through Serper's supported JSON API."""

    provider_name = "serper"

    def search_results(self, query: str, max_results: int = 20) -> list[SearchResult]:
        limit = _result_limit(max_results)
        body: dict[str, Any] = {"q": query, "num": limit}
        if _contains_cjk(query):
            body.update({"gl": "cn", "hl": "zh-cn"})
        response = self._session.post(
            SERPER_ENDPOINT,
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            json=body,
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        response.raise_for_status()
        payload = _json_mapping(response)
        return _search_results(payload.get("organic"), url_key="link", limit=limit)


class BraveWebSearch(_ApiSearchProvider):
    """Web results from Brave Search's supported API."""

    provider_name = "brave"

    def search_results(self, query: str, max_results: int = 20) -> list[SearchResult]:
        limit = _result_limit(max_results)
        params: dict[str, Any] = {"q": query, "count": limit, "safesearch": "moderate"}
        if _contains_cjk(query):
            params.update({"country": "cn", "search_lang": "zh-hans"})
        response = self._session.get(
            BRAVE_SEARCH_ENDPOINT,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
            params=params,
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        response.raise_for_status()
        payload = _json_mapping(response)
        web = payload.get("web")
        values = web.get("results") if isinstance(web, Mapping) else None
        return _search_results(values, url_key="url", limit=limit, snippet_key="description")


class TavilyWebSearch(_ApiSearchProvider):
    """Search results from Tavily's supported JSON API."""

    provider_name = "tavily"

    def search_results(self, query: str, max_results: int = 20) -> list[SearchResult]:
        limit = _result_limit(max_results)
        response = self._session.post(
            TAVILY_ENDPOINT,
            headers={"Content-Type": "application/json"},
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": limit,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        response.raise_for_status()
        payload = _json_mapping(response)
        return _search_results(
            payload.get("results"),
            url_key="url",
            limit=limit,
            snippet_key="content",
        )


class BingApiWebSearch(_ApiSearchProvider):
    """Microsoft Bing Web Search through the configured supported endpoint."""

    provider_name = "bing_api"

    def __init__(self, *, endpoint: str = DEFAULT_BING_SEARCH_ENDPOINT, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.endpoint = (endpoint or DEFAULT_BING_SEARCH_ENDPOINT).strip()

    def search_results(self, query: str, max_results: int = 20) -> list[SearchResult]:
        limit = _result_limit(max_results)
        params: dict[str, Any] = {"q": query, "count": limit, "responseFilter": "Webpages"}
        if _contains_cjk(query):
            params.update({"mkt": "zh-CN", "setLang": "zh-hans"})
        response = self._session.get(
            self.endpoint,
            headers={"Ocp-Apim-Subscription-Key": self.api_key},
            params=params,
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        response.raise_for_status()
        payload = _json_mapping(response)
        web_pages = payload.get("webPages")
        values = web_pages.get("value") if isinstance(web_pages, Mapping) else None
        return _search_results(
            values,
            url_key="url",
            limit=limit,
            snippet_key="snippet",
            title_key="name",
        )


def _json_mapping(response: Any) -> Mapping[str, Any]:
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError("Search API returned a non-object JSON response")
    return payload


def _search_results(
    values: object,
    *,
    url_key: str,
    limit: int,
    title_key: str = "title",
    snippet_key: str = "snippet",
) -> list[SearchResult]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, Mapping)):
        return []

    results: list[SearchResult] = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        url = str(value.get(url_key) or "").strip()
        if not url:
            continue
        results.append(
            SearchResult(
                url=url,
                title=str(value.get(title_key) or "").strip(),
                snippet=str(value.get(snippet_key) or "").strip(),
            )
        )
        if len(results) >= limit:
            break
    return results


def _result_limit(max_results: int) -> int:
    return max(1, min(int(max_results or 20), 50))


def _contains_cjk(text: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in text or "")


__all__ = [
    "BingApiWebSearch",
    "BraveWebSearch",
    "SerperWebSearch",
    "TavilyWebSearch",
]
