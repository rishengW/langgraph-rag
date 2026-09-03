from __future__ import annotations

import logging
from dataclasses import dataclass
from time import monotonic
from typing import Any, cast
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import urlopen

from bs4 import BeautifulSoup

from .common import (
    DUCKDUCKGO_BASE_URL,
    SearchResult,
    search_request,
    urlopen_context,
)

logger = logging.getLogger(__name__)


def load_ddgs() -> Any:
    """Import the installed DuckDuckGo search client."""

    import_errors: list[str] = []
    try:
        from ddgs import DDGS as DdgsClient

        return DdgsClient
    except ImportError as exc:
        import_errors.append(f"ddgs: {exc}")

    try:
        from duckduckgo_search import DDGS as DuckDuckGoSearchClient

        return DuckDuckGoSearchClient
    except ImportError as exc:
        import_errors.append(f"duckduckgo_search: {exc}")

    raise ImportError(
        "Install ddgs or duckduckgo-search for package-based web search "
        f"({'; '.join(import_errors)})"
    )


@dataclass(frozen=True)
class DuckDuckGoWebSearch:
    """Bounded DuckDuckGo HTML provider with optional direct DDGS helpers."""

    region: str = "wt-wt"
    timelimit: str | None = None
    verify_ssl: bool = True
    timeout: float = 8.0

    @property
    def provider_name(self) -> str:
        return "duckduckgo"

    def search(self, query: str, max_results: int = 20) -> list[str]:
        return [result.url for result in self.search_results(query, max_results)]

    def search_results(self, query: str, max_results: int = 20) -> list[SearchResult]:
        # The installed ``ddgs`` package may fan one query out to many search
        # engines and continue working after the chat deadline. Keep it as an
        # explicit compatibility helper, but use DuckDuckGo's own bounded HTML
        # endpoint for normal provider fallback.
        return _dedupe_results(self.search_html_results(query, max_results))[:max_results]

    def search_ddgs(self, query: str, max_results: int) -> list[str]:
        return [result.url for result in self.search_ddgs_results(query, max_results)]

    def search_ddgs_results(self, query: str, max_results: int) -> list[SearchResult]:
        DDGS = load_ddgs()

        search_kwargs = {
            "region": self.region,
            "max_results": max_results,
        }
        if self.timelimit:
            search_kwargs["timelimit"] = self.timelimit

        try:
            client = DDGS(verify=self.verify_ssl, timeout=self.timeout)
        except TypeError:
            # Older client versions do not expose the SSL verification argument.
            try:
                client = DDGS(timeout=self.timeout)
            except TypeError:
                client = DDGS()

        with client as ddgs:
            results = list(ddgs.text(query, **search_kwargs))

        # DDGS clients have used several field names across versions
        # ("href", "link", "url") and ("title") / ("body") for snippet text.
        extracted: list[SearchResult] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            url = ""
            for key in ("href", "link", "url"):
                value = result.get(key)
                if value:
                    url = value
                    break
            if not url:
                continue
            title = str(result.get("title") or "")
            snippet = str(result.get("body") or result.get("snippet") or "")
            extracted.append(SearchResult(url=url, title=title, snippet=snippet))
        return extracted

    def search_html(self, query: str, max_results: int) -> list[str]:
        return [result.url for result in self.search_html_results(query, max_results)]

    def search_html_results(self, query: str, max_results: int) -> list[SearchResult]:
        # urllib/DNS timeouts can overshoot on Windows. Use a conservative
        # fraction of the provider budget so this final fallback still returns
        # before discovery's outer deadline.
        network_budget = max(
            0.1,
            min(float(self.timeout) * 0.4, float(self.timeout) - 1.0),
        )
        deadline_at = monotonic() + network_budget
        params = {
            "q": query,
            "kl": self.region,
        }
        if self.timelimit:
            params["df"] = self.timelimit

        html = ""
        last_error: Exception | None = None
        for base_url in ("https://html.duckduckgo.com/html/",):
            remaining = deadline_at - monotonic()
            if remaining <= 0:
                break
            try:
                with urlopen(
                    search_request(f"{base_url}?{urlencode(params)}"),
                    timeout=max(0.1, min(self.timeout, remaining)),
                    context=urlopen_context(self.verify_ssl),
                ) as response:
                    html = response.read().decode("utf-8", errors="replace")
                break
            except Exception as exc:
                last_error = exc

        if not html:
            raise RuntimeError(f"DuckDuckGo HTML search failed: {last_error}")

        results: list[SearchResult] = []
        seen: set[str] = set()
        for result_block in BeautifulSoup(html, "html.parser").select("div.result, div.web-result"):
            anchor = result_block.select_one("a.result__a")
            if anchor is None:
                continue
            href = cast(str, anchor.get("href", ""))
            if not href:
                continue
            url = unwrap_duckduckgo_redirect(href)
            if not url or url in seen:
                continue
            seen.add(url)
            title = anchor.get_text(" ", strip=True)
            snippet_node = result_block.select_one("a.result__snippet, .result__snippet")
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            results.append(SearchResult(url=url, title=title, snippet=snippet))
            if len(results) >= max_results:
                break

        if results:
            return results

        # Fallback: bare result anchors without recognizable containers.
        for anchor in BeautifulSoup(html, "html.parser").select("a.result__a"):
            href = cast(str, anchor.get("href", ""))
            if not href:
                continue
            url = unwrap_duckduckgo_redirect(href)
            if not url or url in seen:
                continue
            seen.add(url)
            results.append(SearchResult(url=url, title=anchor.get_text(" ", strip=True)))
            if len(results) >= max_results:
                break

        return results


def _dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    deduped: list[SearchResult] = []
    for result in results:
        url = (result.url or "").strip()
        if not url or url in seen:
            continue
        if not url.startswith(("http://", "https://")):
            continue
        seen.add(url)
        deduped.append(result)
    return deduped


def unwrap_duckduckgo_redirect(href: str) -> str:
    if href.startswith("//"):
        href = f"https:{href}"
    elif href.startswith("/"):
        href = f"{DUCKDUCKGO_BASE_URL}{href}"

    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)

    return href
