from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import urlopen

from bs4 import BeautifulSoup

from .common import (
    BAIDU_BASE_URL,
    normalize_urls,
    search_request,
    urlopen_context,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaiduWebSearch:
    """Baidu HTML search provider."""

    verify_ssl: bool = True

    @property
    def provider_name(self) -> str:
        return "baidu"

    def search(self, query: str, max_results: int = 20) -> list[str]:
        params = urlencode({"wd": query, "rn": max(1, max_results)})
        search_url = f"{BAIDU_BASE_URL}/s?{params}"

        with urlopen(
            search_request(search_url),
            timeout=10,
            context=urlopen_context(self.verify_ssl),
        ) as response:
            encoding = response.headers.get_content_charset() or "utf-8"
            html = response.read().decode(encoding, errors="replace")

        urls: list[str] = []
        for href in candidate_baidu_hrefs(BeautifulSoup(html, "html.parser")):
            absolute_url = urljoin(BAIDU_BASE_URL, href)
            if is_baidu_result_redirect(absolute_url):
                absolute_url = self.resolve_redirect(absolute_url)
            urls.append(absolute_url)

            if len(normalize_urls(urls)) >= max_results:
                break

        return normalize_urls(urls)[:max_results]

    def resolve_redirect(self, url: str) -> str:
        """Resolve Baidu's result redirect URL to the target page when possible."""

        if not is_baidu_result_redirect(url):
            return url

        try:
            with urlopen(
                search_request(url),
                timeout=8,
                context=urlopen_context(self.verify_ssl),
            ) as response:
                final_url = response.geturl()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            logger.debug("Could not resolve Baidu redirect %s: %s", url, exc)
            return url

        return final_url if final_url.startswith(("http://", "https://")) else url


def is_baidu_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname == "baidu.com" or hostname.endswith(".baidu.com")


def is_baidu_result_redirect(url: str) -> bool:
    parsed = urlparse(url)
    return is_baidu_url(url) and parsed.path.rstrip("/") == "/link"


def candidate_baidu_hrefs(soup: BeautifulSoup) -> list[str]:
    selectors = [
        "h3.t a[href]",
        ".result h3 a[href]",
        ".result-op h3 a[href]",
        "div.c-container h3 a[href]",
    ]
    hrefs: list[str] = []
    seen: set[str] = set()

    for selector in selectors:
        for anchor in soup.select(selector):
            href = anchor.get("href", "")
            if href and href not in seen:
                seen.add(href)
                hrefs.append(href)

    if hrefs:
        return hrefs

    for anchor in soup.select('a[href*="/link?"]'):
        href = anchor.get("href", "")
        if href and href not in seen:
            seen.add(href)
            hrefs.append(href)

    return hrefs
