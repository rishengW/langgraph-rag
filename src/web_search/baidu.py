from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast
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


class BaiduVerificationError(RuntimeError):
    """Raised when Baidu returns an anti-bot verification page instead of results."""


def is_baidu_verification_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    return (
        hostname in {"wappass.baidu.com", "passport.baidu.com"}
        or "captcha" in path
        or "verify" in path
    )


def is_baidu_verification_page(html: str, final_url: str) -> bool:
    if is_baidu_verification_url(final_url):
        return True
    lowered = html.lower()
    markers = (
        "wappass.baidu.com/static/captcha",
        "百度安全验证",
        "baidu security verification",
        "captcha",
    )
    return any(marker in lowered for marker in markers)


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
            final_url = response.geturl()
            encoding = response.headers.get_content_charset() or "utf-8"
            html = response.read().decode(encoding, errors="replace")

        if is_baidu_verification_page(html, final_url):
            raise BaiduVerificationError(
                "Baidu returned a verification/captcha page; "
                "falling back to another search provider is required."
            )

        soup = BeautifulSoup(html, "html.parser")
        hrefs = candidate_baidu_hrefs(soup)
        if not hrefs:
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            logger.warning(
                "Baidu returned no parseable result links "
                "(final_url=%s, title=%r, html_bytes=%s)",
                final_url,
                title[:120],
                len(html),
            )

        urls: list[str] = []
        for href in hrefs:
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
            href = cast(str, anchor.get("href", ""))
            if href and href not in seen:
                seen.add(href)
                hrefs.append(href)

    if hrefs:
        return hrefs

    for anchor in soup.select('a[href*="/link?"]'):
        href = cast(str, anchor.get("href", ""))
        if href and href not in seen:
            seen.add(href)
            hrefs.append(href)

    return hrefs
