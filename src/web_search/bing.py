from __future__ import annotations

import base64
import binascii
import logging
from dataclasses import dataclass
from typing import cast
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse
from urllib.request import urlopen

from bs4 import BeautifulSoup

from .common import (
    BING_BASE_URL,
    normalize_urls,
    search_request,
    urlopen_context,
)

logger = logging.getLogger(__name__)


class BingVerificationError(RuntimeError):
    """Raised when Bing returns an anti-bot verification page instead of results."""


# REFACTOR: Map shared web_search_timelimit values to Bing HTML filters.
BING_TIMELIMIT_FILTERS = {
    "d": 'ex1:"ez1"',
    "day": 'ex1:"ez1"',
    "w": 'ex1:"ez2"',
    "week": 'ex1:"ez2"',
    "m": 'ex1:"ez3"',
    "month": 'ex1:"ez3"',
}


def normalize_bing_market(region: str | None) -> str:
    """Convert the shared web_search_region setting to a Bing mkt value."""

    text = (region or "").strip()
    if not text or text.lower() == "wt-wt":
        return "zh-CN"

    normalized = text.replace("_", "-")
    parts = normalized.split("-")
    if len(parts) == 2 and all(len(part) == 2 for part in parts):
        left, right = parts
        if left.islower() and right.islower():
            return f"{right.lower()}-{left.upper()}"
        return f"{left.lower()}-{right.upper()}"

    return normalized


def bing_set_language(market: str) -> str:
    return (market.split("-", 1)[0] or "zh").lower()


def bing_timelimit_filter(timelimit: str | None) -> str | None:
    """Convert a shared timelimit value to a Bing filter expression."""

    text = (timelimit or "").strip().lower()
    return BING_TIMELIMIT_FILTERS.get(text)


def build_bing_search_url(
    query: str,
    max_results: int,
    market: str,
    timelimit: str | None = None,
) -> str:
    """Build a Bing HTML search URL with optional freshness filtering."""

    params: dict[str, str | int] = {
        "q": query,
        "count": max(1, max_results),
        "mkt": market,
        "setlang": bing_set_language(market),
    }
    freshness_filter = bing_timelimit_filter(timelimit)
    if freshness_filter:
        params["filters"] = freshness_filter
    return f"{BING_BASE_URL}/search?{urlencode(params)}"


@dataclass(frozen=True)
class BingWebSearch:
    """Bing HTML search provider."""

    market: str = "zh-CN"
    # REFACTOR: Optional shared timelimit used for Bing recency filtering.
    timelimit: str | None = None
    verify_ssl: bool = True

    @property
    def provider_name(self) -> str:
        return "bing"

    def search(self, query: str, max_results: int = 20) -> list[str]:
        search_url = build_bing_search_url(
            query=query,
            max_results=max_results,
            market=self.market,
            timelimit=self.timelimit,
        )

        with urlopen(
            search_request(search_url),
            timeout=15,
            context=urlopen_context(self.verify_ssl),
        ) as response:
            final_url = response.geturl()
            encoding = response.headers.get_content_charset() or "utf-8"
            html = response.read().decode(encoding, errors="replace")

        soup = BeautifulSoup(html, "html.parser")
        hrefs = candidate_bing_hrefs(soup)
        if not hrefs and is_bing_verification_page(html, final_url):
            raise BingVerificationError(
                "Bing returned a verification/captcha or search shell page; "
                "falling back to another search provider is required."
            )

        if not hrefs:
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            logger.warning(
                "Bing returned no parseable result links "
                "(final_url=%s, title=%r, html_bytes=%s)",
                final_url,
                title[:120],
                len(html),
            )

        urls: list[str] = []
        for href in hrefs:
            url = unwrap_bing_redirect(href)
            if url:
                urls.append(url)
            if len(normalize_urls(urls)) >= max_results:
                break

        return normalize_urls(urls)[:max_results]


def candidate_bing_hrefs(soup: BeautifulSoup) -> list[str]:
    selectors = [
        "li.b_algo h2 a[href]",
        "ol#b_results h2 a[href]",
        "main h2 a[href]",
    ]
    hrefs: list[str] = []
    seen: set[str] = set()

    for selector in selectors:
        for anchor in soup.select(selector):
            href = cast(str, anchor.get("href", ""))
            if href and href not in seen:
                seen.add(href)
                hrefs.append(href)

    return hrefs


def is_bing_verification_page(html: str, final_url: str) -> bool:
    parsed = urlparse(final_url)
    path = parsed.path.lower()
    if path.rstrip("/") == "" and "q" in parse_qs(parsed.query):
        return True
    if "captcha" in path or "challenge" in path:
        return True

    lowered = html.lower()
    markers = (
        "captcha",
        "verify you are human",
        "unusual traffic",
        "are you a robot",
    )
    return any(marker in lowered for marker in markers)


def unwrap_bing_redirect(href: str) -> str:
    absolute_url = urljoin(BING_BASE_URL, href)
    parsed = urlparse(absolute_url)
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"bing.com", "www.bing.com", "cn.bing.com"}:
        return absolute_url

    query = parse_qs(parsed.query)
    for key in ("url", "r", "q"):
        target = query.get(key, [""])[0]
        if target.startswith(("http://", "https://")):
            return target

    encoded_target = query.get("u", [""])[0]
    if encoded_target:
        target = decode_bing_target(encoded_target)
        if target:
            return target

    return ""


def decode_bing_target(value: str) -> str:
    target = unquote(value)
    if target.startswith(("http://", "https://")):
        return target

    if target.startswith("a1"):
        target = target[2:]

    padding = "=" * (-len(target) % 4)
    try:
        decoded = base64.urlsafe_b64decode(f"{target}{padding}").decode(
            "utf-8",
            errors="replace",
        )
    except (binascii.Error, ValueError, OSError):
        return ""

    return decoded if decoded.startswith(("http://", "https://")) else ""
