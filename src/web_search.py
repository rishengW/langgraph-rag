"""Web search helpers for discovering RAG source URLs."""

from __future__ import annotations

import logging
import hashlib
import ssl
from dataclasses import replace
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from .config import Settings


logger = logging.getLogger(__name__)

BAIDU_BASE_URL = "https://www.baidu.com"
SEARCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


def _load_ddgs():
    """Import the installed DuckDuckGo search client.

    The package was renamed from `duckduckgo-search` to `ddgs` in newer
    releases. Supporting both keeps the app usable across environments.
    """

    try:
        from ddgs import DDGS

        return DDGS
    except ImportError:
        from duckduckgo_search import DDGS

        return DDGS


def _normalize_urls(urls: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []

    for url in urls:
        clean_url = (url or "").strip()
        if not clean_url or clean_url in seen:
            continue
        if not clean_url.startswith(("http://", "https://")):
            continue

        seen.add(clean_url)
        normalized.append(clean_url)

    return normalized


def _discover_urls_from_duckduckgo(question: str, settings: Settings) -> list[str]:
    DDGS = _load_ddgs()

    search_kwargs = {
        "region": settings.web_search_region,
        "max_results": settings.web_search_max_results,
    }
    if settings.web_search_timelimit:
        search_kwargs["timelimit"] = settings.web_search_timelimit

    with DDGS(verify=settings.web_search_verify_ssl) as ddgs:
        results = list(ddgs.text(question, **search_kwargs))

    urls = _normalize_urls(result.get("href", "") for result in results)
    return urls


def _urlopen_context(settings: Settings) -> ssl.SSLContext | None:
    if settings.web_search_verify_ssl:
        return None
    return ssl._create_unverified_context()


def _search_request(url: str) -> Request:
    return Request(
        url,
        headers={
            "User-Agent": SEARCH_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )


def _is_baidu_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    return hostname == "baidu.com" or hostname.endswith(".baidu.com")


def _is_baidu_result_redirect(url: str) -> bool:
    parsed = urlparse(url)
    return _is_baidu_url(url) and parsed.path.rstrip("/") == "/link"


def _resolve_baidu_redirect(url: str, settings: Settings) -> str:
    """Resolve Baidu's result redirect URL to the target page when possible."""

    if not _is_baidu_result_redirect(url):
        return url

    try:
        with urlopen(
            _search_request(url),
            timeout=8,
            context=_urlopen_context(settings),
        ) as response:
            final_url = response.geturl()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        logger.debug("Could not resolve Baidu redirect %s: %s", url, exc)
        return url

    return final_url if final_url.startswith(("http://", "https://")) else url


def _candidate_baidu_hrefs(soup: BeautifulSoup) -> list[str]:
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


def _discover_urls_from_baidu(question: str, settings: Settings) -> list[str]:
    query = urlencode(
        {
            "wd": question,
            "rn": max(1, settings.web_search_max_results),
        }
    )
    search_url = f"{BAIDU_BASE_URL}/s?{query}"

    with urlopen(
        _search_request(search_url),
        timeout=10,
        context=_urlopen_context(settings),
    ) as response:
        encoding = response.headers.get_content_charset() or "utf-8"
        html = response.read().decode(encoding, errors="replace")

    urls: list[str] = []
    for href in _candidate_baidu_hrefs(BeautifulSoup(html, "html.parser")):
        absolute_url = urljoin(BAIDU_BASE_URL, href)
        if _is_baidu_result_redirect(absolute_url):
            absolute_url = _resolve_baidu_redirect(absolute_url, settings)
        urls.append(absolute_url)

        if len(_normalize_urls(urls)) >= settings.web_search_max_results:
            break

    return _normalize_urls(urls)[: settings.web_search_max_results]


def discover_urls_from_web(question: str, settings: Settings) -> list[str]:
    """Search the web for pages that can be used as RAG sources."""

    if not settings.web_search_enabled:
        return []

    provider = settings.web_search_provider
    if provider in ("duckduckgo", "ddg"):
        urls = _discover_urls_from_duckduckgo(question, settings)
    elif provider == "baidu":
        urls = _discover_urls_from_baidu(question, settings)
    else:
        raise ValueError(
            "Unsupported WEB_SEARCH_PROVIDER "
            f"{settings.web_search_provider!r}; use 'duckduckgo' or 'baidu'."
        )

    logger.info(
        "Discovered %s URL(s) from %s web search",
        len(urls),
        "duckduckgo" if provider == "ddg" else provider,
    )
    return urls


def settings_for_discovered_urls(settings: Settings, urls: list[str]) -> Settings:
    """Create isolated settings for a search-derived source set."""

    fingerprint = hashlib.sha256("\n".join(urls).encode("utf-8")).hexdigest()[:12]
    return replace(
        settings,
        source_urls=urls,
        chroma_dir=settings.chroma_dir / "web-search" / fingerprint,
        collection_name=f"{settings.collection_name}-web-{fingerprint}",
    )
