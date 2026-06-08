from __future__ import annotations

import ssl
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import Request


BAIDU_BASE_URL = "https://www.baidu.com"
BING_BASE_URL = "https://cn.bing.com"
DUCKDUCKGO_BASE_URL = "https://duckduckgo.com"
SEARCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


def normalize_urls(urls: Iterable[str]) -> list[str]:
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


def urlopen_context(verify_ssl: bool) -> ssl.SSLContext | None:
    if verify_ssl:
        return None
    return ssl._create_unverified_context()


def search_request(url: str) -> Request:
    return Request(
        url,
        headers={
            "User-Agent": SEARCH_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )


# Hostnames that almost never produce article-like content useful for RAG.
# These tend to be search-results-of-search-results, login walls, image
# galleries, or tag/category pages. Filter them before fetching so the
# top-K sees real candidate articles.
NOISE_HOSTNAMES = {
    "image.baidu.com",
    "tieba.baidu.com",
    "zhidao.baidu.com",
    "fanyi.baidu.com",
    "map.baidu.com",
    "v.baidu.com",
    "video.baidu.com",
    "wenku.baidu.com",
    "passport.baidu.com",
    "login.baidu.com",
    "bing.com",
    "www.bing.com",
    "cn.bing.com",
}


def is_noise_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    if not hostname:
        return True
    if hostname in NOISE_HOSTNAMES:
        return True
    # Baidu's own search-of-search pages are not article content.
    if hostname.endswith(".baidu.com") and urlparse(url).path.startswith("/s"):
        return True
    return False


def select_top_urls(urls: list[str], top_k: int | None) -> list[str]:
    """Drop noise hosts and keep the top-K URLs in provider-ranked order."""

    filtered = [url for url in urls if not is_noise_url(url)]
    if top_k and top_k > 0:
        return filtered[:top_k]
    return filtered
