from __future__ import annotations

import re
import ssl
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlparse
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

# REFACTOR: URL-level quality gates for provider results before page fetching.
NOISE_PATH_SEGMENTS = {
    "account",
    "accounts",
    "auth",
    "author",
    "authors",
    "categories",
    "category",
    "download",
    "downloads",
    "file",
    "files",
    "login",
    "register",
    "search",
    "searches",
    "signin",
    "signup",
    "tag",
    "tags",
    "user",
    "users",
}
LOW_VALUE_PATH_LEAFS = {
    "about",
    "contact",
    "feed",
    "privacy",
    "rss",
    "sitemap",
    "terms",
}
LOW_VALUE_FILE_EXTENSIONS = {
    ".7z",
    ".apk",
    ".avi",
    ".css",
    ".csv",
    ".dmg",
    ".doc",
    ".docx",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".rss",
    ".svg",
    ".tar",
    ".webp",
    ".wmv",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}
CONTENT_PATH_CUES = {
    "announcements",
    "article",
    "articles",
    "blog",
    "docs",
    "guide",
    "learn",
    "news",
    "post",
    "posts",
    "press",
    "release",
    "releases",
    "report",
    "research",
}
SEARCH_QUERY_KEYS = {"keyword", "keywords", "q", "query", "s", "search", "wd"}
MIN_USABLE_URL_SCORE = 45


def is_noise_url(url: str) -> bool:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    if not hostname:
        return True
    if parsed.scheme not in {"http", "https"}:
        return True
    if hostname in NOISE_HOSTNAMES:
        return True
    # Baidu's own search-of-search pages are not article content.
    if hostname.endswith(".baidu.com") and path.startswith("/s"):
        return True
    if has_noise_path(parsed.path):
        return True
    if has_search_query(parsed.path, parsed.query):
        return True
    return False


def has_noise_path(path: str) -> bool:
    segments = path_segments(path)
    if not segments:
        return False

    leaf = segments[-1]
    if leaf in LOW_VALUE_PATH_LEAFS:
        return True
    if any(leaf.endswith(extension) for extension in LOW_VALUE_FILE_EXTENSIONS):
        return True
    return bool(set(segments) & NOISE_PATH_SEGMENTS)


def has_search_query(path: str, query: str) -> bool:
    if not query:
        return False

    keys = {key.lower() for key in parse_qs(query)}
    if not keys & SEARCH_QUERY_KEYS:
        return False

    segments = path_segments(path)
    return not segments or bool(set(segments) & {"s", "search", "searches"})


def path_segments(path: str) -> list[str]:
    return [
        segment
        for segment in re.split(r"/+", unquote(path).lower().strip("/"))
        if segment
    ]


def canonical_url_key(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]

    path = re.sub(r"/+", "/", unquote(parsed.path).lower()).rstrip("/")
    if path.endswith("/index.html") or path.endswith("/index.htm"):
        path = path.rsplit("/", 1)[0]
    return hostname, path or "/"


def url_quality_score(url: str) -> int:
    if is_noise_url(url):
        return 0

    parsed = urlparse(url)
    segments = path_segments(parsed.path)
    score = 60
    score += 5 if parsed.scheme == "https" else -5
    score += 10 if len(segments) >= 2 else 3
    score -= 30 if not segments else 0
    score -= 5 if parsed.query else 0
    score += 15 if set(segments) & CONTENT_PATH_CUES else 0
    score += 8 if any(re.fullmatch(r"20\d{2}", segment) for segment in segments) else 0
    return score


def select_top_urls(urls: list[str], top_k: int | None) -> list[str]:
    """Keep high-quality URLs after deterministic filtering and deduplication."""

    filtered = ranked_usable_urls(urls)
    if top_k and top_k > 0:
        return filtered[:top_k]
    return filtered


def ranked_usable_urls(urls: Iterable[str]) -> list[str]:
    best_by_key: dict[tuple[str, str], tuple[int, int, str]] = {}

    for position, url in enumerate(normalize_urls(urls)):
        score = url_quality_score(url)
        if score < MIN_USABLE_URL_SCORE:
            continue

        key = canonical_url_key(url)
        current = best_by_key.get(key)
        if current is None or (score, -position) > (current[0], -current[1]):
            best_by_key[key] = (score, position, url)

    ranked = sorted(best_by_key.values(), key=lambda item: (-item[0], item[1], item[2]))
    return [url for _score, _position, url in ranked]
