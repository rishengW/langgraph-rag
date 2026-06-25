from __future__ import annotations

import logging
import re
import ssl
from collections.abc import Iterable
from dataclasses import dataclass, replace
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request


@dataclass(frozen=True)
class SearchResult:
    """A single search engine result with its topical text.

    The ``title`` and ``snippet`` are the strongest signals a provider returns
    for whether a URL is relevant to the query. Keeping them lets the ranker
    score topical relevance instead of guessing from URL shape alone.
    """

    url: str
    title: str = ""
    snippet: str = ""

    @property
    def relevance_text(self) -> str:
        return f"{self.title} {self.snippet}".strip()

BAIDU_BASE_URL = "https://www.baidu.com"
BING_BASE_URL = "https://cn.bing.com"
DUCKDUCKGO_BASE_URL = "https://duckduckgo.com"
SEARCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
logger = logging.getLogger(__name__)


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
# REFACTOR: Keep the default URL quality threshold explicit for Settings wiring.
DEFAULT_MIN_USABLE_URL_SCORE = 45


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
    return bool(has_search_query(parsed.path, parsed.query))


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


def url_quality_score(url: str, *, query: str = "") -> int:
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
    score += _url_query_relevance_bonus(segments, parsed.hostname or "", query)
    return score


def _url_query_relevance_bonus(
    segments: list[str],
    hostname: str,
    query: str,
) -> int:
    if not query:
        return 0
    query_terms = _search_query_terms(query)
    if not query_terms:
        return 0
    bonus = 0
    path_text = " ".join(segments).lower()
    for term in query_terms:
        if term in hostname.lower():
            bonus += 8
        if term in path_text:
            bonus += 5
    return min(bonus, 20)


_SEARCH_QUERY_FILLER_RE = re.compile(
    r"\b(?:a|an|the|is|are|was|were|of|in|on|at|to|for|with|by|about|"
    r"what|when|where|which|who|how|does|do|did|can|could|will|would|"
    r"should|tell|explain|find|show|give|list|please|just|latest|newest|"
    r"current|recent|new|now)\b",
    re.I,
)


def _search_query_terms(query: str) -> list[str]:
    cleaned = _SEARCH_QUERY_FILLER_RE.sub(" ", query)
    return [
        token.lower()
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", cleaned)
        if len(token) >= 3
    ]


# REFACTOR: Title/snippet relevance is the strongest topical signal a provider
# gives us. Weight it heavily so an off-topic but well-shaped URL cannot
# out-rank a genuinely relevant page, and so a result whose snippet shares no
# query terms is pushed below the usability gate.
TEXT_RELEVANCE_MAX_BONUS = 45
TEXT_RELEVANCE_MISS_PENALTY = 60


def text_relevance_delta(text: str, query: str) -> int:
    """Score how well result title/snippet text matches the query.

    Returns a positive bonus proportional to the share of query terms found in
    ``text``, or a negative penalty when the text is present but shares no
    query terms (a strong off-topic signal). Returns ``0`` when either the
    query or the text carries no usable terms, leaving URL-shape scoring intact.
    """

    query_terms = _search_query_terms(query)
    if not query_terms:
        return 0

    normalized = text.lower()
    if not normalized.strip():
        return 0

    matched = sum(1 for term in set(query_terms) if term in normalized)
    if matched == 0:
        return -TEXT_RELEVANCE_MISS_PENALTY

    coverage = matched / len(set(query_terms))
    return round(coverage * TEXT_RELEVANCE_MAX_BONUS)


def result_quality_score(result: SearchResult, *, query: str = "") -> int:
    """Score a search result using URL shape plus title/snippet relevance."""

    score = url_quality_score(result.url, query=query)
    if score <= 0:
        return score
    return score + text_relevance_delta(result.relevance_text, query)


def normalize_results(results: Iterable[SearchResult]) -> list[SearchResult]:
    """Deduplicate and validate search results, preserving title/snippet text."""

    seen: set[str] = set()
    normalized: list[SearchResult] = []

    for result in results:
        clean_url = (result.url or "").strip()
        if not clean_url or clean_url in seen:
            continue
        if not clean_url.startswith(("http://", "https://")):
            continue

        seen.add(clean_url)
        normalized.append(replace(result, url=clean_url))

    return normalized


def select_top_urls(
    urls: list[str],
    top_k: int | None,
    min_score: int = DEFAULT_MIN_USABLE_URL_SCORE,
    *,
    query: str = "",
) -> list[str]:
    """Keep high-quality URLs after deterministic filtering and deduplication."""

    return select_top_results(
        [SearchResult(url=url) for url in urls],
        top_k,
        min_score=min_score,
        query=query,
    )


def select_top_results(
    results: list[SearchResult],
    top_k: int | None,
    min_score: int = DEFAULT_MIN_USABLE_URL_SCORE,
    *,
    query: str = "",
) -> list[str]:
    """Rank results by URL shape + title/snippet relevance, keep the top-k URLs."""

    ranked = ranked_usable_results(results, min_score=min_score, query=query)
    if top_k and top_k > 0:
        return ranked[:top_k]
    return ranked


def ranked_usable_urls(
    urls: Iterable[str],
    min_score: int = DEFAULT_MIN_USABLE_URL_SCORE,
    *,
    query: str = "",
) -> list[str]:
    return ranked_usable_results(
        [SearchResult(url=url) for url in urls],
        min_score=min_score,
        query=query,
    )


def ranked_usable_results(
    results: Iterable[SearchResult],
    min_score: int = DEFAULT_MIN_USABLE_URL_SCORE,
    *,
    query: str = "",
) -> list[str]:
    best_by_key: dict[tuple[str, str], tuple[int, int, str]] = {}

    for position, result in enumerate(normalize_results(results)):
        score = result_quality_score(result, query=query)
        usable = score >= min_score
        logger.debug(
            "Scored web search URL candidate url=%s score=%s min_score=%s usable=%s",
            result.url,
            score,
            min_score,
            usable,
            extra={
                "url": result.url,
                "score": score,
                "min_score": min_score,
                "usable": usable,
            },
        )
        if score < min_score:
            continue

        key = canonical_url_key(result.url)
        current = best_by_key.get(key)
        if current is None or (score, -position) > (current[0], -current[1]):
            best_by_key[key] = (score, position, result.url)

    ranked = sorted(best_by_key.values(), key=lambda item: (-item[0], item[1], item[2]))
    return [url for _score, _position, url in ranked]
