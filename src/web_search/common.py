from __future__ import annotations

import logging
import re
import ssl
import unicodedata
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

# Chinese does not normally separate words with spaces, so the Latin-only
# tokenizer below used to produce no terms at all for Chinese queries. Character
# bigrams give us a small, dependency-free lexical signal: ``南京地铁`` becomes
# ``南京``, ``京地``, and ``地铁``. An unrelated Nanjing homepage may match the
# first bigram, but it will not cover enough of a metro-specific query to pass.
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_CJK_QUERY_FILLERS = (
    "告诉我",
    "是什么",
    "怎么样",
    "有多少",
    "请问",
    "帮我",
    "查询",
    "查找",
    "多少",
    "几个",
    "几条",
)


def _normalize_relevance_text(text: str) -> str:
    """Normalize full-width forms and case for deterministic matching."""

    return unicodedata.normalize("NFKC", text or "").casefold()


def _cjk_bigrams(text: str) -> list[str]:
    terms: list[str] = []
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            terms.append(run)
            continue
        terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    return terms


def _search_query_terms(query: str) -> list[str]:
    cleaned = _normalize_relevance_text(query)
    cleaned = _SEARCH_QUERY_FILLER_RE.sub(" ", cleaned)
    for filler in _CJK_QUERY_FILLERS:
        cleaned = cleaned.replace(filler, " ")

    latin_terms = [
        token.lower()
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", cleaned)
        if len(token) >= 3
    ]
    # Preserve deterministic ordering while avoiding double-counting repeated
    # query words or overlapping runs.
    return list(dict.fromkeys([*latin_terms, *_cjk_bigrams(cleaned)]))


# REFACTOR: Title/snippet relevance is the strongest topical signal a provider
# gives us. Weight it heavily so an off-topic but well-shaped URL cannot
# out-rank a genuinely relevant page, and so a result whose snippet shares no
# distinctive query terms is pushed below the usability gate.
TEXT_RELEVANCE_MAX_BONUS = 45
TEXT_RELEVANCE_MISS_PENALTY = 60

# Very common words carry almost no topical signal: matching them should not
# rescue an otherwise off-topic page. "model" is the canonical failure case —
# it appears in dictionary entries and car news alike, so it must not, on its
# own, mark a page as relevant to a query about a specific protocol/product.
# The second block is low-signal context/result filler (esp. sports queries):
# "Jordan Argentina World Cup result" must rank on the *entities* (jordan,
# argentina), not on the common context words (world, cup, result), otherwise
# Air-Jordan shoe pages and country encyclopedias clear the usability gate.
GENERIC_QUERY_TERMS = frozenset({
    "ai",
    "app",
    "data",
    "guide",
    "info",
    "information",
    "model",
    "models",
    "news",
    "online",
    "page",
    "service",
    "services",
    "site",
    "software",
    "system",
    "tool",
    "tools",
    "version",
    "web",
    "website",
    # Low-signal context / result / sports filler.
    "cup",
    "draw",
    "final",
    "finals",
    "fixture",
    "fixtures",
    "game",
    "games",
    "highlights",
    "loss",
    "match",
    "matches",
    "result",
    "results",
    "round",
    "schedule",
    "score",
    "scores",
    "season",
    "stage",
    "standings",
    "versus",
    "win",
    "wins",
    "world",
})
# A match only counts as "on topic" when the result shares a meaningful share
# of the distinctive query terms. Matching a single generic word is not enough.
TEXT_RELEVANCE_MIN_COVERAGE = 0.34
# REFACTOR: When a query names two or more distinctive entities (e.g. two teams
# in "Argentina vs Jordan"), a usable page must mention at least this many of
# them. Matching only one named entity (e.g. a page about the country Jordan,
# or the Air Jordan shoe brand) is treated as off-topic for a multi-entity
# factual query.
MULTI_ENTITY_MIN_MATCHES = 2


def _acronym_terms(query: str) -> set[str]:
    """Return lowercased tokens that appear as genuine acronyms in ``query``.

    An acronym is an all-uppercase token of length 2-5 in the *original* query
    casing (e.g. "MCP", "LLM", "RAG"). This is the real distinctiveness signal
    -- unlike a blanket "short token" rule, which wrongly boosts ordinary short
    words like "cup", "vs", or "win".
    """

    acronyms: set[str] = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9]*", query or ""):
        if 2 <= len(raw) <= 5 and raw.isupper():
            acronyms.add(raw.lower())
    return acronyms


def _term_weight(term: str, acronyms: frozenset[str] | set[str] = frozenset()) -> float:
    """Weight a query term by how distinctive it is.

    Generic, high-frequency words contribute little; distinctive named entities
    and genuine acronyms contribute the most. This keeps a lone "model" match
    from rescuing an off-topic page while a "mcp"/"anthropic" match still
    counts. Only real uppercase acronyms (passed in ``acronyms``) get the
    acronym boost -- ordinary short words like "cup" stay at the base weight.
    """

    if term in GENERIC_QUERY_TERMS:
        return 0.2
    if term in acronyms:
        return 1.3
    return 1.0


def text_relevance_delta(text: str, query: str) -> int:
    """Score how well result title/snippet text matches the query.

    Returns a positive bonus proportional to the weighted share of query terms
    found in ``text``, or a negative penalty when the text is present but
    shares no meaningful query terms (a strong off-topic signal). Matching only
    generic, high-frequency words (e.g. "model") counts for very little and
    will not, on its own, clear the relevance bar. Returns ``0`` when either
    the query or the text carries no usable terms, leaving URL-shape scoring
    intact.

    Multi-entity guard: when the query names two or more distinctive
    (non-generic) entities, a page that mentions fewer than
    ``MULTI_ENTITY_MIN_MATCHES`` of them is treated as a strong off-topic miss.
    This is what stops an Air-Jordan shoe page (matches only "jordan") or a
    country encyclopedia (matches only "argentina") from clearing the gate for
    a query about the match *between* the two.
    """

    acronyms = _acronym_terms(query)
    query_terms = set(_search_query_terms(query))
    if not query_terms:
        return 0

    normalized = _normalize_relevance_text(text)
    if not normalized.strip():
        return 0

    total_weight = sum(_term_weight(term, acronyms) for term in query_terms)
    if total_weight <= 0:
        return 0

    # Multi-entity conjunction: a page must cover at least two distinct named
    # entities when the query asks about a relationship between them.
    distinctive_terms = {
        term for term in query_terms if _term_weight(term, acronyms) >= 1.0
    }
    if len(distinctive_terms) >= MULTI_ENTITY_MIN_MATCHES:
        matched_distinctive = sum(
            1 for term in distinctive_terms if term in normalized
        )
        if matched_distinctive < MULTI_ENTITY_MIN_MATCHES:
            return -TEXT_RELEVANCE_MISS_PENALTY

    matched_weight = sum(
        _term_weight(term, acronyms) for term in query_terms if term in normalized
    )
    coverage = matched_weight / total_weight

    if coverage < TEXT_RELEVANCE_MIN_COVERAGE:
        # Below the bar: either nothing matched or only generic filler did.
        # Penalize proportionally so a pure miss is punished hardest while a
        # weak generic-only match is nudged down rather than rewarded.
        penalty_fraction = 1.0 - (coverage / TEXT_RELEVANCE_MIN_COVERAGE)
        return -round(TEXT_RELEVANCE_MISS_PENALTY * penalty_fraction)

    return round(coverage * TEXT_RELEVANCE_MAX_BONUS)


def is_page_text_relevant(text: str, query: str, *, title: str = "") -> bool:
    """Return whether fetched page text has enough lexical overlap to ground ``query``.

    Very short queries carry too little evidence for a reliable deterministic
    rejection, so this predicate deliberately abstains when fewer than two
    usable terms are available. Specific Chinese queries produce several CJK
    bigrams and are checked normally. The title is included because a short
    factual page often states the subject there and the answer in its body.
    """

    query_terms = set(_search_query_terms(query))
    if len(query_terms) < 2:
        return True
    combined = f"{title} {text}".strip()
    return text_relevance_delta(combined, query) > 0


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
