from __future__ import annotations

import ipaddress
import logging
import re
import ssl
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Literal
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request

from .evidence import (
    PRICE_INTENT,
    answer_evidence_delta,
    detect_query_intents,
    has_required_answer_evidence,
)
from .query_constraints import (
    contains_cjk,
    extract_query_constraints,
    normalize_constraint_text,
)


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
# Only segments that never carry article content stay hard rejections. Listing
# segments (tags, categories, author archives) frequently do carry usable text
# and are demoted by ``LISTING_PATH_PENALTY`` instead of removed. Distribution
# segments such as ``download`` or ``files`` were removed entirely: official
# release notes and document pages commonly live under them, and unreadable
# binaries are still rejected by ``LOW_VALUE_FILE_EXTENSIONS``.
NOISE_PATH_SEGMENTS = {
    "account",
    "accounts",
    "auth",
    "login",
    "register",
    "search",
    "searches",
    "signin",
    "signup",
}
LISTING_PATH_SEGMENTS = {
    "author",
    "authors",
    "categories",
    "category",
    "tag",
    "tags",
    "user",
    "users",
}
# URL shape is only a tiebreaker now that ``page_structure`` measures link
# density on the page we fetched. Keep the nudge small enough that a listing URL
# with strong query relevance still clears the usability gate.
LISTING_PATH_PENALTY = 6
LOW_VALUE_PATH_LEAFS = {
    "about",
    "contact",
    "feed",
    "privacy",
    "rss",
    "sitemap",
    "terms",
}
# Redirect/doorway scripts are navigation intermediaries, not readable source
# pages. In particular, ``repack.php`` has appeared in provider results with a
# plausible snippet even though the target is only a traffic-forwarding page.
DOORWAY_PATH_LEAFS = {
    "jump.php",
    "outlink.php",
    "redirect.php",
    "redir.php",
    "repack.php",
}
# Binary and media assets the fetcher cannot turn into readable text. ``.pdf``
# is deliberately absent: government notices, standards, and vendor
# whitepapers are usually PDFs, and the lightweight fetcher extracts them
# through ``src/backend/web_search/pdf_loader.py``.
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

# Host provenance is a bounded ranking signal, not a relevance substitute.
# Government and education suffixes have formal ownership rules in China. The
# owner allowlist is intentionally small and limited to domains whose primary
# purpose is publishing first-party product or technical information.
HostAuthorityClass = Literal["government", "education", "recognized_owner", "standard"]
GOVERNMENT_AUTHORITY_BONUS = 24
EDUCATION_AUTHORITY_BONUS = 14
RECOGNIZED_OWNER_AUTHORITY_BONUS = 16
SYNDICATED_HOST_PENALTY = 12
OFFICIAL_SOURCE_BONUS = 18
UNVERIFIED_OFFICIAL_SOURCE_PENALTY = 12
LANGUAGE_MISMATCH_PENALTY = 10
# A hostname that embeds an owner's name without being one of its registered
# domains is a weak impersonation signal, not proof. Substring matching also
# hits genuine first-party and community hosts (``deepseek.net``,
# ``xiaomiev.com``, ``deepseek-ai.github.io``), so this is a ranking penalty
# rather than a pre-fetch rejection.
OWNER_LOOKALIKE_PENALTY = 15

RECOGNIZED_OWNER_DOMAINS = frozenset(
    {
        "alibabacloud.com",
        "aliyun.com",
        "anthropic.com",
        "apple.com",
        "apple.com.cn",
        "byd.com",
        "cloud.baidu.com",
        "cloud.tencent.com",
        "deepseek.com",
        "github.com",
        "huawei.com",
        "langchain.com",
        "mi.com",
        "microsoft.com",
        "nio.com",
        "nvidia.com",
        "openai.com",
        "xiaomi.com",
        "xpeng.com",
    }
)

# A domain that embeds a known owner's name while living outside that owner's
# registered domains is more likely to be an impersonation or affiliate site
# than a first-party source. Ordinary independent media domains remain valid:
# this only targets hosts that mimic the owner's name.
OWNER_IDENTITY_DOMAINS: dict[str, tuple[str, ...]] = {
    "alibaba": ("alibabacloud.com", "aliyun.com"),
    "aliyun": ("alibabacloud.com", "aliyun.com"),
    "anthropic": ("anthropic.com",),
    "apple": ("apple.com", "apple.com.cn"),
    "deepseek": ("deepseek.com",),
    "langchain": ("langchain.com",),
    "microsoft": ("microsoft.com",),
    "nvidia": ("nvidia.com",),
    "openai": ("openai.com",),
    "xiaomi": ("mi.com", "xiaomi.com"),
    "\u5c0f\u7c73": ("mi.com", "xiaomi.com"),
}

_SITE_CONSTRAINT_RE = re.compile(r"(?:^|\s)site:([a-z0-9.-]+)", re.I)
_OFFICIAL_QUERY_RE = re.compile(r"(?:\u5b98\u65b9|\bofficial\b|(?:^|\s)site:)", re.I)

# These hosts frequently syndicate, aggregate, or mass-publish articles. They
# remain eligible when their result is strongly relevant; the modest penalty
# only prevents them from displacing a similarly relevant primary source.
SYNDICATED_CONTENT_DOMAINS = frozenset(
    {
        "163.com",
        "baijiahao.baidu.com",
        "daydaynews.cc",
        "k.sina.com.cn",
        "kuaibao.qq.com",
        "sohu.com",
        "toutiao.com",
        "uc.cn",
    }
)

# A compact public-suffix approximation is sufficient for result diversity.
# It deliberately covers common compound suffixes without pretending to be a
# replacement for the full Public Suffix List.
COMMON_COMPOUND_PUBLIC_SUFFIXES = frozenset(
    {
        "ac.cn",
        "ac.uk",
        "co.jp",
        "co.uk",
        "com.au",
        "com.cn",
        "edu.cn",
        "gov.cn",
        "gov.uk",
        "net.au",
        "net.cn",
        "org.au",
        "org.cn",
        "org.uk",
    }
)


def _hostname_from_url_or_host(url_or_host: str) -> str:
    candidate = (url_or_host or "").strip().lower().rstrip(".")
    if not candidate:
        return ""
    parsed = urlparse(candidate if "://" in candidate else f"//{candidate}")
    return (parsed.hostname or "").lower().rstrip(".")


def _host_matches(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith(f".{domain}")


def host_authority_class(url_or_host: str) -> HostAuthorityClass:
    """Classify conservative first-party authority from a URL or hostname."""

    hostname = _hostname_from_url_or_host(url_or_host)
    if not hostname:
        return "standard"
    if hostname == "gov.cn" or hostname.endswith(".gov.cn"):
        return "government"
    if hostname == "edu.cn" or hostname.endswith(".edu.cn"):
        return "education"
    if any(_host_matches(hostname, domain) for domain in RECOGNIZED_OWNER_DOMAINS):
        return "recognized_owner"
    return "standard"


def host_quality_score(url_or_host: str) -> int:
    """Return a bounded authority bonus or soft syndication penalty."""

    hostname = _hostname_from_url_or_host(url_or_host)
    if not hostname:
        return 0
    if any(_host_matches(hostname, domain) for domain in SYNDICATED_CONTENT_DOMAINS):
        return -SYNDICATED_HOST_PENALTY

    authority = host_authority_class(hostname)
    return {
        "government": GOVERNMENT_AUTHORITY_BONUS,
        "education": EDUCATION_AUTHORITY_BONUS,
        "recognized_owner": RECOGNIZED_OWNER_AUTHORITY_BONUS,
        "standard": 0,
    }[authority]


def prefetch_rejection_reason(result: SearchResult, query: str) -> str | None:
    """Return a hard pre-fetch rejection reason for unsafe or off-topic results."""

    if is_noise_url(result.url):
        return "noise_url"

    hostname = _hostname_from_url_or_host(result.url)
    for required_domain in _SITE_CONSTRAINT_RE.findall(query or ""):
        domain = required_domain.strip(".").lower()
        if domain and not _host_matches(hostname, domain):
            return "site_constraint_mismatch"

    constraints = extract_query_constraints(query)
    evidence_text = result.relevance_text.strip()
    combined = normalize_constraint_text(f"{unquote(result.url)} {evidence_text}")
    if evidence_text:
        if constraints.quoted_phrases and not all(
            phrase in combined for phrase in constraints.quoted_phrases
        ):
            return "missing_quoted_title"
        if constraints.identifiers and not all(
            identifier in combined for identifier in constraints.identifiers
        ):
            return "missing_identifier"
        # CJK entity extraction deliberately keeps long noun phrases. Exact
        # phrase enforcement here would reject legitimate paraphrases, so
        # entity coverage remains a weighted lexical signal below; quoted
        # titles and identifiers are the hard anchors.

    return None


def is_owner_domain_lookalike(url_or_host: str, query: str) -> bool:
    """Return whether a host mimics a known owner's name without owning it."""

    hostname = _hostname_from_url_or_host(url_or_host)
    if not hostname:
        return False

    normalized_query = normalize_constraint_text(query)
    for identity, owner_domains in OWNER_IDENTITY_DOMAINS.items():
        if identity not in normalized_query:
            continue
        latin_identity = identity.encode("ascii", errors="ignore").decode("ascii")
        identity_tokens = (
            {latin_identity}
            if latin_identity
            else {
                domain.split(".", 1)[0]
                for domain in owner_domains
                if len(domain.split(".", 1)[0]) >= 4
            }
        )
        if not any(token and token in hostname for token in identity_tokens):
            continue
        if not any(_host_matches(hostname, domain) for domain in owner_domains):
            return True
    return False


def result_constraint_score(result: SearchResult, query: str) -> int:
    """Score language and first-party authority after hard constraints pass."""

    score = 0
    authority = host_authority_class(result.url)
    if _OFFICIAL_QUERY_RE.search(query or ""):
        score += (
            OFFICIAL_SOURCE_BONUS
            if authority != "standard"
            else -UNVERIFIED_OFFICIAL_SOURCE_PENALTY
        )

    evidence_text = result.relevance_text.strip()
    if contains_cjk(query) and evidence_text and not contains_cjk(evidence_text):
        score -= LANGUAGE_MISMATCH_PENALTY
    if is_owner_domain_lookalike(result.url, query):
        score -= OWNER_LOOKALIKE_PENALTY
    return score


def registrable_domain(url_or_host: str) -> str:
    """Return a stable, registrable-ish key for per-domain result diversity."""

    hostname = _hostname_from_url_or_host(url_or_host)
    if not hostname:
        return ""
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        return hostname

    labels = hostname.split(".")
    if len(labels) <= 2:
        return hostname
    last_two = ".".join(labels[-2:])
    if last_two in COMMON_COMPOUND_PUBLIC_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last_two


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
    # Baidu normally resolves these links before returning a result. Keeping
    # one after resolution failed would expose a search-engine redirect rather
    # than the source page to the fetcher and to users.
    is_baidu_host = hostname == "baidu.com" or hostname.endswith(".baidu.com")
    if is_baidu_host and unquote(path).rstrip("/") == "/link":
        return True
    # Baidu's own search-of-search pages are not article content.
    if hostname.endswith(".baidu.com") and path.startswith("/s"):
        return True
    if has_noise_path(parsed.path):
        return True
    return bool(has_search_query(parsed.path, parsed.query))


def has_listing_path(path: str) -> bool:
    """Return whether a path looks like a tag, category, or author listing."""

    return bool(set(path_segments(path)) & LISTING_PATH_SEGMENTS)


def has_noise_path(path: str) -> bool:
    segments = path_segments(path)
    if not segments:
        return False

    leaf = segments[-1]
    if leaf in DOORWAY_PATH_LEAFS:
        return True
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
    return [segment for segment in re.split(r"/+", unquote(path).lower().strip("/")) if segment]


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
    score -= LISTING_PATH_PENALTY if set(segments) & LISTING_PATH_SEGMENTS else 0
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
    r"current|recent|new|now|"
    # Relative time wording carries no topical signal. Leaving it in made
    # "this"/"year" the distinctive terms of "who wins the world cup this
    # year", so the multi-entity guard rejected the correct page.
    r"this|that|these|those|year|years|month|months|week|weeks|"
    r"day|days|today|yesterday|tomorrow|currently)\b",
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
    # Relative time markers, dropped for the same reason as the Latin ones.
    "今年",
    "去年",
    "明年",
    "本年",
    "目前",
    "现在",
    "最近",
)

_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_QUANTITATIVE_QUERY_RE = re.compile(
    r"(?:\bhow\s+many\b|\bnumber\s+of\b|\bcount\s+of\b|\btotal\s+number\b|"
    r"多少|几(?:个|条|座|项|种|人|家|所|辆|次)|数量|总数|合计)"
)
_METRO_LINE_COUNT_QUERY_RE = re.compile(
    r"(?:地铁|轨道交通).*(?:线路\s*数量|线路数|条数|几条(?:线|线路)?)"
)
_METRO_ENTITY_RE = re.compile(r"([\u3400-\u4dbf\u4e00-\u9fff]{2,8})(?:市)?(?:地铁|轨道交通)")
_LINE_QUANTITY_RE = re.compile(
    r"\d[\d,.]*\s*(?:条(?:线|线路)?|lines?|routes?)",
    re.I,
)
_QUANTITY_UNIT_PATTERN = (
    r"(?:lines?|routes?|stations?|items?|people|persons?|companies|schools?|"
    r"vehicles?|times?|条(?:线|线路)?|个|座|项|种|人|家|所|辆|次)"
)
_QUANTITY_CUE_RE = re.compile(
    rf"(?:\bthere\s+(?:are|were)\b|\b(?:total|number|count)\b|"
    rf"共(?:有|计|运营|开通)?|总计|合计|共有|达到|"
    rf"(?:运营|开通)线路(?:总数)?(?:为|达|达到))"
    rf"[^\d]{{0,20}}"
    rf"((?:\d{{1,3}}(?:,\d{{3}})+)|\d{{1,6}})(?:\.\d+)?"
    rf"(?:\s*{_QUANTITY_UNIT_PATTERN})?",
    re.I,
)
_QUANTITY_STATUS_RE = re.compile(
    rf"(?<!\d)((?:\d{{1,3}}(?:,\d{{3}})+)|\d{{1,6}})(?:\.\d+)?\s*"
    rf"{_QUANTITY_UNIT_PATTERN}\s*"
    rf"(?:are\s+(?:currently\s+)?(?:operational|open|in\s+operation)|"
    rf"投入运营|正在运营|运营中|已开通)",
    re.I,
)

YEAR_MATCH_MAX_BONUS = 20
YEAR_MISSING_PENALTY = 12
YEAR_CONFLICT_PENALTY = 30
QUANTITY_ANSWER_BONUS = 16
QUANTITY_ANSWER_MISSING_PENALTY = 12
PAGE_PRIMARY_WEIGHT = 0.75
PAGE_LEAD_MAX_CHARS = 1200


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


def _requested_years(query: str) -> set[str]:
    return set(_YEAR_RE.findall(_normalize_relevance_text(query)))


def _year_relevance_delta(text: str, query: str) -> int:
    """Reward requested years and penalize absent or conflicting years."""

    requested = _requested_years(query)
    if not requested:
        return 0

    observed = set(_YEAR_RE.findall(_normalize_relevance_text(text)))
    matches = requested & observed
    if matches:
        coverage = len(matches) / len(requested)
        return round(10 + (YEAR_MATCH_MAX_BONUS - 10) * coverage)
    if observed:
        return -YEAR_CONFLICT_PENALTY
    return -YEAR_MISSING_PENALTY


def _is_quantitative_query(query: str) -> bool:
    return PRICE_INTENT not in detect_query_intents(query) and bool(
        _QUANTITATIVE_QUERY_RE.search(_normalize_relevance_text(query))
    )


def _has_quantity_answer(text: str, query: str = "") -> bool:
    """Return whether text states a non-year number as an answer-like fact."""

    normalized = _normalize_relevance_text(text)
    for pattern in (_QUANTITY_CUE_RE, _QUANTITY_STATUS_RE):
        for match in pattern.finditer(normalized):
            value = match.group(1).replace(",", "").split(".", 1)[0]
            if not _YEAR_RE.fullmatch(value) and _quantity_context_matches_query(
                normalized,
                match.start(),
                match.end(),
                query,
                answer_text=match.group(0),
            ):
                return True
    return False


def _quantity_context_matches_query(
    text: str,
    start: int,
    end: int,
    query: str,
    *,
    answer_text: str,
) -> bool:
    if not query:
        return True

    normalized_query = _normalize_relevance_text(query)
    context = text[max(0, start - 120) : min(len(text), end + 120)]
    requested_years = _requested_years(normalized_query)
    if requested_years and not requested_years & set(_YEAR_RE.findall(context)):
        return False

    if _METRO_LINE_COUNT_QUERY_RE.search(normalized_query):
        if not _LINE_QUANTITY_RE.search(answer_text):
            return False
        entity_match = _METRO_ENTITY_RE.search(normalized_query)
        if entity_match:
            entity_context = text[max(0, start - 80) : min(len(text), end + 80)]
            if entity_match.group(1) not in entity_context:
                return False

    if "运营线路总数" in normalized_query:
        future_or_construction = re.search(
            r"(?:在建|规划|计划|预计|建设|新线|将开通|待开通)",
            context,
        )
        operating_scope = re.search(
            r"(?:运营|已开通|开通运营|投入运营|运营中|"
            r"operational|in\s+operation)",
            context,
            re.I,
        )
        explicit_total = "线路" in context and bool(
            re.search(r"(?:共有|共计|总计|合计|总数)", context)
        )
        if future_or_construction and not operating_scope:
            return False
        if not operating_scope and not explicit_total:
            return False

    query_terms = {term for term in _search_query_terms(query) if term not in GENERIC_QUERY_TERMS}
    if not query_terms:
        return True

    matched_terms = sum(1 for term in query_terms if term in context)
    required_matches = 2 if len(query_terms) >= 2 else 1
    return matched_terms >= required_matches


def _answer_completeness_delta(text: str, query: str) -> int:
    if not _is_quantitative_query(query):
        return 0
    if _has_quantity_answer(text, query):
        return QUANTITY_ANSWER_BONUS
    return -QUANTITY_ANSWER_MISSING_PENALTY


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
GENERIC_QUERY_TERMS = frozenset(
    {
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
    }
)
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
    distinctive_terms = {term for term in query_terms if _term_weight(term, acronyms) >= 1.0}
    if len(distinctive_terms) >= MULTI_ENTITY_MIN_MATCHES:
        matched_distinctive = sum(1 for term in distinctive_terms if term in normalized)
        if matched_distinctive < MULTI_ENTITY_MIN_MATCHES:
            return -TEXT_RELEVANCE_MISS_PENALTY

    matched_weight = sum(_term_weight(term, acronyms) for term in query_terms if term in normalized)
    coverage = matched_weight / total_weight

    if coverage < TEXT_RELEVANCE_MIN_COVERAGE:
        # Below the bar: either nothing matched or only generic filler did.
        # Penalize proportionally so a pure miss is punished hardest while a
        # weak generic-only match is nudged down rather than rewarded.
        penalty_fraction = 1.0 - (coverage / TEXT_RELEVANCE_MIN_COVERAGE)
        lexical_delta = -round(TEXT_RELEVANCE_MISS_PENALTY * penalty_fraction)
    else:
        lexical_delta = round(coverage * TEXT_RELEVANCE_MAX_BONUS)

    return (
        lexical_delta
        + _year_relevance_delta(normalized, query)
        + _answer_completeness_delta(normalized, query)
        + answer_evidence_delta(normalized, query)
    )


def page_relevance_score(
    text: str,
    query: str,
    *,
    title: str = "",
    h1: str = "",
) -> int:
    """Score fetched content with early-page evidence weighted most heavily.

    HTML extraction preserves article order, so the first portion normally
    contains the H1 and lead even when the loader does not expose H1 metadata.
    An explicit ``h1`` can still be supplied by loaders that retain it.
    """

    normalized_text = _normalize_relevance_text(text)
    all_evidence = _normalize_relevance_text(
        " ".join(part for part in (title, h1, normalized_text) if part)
    )
    requested_years = _requested_years(query)
    if requested_years and not requested_years & set(_YEAR_RE.findall(all_evidence)):
        return -TEXT_RELEVANCE_MISS_PENALTY
    if _is_quantitative_query(query) and not _has_quantity_answer(all_evidence):
        return -TEXT_RELEVANCE_MISS_PENALTY
    if not has_required_answer_evidence(all_evidence, query):
        return -TEXT_RELEVANCE_MISS_PENALTY

    lead = normalized_text[:PAGE_LEAD_MAX_CHARS]
    primary = " ".join(part for part in (title, h1, lead) if part).strip()
    full = " ".join(part for part in (title, h1, normalized_text) if part).strip()
    if not primary:
        return text_relevance_delta(full, query)

    primary_delta = text_relevance_delta(primary, query)
    full_delta = text_relevance_delta(full, query)
    return round(PAGE_PRIMARY_WEIGHT * primary_delta + (1.0 - PAGE_PRIMARY_WEIGHT) * full_delta)


def is_page_text_relevant(
    text: str,
    query: str,
    *,
    title: str = "",
    h1: str = "",
) -> bool:
    """Return whether fetched page content has enough evidence to ground ``query``.

    Very short queries carry too little evidence for a reliable deterministic
    rejection, so this predicate deliberately abstains when fewer than two
    usable terms are available. Specific Chinese queries produce several CJK
    bigrams and are checked normally. Explicit years and requested quantities
    are never optional: a page must contain those answer facts even when the
    lexical predicate would otherwise abstain.
    """

    combined = _normalize_relevance_text(
        " ".join(part for part in (title, h1, text) if part).strip()
    )
    requested_years = _requested_years(query)
    if requested_years and not requested_years & set(_YEAR_RE.findall(combined)):
        return False
    if _is_quantitative_query(query) and not _has_quantity_answer(combined, query):
        return False
    if not has_required_answer_evidence(combined, query):
        return False

    query_terms = set(_search_query_terms(query))
    if len(query_terms) < 2:
        return True
    return page_relevance_score(text, query, title=title, h1=h1) > 0


def result_quality_score(result: SearchResult, *, query: str = "") -> int:
    """Score a search result using URL shape plus title/snippet relevance."""

    if prefetch_rejection_reason(result, query) is not None:
        return 0
    score = url_quality_score(result.url, query=query)
    if score <= 0:
        return score
    return (
        score
        + host_quality_score(result.url)
        + result_constraint_score(result, query)
        + text_relevance_delta(result.relevance_text, query)
    )


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
