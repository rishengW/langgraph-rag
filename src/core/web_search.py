"""Web search helpers for discovering RAG source URLs."""

from __future__ import annotations

import hashlib
import logging
import ssl
from dataclasses import replace
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from .config import Settings


logger = logging.getLogger(__name__)

BAIDU_BASE_URL = "https://www.baidu.com"
DUCKDUCKGO_BASE_URL = "https://duckduckgo.com"
SEARCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


def _load_ddgs():
    """Import the installed DuckDuckGo search client."""

    import_errors: list[str] = []
    try:
        from ddgs import DDGS

        return DDGS
    except ImportError as exc:
        import_errors.append(f"ddgs: {exc}")

    try:
        from duckduckgo_search import DDGS

        return DDGS
    except ImportError as exc:
        import_errors.append(f"duckduckgo_search: {exc}")

    raise ImportError(
        "Install ddgs or duckduckgo-search for package-based web search "
        f"({'; '.join(import_errors)})"
    )


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


def _ddgs_search(question: str, settings: Settings) -> list[str]:
    DDGS = _load_ddgs()

    search_kwargs = {
        "region": settings.web_search_region,
        "max_results": settings.web_search_max_results,
    }
    if settings.web_search_timelimit:
        search_kwargs["timelimit"] = settings.web_search_timelimit

    try:
        client = DDGS(verify=settings.web_search_verify_ssl)
    except TypeError:
        # Older client versions do not expose the SSL verification argument.
        client = DDGS()

    with client as ddgs:
        results = list(ddgs.text(question, **search_kwargs))

    # DDGS clients have used several field names across versions
    # ("href", "link", "url"). Read the first one that's present.
    extracted: list[str] = []
    for result in results:
        for key in ("href", "link", "url"):
            value = result.get(key) if isinstance(result, dict) else None
            if value:
                extracted.append(value)
                break
    return extracted


def _unwrap_duckduckgo_redirect(href: str) -> str:
    if href.startswith("//"):
        href = f"https:{href}"
    elif href.startswith("/"):
        href = f"{DUCKDUCKGO_BASE_URL}{href}"

    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)

    return href


def _duckduckgo_html_search(question: str, settings: Settings) -> list[str]:
    params = {
        "q": question,
        "kl": settings.web_search_region,
    }
    if settings.web_search_timelimit:
        params["df"] = settings.web_search_timelimit

    html = ""
    last_error: Exception | None = None
    for base_url in ("https://html.duckduckgo.com/html/", "https://duckduckgo.com/html/"):
        try:
            with urlopen(
                _search_request(f"{base_url}?{urlencode(params)}"),
                timeout=30,
                context=_urlopen_context(settings),
            ) as response:
                html = response.read().decode("utf-8", errors="replace")
            break
        except Exception as exc:
            last_error = exc

    if not html:
        raise RuntimeError(f"DuckDuckGo HTML search failed: {last_error}")

    urls: list[str] = []
    for anchor in BeautifulSoup(html, "html.parser").select("a.result__a"):
        href = anchor.get("href", "")
        if href:
            urls.append(_unwrap_duckduckgo_redirect(href))
        if len(_normalize_urls(urls)) >= settings.web_search_max_results:
            break

    return urls


def _discover_urls_from_duckduckgo(question: str, settings: Settings) -> list[str]:
    ddgs_error: Exception | None = None
    try:
        urls = _normalize_urls(_ddgs_search(question, settings))
    except Exception as exc:
        ddgs_error = exc
        logger.warning("DDGS search failed; trying DuckDuckGo HTML fallback: %s", exc)
        urls = []

    if urls:
        return urls[: settings.web_search_max_results]

    try:
        return _normalize_urls(_duckduckgo_html_search(question, settings))[
            : settings.web_search_max_results
        ]
    except Exception as exc:
        if ddgs_error:
            raise RuntimeError(
                f"DDGS search failed ({ddgs_error}); "
                f"DuckDuckGo HTML fallback also failed ({exc})"
            ) from exc
        raise


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
}


def _is_noise_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    if not hostname:
        return True
    if hostname in NOISE_HOSTNAMES:
        return True
    # Baidu's own search-of-search pages are not article content.
    if hostname.endswith(".baidu.com") and urlparse(url).path.startswith("/s"):
        return True
    return False


def _select_top_urls(urls: list[str], settings: Settings) -> list[str]:
    """Drop noise hosts and keep the top-K most relevant URLs as ranked by
    the search engine. The engine's ordering is treated as the relevance
    ranking; we just trim and de-noise."""

    filtered = [url for url in urls if not _is_noise_url(url)]
    top_k = getattr(settings, "web_search_top_k", 0) or 0
    if top_k > 0:
        return filtered[:top_k]
    return filtered


def discover_urls_from_web(question: str, settings: Settings) -> list[str]:
    """Search the web for pages that can be used as RAG sources."""

    if not settings.web_search_enabled:
        return []

    provider = getattr(settings, "web_search_provider", "baidu").strip().lower()
    if provider in ("duckduckgo", "ddg"):
        urls = _discover_urls_from_duckduckgo(question, settings)
    elif provider == "baidu":
        urls = _discover_urls_from_baidu(question, settings)
    else:
        raise ValueError(
            "Unsupported WEB_SEARCH_PROVIDER "
            f"{settings.web_search_provider!r}; use 'duckduckgo' or 'baidu'."
        )

    selected = _select_top_urls(urls, settings)

    logger.info(
        "Discovered %s URL(s) from %s; keeping top %s after filtering",
        len(urls),
        "duckduckgo" if provider == "ddg" else provider,
        len(selected),
    )
    return selected


def settings_for_discovered_urls(settings: Settings, urls: list[str]) -> Settings:
    """Create isolated settings for a search-derived source set."""

    fingerprint = hashlib.sha256("\n".join(urls).encode("utf-8")).hexdigest()[:12]
    return replace(
        settings,
        source_urls=urls,
        chroma_dir=settings.chroma_dir / "web-search" / fingerprint,
        collection_name=f"{settings.collection_name}-web-{fingerprint}",
    )
