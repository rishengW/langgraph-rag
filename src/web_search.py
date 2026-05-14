"""Web search helpers for discovering RAG source URLs."""

from __future__ import annotations

import logging
import hashlib
import ssl
from dataclasses import replace
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import Request, urlopen

from .config import Settings


logger = logging.getLogger(__name__)


def _load_ddgs():
    """Import the installed DuckDuckGo search client.

    The package was renamed from `duckduckgo-search` to `ddgs` in newer
    releases. Supporting both keeps the app usable across environments.
    """

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

    return [result.get("href", "") for result in results]


def _unwrap_duckduckgo_redirect(href: str) -> str:
    if href.startswith("//"):
        href = f"https:{href}"
    elif href.startswith("/"):
        href = f"https://duckduckgo.com{href}"

    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)

    return href


def _duckduckgo_html_search(question: str, settings: Settings) -> list[str]:
    from bs4 import BeautifulSoup

    params = {
        "q": question,
        "kl": settings.web_search_region,
    }
    if settings.web_search_timelimit:
        params["df"] = settings.web_search_timelimit

    ssl_context = None
    if not settings.web_search_verify_ssl:
        ssl_context = ssl._create_unverified_context()

    html = ""
    last_error: Exception | None = None
    for base_url in ("https://html.duckduckgo.com/html/", "https://duckduckgo.com/html/"):
        request = Request(
            f"{base_url}?{urlencode(params)}",
            headers={"User-Agent": "rag-langgraph-local/1.0"},
        )
        try:
            with urlopen(request, timeout=30, context=ssl_context) as response:
                html = response.read().decode("utf-8", errors="replace")
            break
        except Exception as exc:
            last_error = exc

    if not html:
        raise RuntimeError(f"DuckDuckGo HTML search failed: {last_error}")

    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    for anchor in soup.select("a.result__a"):
        href = anchor.get("href", "")
        if href:
            urls.append(_unwrap_duckduckgo_redirect(href))
        if len(urls) >= settings.web_search_max_results:
            break

    return urls


def discover_urls_from_web(question: str, settings: Settings) -> list[str]:
    """Search the web for pages that can be used as RAG sources."""

    if not settings.web_search_enabled:
        return []

    ddgs_error: Exception | None = None
    try:
        urls = _normalize_urls(_ddgs_search(question, settings))
    except Exception as exc:
        ddgs_error = exc
        logger.warning("DDGS search failed; trying DuckDuckGo HTML fallback: %s", exc)
        urls = []

    if not urls:
        try:
            urls = _normalize_urls(_duckduckgo_html_search(question, settings))
        except Exception as exc:
            if ddgs_error:
                raise RuntimeError(
                    f"DDGS search failed ({ddgs_error}); "
                    f"DuckDuckGo HTML fallback also failed ({exc})"
                ) from exc
            raise

    logger.info("Discovered %s URL(s) from web search", len(urls))
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
