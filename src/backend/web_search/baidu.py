from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from time import monotonic
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import urlopen

from bs4 import BeautifulSoup

from .common import (
    BAIDU_BASE_URL,
    SearchResult,
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
    timeout: float = 8.0

    @property
    def provider_name(self) -> str:
        return "baidu"

    def search(self, query: str, max_results: int = 20) -> list[str]:
        return [result.url for result in self.search_results(query, max_results)]

    def search_results(self, query: str, max_results: int = 20) -> list[SearchResult]:
        provider_budget = max(0.1, float(self.timeout) - 0.5)
        deadline_at = monotonic() + provider_budget
        params = urlencode({"wd": query, "rn": max(1, max_results)})
        search_url = f"{BAIDU_BASE_URL}/s?{params}"

        with urlopen(
            search_request(search_url),
            timeout=provider_budget,
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
        raw_results = candidate_baidu_results(soup)
        if not raw_results:
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            logger.warning(
                "Baidu returned no parseable result links "
                "(final_url=%s, title=%r, html_bytes=%s)",
                final_url,
                title[:120],
                len(html),
            )

        candidates = [
            (urljoin(BAIDU_BASE_URL, href), title, snippet)
            for href, title, snippet in raw_results[: max(1, max_results)]
        ]
        resolved_urls = [url for url, _title, _snippet in candidates]
        redirect_indexes = [
            index
            for index, url in enumerate(resolved_urls)
            if is_baidu_result_redirect(url)
        ]
        remaining = deadline_at - monotonic()
        if redirect_indexes and remaining > 0.1:
            executor = ThreadPoolExecutor(
                max_workers=min(4, len(redirect_indexes)),
                thread_name_prefix="baidu-redirect",
            )
            futures: dict[Future[str], int] = {
                executor.submit(
                    self.resolve_redirect,
                    resolved_urls[index],
                    remaining,
                ): index
                for index in redirect_indexes
            }
            try:
                done, not_done = wait(futures, timeout=remaining)
                for future in done:
                    index = futures[future]
                    try:
                        resolved_urls[index] = future.result()
                    except Exception as exc:
                        logger.debug(
                            "Could not resolve Baidu redirect %s: %s",
                            resolved_urls[index],
                            exc,
                        )
                for future in not_done:
                    future.cancel()
                if not_done:
                    logger.debug(
                        "Baidu redirect deadline left %s URL(s) unresolved",
                        len(not_done),
                    )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        results: list[SearchResult] = []
        seen: set[str] = set()
        for absolute_url, (_original_url, title, snippet) in zip(
            resolved_urls,
            candidates,
            strict=True,
        ):
            if not absolute_url.startswith(("http://", "https://")):
                continue
            if absolute_url in seen:
                continue
            seen.add(absolute_url)
            results.append(SearchResult(url=absolute_url, title=title, snippet=snippet))

        return results

    def resolve_redirect(self, url: str, timeout: float | None = None) -> str:
        """Resolve Baidu's result redirect URL to the target page when possible."""

        if not is_baidu_result_redirect(url):
            return url

        request_timeout = self.timeout if timeout is None else min(self.timeout, timeout)
        try:
            with urlopen(
                search_request(url),
                timeout=max(0.1, request_timeout),
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
    return [href for href, _title, _snippet in candidate_baidu_results(soup)]


def candidate_baidu_results(soup: BeautifulSoup) -> list[tuple[str, str, str]]:
    """Extract (href, title, snippet) tuples from Baidu result containers.

    Baidu wraps each organic result in ``div.c-container`` (or ``.result`` /
    ``.result-op``) with the link in an ``h3`` and the abstract in
    ``.c-abstract``. Reading the abstract gives the ranker topical text.
    """

    results: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    for container in soup.select("div.c-container, .result, .result-op"):
        anchor = container.select_one("h3 a[href]")
        if anchor is None:
            continue
        href = cast(str, anchor.get("href", ""))
        if not href or href in seen:
            continue
        seen.add(href)
        title = anchor.get_text(" ", strip=True)
        snippet_node = container.select_one(
            ".c-abstract, [class*='content-right'], .c-span-last"
        )
        snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
        results.append((href, title, snippet))

    if results:
        return results

    # Fallback: bare result-link anchors without recognizable containers.
    for selector in ("h3.t a[href]", "div.c-container h3 a[href]", 'a[href*="/link?"]'):
        for anchor in soup.select(selector):
            href = cast(str, anchor.get("href", ""))
            if href and href not in seen:
                seen.add(href)
                results.append((href, anchor.get_text(" ", strip=True), ""))

    return results
