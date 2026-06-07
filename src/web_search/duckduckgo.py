from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import urlopen

from bs4 import BeautifulSoup

from .common import (
    DUCKDUCKGO_BASE_URL,
    normalize_urls,
    search_request,
    urlopen_context,
)


logger = logging.getLogger(__name__)


def load_ddgs():
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


@dataclass(frozen=True)
class DuckDuckGoWebSearch:
    """DuckDuckGo search provider using ddgs with an HTML fallback."""

    region: str = "wt-wt"
    timelimit: str | None = None
    verify_ssl: bool = True

    @property
    def provider_name(self) -> str:
        return "duckduckgo"

    def search(self, query: str, max_results: int = 20) -> list[str]:
        ddgs_error: Exception | None = None
        try:
            urls = normalize_urls(self.search_ddgs(query, max_results))
        except Exception as exc:
            ddgs_error = exc
            logger.warning("DDGS search failed; trying DuckDuckGo HTML fallback: %s", exc)
            urls = []

        if urls:
            return urls[:max_results]

        try:
            return normalize_urls(self.search_html(query, max_results))[:max_results]
        except Exception as exc:
            if ddgs_error:
                raise RuntimeError(
                    f"DDGS search failed ({ddgs_error}); "
                    f"DuckDuckGo HTML fallback also failed ({exc})"
                ) from exc
            raise

    def search_ddgs(self, query: str, max_results: int) -> list[str]:
        DDGS = load_ddgs()

        search_kwargs = {
            "region": self.region,
            "max_results": max_results,
        }
        if self.timelimit:
            search_kwargs["timelimit"] = self.timelimit

        try:
            client = DDGS(verify=self.verify_ssl)
        except TypeError:
            # Older client versions do not expose the SSL verification argument.
            client = DDGS()

        with client as ddgs:
            results = list(ddgs.text(query, **search_kwargs))

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

    def search_html(self, query: str, max_results: int) -> list[str]:
        params = {
            "q": query,
            "kl": self.region,
        }
        if self.timelimit:
            params["df"] = self.timelimit

        html = ""
        last_error: Exception | None = None
        for base_url in (
            "https://html.duckduckgo.com/html/",
            "https://duckduckgo.com/html/",
        ):
            try:
                with urlopen(
                    search_request(f"{base_url}?{urlencode(params)}"),
                    timeout=30,
                    context=urlopen_context(self.verify_ssl),
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
                urls.append(unwrap_duckduckgo_redirect(href))
            if len(normalize_urls(urls)) >= max_results:
                break

        return urls


def unwrap_duckduckgo_redirect(href: str) -> str:
    if href.startswith("//"):
        href = f"https:{href}"
    elif href.startswith("/"):
        href = f"{DUCKDUCKGO_BASE_URL}{href}"

    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)

    return href
