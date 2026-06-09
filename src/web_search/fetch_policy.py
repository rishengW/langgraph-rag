# REFACTOR: Domain-aware fetch policy for lightweight web-search page loading.
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from urllib.parse import urlparse

DEFAULT_JS_FALLBACK_DOMAINS = [
    "baike.baidu.com",
    "zhuanlan.zhihu.com",
    "apps.microsoft.com",
    "deepseek.net",
]

BROWSER_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


@dataclass(frozen=True)
class FetchPolicy:
    """Per-URL loading policy for HTTP and optional JS fallback."""

    force_js: bool = False
    retry_js_on_low_text: bool = False
    request_headers: dict[str, str] = field(default_factory=dict)


def resolve_fetch_policy(
    url: str,
    *,
    js_fallback_enabled: bool = False,
    js_fallback_domains: Sequence[str] | None = None,
    js_force_domains: Sequence[str] | None = None,
) -> FetchPolicy:
    """Return the domain-specific fetch policy for a URL."""

    fallback_domains = (
        DEFAULT_JS_FALLBACK_DOMAINS if js_fallback_domains is None else list(js_fallback_domains)
    )
    force_domains = list(js_force_domains or [])
    host = url_host(url)
    fallback_match = domain_matches(host, fallback_domains)
    force_match = domain_matches(host, force_domains)
    browser_profile = fallback_match or force_match

    return FetchPolicy(
        force_js=bool(js_fallback_enabled and force_match),
        retry_js_on_low_text=bool(js_fallback_enabled and browser_profile),
        request_headers=dict(BROWSER_REQUEST_HEADERS) if browser_profile else {},
    )


def url_host(url: str) -> str:
    """Extract a lowercase hostname from a URL or host string."""

    candidate = (url or "").strip()
    if not candidate:
        return ""
    parsed = urlparse(candidate if "://" in candidate else f"https://{candidate}")
    return (parsed.hostname or "").strip(".").lower()


def domain_matches(host_or_url: str, domains: Sequence[str]) -> bool:
    """Return whether a host exactly matches, or is below, a configured domain."""

    host = url_host(host_or_url)
    if not host:
        return False

    for domain in domains:
        normalized = url_host(str(domain))
        if normalized and (host == normalized or host.endswith(f".{normalized}")):
            return True
    return False


__all__ = [
    "BROWSER_REQUEST_HEADERS",
    "DEFAULT_JS_FALLBACK_DOMAINS",
    "FetchPolicy",
    "domain_matches",
    "resolve_fetch_policy",
    "url_host",
]
