from __future__ import annotations

from typing import Any

from .baidu import BaiduWebSearch
from .duckduckgo import DuckDuckGoWebSearch
from .protocol import WebSearchProvider


def normalize_provider_name(name: str | None) -> str:
    return (name or "baidu").strip().lower()


def get_search_provider(
    name: str | None = None,
    config: Any | None = None,
) -> WebSearchProvider:
    """Create a web search provider from a provider name and settings-like config."""

    if config is None and name is not None and not isinstance(name, str):
        config = name
        name = None

    provider_name = normalize_provider_name(
        name if name is not None else getattr(config, "web_search_provider", "baidu")
    )
    verify_ssl = getattr(config, "web_search_verify_ssl", True)

    if provider_name in ("duckduckgo", "ddg"):
        return DuckDuckGoWebSearch(
            region=getattr(config, "web_search_region", "wt-wt"),
            timelimit=getattr(config, "web_search_timelimit", None),
            verify_ssl=verify_ssl,
        )
    if provider_name == "baidu":
        return BaiduWebSearch(verify_ssl=verify_ssl)

    raw_name = (
        name if name is not None else getattr(config, "web_search_provider", provider_name)
    )
    raise ValueError(
        f"Unsupported WEB_SEARCH_PROVIDER {raw_name!r}; use 'duckduckgo' or 'baidu'."
    )
