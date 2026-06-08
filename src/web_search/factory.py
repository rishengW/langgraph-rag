from __future__ import annotations

from typing import Any

from .baidu import BaiduWebSearch
from .bing import BingWebSearch, normalize_bing_market
from .duckduckgo import DuckDuckGoWebSearch
from .protocol import WebSearchProvider


def normalize_provider_name(name: str | None) -> str:
    normalized = (name or "bing").strip().lower()
    aliases = {
        "ddg": "duckduckgo",
        "msn": "bing",
    }
    return aliases.get(normalized, normalized)


def get_search_provider(
    name: str | None = None,
    config: Any | None = None,
) -> WebSearchProvider:
    """Create a web search provider from a provider name and settings-like config."""

    if config is None and name is not None and not isinstance(name, str):
        config = name
        name = None

    provider_name = normalize_provider_name(
        name if name is not None else getattr(config, "web_search_provider", "bing")
    )
    verify_ssl = getattr(config, "web_search_verify_ssl", True)

    if provider_name == "bing":
        return BingWebSearch(
            market=normalize_bing_market(getattr(config, "web_search_region", None)),
            verify_ssl=verify_ssl,
        )
    if provider_name == "duckduckgo":
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
        f"Unsupported WEB_SEARCH_PROVIDER {raw_name!r}; use 'bing', 'baidu', "
        "or 'duckduckgo'."
    )
