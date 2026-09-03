from __future__ import annotations

from typing import Any

from .api_providers import (
    BingApiWebSearch,
    BraveWebSearch,
    SerperWebSearch,
    TavilyWebSearch,
)
from .baidu import BaiduWebSearch
from .bing import BingWebSearch, normalize_bing_market
from .duckduckgo import DuckDuckGoWebSearch
from .protocol import WebSearchProvider

API_PROVIDER_NAMES = ("serper", "brave", "tavily", "bing_api")
HTML_PROVIDER_NAMES = ("bing", "baidu", "duckduckgo")
SUPPORTED_PROVIDER_NAMES = (*API_PROVIDER_NAMES, *HTML_PROVIDER_NAMES)
_API_KEY_FIELDS = {
    "serper": "serper_api_key",
    "brave": "brave_search_api_key",
    "tavily": "tavily_api_key",
    "bing_api": "bing_search_api_key",
}


def normalize_provider_name(name: str | None) -> str:
    normalized = (name or "bing").strip().lower()
    aliases = {
        "azure_bing": "bing_api",
        "bing-api": "bing_api",
        "bingapi": "bing_api",
        "brave_search": "brave",
        "ddg": "duckduckgo",
        "google": "serper",
        "msn": "bing",
        "serperdev": "serper",
        "tavily_search": "tavily",
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
    timeout = getattr(config, "web_search_provider_timeout_seconds", 8)
    api_timeout = getattr(config, "web_search_api_timeout_seconds", 20)

    if provider_name == "bing":
        return BingWebSearch(
            market=normalize_bing_market(getattr(config, "web_search_region", None)),
            # REFACTOR: Propagate shared recency setting into Bing.
            timelimit=getattr(config, "web_search_timelimit", None),
            verify_ssl=verify_ssl,
            timeout=timeout,
        )
    if provider_name == "duckduckgo":
        return DuckDuckGoWebSearch(
            region=getattr(config, "web_search_region", "wt-wt"),
            timelimit=getattr(config, "web_search_timelimit", None),
            verify_ssl=verify_ssl,
            timeout=timeout,
        )
    if provider_name == "baidu":
        return BaiduWebSearch(verify_ssl=verify_ssl, timeout=timeout)
    if provider_name == "serper":
        return SerperWebSearch(
            api_key=getattr(config, "serper_api_key", ""),
            verify_ssl=verify_ssl,
            timeout=api_timeout,
        )
    if provider_name == "brave":
        return BraveWebSearch(
            api_key=getattr(config, "brave_search_api_key", ""),
            verify_ssl=verify_ssl,
            timeout=api_timeout,
        )
    if provider_name == "tavily":
        return TavilyWebSearch(
            api_key=getattr(config, "tavily_api_key", ""),
            verify_ssl=verify_ssl,
            timeout=api_timeout,
        )
    if provider_name == "bing_api":
        return BingApiWebSearch(
            api_key=getattr(config, "bing_search_api_key", ""),
            endpoint=getattr(config, "bing_search_endpoint", ""),
            verify_ssl=verify_ssl,
            timeout=api_timeout,
        )

    raw_name = name if name is not None else getattr(config, "web_search_provider", provider_name)
    choices = ", ".join(repr(provider) for provider in SUPPORTED_PROVIDER_NAMES)
    raise ValueError(f"Unsupported WEB_SEARCH_PROVIDER {raw_name!r}; use {choices}.")


def is_search_provider_configured(name: str, config: Any | None) -> bool:
    """Return whether a provider can be constructed without exposing secrets."""

    provider_name = normalize_provider_name(name)
    key_field = _API_KEY_FIELDS.get(provider_name)
    if key_field is None:
        return provider_name in HTML_PROVIDER_NAMES
    return bool(str(getattr(config, key_field, "") or "").strip())


def configured_api_provider_names(config: Any | None) -> list[str]:
    """Return configured key-backed providers in stable preference order."""

    return [
        name
        for name in API_PROVIDER_NAMES
        if is_search_provider_configured(name, config)
    ]


__all__ = [
    "API_PROVIDER_NAMES",
    "HTML_PROVIDER_NAMES",
    "SUPPORTED_PROVIDER_NAMES",
    "configured_api_provider_names",
    "get_search_provider",
    "is_search_provider_configured",
    "normalize_provider_name",
]
