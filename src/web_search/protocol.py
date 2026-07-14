from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable

from .common import SearchResult


@dataclass(frozen=True)
class RankedSearchResult:
    """A usable provider result with ranking signals preserved end to end."""

    url: str
    title: str = ""
    snippet: str = ""
    provider: str = ""
    provider_rank: int = 0
    relevance_score: int = 0
    quality_score: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return a LangChain-artifact-safe representation."""

        return asdict(self)


@runtime_checkable
class WebSearchProvider(Protocol):
    """Search provider interface for URL discovery."""

    @property
    def provider_name(self) -> str:
        """Stable provider identifier used in logs and factory selection."""
        ...

    def search(self, query: str, max_results: int = 20) -> list[str]:
        """Return candidate source URLs for a query."""
        ...

    def search_results(self, query: str, max_results: int = 20) -> list[SearchResult]:
        """Return candidate results with title/snippet text for relevance ranking."""
        ...


__all__ = ["RankedSearchResult", "WebSearchProvider"]
