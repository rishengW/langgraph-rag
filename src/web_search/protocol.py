from __future__ import annotations

from typing import Protocol, runtime_checkable


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
