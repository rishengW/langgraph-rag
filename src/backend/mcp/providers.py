"""Tool providers for injected, retrieval, memory, builtin, and session tools."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from src.errors import ConfigurationError
from .catalog import compose_snapshot, descriptor_from_tool
from .models import (
    ProviderHealth,
    RiskLevel,
    ToolCatalogSnapshot,
    ToolDescriptor,
    ToolSource,
)

if TYPE_CHECKING:
    from src.config import Settings

ToolEntry = tuple[BaseTool, ToolDescriptor]
RiskResolver = Callable[[BaseTool], RiskLevel]


class FactoryToolProvider:
    """Lifecycle adapter over a deterministic local BaseTool factory."""

    def __init__(
        self,
        name: str,
        factory: Callable[[], Sequence[BaseTool]],
        *,
        source: ToolSource,
        risk: RiskLevel | RiskResolver,
    ) -> None:
        self.name = name
        self._factory = factory
        self._source = source
        self._risk = risk
        self._snapshot: ToolCatalogSnapshot | None = None
        self._status = "closed"

    def tools(self) -> tuple[BaseTool, ...]:
        """Build one detached raw contribution for compatibility callers."""

        return tuple(self._factory())

    def entries(self) -> tuple[ToolEntry, ...]:
        """Build and validate one detached local provider contribution."""

        entries: list[ToolEntry] = []
        for tool in self.tools():
            if not isinstance(tool, BaseTool):
                raise ConfigurationError(f"Provider {self.name!r} returned a non-BaseTool")
            risk = self._risk(tool) if callable(self._risk) else self._risk
            entries.append(
                (
                    tool,
                    descriptor_from_tool(tool, source=self._source, risk_level=risk),
                )
            )
        return tuple(entries)

    async def start(self) -> None:
        """Build a complete provider snapshot before reporting readiness."""

        self._snapshot = compose_snapshot(self.entries(), generation=1)
        self._status = "ready"

    async def snapshot(self) -> ToolCatalogSnapshot:
        """Return the provider's current immutable snapshot."""

        if self._snapshot is None:
            raise ConfigurationError(f"Provider {self.name!r} is not started")
        return self._snapshot

    async def health(self) -> ProviderHealth:
        """Return bounded provider readiness metadata."""

        generation = self._snapshot.generation if self._snapshot is not None else 0
        return ProviderHealth(
            provider=self.name,
            status=self._status,  # type: ignore[arg-type]
            generation=generation,
        )

    async def close(self) -> None:
        """Mark this local provider closed; snapshots remain immutable."""

        self._status = "closed"


class InjectedToolProvider(FactoryToolProvider):
    """Provider preserving the existing GraphProviders.tools injection seam."""

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        captured = tuple(tools)
        super().__init__(
            "injected",
            lambda: captured,
            source="local",
            risk=_injected_risk,
        )


class RetrieverToolProvider(FactoryToolProvider):
    """Heavy-path Chroma retriever provider."""

    def __init__(self, settings: Settings, *, rebuild: bool) -> None:
        def build() -> Sequence[BaseTool]:
            from ..core.retriever import build_retriever_tool

            return (build_retriever_tool(settings, rebuild=rebuild),)

        super().__init__("retriever", build, source="local", risk="read")


class WebSearchToolProvider(FactoryToolProvider):
    """Live web-search provider, required by the lightweight graph."""

    def __init__(self, settings: Settings, *, required: bool) -> None:
        def build() -> Sequence[BaseTool]:
            from ..tools import build_web_search_tool

            if not required and not settings.web_search_enabled:
                return ()
            return (build_web_search_tool(settings),)

        super().__init__("web-search", build, source="builtin", risk="read")


class MemoryToolProvider(FactoryToolProvider):
    """Optional long-term memory provider with read/write risk metadata."""

    def __init__(self, settings: Settings) -> None:
        def build() -> Sequence[BaseTool]:
            from ..tools import build_memory_tools

            return build_memory_tools(settings) if settings.memory_enabled else ()

        super().__init__("memory", build, source="local", risk=_memory_risk)


class BuiltinToolProvider(FactoryToolProvider):
    """Configured builtin network-read and local-compute tools."""

    def __init__(self, settings: Settings) -> None:
        def build() -> Sequence[BaseTool]:
            from src.backend import tools

            built: list[BaseTool] = []
            factories = (
                (settings.weather_enabled, tools.build_weather_tool),
                (settings.stock_enabled, tools.build_stock_tool),
                (settings.currency_enabled, tools.build_currency_tool),
                (settings.wikipedia_enabled, tools.build_wikipedia_tool),
                (settings.directions_enabled, tools.build_directions_tool),
                (settings.map_enabled, tools.build_map_tool),
                (settings.math_enabled, tools.build_math_tool),
                (settings.statistics_enabled, tools.build_statistics_tool),
                (settings.linalg_enabled, tools.build_linalg_tool),
                (settings.number_theory_enabled, tools.build_number_theory_tool),
                (settings.datetime_enabled, tools.build_datetime_tool),
                (settings.summarize_url_enabled, tools.build_summarize_url_tool),
            )
            for enabled, factory in factories:
                if enabled:
                    built.append(factory(settings))
            return built

        super().__init__("builtin", build, source="builtin", risk=_builtin_risk)


class DocumentToolProvider(FactoryToolProvider):
    """Configured local document readers confined by existing factories."""

    def __init__(self, settings: Settings) -> None:
        def build() -> Sequence[BaseTool]:
            from src.backend import tools

            if not settings.file_read_enabled:
                return ()
            return (
                tools.build_text_file_tool(settings),
                tools.build_markdown_file_tool(settings),
                tools.build_word_tool(settings),
                tools.build_excel_tool(settings),
                tools.build_pdf_tool(settings),
            )

        super().__init__("documents", build, source="local", risk="read")


class SessionEditingToolProvider(FactoryToolProvider):
    """Chat-only editors whose existing factories enforce session confinement."""

    def __init__(
        self,
        settings: Settings,
        *,
        session_root: Path | None,
        thread_id: str,
    ) -> None:
        def build() -> Sequence[BaseTool]:
            from src.backend import tools

            keyword = {"session_root": session_root, "thread_id": thread_id}
            return (
                *tools.build_word_edit_tools(settings, **keyword),
                *tools.build_text_edit_tools(settings, **keyword),
                *tools.build_markdown_edit_tools(settings, **keyword),
                *tools.build_csv_edit_tools(settings, **keyword),
                *tools.build_excel_create_tools(settings, **keyword),
                *tools.build_excel_edit_tools(settings, **keyword),
                *tools.build_powerpoint_edit_tools(settings, **keyword),
            )

        super().__init__("session-editing", build, source="local", risk="write")


class DisabledOutboundMCPProvider:
    """Explicit future seam that cannot activate outbound MCP in Phase 3."""

    def __init__(self, *, enabled: bool = False) -> None:
        if enabled:
            raise ConfigurationError("Outbound MCP providers are disabled in this release")
        self._closed = False
        self._snapshot = ToolCatalogSnapshot(generation=1, tools=(), descriptors=())

    def entries(self) -> tuple[ToolEntry, ...]:
        """Return no tools while outbound MCP remains separately prohibited."""

        return ()

    async def start(self) -> None:
        """Start the disabled seam without opening transports."""

        self._closed = False

    async def snapshot(self) -> ToolCatalogSnapshot:
        """Return the empty immutable outbound contribution."""

        return self._snapshot

    async def health(self) -> ProviderHealth:
        """Report disabled rather than degraded or ready."""

        return ProviderHealth(
            provider="outbound-mcp",
            status="closed" if self._closed else "disabled",
            generation=0,
        )

    async def close(self) -> None:
        """Close without any transport lifecycle side effects."""

        self._closed = True


def default_provider_entries(
    settings: Settings,
    *,
    lightweight: bool,
    rebuild_vectorstore: bool,
    session_root: Path | None,
    thread_id: str,
) -> tuple[ToolEntry, ...]:
    """Compose full/lightweight entries once, differing only at retrieval/search."""

    providers = _default_providers(
        settings,
        lightweight=lightweight,
        rebuild_vectorstore=rebuild_vectorstore,
        session_root=session_root,
        thread_id=thread_id,
    )
    return tuple(entry for provider in providers for entry in provider.entries())


def default_provider_tools(
    settings: Settings,
    *,
    lightweight: bool,
    rebuild_vectorstore: bool,
    session_root: Path | None,
    thread_id: str,
) -> tuple[BaseTool, ...]:
    """Return raw tools through the same provider composition for compatibility."""

    providers = _default_providers(
        settings,
        lightweight=lightweight,
        rebuild_vectorstore=rebuild_vectorstore,
        session_root=session_root,
        thread_id=thread_id,
    )
    return tuple(tool for provider in providers for tool in provider.tools())


def _default_providers(
    settings: Settings,
    *,
    lightweight: bool,
    rebuild_vectorstore: bool,
    session_root: Path | None,
    thread_id: str,
) -> tuple[FactoryToolProvider, ...]:
    providers: list[FactoryToolProvider] = []
    if not lightweight:
        providers.append(RetrieverToolProvider(settings, rebuild=rebuild_vectorstore))
    providers.extend(
        (
            WebSearchToolProvider(settings, required=lightweight),
            MemoryToolProvider(settings),
            BuiltinToolProvider(settings),
            DocumentToolProvider(settings),
            SessionEditingToolProvider(
                settings,
                session_root=session_root,
                thread_id=thread_id,
            ),
        )
    )
    return tuple(providers)


def _memory_risk(tool: BaseTool) -> RiskLevel:
    return "read" if tool.name == "recall_memory" else "write"


def _builtin_risk(tool: BaseTool) -> RiskLevel:
    if tool.name in {
        "solve_math",
        "compute_statistics",
        "linear_algebra",
        "number_theory",
        "date_time",
    }:
        return "execute"
    return "read"


def _injected_risk(tool: BaseTool) -> RiskLevel:
    metadata = tool.metadata or {}
    candidate = metadata.get("risk_level")
    if candidate in {"read", "write", "execute", "admin"}:
        return candidate  # type: ignore[return-value]
    return "read"


__all__ = [
    "BuiltinToolProvider",
    "DisabledOutboundMCPProvider",
    "DocumentToolProvider",
    "FactoryToolProvider",
    "InjectedToolProvider",
    "MemoryToolProvider",
    "RetrieverToolProvider",
    "SessionEditingToolProvider",
    "ToolEntry",
    "WebSearchToolProvider",
    "default_provider_entries",
    "default_provider_tools",
]
