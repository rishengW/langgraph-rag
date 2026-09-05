"""Transport-neutral orchestration for stateless RAG questions."""

from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from src.config import Settings
from src.errors import RAGError
from src.utils.urls import parse_url_input

from .errors import RagApplicationError
from .models import RagAnswer, RagRequest, SourceReference

logger = logging.getLogger(__name__)

BuildGraph = Callable[[Settings, bool], Any]
BuildLightweightGraph = Callable[[Settings], Any]
DiscoverUrls = Callable[[str, Settings], list[str]]
SettingsForDiscoveredUrls = Callable[[Settings, list[str]], Settings]
RunRagQuery = Callable[..., dict[str, Any]]


@dataclass(frozen=True, slots=True)
class RagServiceDependencies:
    """Infrastructure seams used by :class:`RagApplicationService`."""

    build_graph: BuildGraph
    build_lightweight_graph: BuildLightweightGraph
    discover_urls: DiscoverUrls
    settings_for_discovered_urls: SettingsForDiscoveredUrls
    run_query: RunRagQuery


@dataclass(frozen=True, slots=True)
class RagGraphState:
    """Callbacks for the optional process-global QA graph."""

    current_graph: Callable[[], Any]
    current_settings: Callable[[], Settings]
    clear: Callable[[], None]
    promote: Callable[[Any, Settings], None]


class RagApplicationService:
    """Execute the existing non-streaming QA flow without a transport import."""

    def __init__(
        self,
        *,
        settings: Settings,
        graph: Any,
        rebuild_lock: asyncio.Lock,
        graph_state: RagGraphState,
        dependencies: RagServiceDependencies,
    ) -> None:
        self._settings = settings
        self._graph = graph
        self._rebuild_lock = rebuild_lock
        self._graph_state = graph_state
        self._dependencies = dependencies

    async def ask(self, request: RagRequest) -> RagAnswer:
        """Resolve sources, select/build a graph, and shape a stable answer."""

        try:
            return await self._ask(request)
        except RAGError:
            raise
        except Exception as exc:
            logger.exception("Unexpected RAG failure (request_id=%s)", request.request_id)
            raise RagApplicationError(
                "Internal server error.",
                request_id=request.request_id,
                internal_cause=exc,
            ) from exc

    async def _ask(self, request: RagRequest) -> RagAnswer:
        urls = parse_url_input(request.urls)
        discovered_from_search = False
        search_failed = False

        if urls is None and request.web_search and self._settings.web_search_enabled:
            try:
                discovered_urls = self._dependencies.discover_urls(request.question, self._settings)
                if discovered_urls:
                    urls = discovered_urls
                    discovered_from_search = True
                else:
                    search_failed = True
            except Exception as exc:
                search_failed = True
                logger.warning(
                    "Web search failed; using configured URLs (request_id=%s, error=%s)",
                    request.request_id,
                    type(exc).__name__,
                )

        source_mode, source_note = self._source_metadata(
            request=request,
            urls=urls,
            discovered_from_search=discovered_from_search,
            search_failed=search_failed,
        )
        urls_changed = urls is not None and urls != self._settings.source_urls
        needs_new_graph = request.rebuild or urls_changed or discovered_from_search
        effective_rebuild = request.rebuild or urls_changed or discovered_from_search
        if urls_changed and not request.rebuild:
            logger.info("URLs changed; enabling rebuild to ingest them")

        logger.info("Processing RAG query (request_id=%s)", request.request_id)
        if urls:
            logger.info("Using %d custom URLs (request_id=%s)", len(urls), request.request_id)

        graph_to_use = self._graph
        settings_to_use = self._settings
        use_lightweight = discovered_from_search and self._settings.web_search_lightweight
        if use_lightweight and urls is not None:
            settings_to_use = replace(self._settings, source_urls=urls)
            graph_to_use = self._dependencies.build_lightweight_graph(settings_to_use)
        elif needs_new_graph:
            settings_to_use = self._settings_for_urls(urls, discovered_from_search)
            if effective_rebuild:
                graph_to_use, settings_to_use = await self._rebuild_graph(
                    settings_to_use,
                    discovered_from_search=discovered_from_search,
                )
            else:
                graph_to_use = self._dependencies.build_graph(settings_to_use, False)

            if effective_rebuild and not discovered_from_search:
                self._graph_state.promote(graph_to_use, settings_to_use)

        result = self._dependencies.run_query(
            question=request.question,
            urls=urls,
            settings=settings_to_use,
            rebuild_vectorstore=False,
            graph=graph_to_use,
            verbose=request.debug,
        )
        source_urls = list(settings_to_use.source_urls)
        sources = tuple(
            SourceReference(url=url, citation_id=f"source-{index}")
            for index, url in enumerate(source_urls, start=1)
        )
        if result["error"]:
            logger.error("RAG query failed (request_id=%s)", request.request_id)
            return RagAnswer(
                answer=None,
                error=str(result["error"]),
                success=False,
                messages=None,
                source_urls=source_urls,
                source_mode=source_mode,
                source_note=source_note,
                request_id=request.request_id,
                sources=sources,
            )

        messages: list[str] | None = None
        if request.debug:
            raw_messages = result.get("messages", []) or []
            messages = [getattr(message, "content", str(message)) for message in raw_messages]
        answer = result["answer"]
        if search_failed:
            answer = (
                f"Web search failed, so I used the configured default URLs instead.\n\n{answer}"
            )
        return RagAnswer(
            answer=answer,
            error=None,
            success=True,
            messages=messages,
            source_urls=source_urls,
            source_mode=source_mode,
            source_note=source_note,
            request_id=request.request_id,
            sources=sources,
        )

    def _settings_for_urls(self, urls: list[str] | None, discovered_from_search: bool) -> Settings:
        if discovered_from_search and urls is not None:
            return self._dependencies.settings_for_discovered_urls(self._settings, urls)
        return replace(self._settings, source_urls=urls) if urls is not None else self._settings

    async def _rebuild_graph(
        self,
        settings: Settings,
        *,
        discovered_from_search: bool,
    ) -> tuple[Any, Settings]:
        async with self._rebuild_lock:
            replacing_global_graph = not discovered_from_search
            previous_graph = self._graph_state.current_graph()
            previous_settings = self._graph_state.current_settings()
            graph_to_use: Any = None
            try:
                if replacing_global_graph:
                    self._graph_state.clear()
                    gc.collect()
                graph_to_use = self._dependencies.build_graph(settings, True)
                return graph_to_use, settings
            except Exception:
                if replacing_global_graph:
                    self._graph_state.promote(previous_graph, previous_settings)
                raise
            finally:
                gc.collect()

    def _source_metadata(
        self,
        *,
        request: RagRequest,
        urls: list[str] | None,
        discovered_from_search: bool,
        search_failed: bool,
    ) -> tuple[str, str | None]:
        if urls is not None and not discovered_from_search:
            return "explicit", None
        if discovered_from_search:
            return "web_search", None
        if request.web_search and self._settings.web_search_enabled:
            return "defaults", "web_search_failed" if search_failed else "web_search_no_results"
        if request.web_search:
            return "defaults", "web_search_disabled"
        return "defaults", None
