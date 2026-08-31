"""FastAPI application for the only Subcribers RAG agent."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, TypeAlias

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..api.auth import require_api_key
from ..api.dependencies import (
    clear_qa_graph,
    configure_cors,
    get_config,
    get_metrics,
    get_qa_graph,
    get_rebuild_lock,
    initialize_qa_app_state,
    qa_readiness_response,
    update_qa_graph_state,
)
from ..api.errors import register_error_handlers
from ..api.models import QueryRequest, QueryResponse
from ..api.streaming import format_sse
from ..application import (
    RagApplicationError,
    RagApplicationService,
    RagGraphState,
    RagRequest,
    RagServiceDependencies,
)
from ..config import Settings, load_cors_allow_origins, load_settings
from ..core.graph import build_graph
from ..core.graph_executor import run_rag_query
from ..core.web_search import discover_urls_from_web, settings_for_discovered_urls
from ..errors import RAGError
from ..graph.builder import build_lightweight_graph
from ..graph.executor import GraphExecutor
from ..graph.metrics import MetricsCollector, MetricsSnapshot
from ..utils.urls import parse_url_input

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QaGraphDep: TypeAlias = Annotated[Any, Depends(get_qa_graph)]
SettingsDep: TypeAlias = Annotated[Settings, Depends(get_config)]
RebuildLockDep: TypeAlias = Annotated[asyncio.Lock, Depends(get_rebuild_lock)]
MetricsDep: TypeAlias = Annotated[MetricsCollector, Depends(get_metrics)]


def _parse_request_urls(raw_urls: str | list[str] | None) -> list[str] | None:
    """Normalize request URL input.

    Empty UI fields usually arrive as null or an empty string, while API
    clients may send an empty list. Treat all of those as "no URLs supplied"
    so the web-search path can run.
    """

    return parse_url_input(raw_urls)


def create_app(
    rebuild_db: bool = False,
    api_host: str = "127.0.0.1",
    api_port: int = 8000,
    config_file: str | Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        rebuild_db: Whether to rebuild the vector database on startup.
        api_host: Host to bind to.
        api_port: Port to bind to.

    Returns:
        Configured FastAPI app instance.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            logger.info("Loading settings and building graph...")
            settings = (
                load_settings() if config_file is None else load_settings(config_file=config_file)
            )
            graph = build_graph(settings, rebuild_vectorstore=rebuild_db)
            initialize_qa_app_state(app, settings=settings, graph=graph)
            logger.info("Graph built successfully")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
        try:
            yield
        finally:
            # No teardown needed today; placeholder for future cleanup
            # (e.g., releasing the Chroma system or persistent clients).
            pass

    app = FastAPI(
        title="only Subcribers API",
        description="API for the only Subcribers local RAG agent",
        version="1.0.0",
        lifespan=lifespan,
    )
    # REFACTOR: Register typed RAG error responses for this app instance.
    register_error_handlers(app)
    cors_origins = (
        load_cors_allow_origins() if config_file is None else load_cors_allow_origins(config_file)
    )
    configure_cors(app, cors_origins)
    initialize_qa_app_state(app)

    # Serve static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Root endpoint - serve index.html
    @app.get("/", response_model=None)
    async def root() -> FileResponse | dict[str, str]:
        """Serve the main UI page."""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file, media_type="text/html")
        return {"message": "only Subcribers API - Use /query to ask questions"}

    # Health check
    @app.get("/health")
    async def health_check() -> dict[str, object]:
        """Health check endpoint."""
        return {
            "status": "ok",
            "graph_ready": getattr(app.state, "qa_graph", None) is not None,
        }

    @app.get("/ready")
    async def ready_check(request: Request) -> JSONResponse:
        """Readiness check endpoint with only local state checks."""

        return qa_readiness_response(request)

    # Main query endpoint
    @app.post("/query", dependencies=[Depends(require_api_key)])
    async def query(
        request: QueryRequest,
        fastapi_request: Request,
        graph: QaGraphDep,
        settings: SettingsDep,
        rebuild_lock: RebuildLockDep,
    ) -> QueryResponse:
        """Execute a RAG query.

        Args:
            request: Query request with question and optional URLs.

        Returns:
            QueryResponse with answer or error message.
        """

        service = RagApplicationService(
            settings=settings,
            graph=graph,
            rebuild_lock=rebuild_lock,
            graph_state=RagGraphState(
                current_graph=lambda: getattr(fastapi_request.app.state, "qa_graph", None),
                current_settings=lambda: getattr(fastapi_request.app.state, "settings", settings),
                clear=lambda: clear_qa_graph(fastapi_request.app),
                promote=lambda new_graph, new_settings: update_qa_graph_state(
                    fastapi_request.app,
                    graph=new_graph,
                    settings=new_settings,
                ),
            ),
            dependencies=RagServiceDependencies(
                build_graph=lambda graph_settings, rebuild: build_graph(
                    graph_settings, rebuild_vectorstore=rebuild
                ),
                build_lightweight_graph=build_lightweight_graph,
                discover_urls=discover_urls_from_web,
                settings_for_discovered_urls=settings_for_discovered_urls,
                run_query=run_rag_query,
            ),
        )
        try:
            answer = await service.ask(
                RagRequest(
                    question=request.question,
                    urls=request.urls,
                    rebuild=request.rebuild,
                    web_search=request.web_search,
                    debug=request.debug,
                )
            )
        except RagApplicationError as exc:
            raise RAGError(exc.public_detail) from exc.internal_cause

        return QueryResponse(
            answer=answer.answer,
            error=answer.error,
            success=answer.success,
            messages=answer.messages,
            source_urls=answer.source_urls,
            source_mode=answer.source_mode,
            source_note=answer.source_note,
        )

    @app.post("/query/stream", dependencies=[Depends(require_api_key)])
    async def query_stream(
        request: QueryRequest,
        graph: QaGraphDep,
        settings: SettingsDep,
        metrics: MetricsDep,
    ) -> StreamingResponse:
        """Stream typed graph events as Server-Sent Events."""

        urls = _parse_request_urls(request.urls)
        graph_to_use = graph
        settings_to_use = settings
        rebuild = request.rebuild

        discovered_from_search = False
        if urls is None and request.web_search and settings.web_search_enabled:
            try:
                discovered_urls = discover_urls_from_web(request.question, settings)
            except Exception as exc:
                discovered_urls = []
                logger.warning("Web search failed during stream: %s", exc)
            if discovered_urls:
                urls = discovered_urls
                discovered_from_search = True
                if settings.web_search_lightweight:
                    settings_to_use = replace(settings, source_urls=urls)
                    graph_to_use = build_lightweight_graph(settings_to_use)
                    rebuild = False
                else:
                    settings_to_use = settings_for_discovered_urls(settings, urls)
                    rebuild = True

        if (
            urls is not None
            and urls != settings.source_urls
            and not (discovered_from_search and settings.web_search_lightweight)
        ):
            settings_to_use = replace(settings, source_urls=urls)
            rebuild = True

        if rebuild:
            graph_to_use = await asyncio.to_thread(
                build_graph,
                settings_to_use,
                rebuild,
            )

        def event_iter() -> Iterator[str]:
            executor = GraphExecutor(graph_to_use, metrics=metrics)
            inputs = {"messages": [("user", request.question)]}
            for event in executor.stream(inputs):
                yield format_sse(event)

        return StreamingResponse(event_iter(), media_type="text/event-stream")

    @app.get("/metrics")
    async def metrics(metrics: MetricsDep) -> MetricsSnapshot:
        """Return in-process graph metrics."""

        return metrics.snapshot()

    return app
