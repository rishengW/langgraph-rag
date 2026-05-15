"""FastAPI application for the RAG LangGraph agent."""

from __future__ import annotations

import logging
import gc
from dataclasses import replace
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import load_settings
from .graph import build_graph
from .graph_executor import run_rag_query
from .web_search import discover_urls_from_web, settings_for_discovered_urls

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    question: str = Field(..., min_length=1, description="The question to ask")
    urls: Optional[str] = Field(
        None,
        description="Comma-separated URLs for RAG sources. If not provided, uses defaults from .env",
    )
    rebuild: bool = Field(
        False,
        description="Whether to rebuild the vector database",
    )
    web_search: bool = Field(
        True,
        description="If true and URLs are empty, discover source URLs from web search",
    )
    debug: bool = Field(
        False,
        description="If true, include intermediate messages for debugging",
    )


class QueryResponse(BaseModel):
    """Response model for RAG queries."""

    answer: Optional[str] = Field(None, description="The generated answer")
    error: Optional[str] = Field(None, description="Error message if query failed")
    success: bool = Field(True, description="Whether the query was successful")
    messages: Optional[list[str]] = Field(None, description="Intermediate messages (debug)")
    source_urls: Optional[list[str]] = Field(None, description="URLs used for retrieval")
    source_mode: Optional[str] = Field(None, description="Source mode used (explicit, web_search, defaults)")
    source_note: Optional[str] = Field(None, description="Additional detail about source selection")


# Global state
_graph = None
_settings = None
_rebuild_lock = None


def create_app(rebuild_db: bool = False, api_host: str = "127.0.0.1", api_port: int = 8000):
    """Create and configure the FastAPI application.
    
    Args:
        rebuild_db: Whether to rebuild the vector database on startup.
        api_host: Host to bind to.
        api_port: Port to bind to.
    
    Returns:
        Configured FastAPI app instance.
    """
    
    global _graph, _settings
    
    app = FastAPI(
        title="RAG LangGraph API",
        description="API for the local RAG LangGraph agent",
        version="1.0.0",
    )
    
    # Initialize settings and graph on startup
    @app.on_event("startup")
    async def startup_event():
        global _graph, _settings, _rebuild_lock
        try:
            logger.info("Loading settings and building graph...")
            _settings = load_settings()
            _graph = build_graph(_settings, rebuild_vectorstore=rebuild_db)
            # Lazily created module-level lock to avoid import-time asyncio creation.
            import asyncio

            _rebuild_lock = asyncio.Lock()
            logger.info("Graph built successfully")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise
    
    # Serve static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Root endpoint - serve index.html
    @app.get("/")
    async def root():
        """Serve the main UI page."""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file, media_type="text/html")
        return {"message": "RAG LangGraph API - Use /query to ask questions"}
    
    # Health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "graph_ready": _graph is not None}
    
    # Main query endpoint
    @app.post("/query")
    async def query(request: QueryRequest) -> QueryResponse:
        """Execute a RAG query.
        
        Args:
            request: Query request with question and optional URLs.
        
        Returns:
            QueryResponse with answer or error message.
        """

        global _graph, _settings, _rebuild_lock
        
        if not _graph or not _settings:
            raise HTTPException(
                status_code=503,
                detail="Graph not initialized. Try again in a moment.",
            )
        
        try:
            # Parse URLs if provided
            urls: list[str] | None = None
            if request.urls and request.urls.strip():
                urls = [url.strip() for url in request.urls.split(",") if url.strip()]

            discovered_from_search = False
            search_error: str | None = None

            if urls is None and request.web_search and _settings.web_search_enabled:
                try:
                    discovered_urls = discover_urls_from_web(request.question, _settings)
                    if discovered_urls:
                        urls = discovered_urls
                        discovered_from_search = True
                except Exception as exc:
                    search_error = str(exc)
                    logger.warning("Web search failed; falling back to configured URLs: %s", exc)

            source_mode: str | None = None
            source_note: str | None = None
            if request.urls and request.urls.strip():
                source_mode = "explicit"
            elif discovered_from_search:
                source_mode = "web_search"
            else:
                source_mode = "defaults"
                if request.web_search and _settings.web_search_enabled:
                    if search_error:
                        source_note = "web_search_failed"
                    else:
                        source_note = "web_search_no_results"
                elif request.web_search and not _settings.web_search_enabled:
                    source_note = "web_search_disabled"

            # Decide whether we must rebuild/refresh the graph.
            # Important: `_graph` is built with a specific retriever (and URL set). If a
            # request asks for a rebuild, or supplies a different URL list, we must rebuild
            # the graph so the retriever tool points at the right Chroma collection.
            urls_changed = urls is not None and urls != _settings.source_urls
            needs_new_graph = request.rebuild or urls_changed or discovered_from_search

            # If the user supplied different URLs but forgot to tick rebuild, auto-upgrade.
            # Otherwise Chroma would be loaded from disk and the new URLs would never be ingested.
            effective_rebuild = request.rebuild or urls_changed or discovered_from_search
            if urls_changed and not request.rebuild:
                logger.info("URLs changed; enabling rebuild to ingest them")
            
            logger.info(f"Processing query: {request.question}")
            if urls:
                logger.info(f"Using {len(urls)} custom URLs")

            # Build a per-request graph when required; optionally promote it to the global graph
            # after a rebuild so subsequent requests use the refreshed database.
            graph_to_use = _graph
            settings_to_use = _settings

            if needs_new_graph:
                if discovered_from_search and urls is not None:
                    settings_to_use = settings_for_discovered_urls(_settings, urls)
                else:
                    settings_to_use = replace(_settings, source_urls=urls) if urls is not None else _settings

                if effective_rebuild:
                    logger.info(
                        "Rebuilding vector DB in %s (collection=%s)",
                        str(settings_to_use.chroma_dir),
                        settings_to_use.collection_name,
                    )

                if effective_rebuild and _rebuild_lock is not None:
                    async with _rebuild_lock:
                        # Save settings before cleanup
                        new_settings = settings_to_use
                        
                        # Clear old graph reference and force garbage collection to release file handles
                        if _graph is not None:
                            logger.debug("Clearing old graph to release file handles")
                            del graph_to_use
                        gc.collect()
                        
                        # Rebuild with fresh graph
                        graph_to_use = build_graph(new_settings, rebuild_vectorstore=True)
                        settings_to_use = new_settings
                else:
                    graph_to_use = build_graph(settings_to_use, rebuild_vectorstore=effective_rebuild)

                if effective_rebuild and not discovered_from_search:
                    # Promote rebuilt graph/settings globally.
                    _graph = graph_to_use
                    _settings = settings_to_use
            
            # Run the query
            result = run_rag_query(
                question=request.question,
                urls=urls,
                settings=settings_to_use,
                rebuild_vectorstore=False,
                graph=graph_to_use,
                verbose=request.debug,
            )
            
            if result["error"]:
                logger.error(f"Query error: {result['error']}")
                return QueryResponse(
                    answer=None,
                    error=result["error"],
                    success=False,
                    source_urls=settings_to_use.source_urls,
                    source_mode=source_mode,
                    source_note=source_note,
                )

            # Optionally include simplified intermediate messages for debugging
            messages: Optional[list[str]] = None
            if request.debug:
                raw_messages = result.get("messages", []) or []
                messages = [getattr(m, "content", str(m)) for m in raw_messages]

            logger.info("Query processed successfully")
            answer = result["answer"]
            if search_error:
                answer = (
                    f"Web search failed, so I used the configured default URLs instead.\n\n"
                    f"{answer}"
                )
            return QueryResponse(
                answer=answer,
                error=None,
                success=True,
                messages=messages,
                source_urls=settings_to_use.source_urls,
                source_mode=source_mode,
                source_note=source_note,
            )
        
        except Exception as e:
            logger.error(f"Unexpected error during query: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}",
            )
    
    return app
