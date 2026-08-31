"""FastAPI application for the only Subcribers chat agent.

Endpoints
---------

* ``POST /chat``  — start a new chat session. Optionally accepts
  ``urls`` (explicit sources) or triggers web search.
  Returns a ``thread_id`` the client uses for subsequent turns.
* ``POST /chat/{thread_id}/message`` — send the next user turn,
  receive the assistant's reply.
* ``GET  /chat/{thread_id}/history`` — read the full transcript so
  the UI can repaint after a reload.
* ``DELETE /chat/{thread_id}`` — drop a session from memory.
* ``GET /health``  — liveness probe.
* ``GET /``        — serves the chat UI.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, TypeAlias

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage

from ..api.amap_proxy import (
    AMapProxyError,
    build_amap_client_config,
    fetch_amap_proxy_response,
)
from ..api.auth import require_api_key
from ..api.dependencies import (
    chat_readiness_response,
    configure_cors,
    get_chat_graph_factory_lock,
    get_config,
    get_metrics,
    get_session_registry,
    initialize_chat_app_state,
)
from ..api.errors import register_error_handlers
from ..api.models import (
    HistoryResponse,
    HistoryTurn,
    MessageRequest,
    MessageResponse,
    StartChatRequest,
    StartChatResponse,
    UploadedFile,
    UploadResponse,
)
from ..api.streaming import format_sse
from ..application import (
    ChatApplicationService,
    SessionLifecycleDependencies,
    SessionLifecycleService,
    StartSessionRequest,
    TurnExecutionDependencies,
    TurnExecutionService,
    TurnRequest,
    serialize_history,
)
from ..config import Settings, load_cors_allow_origins, load_settings
from ..core.web_search import discover_urls_from_web
from ..errors import ResourceNotFoundError
from ..graph.builder import build_lightweight_graph
from ..graph.metrics import MetricsCollector, MetricsSnapshot
from ..graph.nodes.condense import condense_followup_question
from ..memory.recall import build_turn_messages
from ..memory.store import get_memory_store
from ..sessions import (
    ChatSession,
    ChatSessionRegistry,
    SQLiteMemorySaver,
    SQLiteStorage,
    _settings_for_session,
)
from .graph import build_chat_graph
from .memory_hooks import (
    after_turn,
    build_extraction_runtime,
    on_session_start,
)
from .uploads import (
    ALLOWED_UPLOAD_SUFFIXES,
    UploadError,
    build_upload_context_note,
    list_session_uploads,
    sanitize_filename,
    save_upload_stream,
    session_upload_dir,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Content types for session-file downloads. Anything absent falls back to
# application/octet-stream so an unexpected type is never served as inline
# HTML/script by a browser.
DOWNLOAD_MEDIA_TYPES = {
    ".csv": "text/csv",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".log": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}

SettingsDep: TypeAlias = Annotated[Settings, Depends(get_config)]
SessionRegistryDep: TypeAlias = Annotated[ChatSessionRegistry, Depends(get_session_registry)]
GraphFactoryLockDep: TypeAlias = Annotated[asyncio.Lock, Depends(get_chat_graph_factory_lock)]
MetricsDep: TypeAlias = Annotated[MetricsCollector, Depends(get_metrics)]


# ---- helpers --------------------------------------------------------------


def _serialize_messages(messages: Iterable[Any]) -> list[HistoryTurn]:
    """Compatibility wrapper around the application history serializer."""

    return [
        HistoryTurn(
            role=turn.role,
            content=turn.content,
            artifacts=list(turn.artifacts),
        )
        for turn in serialize_history(messages)
    ]


def _session_database_paths(settings: Settings) -> tuple[Path, Path]:
    base_dir = Path(settings.chroma_dir) / "chat"
    return base_dir / "sessions.sqlite3", base_dir / "checkpoints.sqlite3"


def _session_edit_root(settings: Settings, thread_id: str) -> Path | None:
    """Return the per-session upload dir used to confine document editing.

    Returns ``None`` when there is no thread to scope to, which makes the
    editing tools unavailable rather than falling back to a shared root.
    """

    if not thread_id:
        return None
    return session_upload_dir(settings, thread_id)


def _call_graph_factory(
    factory: Any,
    settings: Settings,
    **kwargs: Any,
) -> Any:
    """Call a graph factory with only the keywords its signature accepts.

    The chat API is intentionally compatible with injected legacy factories
    used by downstream callers and tests. Signature filtering preserves that
    seam without catching a ``TypeError`` raised from inside the factory.
    """

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(settings, **kwargs)

    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    accepted_names = set(parameters)
    supported = (
        kwargs
        if accepts_kwargs
        else {name: value for name, value in kwargs.items() if name in accepted_names}
    )
    settings_parameter = parameters.get("settings")
    if settings_parameter is not None and settings_parameter.kind is inspect.Parameter.KEYWORD_ONLY:
        return factory(settings=settings, **supported)
    return factory(settings, **supported)


def _build_chat_graph_for_session(
    settings: Settings,
    rebuild_vectorstore: bool,
    checkpointer: Any,
    thread_id: str = "",
) -> Any:
    session_root = _session_edit_root(settings, thread_id)
    return _call_graph_factory(
        build_chat_graph,
        settings,
        rebuild_vectorstore=rebuild_vectorstore,
        checkpointer=checkpointer,
        session_root=session_root,
        thread_id=thread_id,
    )


def _build_lightweight_chat_graph_for_session(
    settings: Settings,
    checkpointer: Any,
    thread_id: str = "",
) -> Any:
    """Build the direct web-search chat graph for discovered one-shot URLs."""

    return _call_graph_factory(
        build_lightweight_graph,
        settings,
        mode="chat",
        checkpointer=checkpointer,
        session_root=_session_edit_root(settings, thread_id),
        thread_id=thread_id,
    )


def _restore_persisted_sessions(
    registry: ChatSessionRegistry,
    storage: SQLiteStorage,
    base_settings: Settings,
    checkpointer: Any,
) -> int:
    restored = 0
    for metadata in storage.list_metadata():
        urls = list(metadata.source_urls) or list(base_settings.source_urls)
        if metadata.source_mode == "web_search" and base_settings.web_search_lightweight:
            # Lightweight chat owns discovery inside the graph. Persisted URLs
            # remain session metadata for the UI, but must not become the next
            # turn's implicit source set.
            settings = replace(base_settings, source_urls=[])
            graph = _build_lightweight_chat_graph_for_session(
                settings, checkpointer, metadata.thread_id
            )
        else:
            graph_settings = (
                replace(base_settings, web_search_enabled=False)
                if metadata.source_mode == "web_search"
                else base_settings
            )
            settings = _settings_for_session(
                graph_settings,
                urls,
                metadata.thread_id,
                metadata.isolated_chroma,
            )
            graph = _build_chat_graph_for_session(
                settings,
                rebuild_vectorstore=False,
                checkpointer=checkpointer,
                thread_id=metadata.thread_id,
            )
        registry.restore(graph=graph, settings=settings, metadata=metadata)
        restored += 1
    return restored


def _graph_inputs_for_turn(
    session: ChatSession,
    message: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Build graph inputs for one chat turn.

    The session's current ``source_urls`` are seeded into graph state so a
    freshly web-search-refreshed URL set overwrites any stale ``source_urls``
    left in the per-thread checkpoint by a previous turn. Without this, the
    ``merge`` node re-ranks the previous turn's URLs ahead of the current
    turn's freshly discovered URLs (equal hit counts, but better provider
    rank), so every later question keeps fetching the first turn's pages.

    When the session has newly uploaded files (present on disk but not yet
    announced to the model), a ``SystemMessage`` listing their tool-ready
    paths is prepended so the LLM knows the exact path to pass to the file
    tools. Already-announced files are tracked on the session so the note is
    not repeated every turn.

    When long-term memory is enabled with automatic recall, a memory
    ``SystemMessage`` is placed ahead of the upload note. Both are stripped
    from the transcript the history endpoint returns.
    """

    turn_messages: list[Any] = []
    if settings is not None:
        turn_messages = build_turn_messages(
            settings,
            thread_id=getattr(session, "thread_id", None),
            message=message,
            upload_note=_new_upload_context(session, settings),
        )
    else:
        turn_messages.append(HumanMessage(content=message))

    inputs: dict[str, Any] = {"messages": turn_messages}
    if (
        session.source_mode == "web_search"
        and settings is not None
        and settings.web_search_lightweight
    ):
        # A web-search turn must not inherit the previous question's pages
        # from the checkpoint. The graph will populate this list from the
        # current turn's search results.
        inputs["current_question"] = message
        inputs["source_urls"] = []
        inputs["source_mode"] = "web_search"
        inputs["sub_questions"] = []
        inputs["expanded_queries"] = []
        inputs["search_queries"] = []
        inputs["web_search_results"] = []
        inputs["web_search_result_metadata"] = []
        inputs["web_answer_attempts"] = 0
        inputs["web_answer_no_readable_content"] = False
        inputs["expansion_attempted"] = False
    elif session.source_urls:
        inputs["source_urls"] = list(session.source_urls)
        inputs["source_mode"] = session.source_mode
    return inputs


def _new_upload_context(session: ChatSession, settings: Settings) -> str | None:
    """Return an upload-context note for files not yet announced, else None."""

    if not settings.file_read_enabled:
        return None
    try:
        available = list_session_uploads(settings, session.thread_id)
    except Exception as exc:
        logger.warning("Could not list uploads for thread %s: %s", session.thread_id, exc)
        return None

    new_paths = [p for p in available if p not in session.announced_uploads]
    if not new_paths:
        return None

    session.announced_uploads.update(new_paths)
    # Announce the full current set so the model always has every path, even
    # if an earlier note scrolled out of its effective context window.
    return build_upload_context_note(
        available,
        word_edit_enabled=settings.word_edit_enabled,
        text_edit_enabled=settings.text_edit_enabled,
        markdown_edit_enabled=settings.markdown_edit_enabled,
        powerpoint_edit_enabled=settings.powerpoint_edit_enabled,
        excel_edit_enabled=settings.excel_edit_enabled,
    )


def _build_chat_application_service(
    *,
    settings: Settings,
    sessions: ChatSessionRegistry,
    graph_factory_lock: asyncio.Lock,
    checkpointer: Any,
    extraction_runtime: Any,
    metrics: MetricsCollector | None = None,
) -> ChatApplicationService:
    """Compose transport-neutral chat services from existing infrastructure."""

    def purge_session_memory(memory_settings: Settings, thread_id: str) -> None:
        get_memory_store(memory_settings).purge_session(thread_id)

    lifecycle = SessionLifecycleService(
        settings=settings,
        sessions=sessions,
        graph_factory_lock=graph_factory_lock,
        checkpointer=checkpointer,
        dependencies=SessionLifecycleDependencies(
            discover_urls=discover_urls_from_web,
            build_graph=_build_chat_graph_for_session,
            build_lightweight_graph=_build_lightweight_chat_graph_for_session,
            settings_for_session=_settings_for_session,
            condense_question=condense_followup_question,
            on_session_start=lambda thread_id: on_session_start(
                extraction_runtime,
                new_thread_id=thread_id,
            ),
            upload_directory=session_upload_dir,
            purge_memory=purge_session_memory,
        ),
    )
    turns = TurnExecutionService(
        TurnExecutionDependencies(
            build_inputs=lambda session, message: _graph_inputs_for_turn(
                session,
                message,
                settings,
            ),
            checkpointer=checkpointer,
            metrics=metrics,
            after_turn=lambda thread_id: after_turn(
                extraction_runtime,
                thread_id=thread_id,
            ),
            sync_sources=lifecycle.sync_graph_owned_sources,
        )
    )
    return ChatApplicationService(lifecycle=lifecycle, turns=turns)


# ---- app factory ----------------------------------------------------------


def create_app(
    api_host: str = "127.0.0.1",
    api_port: int = 8001,
    config_file: str | Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        extraction_runtime = None
        try:
            logger.info("Loading settings for chat app...")
            settings = (
                load_settings() if config_file is None else load_settings(config_file=config_file)
            )
            metadata_path, checkpoint_path = _session_database_paths(settings)
            storage = SQLiteStorage(metadata_path)
            registry = ChatSessionRegistry(storage=storage)
            checkpointer = SQLiteMemorySaver(checkpoint_path)
            initialize_chat_app_state(
                app,
                settings=settings,
                session_registry=registry,
            )
            app.state.chat_session_storage = storage
            app.state.chat_checkpointer = checkpointer
            restored = await asyncio.to_thread(
                _restore_persisted_sessions,
                registry,
                storage,
                settings,
                checkpointer,
            )
            extraction_runtime = build_extraction_runtime(
                settings,
                checkpointer=checkpointer,
                registry=registry,
                storage=storage,
            )
            app.state.extraction_runtime = extraction_runtime
            logger.info("Chat app ready with %d restored session(s)", restored)
        except Exception as exc:
            logger.error(f"Failed to initialize chat app: {exc}")
            raise
        try:
            yield
        finally:
            if extraction_runtime is not None:
                extraction_runtime.scheduler.shutdown()

    app = FastAPI(
        title="only Subcribers Chat API",
        description="Multi-turn chat API on top of the only Subcribers RAG engine",
        version="1.0.0",
        lifespan=lifespan,
    )
    # REFACTOR: Register typed RAG error responses for this app instance.
    register_error_handlers(app)
    cors_origins = (
        load_cors_allow_origins() if config_file is None else load_cors_allow_origins(config_file)
    )
    configure_cors(app, cors_origins)
    initialize_chat_app_state(app)

    # ---- static UI ------------------------------------------------------
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_model=None)
    async def root() -> FileResponse | dict[str, str]:
        index = static_dir / "index.html"
        if index.exists():
            return FileResponse(index, media_type="text/html")
        return {"message": "only Subcribers Chat API - POST /chat to start"}

    @app.get("/health")
    async def health() -> dict[str, object]:
        registry = getattr(app.state, "session_registry", None)
        return {
            "status": "ok",
            "settings_loaded": getattr(app.state, "settings", None) is not None,
            "sessions": len(registry) if registry is not None else 0,
        }

    @app.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        """Readiness check endpoint with only local state checks."""

        return chat_readiness_response(request)

    @app.get("/chat/config")
    async def chat_client_config(settings: SettingsDep) -> JSONResponse:
        """Return the browser-safe AMap client configuration only."""

        return JSONResponse(
            content=build_amap_client_config(settings),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/_AMapService/{proxied_path:path}", response_model=None)
    async def amap_service_proxy(
        proxied_path: str,
        request: Request,
        settings: SettingsDep,
    ) -> Response:
        """Forward one bounded request to AMap's fixed JS API service hosts."""

        try:
            result = await asyncio.to_thread(
                fetch_amap_proxy_response,
                proxied_path,
                request.query_params.multi_items(),
                security_code=settings.amap_js_security_code,
                timeout_seconds=settings.amap_api_timeout_seconds,
            )
        except AMapProxyError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": str(exc)},
                headers={"Cache-Control": "no-store"},
            )
        return Response(
            content=result.content,
            media_type=result.media_type,
            status_code=result.status_code,
            headers={"Cache-Control": "no-store"},
        )

    # ---- chat endpoints -------------------------------------------------
    @app.post(
        "/chat",
        response_model=StartChatResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def start_chat(
        request: StartChatRequest,
        settings: SettingsDep,
        sessions: SessionRegistryDep,
        graph_factory_lock: GraphFactoryLockDep,
    ) -> StartChatResponse:
        service = _build_chat_application_service(
            settings=settings,
            sessions=sessions,
            graph_factory_lock=graph_factory_lock,
            checkpointer=getattr(app.state, "chat_checkpointer", None),
            extraction_runtime=getattr(app.state, "extraction_runtime", None),
        )
        result = await service.start(
            StartSessionRequest(
                urls=request.urls,
                web_search=request.web_search,
                seed_question=request.seed_question,
            )
        )
        return StartChatResponse(
            thread_id=result.thread_id,
            source_urls=list(result.source_urls),
            source_mode=result.source_mode,
            source_note=result.source_note,
        )

    @app.post(
        "/chat/{thread_id}/message",
        response_model=MessageResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def post_message(
        thread_id: str,
        request: MessageRequest,
        fastapi_request: Request,
        sessions: SessionRegistryDep,
    ) -> MessageResponse:
        # Preserve the legacy typed 404 even when app configuration has not
        # completed initialization yet.
        if sessions.get(thread_id) is None:
            raise ResourceNotFoundError(f"Unknown chat thread {thread_id!r}")
        settings = get_config(fastapi_request)
        service = _build_chat_application_service(
            settings=settings,
            sessions=sessions,
            graph_factory_lock=get_chat_graph_factory_lock(fastapi_request),
            checkpointer=getattr(fastapi_request.app.state, "chat_checkpointer", None),
            extraction_runtime=getattr(
                fastapi_request.app.state,
                "extraction_runtime",
                None,
            ),
        )
        result = await service.complete_turn(
            TurnRequest(thread_id=thread_id, message=request.message)
        )
        return MessageResponse(
            thread_id=result.thread_id,
            answer=result.answer,
            error=result.error,
            artifacts=list(result.artifacts),
        )

    @app.post(
        "/chat/{thread_id}/message/stream",
        dependencies=[Depends(require_api_key)],
    )
    async def post_message_stream(
        thread_id: str,
        request: MessageRequest,
        fastapi_request: Request,
        sessions: SessionRegistryDep,
        metrics: MetricsDep,
        tokens: bool = True,
    ) -> StreamingResponse:
        # Resolve the session before constructing StreamingResponse so the
        # existing typed 404 remains an HTTP response rather than an SSE error.
        if sessions.get(thread_id) is None:
            raise ResourceNotFoundError(f"Unknown chat thread {thread_id!r}")
        service = _build_chat_application_service(
            settings=get_config(fastapi_request),
            sessions=sessions,
            graph_factory_lock=get_chat_graph_factory_lock(fastapi_request),
            checkpointer=getattr(fastapi_request.app.state, "chat_checkpointer", None),
            extraction_runtime=getattr(
                fastapi_request.app.state,
                "extraction_runtime",
                None,
            ),
            metrics=metrics,
        )

        async def event_iter() -> AsyncIterator[str]:
            events = service.stream_turn(
                TurnRequest(
                    thread_id=thread_id,
                    message=request.message,
                    stream_tokens=tokens,
                )
            )
            async for event in events:
                # SSE serialization belongs exclusively to this HTTP adapter.
                yield format_sse(event)

        return StreamingResponse(event_iter(), media_type="text/event-stream")

    @app.get(
        "/chat/{thread_id}/history",
        response_model=HistoryResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def get_history(
        thread_id: str,
        fastapi_request: Request,
        sessions: SessionRegistryDep,
    ) -> HistoryResponse:
        # Preserve the pre-service route contract: unknown threads are 404s
        # even if application configuration is unavailable.
        if sessions.get(thread_id) is None:
            raise ResourceNotFoundError(f"Unknown chat thread {thread_id!r}")
        service = _build_chat_application_service(
            settings=get_config(fastapi_request),
            sessions=sessions,
            graph_factory_lock=get_chat_graph_factory_lock(fastapi_request),
            checkpointer=getattr(fastapi_request.app.state, "chat_checkpointer", None),
            extraction_runtime=getattr(
                fastapi_request.app.state,
                "extraction_runtime",
                None,
            ),
        )
        result = await service.history(thread_id)
        return HistoryResponse(
            thread_id=result.thread_id,
            turns=[
                HistoryTurn(
                    role=turn.role,
                    content=turn.content,
                    artifacts=list(turn.artifacts),
                )
                for turn in result.turns
            ],
            source_urls=list(result.source_urls),
            source_mode=result.source_mode,
        )

    @app.post(
        "/chat/{thread_id}/upload",
        response_model=UploadResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def upload_files(
        thread_id: str,
        fastapi_request: Request,
        sessions: SessionRegistryDep,
        files: list[UploadFile] = File(...),  # noqa: B008 - FastAPI dependency default
    ) -> UploadResponse:
        session = sessions.get(thread_id)
        if session is None:
            raise ResourceNotFoundError(f"Unknown chat thread {thread_id!r}")

        settings = get_config(fastapi_request)
        # Keep upload work bounded so large batches do not exhaust the shared
        # worker pool. Per-name locks preserve deterministic overwrite order
        # when one multipart request contains duplicate filenames.
        semaphore = asyncio.Semaphore(min(4, max(1, len(files))))
        name_locks: dict[str, asyncio.Lock] = {}

        async def process_upload(
            upload: UploadFile,
        ) -> tuple[UploadedFile | None, str | None]:
            name = upload.filename or "upload"
            lock_key = sanitize_filename(name).casefold() or name.casefold()
            name_lock = name_locks.setdefault(lock_key, asyncio.Lock())
            try:
                async with semaphore, name_lock:
                    result = await asyncio.to_thread(
                        save_upload_stream,
                        settings=settings,
                        thread_id=thread_id,
                        filename=name,
                        stream=upload.file,
                    )
            except UploadError as exc:
                return None, f"{name}: {exc}"
            except Exception as exc:
                logger.warning("Upload failed for %s on thread %s: %s", name, thread_id, exc)
                return None, f"{name}: could not process the file."
            finally:
                await upload.close()

            uploaded = UploadedFile(
                filename=result.filename,
                relative_path=result.relative_path,
                size_bytes=result.size_bytes,
            )
            logger.info(
                "Stored upload %r (%d bytes) for thread %s at %s",
                result.filename,
                result.size_bytes,
                thread_id,
                result.relative_path,
            )
            return uploaded, None

        results = await asyncio.gather(*(process_upload(upload) for upload in files))
        saved = [uploaded for uploaded, _ in results if uploaded is not None]
        errors = [error for _, error in results if error is not None]
        return UploadResponse(thread_id=thread_id, files=saved, errors=errors)

    @app.get(
        "/chat/{thread_id}/files/{filename}",
        dependencies=[Depends(require_api_key)],
    )
    async def download_session_file(
        thread_id: str,
        filename: str,
        fastapi_request: Request,
        sessions: SessionRegistryDep,
    ) -> FileResponse:
        """Serve one file from a chat session's own upload directory.

        Access is bound to the session: an unknown thread is a 404, and the
        requested name is reduced to a basename and re-checked against the
        session directory so no path can reach another thread's files.
        """

        session = sessions.get(thread_id)
        if session is None:
            raise ResourceNotFoundError(f"Unknown chat thread {thread_id!r}")

        settings = get_config(fastapi_request)
        safe_name = sanitize_filename(filename)
        if not safe_name or safe_name != filename:
            raise ResourceNotFoundError(f"Unknown file {filename!r}")

        suffix = Path(safe_name).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            raise ResourceNotFoundError(f"Unknown file {filename!r}")

        upload_dir = session_upload_dir(settings, thread_id)
        try:
            base = upload_dir.resolve()
            target = (base / safe_name).resolve()
        except OSError:
            raise ResourceNotFoundError(f"Unknown file {filename!r}") from None

        # Defence in depth: sanitize_filename already strips directory parts,
        # so the resolved parent must be the session directory itself.
        if target.parent != base or not target.is_file():
            raise ResourceNotFoundError(f"Unknown file {filename!r}")

        return FileResponse(
            target,
            media_type=DOWNLOAD_MEDIA_TYPES.get(suffix, "application/octet-stream"),
            filename=safe_name,
        )

    @app.delete("/chat/{thread_id}", dependencies=[Depends(require_api_key)])
    async def delete_chat(
        thread_id: str,
        fastapi_request: Request,
        sessions: SessionRegistryDep,
    ) -> dict[str, str]:
        # Preserve the pre-service route contract: unknown threads are 404s
        # even if application configuration is unavailable.
        if sessions.get(thread_id) is None:
            raise ResourceNotFoundError(f"Unknown chat thread {thread_id!r}")
        service = _build_chat_application_service(
            settings=get_config(fastapi_request),
            sessions=sessions,
            graph_factory_lock=get_chat_graph_factory_lock(fastapi_request),
            checkpointer=getattr(fastapi_request.app.state, "chat_checkpointer", None),
            extraction_runtime=getattr(
                fastapi_request.app.state,
                "extraction_runtime",
                None,
            ),
        )
        result = await service.delete(thread_id)
        return {"status": result.status, "thread_id": result.thread_id}

    @app.get("/metrics")
    async def metrics(metrics: MetricsDep) -> MetricsSnapshot:
        """Return in-process graph metrics."""

        return metrics.snapshot()

    return app
