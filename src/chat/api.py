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
import shutil
from collections.abc import AsyncIterator, Iterable, Iterator
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
from ..config import Settings, load_cors_allow_origins, load_settings
from ..core.web_search import discover_urls_from_web
from ..errors import RAGError, ResourceNotFoundError, RetrieverError
from ..graph.artifacts import extract_artifacts_from_messages
from ..graph.builder import build_lightweight_graph
from ..graph.events import ErrorEvent
from ..graph.executor import GraphExecutor
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
from ..utils.urls import parse_url_input
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
    save_upload,
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


def _parse_urls(raw: str | list[str] | None) -> list[str] | None:
    return parse_url_input(raw)


def _serialize_messages(messages: Iterable[Any]) -> list[HistoryTurn]:
    """Convert LangChain messages into visible turns with attached artifacts."""

    turns: list[HistoryTurn] = []
    pending_artifacts: list[dict[str, Any]] = []
    for msg in messages:
        kind = getattr(msg, "type", None) or msg.__class__.__name__.lower()
        content = getattr(msg, "content", str(msg))
        if kind.startswith("human") or kind == "user":
            role = "user"
        elif kind.startswith("ai") or kind == "assistant":
            role = "assistant"
        elif kind in ("tool", "function"):
            role = "tool"
        elif kind.startswith("system"):
            role = "system"
        else:
            role = kind

        if role == "tool":
            pending_artifacts.extend(extract_artifacts_from_messages([msg]))
            continue
        if role == "system":
            continue
        if role == "user":
            # A new human turn is a hard boundary: never attach a stale tool
            # artifact to a later assistant response.
            pending_artifacts = []
        if role == "assistant":
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls or not (content or "").strip():
                continue

        turn_content = content if isinstance(content, str) else str(content)
        artifacts = pending_artifacts if role == "assistant" else []
        turns.append(
            HistoryTurn(
                role=role,
                content=turn_content,
                artifacts=list(artifacts),
            )
        )
        if role == "assistant":
            pending_artifacts = []
    return turns


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
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    accepted_names = set(parameters)
    supported = (
        kwargs
        if accepts_kwargs
        else {name: value for name, value in kwargs.items() if name in accepted_names}
    )
    settings_parameter = parameters.get("settings")
    if (
        settings_parameter is not None
        and settings_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ):
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
        logger.warning(
            "Could not list uploads for thread %s: %s", session.thread_id, exc
        )
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
        powerpoint_edit_enabled=settings.powerpoint_edit_enabled,
    )


def _source_refresh_allowed(session: ChatSession, settings: Settings) -> bool:
    """Return whether chat should refresh this session from web search."""

    return (
        session.source_mode != "explicit"
        and settings.web_search_enabled
        and not settings.web_search_lightweight
    )


def _sync_graph_owned_web_sources(
    *,
    session: ChatSession,
    values: Any,
    sessions: ChatSessionRegistry,
) -> ChatSession:
    """Persist graph-discovered URLs as session metadata without rebuilding."""

    if session.source_mode != "web_search" or not session.settings.web_search_lightweight:
        return session
    if not isinstance(values, dict):
        return session

    raw_urls = values.get("source_urls")
    if not isinstance(raw_urls, list):
        return session
    urls = [str(url).strip() for url in raw_urls if str(url).strip()]
    if urls == session.source_urls:
        return session

    return sessions.update_sources(
        session.thread_id,
        graph=session.graph,
        settings=session.settings,
        source_urls=urls,
        source_mode="web_search",
        isolated_chroma=False,
    ) or session


async def _condense_query_for_refresh(
    *,
    session: ChatSession,
    message: str,
    settings: Settings,
) -> str:
    """Contextualize a follow-up message against the session transcript.

    The web-search chat path runs the lightweight graph, which has no
    ``condense`` node, so a vague follow-up ("Argentina and Jordan", "group
    stage not knockout") would otherwise drive the source search with no
    conversation context and surface off-topic pages. We read the prior turns
    from the per-thread checkpoint and rewrite the message into a standalone
    question before searching. Falls back to the raw message on any error.
    """

    try:
        config = {"configurable": {"thread_id": session.thread_id}}
        snapshot = await asyncio.to_thread(session.graph.get_state, config)
        values = getattr(snapshot, "values", {}) or {}
        prior_messages = (
            values.get("messages", []) or [] if isinstance(values, dict) else []
        )
    except Exception as exc:
        logger.debug(
            "Could not read prior messages for condense on thread %s: %s",
            session.thread_id,
            exc,
        )
        return message

    if not prior_messages:
        return message

    try:
        standalone = await asyncio.to_thread(
            condense_followup_question,
            prior_messages,
            message,
            settings,
        )
    except Exception as exc:
        logger.warning(
            "Follow-up condense failed for thread %s; using raw message: %s",
            session.thread_id,
            exc,
        )
        return message

    if standalone and standalone.strip() and standalone.strip() != message.strip():
        logger.info(
            "Condensed chat follow-up for search: original=%r → standalone=%r",
            message,
            standalone,
        )
    return standalone or message


async def _refresh_session_sources_from_web(
    *,
    session: ChatSession,
    query: str,
    settings: Settings,
    sessions: ChatSessionRegistry,
    graph_factory_lock: asyncio.Lock,
    checkpointer: Any,
) -> ChatSession:
    """Refresh a chat session's retriever sources from web search for a turn."""

    if not _source_refresh_allowed(session, settings):
        return session

    search_query = await _condense_query_for_refresh(
        session=session,
        message=query,
        settings=settings,
    )

    try:
        urls = discover_urls_from_web(search_query, settings)
    except Exception as exc:
        logger.warning(
            "Web search failed during chat turn for thread %s: %s",
            session.thread_id,
            exc,
        )
        return session

    if not urls:
        logger.info(
            "Web search returned no usable URLs for chat thread %s",
            session.thread_id,
        )
        return session

    if session.source_mode == "web_search" and urls == session.source_urls:
        return session

    if settings.web_search_lightweight:
        session_settings = replace(settings, source_urls=urls)
        isolated_chroma = False
    else:
        session_settings = _settings_for_session(
            replace(settings, web_search_enabled=False),
            urls,
            session.thread_id,
            isolated=True,
        )
        isolated_chroma = True
    try:
        async with graph_factory_lock:
            if settings.web_search_lightweight:
                graph = await asyncio.to_thread(
                    _build_lightweight_chat_graph_for_session,
                    session_settings,
                    checkpointer,
                    session.thread_id,
                )
            else:
                graph = await asyncio.to_thread(
                    _build_chat_graph_for_session,
                    session_settings,
                    True,
                    checkpointer,
                    session.thread_id,
                )
    except RAGError:
        raise
    except Exception as exc:
        logger.warning(
            "Failed to rebuild chat graph from web search for thread %s: %s",
            session.thread_id,
            exc,
            exc_info=True,
        )
        return session

    return sessions.update_sources(
        session.thread_id,
        graph=graph,
        settings=session_settings,
        source_urls=urls,
        source_mode="web_search",
        isolated_chroma=isolated_chroma,
    ) or session


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
                load_settings()
                if config_file is None
                else load_settings(config_file=config_file)
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
        load_cors_allow_origins()
        if config_file is None
        else load_cors_allow_origins(config_file)
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
        urls = _parse_urls(request.urls)
        discovered_from_search = False
        search_error: str | None = None
        graph_owned_web = (
            urls is None
            and settings.web_search_enabled
            and settings.web_search_lightweight
        )

        if urls is None and settings.web_search_enabled and not graph_owned_web:
            seed = (request.seed_question or "").strip()
            if seed:
                try:
                    found = discover_urls_from_web(seed, settings)
                    if found:
                        urls = found
                        discovered_from_search = True
                    else:
                        search_error = "web search returned no usable URLs"
                except Exception as exc:
                    search_error = str(exc)
                    logger.warning(f"Web search failed during chat start: {exc}")

        if request.urls:
            source_mode = "explicit"
        elif graph_owned_web or discovered_from_search:
            source_mode = "web_search"
        else:
            source_mode = "defaults"

        # Settle on the URL list. For "defaults" we fall back to the configured
        # source URLs from ``Settings``.
        if graph_owned_web:
            urls = []
            isolated = False
        elif urls is None:
            urls = list(settings.source_urls)
            isolated = False
        else:
            isolated = True

        # Allocate a thread id, then build a graph against settings keyed on it
        # so isolated-session Chroma directories live under .chroma/chat/<id>/.
        from uuid import uuid4

        thread_id = uuid4().hex
        if graph_owned_web:
            session_settings = replace(settings, source_urls=[])
        else:
            graph_settings = (
                replace(settings, web_search_enabled=False)
                if discovered_from_search
                else settings
            )
            session_settings = _settings_for_session(
                graph_settings, urls, thread_id, isolated
            )

        checkpointer = getattr(app.state, "chat_checkpointer", None)

        # Build the graph (this triggers indexing if Chroma needs to be created).
        # Run in a thread so we don't block the event loop.
        build_failed_for_web_search = False
        try:
            async with graph_factory_lock:
                if graph_owned_web:
                    graph = await asyncio.to_thread(
                        _build_lightweight_chat_graph_for_session,
                        session_settings,
                        checkpointer,
                        thread_id,
                    )
                else:
                    graph = await asyncio.to_thread(
                        _build_chat_graph_for_session,
                        session_settings,
                        isolated,  # rebuild_vectorstore for fresh isolated stores
                        checkpointer,
                        thread_id,
                    )
        except RAGError:
            raise
        except Exception as exc:
            # Web-search-discovered URLs are unreliable (timeouts, thin or
            # junk pages). When they yield no indexable content, fall back to
            # the configured default sources instead of failing the session.
            if discovered_from_search:
                logger.warning(
                    "Failed to build chat graph from web search sources; "
                    "falling back to configured source URLs: %s",
                    exc,
                    exc_info=True,
                )
                build_failed_for_web_search = True
                urls = list(settings.source_urls)
                isolated = False
                source_mode = "defaults"
                discovered_from_search = False
                session_settings = _settings_for_session(
                    settings, urls, thread_id, isolated
                )
                try:
                    async with graph_factory_lock:
                        graph = await asyncio.to_thread(
                            _build_chat_graph_for_session,
                            session_settings,
                            isolated,
                            checkpointer,
                            thread_id,
                        )
                except RAGError:
                    raise
                except Exception as fallback_exc:
                    logger.error(
                        f"Failed to build chat graph: {fallback_exc}",
                        exc_info=True,
                    )
                    raise RetrieverError(
                        f"Failed to build chat graph: {fallback_exc}"
                    ) from fallback_exc
            else:
                logger.error(f"Failed to build chat graph: {exc}", exc_info=True)
                raise RetrieverError(f"Failed to build chat graph: {exc}") from exc

        session = sessions.create(
            graph=graph,
            settings=session_settings,
            source_urls=urls,
            source_mode=source_mode,
            thread_id=thread_id,
            isolated_chroma=isolated,
        )

        note: str | None = None
        if source_mode == "defaults" and (search_error or build_failed_for_web_search):
            note = "web_search_failed"

        try:
            on_session_start(
                getattr(app.state, "extraction_runtime", None),
                new_thread_id=session.thread_id,
            )
        except Exception:
            # The hook is deliberately best-effort; a session response must win.
            logger.debug("memory extraction session-start hook failed", exc_info=True)

        return StartChatResponse(
            thread_id=session.thread_id,
            source_urls=session.source_urls,
            source_mode=session.source_mode,
            source_note=note,
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
        session = sessions.get(thread_id)
        if session is None:
            raise ResourceNotFoundError(f"Unknown chat thread {thread_id!r}")

        # REFACTOR: Resolve app config after session lookup so typed 404 wins.
        settings = get_config(fastapi_request)
        graph_factory_lock = get_chat_graph_factory_lock(fastapi_request)
        session = await _refresh_session_sources_from_web(
            session=session,
            query=request.message,
            settings=settings,
            sessions=sessions,
            graph_factory_lock=graph_factory_lock,
            checkpointer=getattr(fastapi_request.app.state, "chat_checkpointer", None),
        )

        config = {"configurable": {"thread_id": thread_id}}
        inputs = _graph_inputs_for_turn(session, request.message, settings)

        # Snapshot how many messages exist before this turn so we can isolate
        # the messages produced *during* this turn when extracting the answer.
        try:
            prev_snapshot = await asyncio.to_thread(session.graph.get_state, config)
            prev_values = getattr(prev_snapshot, "values", {}) or {}
            prev_count = (
                len(prev_values.get("messages", []) or [])
                if isinstance(prev_values, dict)
                else 0
            )
        except Exception:
            prev_count = 0

        try:
            # Run the graph in a thread (LangGraph invocation is sync-bound).
            result = await asyncio.to_thread(session.graph.invoke, inputs, config)
        except Exception as exc:
            logger.error(
                f"Chat invocation error in thread {thread_id}: {exc}",
                exc_info=True,
            )
            return MessageResponse(
                thread_id=thread_id,
                answer="",
                error=str(exc),
                artifacts=[],
            )

        session = _sync_graph_owned_web_sources(
            session=session,
            values=result,
            sessions=sessions,
        )

        messages = result.get("messages", []) if isinstance(result, dict) else []
        # Only consider messages added during this turn. This prevents echoing
        # the user's own question (the rewrite node may append an AIMessage
        # carrying the question text as a fallback) or a prior turn's reply.
        new_messages = messages[prev_count:] if prev_count <= len(messages) else messages
        artifacts = extract_artifacts_from_messages(new_messages)

        # The assistant's reply is the last AI message with non-empty content
        # that is NOT a tool-call carrier.
        answer = ""
        for msg in reversed(new_messages):
            kind = getattr(msg, "type", None) or msg.__class__.__name__.lower()
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                # AI message that only selects a tool; not a user-facing answer.
                continue
            if (kind.startswith("ai") or kind == "assistant") and (content or "").strip():
                answer = content if isinstance(content, str) else str(content)
                break

        if not answer:
            response = MessageResponse(
                thread_id=thread_id,
                answer="",
                error="No assistant reply was produced.",
                artifacts=artifacts,
            )
            try:
                after_turn(
                    getattr(fastapi_request.app.state, "extraction_runtime", None),
                    thread_id=thread_id,
                )
            except Exception:
                logger.debug("memory extraction after-turn hook failed", exc_info=True)
            return response

        response = MessageResponse(
            thread_id=thread_id,
            answer=answer,
            artifacts=artifacts,
        )
        try:
            after_turn(
                getattr(fastapi_request.app.state, "extraction_runtime", None),
                thread_id=thread_id,
            )
        except Exception:
            logger.debug("memory extraction after-turn hook failed", exc_info=True)
        return response

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
        session = sessions.get(thread_id)
        if session is None:
            raise ResourceNotFoundError(f"Unknown chat thread {thread_id!r}")

        # REFACTOR: Resolve app config after session lookup so typed 404 wins.
        settings = get_config(fastapi_request)
        graph_factory_lock = get_chat_graph_factory_lock(fastapi_request)
        session = await _refresh_session_sources_from_web(
            session=session,
            query=request.message,
            settings=settings,
            sessions=sessions,
            graph_factory_lock=graph_factory_lock,
            checkpointer=getattr(fastapi_request.app.state, "chat_checkpointer", None),
        )

        config = {"configurable": {"thread_id": thread_id}}
        inputs = _graph_inputs_for_turn(session, request.message, settings)

        def event_iter() -> Iterator[str]:
            executor = GraphExecutor(session.graph, metrics=metrics)
            # REFACTOR: ``tokens`` (default True) enables per-token ``TokenEvent``
            # deltas from the answer nodes in addition to node lifecycle events.
            # Pass ``?tokens=false`` to fall back to node-update-only streaming.
            completed = False
            failed = False
            try:
                for event in executor.stream(
                    inputs, config=config, stream_tokens=tokens
                ):
                    if isinstance(event, ErrorEvent):
                        failed = True
                    yield format_sse(event)
                completed = not failed
            finally:
                try:
                    snapshot = session.graph.get_state(config)
                    values = getattr(snapshot, "values", {}) or {}
                    _sync_graph_owned_web_sources(
                        session=session,
                        values=values,
                        sessions=sessions,
                    )
                except Exception as exc:
                    logger.debug(
                        "Could not synchronize graph-owned web sources for %s: %s",
                        session.thread_id,
                        exc,
                    )
                if completed:
                    try:
                        after_turn(
                            getattr(
                                fastapi_request.app.state,
                                "extraction_runtime",
                                None,
                            ),
                            thread_id=thread_id,
                        )
                    except Exception:
                        logger.debug(
                            "memory extraction after-turn hook failed",
                            exc_info=True,
                        )

        return StreamingResponse(event_iter(), media_type="text/event-stream")

    @app.get("/chat/{thread_id}/history", response_model=HistoryResponse)
    async def get_history(
        thread_id: str,
        sessions: SessionRegistryDep,
    ) -> HistoryResponse:
        session = sessions.get(thread_id)
        if session is None:
            raise ResourceNotFoundError(f"Unknown chat thread {thread_id!r}")

        config = {"configurable": {"thread_id": thread_id}}
        # Pull the latest checkpoint state from MemorySaver.
        snapshot = await asyncio.to_thread(session.graph.get_state, config)
        values = getattr(snapshot, "values", {}) or {}
        messages = values.get("messages", []) if isinstance(values, dict) else []

        return HistoryResponse(
            thread_id=thread_id,
            turns=_serialize_messages(messages),
            source_urls=session.source_urls,
            source_mode=session.source_mode,
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
        saved: list[UploadedFile] = []
        errors: list[str] = []

        for upload in files:
            name = upload.filename or "upload"
            try:
                content = await upload.read()
                result = await asyncio.to_thread(
                    save_upload,
                    settings=settings,
                    thread_id=thread_id,
                    filename=name,
                    content=content,
                )
            except UploadError as exc:
                errors.append(f"{name}: {exc}")
                continue
            except Exception as exc:
                logger.warning(
                    "Upload failed for %s on thread %s: %s", name, thread_id, exc
                )
                errors.append(f"{name}: could not process the file.")
                continue
            finally:
                await upload.close()

            saved.append(
                UploadedFile(
                    filename=result.filename,
                    relative_path=result.relative_path,
                    size_bytes=result.size_bytes,
                )
            )
            logger.info(
                "Stored upload %r (%d bytes) for thread %s at %s",
                result.filename,
                result.size_bytes,
                thread_id,
                result.relative_path,
            )

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
        deleted = sessions.delete(thread_id)
        if not deleted:
            raise ResourceNotFoundError(f"Unknown chat thread {thread_id!r}")
        # Remove any files uploaded for this thread so they are not orphaned.
        try:
            settings = get_config(fastapi_request)
            upload_dir = session_upload_dir(settings, thread_id)
            if upload_dir.exists():
                await asyncio.to_thread(shutil.rmtree, upload_dir, True)
        except Exception as exc:
            logger.warning("Failed to remove uploads for thread %s: %s", thread_id, exc)
        # Drop this session's session-scoped memories. Global memories survive.
        # Best effort: a purge failure must not fail the session deletion.
        try:
            settings = get_config(fastapi_request)
            if settings.memory_enabled:
                await asyncio.to_thread(
                    get_memory_store(settings).purge_session, thread_id
                )
        except Exception as exc:
            logger.warning(
                "Failed to purge session memory for thread %s: %s", thread_id, exc
            )
        return {"status": "deleted", "thread_id": thread_id}

    @app.get("/metrics")
    async def metrics(metrics: MetricsDep) -> MetricsSnapshot:
        """Return in-process graph metrics."""

        return metrics.snapshot()

    return app
