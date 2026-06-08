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
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage

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
)
from ..api.streaming import format_sse
from ..config import Settings, load_cors_allow_origins, load_settings
from ..core.web_search import discover_urls_from_web
from ..errors import RAGError, ResourceNotFoundError, RetrieverError
from ..graph.executor import GraphExecutor
from ..graph.metrics import MetricsCollector
from ..utils.urls import parse_url_input
from .graph import build_chat_graph
from ..sessions import (
    ChatSession,
    ChatSessionRegistry,
    SQLiteMemorySaver,
    SQLiteStorage,
    _settings_for_session,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---- helpers --------------------------------------------------------------


def _parse_urls(raw: str | list[str] | None) -> list[str] | None:
    return parse_url_input(raw)


def _serialize_messages(messages) -> list[HistoryTurn]:
    """Convert LangChain message objects to the wire format."""

    turns: list[HistoryTurn] = []
    for msg in messages:
        kind = getattr(msg, "type", None) or msg.__class__.__name__.lower()
        content = getattr(msg, "content", str(msg))
        if kind.startswith("human") or kind == "user":
            role = "user"
        elif kind.startswith("ai") or kind == "assistant":
            role = "assistant"
        elif kind in ("tool", "function"):
            role = "tool"
        else:
            role = kind
        # Tool messages and empty AI messages (tool-call carriers) are
        # uninteresting to the UI; skip them so the transcript stays clean.
        if role == "tool":
            continue
        if role == "assistant" and not (content or "").strip():
            continue
        turns.append(HistoryTurn(role=role, content=content if isinstance(content, str) else str(content)))
    return turns


def _session_database_paths(settings: Settings) -> tuple[Path, Path]:
    base_dir = Path(settings.chroma_dir) / "chat"
    return base_dir / "sessions.sqlite3", base_dir / "checkpoints.sqlite3"


def _build_chat_graph_for_session(
    settings: Settings,
    rebuild_vectorstore: bool,
    checkpointer,
):
    try:
        return build_chat_graph(
            settings,
            rebuild_vectorstore=rebuild_vectorstore,
            checkpointer=checkpointer,
        )
    except TypeError:
        return build_chat_graph(settings, rebuild_vectorstore=rebuild_vectorstore)


def _restore_persisted_sessions(
    registry: ChatSessionRegistry,
    storage: SQLiteStorage,
    base_settings: Settings,
    checkpointer,
) -> int:
    restored = 0
    for metadata in storage.list_metadata():
        urls = list(metadata.source_urls) or list(base_settings.source_urls)
        settings = _settings_for_session(
            base_settings,
            urls,
            metadata.thread_id,
            metadata.isolated_chroma,
        )
        graph = _build_chat_graph_for_session(
            settings,
            rebuild_vectorstore=False,
            checkpointer=checkpointer,
        )
        registry.restore(graph=graph, settings=settings, metadata=metadata)
        restored += 1
    return restored


def _source_refresh_allowed(session: ChatSession, settings: Settings) -> bool:
    """Return whether chat should refresh this session from web search."""

    return session.source_mode != "explicit" and settings.web_search_enabled


async def _refresh_session_sources_from_web(
    *,
    session: ChatSession,
    query: str,
    settings: Settings,
    sessions: ChatSessionRegistry,
    graph_factory_lock: asyncio.Lock,
    checkpointer,
) -> ChatSession:
    """Refresh a chat session's retriever sources from web search for a turn."""

    if not _source_refresh_allowed(session, settings):
        return session

    try:
        urls = discover_urls_from_web(query, settings)
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

    session_settings = _settings_for_session(
        settings,
        urls,
        session.thread_id,
        isolated=True,
    )
    try:
        async with graph_factory_lock:
            graph = await asyncio.to_thread(
                _build_chat_graph_for_session,
                session_settings,
                True,
                checkpointer,
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
        isolated_chroma=True,
    ) or session


# ---- app factory ----------------------------------------------------------


def create_app(
    api_host: str = "127.0.0.1",
    api_port: int = 8001,
    config_file: str | Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
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
            logger.info("Chat app ready with %d restored session(s)", restored)
        except Exception as exc:
            logger.error(f"Failed to initialize chat app: {exc}")
            raise
        yield

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

    @app.get("/")
    async def root():
        index = static_dir / "index.html"
        if index.exists():
            return FileResponse(index, media_type="text/html")
        return {"message": "only Subcribers Chat API - POST /chat to start"}

    @app.get("/health")
    async def health():
        registry = getattr(app.state, "session_registry", None)
        return {
            "status": "ok",
            "settings_loaded": getattr(app.state, "settings", None) is not None,
            "sessions": len(registry) if registry is not None else 0,
        }

    @app.get("/ready")
    async def ready(request: Request):
        """Readiness check endpoint with only local state checks."""

        return chat_readiness_response(request)

    # ---- chat endpoints -------------------------------------------------
    @app.post(
        "/chat",
        response_model=StartChatResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def start_chat(
        request: StartChatRequest,
        settings: Settings = Depends(get_config),
        sessions: ChatSessionRegistry = Depends(get_session_registry),
        graph_factory_lock: asyncio.Lock = Depends(get_chat_graph_factory_lock),
    ) -> StartChatResponse:
        urls = _parse_urls(request.urls)
        discovered_from_search = False
        search_error: str | None = None

        if urls is None and settings.web_search_enabled:
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
        elif discovered_from_search:
            source_mode = "web_search"
        else:
            source_mode = "defaults"

        # Settle on the URL list. For "defaults" we fall back to the configured
        # source URLs from ``Settings``.
        if urls is None:
            urls = list(settings.source_urls)
            isolated = False
        else:
            isolated = True

        # Allocate a thread id, then build a graph against settings keyed on it
        # so isolated-session Chroma directories live under .chroma/chat/<id>/.
        from uuid import uuid4

        thread_id = uuid4().hex
        session_settings = _settings_for_session(settings, urls, thread_id, isolated)

        # Build the graph (this triggers indexing if Chroma needs to be created).
        # Run in a thread so we don't block the event loop.
        try:
            async with graph_factory_lock:
                graph = await asyncio.to_thread(
                    _build_chat_graph_for_session,
                    session_settings,
                    isolated,  # rebuild_vectorstore for fresh isolated stores
                    getattr(app.state, "chat_checkpointer", None),
                )
        except RAGError:
            raise
        except Exception as exc:
            logger.error(f"Failed to build chat graph: {exc}", exc_info=True)
            raise RetrieverError(f"Failed to build chat graph: {exc}") from exc

        session = sessions.create(
            graph=graph,
            settings=session_settings,
            source_urls=urls,
            source_mode=source_mode,
            thread_id=thread_id,
        )

        note: str | None = None
        if search_error and source_mode == "defaults":
            note = "web_search_failed"

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
        settings: Settings = Depends(get_config),
        sessions: ChatSessionRegistry = Depends(get_session_registry),
        graph_factory_lock: asyncio.Lock = Depends(get_chat_graph_factory_lock),
    ) -> MessageResponse:
        session = sessions.get(thread_id)
        if session is None:
            raise ResourceNotFoundError(f"Unknown chat thread {thread_id!r}")

        session = await _refresh_session_sources_from_web(
            session=session,
            query=request.message,
            settings=settings,
            sessions=sessions,
            graph_factory_lock=graph_factory_lock,
            checkpointer=getattr(fastapi_request.app.state, "chat_checkpointer", None),
        )

        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=request.message)]}

        # Snapshot how many messages exist before this turn so we can isolate
        # the messages produced *during* this turn when extracting the answer.
        try:
            prev_snapshot = await asyncio.to_thread(session.graph.get_state, config)
            prev_values = getattr(prev_snapshot, "values", {}) or {}
            prev_count = len(prev_values.get("messages", []) or []) if isinstance(prev_values, dict) else 0
        except Exception:
            prev_count = 0

        try:
            # Run the graph in a thread (LangGraph invocation is sync-bound).
            result = await asyncio.to_thread(session.graph.invoke, inputs, config)
        except Exception as exc:
            logger.error(f"Chat invocation error in thread {thread_id}: {exc}", exc_info=True)
            return MessageResponse(thread_id=thread_id, answer="", error=str(exc))

        messages = result.get("messages", []) if isinstance(result, dict) else []
        # Only consider messages added during this turn. This prevents echoing
        # the user's own question (the rewrite node may append an AIMessage
        # carrying the question text as a fallback) or a prior turn's reply.
        new_messages = messages[prev_count:] if prev_count <= len(messages) else messages

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
            return MessageResponse(
                thread_id=thread_id,
                answer="",
                error="No assistant reply was produced.",
            )

        return MessageResponse(thread_id=thread_id, answer=answer)

    @app.post(
        "/chat/{thread_id}/message/stream",
        dependencies=[Depends(require_api_key)],
    )
    async def post_message_stream(
        thread_id: str,
        request: MessageRequest,
        fastapi_request: Request,
        settings: Settings = Depends(get_config),
        sessions: ChatSessionRegistry = Depends(get_session_registry),
        graph_factory_lock: asyncio.Lock = Depends(get_chat_graph_factory_lock),
        metrics: MetricsCollector = Depends(get_metrics),
    ) -> StreamingResponse:
        session = sessions.get(thread_id)
        if session is None:
            raise ResourceNotFoundError(f"Unknown chat thread {thread_id!r}")

        session = await _refresh_session_sources_from_web(
            session=session,
            query=request.message,
            settings=settings,
            sessions=sessions,
            graph_factory_lock=graph_factory_lock,
            checkpointer=getattr(fastapi_request.app.state, "chat_checkpointer", None),
        )

        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=request.message)]}

        def event_iter():
            executor = GraphExecutor(session.graph, metrics=metrics)
            for event in executor.stream(inputs, config=config):
                yield format_sse(event)

        return StreamingResponse(event_iter(), media_type="text/event-stream")

    @app.get("/chat/{thread_id}/history", response_model=HistoryResponse)
    async def get_history(
        thread_id: str,
        sessions: ChatSessionRegistry = Depends(get_session_registry),
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

    @app.delete("/chat/{thread_id}", dependencies=[Depends(require_api_key)])
    async def delete_chat(
        thread_id: str,
        sessions: ChatSessionRegistry = Depends(get_session_registry),
    ):
        deleted = sessions.delete(thread_id)
        if not deleted:
            raise ResourceNotFoundError(f"Unknown chat thread {thread_id!r}")
        return {"status": "deleted", "thread_id": thread_id}

    @app.get("/metrics")
    async def metrics(metrics: MetricsCollector = Depends(get_metrics)):
        """Return in-process graph metrics."""

        return metrics.snapshot()

    return app
