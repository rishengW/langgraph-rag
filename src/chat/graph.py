"""Chat-compatible LangGraph workflow exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._compat import warn_deprecated_import
from ..config import Settings
from ..graph.builder import build_graph as _build_graph
from ..graph.builder import build_memory_saver as _build_memory_saver

warn_deprecated_import("src.chat.graph", "src.graph.builder")


def build_chat_graph(
    settings: Settings,
    rebuild_vectorstore: bool = False,
    checkpointer: Any = None,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> Any:
    """Compile and return the chat workflow.

    The compiled graph is bound to the supplied ``settings.source_urls``
    via the retriever tool. Each chat session that has a different source
    set must call ``build_chat_graph`` again with the appropriate
    settings.

    ``session_root`` and ``thread_id`` confine session-bound document
    editing tools to one chat session's upload directory. Omitting them
    builds a graph without those tools."""

    build_kwargs: dict[str, Any] = {
        "settings": settings,
        "rebuild_vectorstore": rebuild_vectorstore,
    }
    if checkpointer is not None:
        build_kwargs["checkpointer"] = checkpointer
    if session_root is not None:
        build_kwargs["session_root"] = session_root
    if thread_id:
        build_kwargs["thread_id"] = thread_id
    return _build_graph(**build_kwargs)


__all__ = ["_build_memory_saver", "build_chat_graph"]
