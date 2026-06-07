"""Chat-compatible LangGraph workflow exports."""

from __future__ import annotations

from ..core.config import Settings
from ..graph.builder import build_graph as _build_graph
from ..graph.builder import build_memory_saver as _build_memory_saver


def build_chat_graph(settings: Settings, rebuild_vectorstore: bool = False):
    """Compile and return the chat workflow.

    The compiled graph is bound to the supplied ``settings.source_urls``
    via the retriever tool. Each chat session that has a different source
    set must call ``build_chat_graph`` again with the appropriate
    settings."""

    return _build_graph(
        mode="chat",
        settings=settings,
        rebuild_vectorstore=rebuild_vectorstore,
    )


__all__ = ["_build_memory_saver", "build_chat_graph"]
