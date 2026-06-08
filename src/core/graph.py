from __future__ import annotations

from .._compat import warn_deprecated_import
from .config import Settings
from ..graph.builder import build_graph as _build_graph

warn_deprecated_import("src.core.graph", "src.graph.builder")


def build_graph(settings: Settings, rebuild_vectorstore: bool = False):
    """Compile and return the single-shot QA workflow."""

    return _build_graph(
        mode="qa",
        settings=settings,
        rebuild_vectorstore=rebuild_vectorstore,
    )
