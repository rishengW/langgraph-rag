from __future__ import annotations

from .._compat import warn_deprecated_import
from ..graph.state import ChatState, RAGState

warn_deprecated_import("src.chat.state", "src.graph.state")

__all__ = ["ChatState", "RAGState"]
