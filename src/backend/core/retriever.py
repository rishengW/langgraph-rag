from __future__ import annotations

from src._compat import warn_deprecated_import
from ..rag.chroma_retriever import (
    EMBEDDING_CONFIG_FILENAME,
    ChromaRetriever,
    _embedding_config,
    _embedding_config_matches,
    _embedding_config_path,
    _paths_match,
    _persisted_chroma_exists,
    _release_chroma_system,
    _rmtree_with_retry,
    _write_embedding_config,
    build_retriever,
    build_retriever_tool,
)
from ..rag.retriever import Retriever

warn_deprecated_import("src.backend.core.retriever", "src.backend.rag")

__all__ = [
    "EMBEDDING_CONFIG_FILENAME",
    "ChromaRetriever",
    "Retriever",
    "_embedding_config",
    "_embedding_config_matches",
    "_embedding_config_path",
    "_paths_match",
    "_persisted_chroma_exists",
    "_release_chroma_system",
    "_rmtree_with_retry",
    "_write_embedding_config",
    "build_retriever",
    "build_retriever_tool",
]
