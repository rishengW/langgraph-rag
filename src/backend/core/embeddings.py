from __future__ import annotations

from src._compat import warn_deprecated_import
from ..rag.dashscope_embeddings import DashScopeEmbeddings, DashScopeTextEmbeddings
from ..rag.embeddings import EmbeddingModel, build_embeddings, is_dashscope_embedding_model
from ..rag.hf_embeddings import HuggingFaceEmbeddingModel, build_huggingface_embeddings

warn_deprecated_import("src.backend.core.embeddings", "src.backend.rag")

__all__ = [
    "DashScopeEmbeddings",
    "DashScopeTextEmbeddings",
    "EmbeddingModel",
    "HuggingFaceEmbeddingModel",
    "build_embeddings",
    "build_huggingface_embeddings",
    "is_dashscope_embedding_model",
]
