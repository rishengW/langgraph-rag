from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.embeddings import Embeddings

from src.config import Settings


@runtime_checkable
class EmbeddingModel(Protocol):
    """Provider boundary for embedding implementations."""

    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a retrieval query."""


def is_dashscope_embedding_model(model_name: str) -> bool:
    return model_name.strip().lower().startswith("text-embedding-")


def build_embeddings(settings: Settings) -> Embeddings:
    """Build the LangChain-compatible embedding model selected by settings."""

    if is_dashscope_embedding_model(settings.embedding_model):
        from .dashscope_embeddings import DashScopeEmbeddings

        return DashScopeEmbeddings(
            api_key=settings.dashscope_api_key,
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
            request_timeout=settings.dashscope_request_timeout,
            max_retries=settings.dashscope_max_retries,
            base_url=settings.dashscope_http_base_url,
        )

    from .hf_embeddings import build_huggingface_embeddings

    return build_huggingface_embeddings(settings.embedding_model)
