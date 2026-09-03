from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings


class HuggingFaceEmbeddingModel(Embeddings):
    """Small adapter that gives HuggingFace embeddings the project protocol shape."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings

        self.model_name = model_name
        self._backend = HuggingFaceEmbeddings(model_name=model_name, **kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._backend.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._backend.embed_query(text)


def build_huggingface_embeddings(model_name: str, **kwargs: Any) -> HuggingFaceEmbeddingModel:
    return HuggingFaceEmbeddingModel(model_name, **kwargs)
