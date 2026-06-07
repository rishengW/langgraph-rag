from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from ..utils.networking import DEFAULT_DASHSCOPE_HTTP_BASE_URL
from ..utils.retry import dashscope_call_with_retry


class DashScopeTextEmbeddings(Embeddings):
    """LangChain-compatible wrapper around DashScope text embeddings."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimension: int | None,
        batch_size: int,
        request_timeout: int,
        max_retries: int,
        base_url: str = "",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.model_name = model
        self.dimension = dimension
        self.batch_size = max(1, batch_size)
        self.request_timeout = request_timeout
        self.max_retries = max(1, max_retries)
        self.base_url = base_url

        import dashscope

        dashscope.base_http_api_url = self.base_url or DEFAULT_DASHSCOPE_HTTP_BASE_URL

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings.extend(self._embed_batch(batch, text_type="document"))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text], text_type="query")[0]

    def _embed_batch(self, texts: list[str], *, text_type: str) -> list[list[float]]:
        if not texts:
            return []

        from dashscope import TextEmbedding

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": [text if text.strip() else " " for text in texts],
            "api_key": self.api_key,
            "text_type": text_type,
            "output_type": "dense",
            "request_timeout": self.request_timeout,
        }
        if self.dimension is not None:
            kwargs["dimension"] = self.dimension

        response = self._call_with_retry(TextEmbedding, kwargs)
        records = (response.output or {}).get("embeddings", [])
        ordered: list[list[float] | None] = [None] * len(texts)

        for fallback_index, record in enumerate(records):
            text_index = int(record.get("text_index", fallback_index))
            if text_index < 0 or text_index >= len(texts):
                raise RuntimeError(
                    f"DashScope returned embedding index {text_index}, "
                    f"but batch size is {len(texts)}."
                )
            ordered[text_index] = [float(value) for value in record["embedding"]]

        missing = [index for index, value in enumerate(ordered) if value is None]
        if missing:
            raise RuntimeError(f"DashScope did not return embeddings for indices: {missing}")

        return [embedding for embedding in ordered if embedding is not None]

    def _call_with_retry(self, client: Any, kwargs: dict[str, Any]) -> Any:
        return dashscope_call_with_retry(
            client,
            kwargs,
            max_retries=self.max_retries,
        )


DashScopeEmbeddings = DashScopeTextEmbeddings
