from __future__ import annotations

import logging
import ssl
import time
from http import HTTPStatus
from typing import Any

from langchain_core.embeddings import Embeddings
from requests.exceptions import RequestException

from .config import DEFAULT_DASHSCOPE_HTTP_BASE_URL, Settings


logger = logging.getLogger(__name__)


def is_dashscope_embedding_model(model_name: str) -> bool:
    return model_name.strip().lower().startswith("text-embedding-")


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

    def _call_with_retry(self, client, kwargs: dict[str, Any]):
        last_error: Exception | None = None
        retryable_statuses = {
            HTTPStatus.TOO_MANY_REQUESTS,
            HTTPStatus.INTERNAL_SERVER_ERROR,
            HTTPStatus.BAD_GATEWAY,
            HTTPStatus.SERVICE_UNAVAILABLE,
            HTTPStatus.GATEWAY_TIMEOUT,
        }

        for attempt in range(self.max_retries):
            try:
                response = client.call(**kwargs)
                if response.status_code == HTTPStatus.OK:
                    return response

                error = RuntimeError(
                    "DashScope embedding error "
                    f"({response.status_code}, {response.code}): {response.message}"
                )
                if response.status_code not in retryable_statuses:
                    raise error
                last_error = error
            except (OSError, ConnectionError, TimeoutError, RequestException, ssl.SSLError) as exc:
                last_error = exc

            if attempt < self.max_retries - 1:
                delay = 1.0 * (2**attempt)
                logger.warning(
                    "DashScope embedding attempt %d/%d failed; retrying in %.1fs: %s",
                    attempt + 1,
                    self.max_retries,
                    delay,
                    last_error,
                )
                time.sleep(delay)

        if last_error is not None:
            raise last_error
        raise RuntimeError("DashScope embedding call failed without an error response.")


def build_embeddings(settings: Settings) -> Embeddings:
    if is_dashscope_embedding_model(settings.embedding_model):
        return DashScopeTextEmbeddings(
            api_key=settings.dashscope_api_key,
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
            request_timeout=settings.dashscope_request_timeout,
            max_retries=settings.dashscope_max_retries,
            base_url=settings.dashscope_http_base_url,
        )

    from langchain_huggingface.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
