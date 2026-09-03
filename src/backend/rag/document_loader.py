from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import date
from typing import Any

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.web_base import _build_metadata
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.networking import ensure_user_agent
from .document_quality import DocumentQualityConfig, filter_quality_documents

ensure_user_agent()

logger = logging.getLogger(__name__)

LoaderFactory = Callable[[str, int], Any]
SplitterFactory = Callable[[int, int], Any]
LoadedDocuments = tuple[int, list[Document]]
CacheKey = tuple[str, int]


class SourceDocumentCache:
    """Small in-process cache for successfully loaded source documents."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._entries: dict[CacheKey, tuple[float, list[Document]]] = {}
        self._load_locks: dict[CacheKey, threading.Lock] = {}

    def get(
        self,
        *,
        url: str,
        page_timeout: int,
        ttl_seconds: int,
    ) -> list[Document] | None:
        """Return cached documents when the entry is present and still valid.

        Args:
            url: Source URL used for the load.
            page_timeout: Request timeout used for the load.
            ttl_seconds: Cache entry time-to-live in seconds.
        """

        if ttl_seconds <= 0:
            return None

        key = (url, page_timeout)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            loaded_at, documents = entry
            if self._clock() - loaded_at > ttl_seconds:
                self._entries.pop(key, None)
                return None

            return _copy_documents(documents)

    def set(
        self,
        *,
        url: str,
        page_timeout: int,
        documents: Sequence[Document],
        ttl_seconds: int,
    ) -> None:
        """Cache successful document loads for the configured TTL."""

        if ttl_seconds <= 0 or not documents:
            return

        key = (url, page_timeout)
        with self._lock:
            self._entries[key] = (self._clock(), _copy_documents(documents))

    def clear(self) -> None:
        """Remove all cached source documents."""

        with self._lock:
            self._entries.clear()
            self._load_locks.clear()

    def _load_lock(self, *, url: str, page_timeout: int) -> threading.Lock:
        key = (url, page_timeout)
        with self._lock:
            return self._load_locks.setdefault(key, threading.Lock())


DEFAULT_SOURCE_DOCUMENT_CACHE = SourceDocumentCache()


def _copy_document(document: Document) -> Document:
    if hasattr(document, "model_copy"):
        return document.model_copy(deep=True)
    if hasattr(document, "copy"):
        return document.copy(deep=True)
    return deepcopy(document)


def _copy_documents(documents: Sequence[Document]) -> list[Document]:
    return [_copy_document(document) for document in documents]


class DateAwareWebBaseLoader(WebBaseLoader):
    """WebBaseLoader variant that preserves publication-date metadata."""

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text and add publication metadata extracted from raw HTML."""

        for path in self.web_paths:
            soup = self._scrape(path, bs_kwargs=self.bs_kwargs)
            text = soup.get_text(**self.bs_get_text_kwargs)
            metadata = _build_metadata(soup, path)
            published_date = _extract_publication_date(str(soup), metadata)
            if published_date is not None:
                metadata["publication_date"] = published_date.isoformat()
            yield Document(page_content=text, metadata=metadata)


def _extract_publication_date(html: str, metadata: dict[str, Any]) -> date | None:
    from ..web_search.date_extractor import extract_publication_date

    return extract_publication_date(html, metadata)


class PdfAwareLoader:
    """Loader wrapper that extracts PDF sources instead of indexing raw bytes.

    A `.pdf` URL goes straight to the PDF reader. Extension-less endpoints that
    serve PDFs (``arxiv.org/pdf/1706.03762``, CMS download handlers) are only
    detectable from the response, so the HTML result is inspected for the PDF
    file header and re-read when it matches. Without this, the HTML loader
    decodes the binary body into megabytes of noise and indexes it as prose.
    """

    def __init__(self, url: str, page_timeout: int) -> None:
        self.url = url
        self.page_timeout = page_timeout

    def load(self) -> list[Document]:
        from ..web_search.pdf_loader import PdfPageLoader, is_pdf_url, looks_like_pdf_payload

        if is_pdf_url(self.url):
            return PdfPageLoader(self.url, self.page_timeout).load()

        documents = list(
            DateAwareWebBaseLoader(
                self.url,
                requests_kwargs={"timeout": self.page_timeout},
            ).load()
        )
        if any(looks_like_pdf_payload(document.page_content or "") for document in documents):
            logger.info("Re-reading %s as a PDF after detecting a PDF response body", self.url)
            return PdfPageLoader(self.url, self.page_timeout).load()
        return documents


def default_loader_factory(url: str, page_timeout: int) -> PdfAwareLoader:
    return PdfAwareLoader(url, page_timeout)


def default_splitter_factory(
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


# REFACTOR: Load source URLs concurrently while preserving input order.
def _load_url_documents(
    index: int,
    url: str,
    page_timeout: int,
    loader_factory: LoaderFactory,
    *,
    cache_ttl_seconds: int,
    document_cache: SourceDocumentCache | None,
) -> LoadedDocuments:
    if document_cache is None:
        loader = loader_factory(url, page_timeout)
        return index, list(loader.load())

    with document_cache._load_lock(url=url, page_timeout=page_timeout):
        cached_documents = document_cache.get(
            url=url,
            page_timeout=page_timeout,
            ttl_seconds=cache_ttl_seconds,
        )
        if cached_documents is not None:
            logger.debug("Using cached source documents for %s", url)
            return index, cached_documents

        loader = loader_factory(url, page_timeout)
        loaded_documents = list(loader.load())
        document_cache.set(
            url=url,
            page_timeout=page_timeout,
            documents=loaded_documents,
            ttl_seconds=cache_ttl_seconds,
        )
    return index, loaded_documents


def _load_url_documents_batch(
    source_urls: Sequence[str],
    *,
    page_timeout: int,
    max_concurrent_loads: int,
    cache_ttl_seconds: int,
    document_cache: SourceDocumentCache | None,
    loader_factory: LoaderFactory,
) -> tuple[list[list[Document]], list[str]]:
    if not source_urls:
        return [], []

    worker_count = min(len(source_urls), max(1, int(max_concurrent_loads)))
    docs_by_index: dict[int, list[Document]] = {}
    failed_by_index: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_source = {
            executor.submit(
                _load_url_documents,
                index,
                url,
                page_timeout,
                loader_factory,
                cache_ttl_seconds=cache_ttl_seconds,
                document_cache=document_cache,
            ): (index, url)
            for index, url in enumerate(source_urls)
        }
        for future in as_completed(future_to_source):
            index, url = future_to_source[future]
            try:
                loaded_index, loaded_docs = future.result()
                docs_by_index[loaded_index] = loaded_docs
            except Exception as exc:
                failed_by_index[index] = url
                logger.warning(
                    "Skipping unreachable source URL %s after %ss timeout: %s",
                    url,
                    page_timeout,
                    exc,
                )

    docs_nested = [
        docs_by_index[index] for index in range(len(source_urls)) if index in docs_by_index
    ]
    failed_urls = [
        failed_by_index[index] for index in range(len(source_urls)) if index in failed_by_index
    ]
    return docs_nested, failed_urls


def load_source_documents(
    source_urls: Sequence[str],
    *,
    page_load_timeout: int,
    max_concurrent_loads: int = 4,
    page_load_cache_ttl_seconds: int = 0,
    document_cache: SourceDocumentCache | None = None,
    loader_factory: LoaderFactory = default_loader_factory,
) -> list[Document]:
    """Load documents from URLs, skipping failed sources as the legacy path did."""

    logger.info("LOAD WEB DOCUMENTS")
    page_timeout = max(1, int(page_load_timeout))
    cache_ttl_seconds = max(0, int(page_load_cache_ttl_seconds))
    cache = None
    if cache_ttl_seconds > 0:
        cache = document_cache or DEFAULT_SOURCE_DOCUMENT_CACHE
    docs_nested, failed_urls = _load_url_documents_batch(
        source_urls,
        page_timeout=page_timeout,
        max_concurrent_loads=max_concurrent_loads,
        cache_ttl_seconds=cache_ttl_seconds,
        document_cache=cache,
        loader_factory=loader_factory,
    )

    docs = [doc for sublist in docs_nested for doc in sublist]
    if not docs:
        failed = ", ".join(failed_urls) if failed_urls else "none"
        raise RuntimeError(f"No source documents could be loaded. Failed URLs: {failed}")

    if failed_urls:
        logger.warning(
            "Indexing continued without %d unreachable URL(s): %s",
            len(failed_urls),
            ", ".join(failed_urls),
        )

    return docs


def split_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    splitter_factory: SplitterFactory = default_splitter_factory,
) -> list[Document]:
    """Split loaded documents with the project token splitter settings."""

    logger.info("SPLIT DOCUMENTS")
    text_splitter = splitter_factory(chunk_size, chunk_overlap)
    return list(text_splitter.split_documents(list(documents)))


def load_and_split_documents(
    source_urls: Sequence[str],
    *,
    page_load_timeout: int,
    max_concurrent_loads: int = 4,
    page_load_cache_ttl_seconds: int = 0,
    document_cache: SourceDocumentCache | None = None,
    chunk_size: int,
    chunk_overlap: int,
    quality_config: DocumentQualityConfig | None = None,
    embeddings: Any | None = None,
    loader_factory: LoaderFactory = default_loader_factory,
    splitter_factory: SplitterFactory = default_splitter_factory,
) -> list[Document]:
    docs = load_source_documents(
        source_urls,
        page_load_timeout=page_load_timeout,
        max_concurrent_loads=max_concurrent_loads,
        page_load_cache_ttl_seconds=page_load_cache_ttl_seconds,
        document_cache=document_cache,
        loader_factory=loader_factory,
    )
    # REFACTOR: Drop clearly poor loaded pages before splitting and embedding.
    docs = filter_quality_documents(docs, quality_config, embeddings=embeddings)
    if not docs:
        raise RuntimeError(
            "All loaded source documents were filtered out before indexing. "
            "No documents met the configured content quality thresholds."
        )
    return split_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        splitter_factory=splitter_factory,
    )
