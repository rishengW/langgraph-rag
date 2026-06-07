from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from ..utils.networking import ensure_user_agent

ensure_user_agent()

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

LoaderFactory = Callable[[str, int], Any]
SplitterFactory = Callable[[int, int], Any]


def default_loader_factory(url: str, page_timeout: int) -> WebBaseLoader:
    return WebBaseLoader(
        url,
        requests_kwargs={"timeout": page_timeout},
    )


def default_splitter_factory(
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def load_source_documents(
    source_urls: Sequence[str],
    *,
    page_load_timeout: int,
    loader_factory: LoaderFactory = default_loader_factory,
) -> list[Document]:
    """Load documents from URLs, skipping failed sources as the legacy path did."""

    logger.info("LOAD WEB DOCUMENTS")
    docs_nested: list[list[Document]] = []
    failed_urls: list[str] = []
    page_timeout = max(1, int(page_load_timeout))

    for url in source_urls:
        try:
            loader = loader_factory(url, page_timeout)
            docs_nested.append(list(loader.load()))
        except Exception as exc:
            failed_urls.append(url)
            logger.warning(
                "Skipping unreachable source URL %s after %ss timeout: %s",
                url,
                page_timeout,
                exc,
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
    chunk_size: int,
    chunk_overlap: int,
    loader_factory: LoaderFactory = default_loader_factory,
    splitter_factory: SplitterFactory = default_splitter_factory,
) -> list[Document]:
    docs = load_source_documents(
        source_urls,
        page_load_timeout=page_load_timeout,
        loader_factory=loader_factory,
    )
    return split_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        splitter_factory=splitter_factory,
    )
