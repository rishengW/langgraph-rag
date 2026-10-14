"""Upload-to-index orchestration shared by the upload/delete endpoints.

Keeps the endpoints thin: resolve the session's indexing scope, parse+chunk
the saved file, and push it into the session's isolated Chroma store.
Migration to an isolated store is a coroutine the endpoint awaits on the
event-loop thread (the lifecycle service internally offloads its heavy work);
parsing and indexing are synchronous and run in ``asyncio.to_thread``.
Indexing failures must never fail the file save itself.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.backend.rag.chroma_retriever import ChromaRetriever
from src.backend.rag.upload_indexer import (
    UploadIndexingError,
    parse_and_chunk_upload,
)
from src.config import Settings

logger = logging.getLogger(__name__)


def index_saved_upload(
    *,
    session: Any,
    relative_path: str,
) -> int:
    """Index one saved upload into the session's isolated Chroma store.

    Returns the chunk count. The caller must have migrated a defaults-mode
    session to an isolated store first (``lifecycle.migrate_to_isolated_store``)
    so uploads never pollute the shared corpus. Synchronous: run in a worker
    thread.
    """

    settings: Settings = session.settings
    if not settings.upload_indexing_enabled:
        raise UploadIndexingError("upload indexing is disabled.")

    root = Path(settings.file_read_root).expanduser().resolve()
    path = (root / relative_path).resolve()
    if root not in path.parents:
        raise UploadIndexingError("upload path escaped the file-read root.")

    result = parse_and_chunk_upload(
        path,
        settings=settings,
        source=relative_path,
    )

    provider = ChromaRetriever(settings)
    added = provider.index_documents(
        result.chunks,
        parent_docstore_additions=result.parent_docstore_additions,
    )
    logger.info(
        "Indexed upload %r for thread %s: %d chunk(s) (hierarchical=%s)",
        relative_path,
        session.thread_id,
        added,
        result.parent_docstore_additions is not None,
    )
    return added


def delete_indexed_upload(*, session: Any, relative_path: str) -> int:
    """Remove one upload's chunks from the session store; 0 when disabled."""

    settings = session.settings
    if not getattr(settings, "upload_indexing_enabled", False) or not session.isolated_chroma:
        return 0
    try:
        provider = ChromaRetriever(settings)
        return provider.delete_documents_by_source(relative_path)
    except Exception as exc:
        logger.warning(
            "Could not remove indexed chunks for %r on thread %s: %s",
            relative_path,
            session.thread_id,
            exc,
        )
        return 0


__all__ = [
    "delete_indexed_upload",
    "index_saved_upload",
]
