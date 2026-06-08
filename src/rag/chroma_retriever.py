from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool
from langchain_core.tools.retriever import create_retriever_tool

from ..config import Settings
from ..utils.retry import call_with_retry, remove_tree_with_retry
from .document_loader import load_and_split_documents
from .embeddings import build_embeddings
from .retriever import Retriever

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


logger = logging.getLogger(__name__)

EMBEDDING_CONFIG_FILENAME = "embedding_config.json"
RESERVED_CHROMA_CHILD_DIRS = {"chat", "web-search"}


def _persisted_chroma_exists(chroma_dir: Path) -> bool:
    if not chroma_dir.exists():
        return False
    if (chroma_dir / "chroma.sqlite3").exists():
        return True
    return any(
        child.name not in RESERVED_CHROMA_CHILD_DIRS
        and child.name != EMBEDDING_CONFIG_FILENAME
        for child in chroma_dir.iterdir()
    )


def _embedding_config_path(chroma_dir: Path) -> Path:
    return chroma_dir / EMBEDDING_CONFIG_FILENAME


def _embedding_config(settings: Settings) -> dict[str, int | str | None]:
    return {
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
    }


def _embedding_config_matches(settings: Settings) -> bool:
    path = _embedding_config_path(settings.chroma_dir)
    if not path.exists():
        logger.info(
            "Existing Chroma store has no embedding metadata; rebuilding for %s",
            settings.embedding_model,
        )
        return False

    try:
        stored_config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.info("Could not read Chroma embedding metadata; rebuilding: %s", exc)
        return False

    expected_config = _embedding_config(settings)
    if stored_config != expected_config:
        logger.info(
            "Embedding config changed from %s to %s; rebuilding Chroma",
            stored_config,
            expected_config,
        )
        return False

    return True


def _write_embedding_config(settings: Settings) -> None:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    path = _embedding_config_path(settings.chroma_dir)
    path.write_text(
        json.dumps(_embedding_config(settings), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _paths_match(left: str | os.PathLike | None, right: Path) -> bool:
    if not left:
        return False

    try:
        return Path(left).resolve() == right.resolve()
    except (OSError, RuntimeError, ValueError):
        return str(left) == str(right)


def _release_chroma_system(chroma_dir: Path) -> None:
    """Stop Chroma's shared persistent client for a directory before deletion."""

    try:
        from chromadb.api.shared_system_client import SharedSystemClient
    except Exception as exc:
        logger.debug("Could not import Chroma shared-system client: %s", exc)
        return

    systems = getattr(SharedSystemClient, "_identifier_to_system", None)
    if not isinstance(systems, dict):
        logger.debug("Chroma shared-system cache is unavailable; skipping targeted release")
        return

    lock = getattr(SharedSystemClient, "_refcount_lock", None)
    refcounts = getattr(SharedSystemClient, "_identifier_to_refcount", None)
    context = lock if lock is not None else nullcontext()

    systems_to_stop = []
    with context:
        for identifier, system in list(systems.items()):
            persist_directory = getattr(system.settings, "persist_directory", None)
            if _paths_match(identifier, chroma_dir) or _paths_match(persist_directory, chroma_dir):
                systems_to_stop.append((identifier, system))
                systems.pop(identifier, None)
                if isinstance(refcounts, dict):
                    refcounts.pop(identifier, None)

    for identifier, system in systems_to_stop:
        try:
            system.stop()
            logger.info("Released Chroma system for %s before rebuild", identifier)
        except Exception as exc:
            logger.warning("Failed to stop Chroma system for %s: %s", identifier, exc)


def _rmtree_with_retry(path: Path, max_retries: int = 10, delay: float = 1.0) -> None:
    """Remove a directory tree with retries for Windows file locking."""

    remove_tree_with_retry(
        path,
        max_retries=max_retries,
        delay=delay,
        operation_logger=logger,
    )


def _unlink_with_retry(path: Path, max_retries: int = 10, delay: float = 1.0) -> None:
    """Remove a file with retries for Windows file locking."""

    call_with_retry(
        lambda: path.unlink(missing_ok=True),
        max_retries=max_retries,
        base_delay=delay,
        retryable=lambda exc: isinstance(exc, (PermissionError, OSError)),
        log_label=f"remove {path}",
    )


def _clear_chroma_store(chroma_dir: Path) -> None:
    """Clear Chroma files while preserving nested chat/web-search stores."""

    if not chroma_dir.exists():
        return

    for child in list(chroma_dir.iterdir()):
        if child.name in RESERVED_CHROMA_CHILD_DIRS:
            continue
        if child.is_dir():
            _rmtree_with_retry(child)
        else:
            _unlink_with_retry(child)


class ChromaRetriever:
    """Retriever provider that owns the Chroma vectorstore lifecycle."""

    def __init__(
        self,
        settings: Settings,
        *,
        rebuild: bool = False,
        embeddings: Embeddings | None = None,
        chroma_cls: Any | None = None,
    ) -> None:
        self.settings = settings
        self._embeddings = embeddings
        self._chroma_cls = chroma_cls or Chroma
        self._retriever = self._build_langchain_retriever(rebuild=rebuild)

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        retriever = self._retriever
        original_search_kwargs: dict[str, Any] | None = None

        if hasattr(retriever, "search_kwargs"):
            try:
                original_search_kwargs = dict(getattr(retriever, "search_kwargs") or {})
                retriever.search_kwargs = {**original_search_kwargs, "k": k}
            except Exception:
                original_search_kwargs = None

        try:
            if hasattr(retriever, "invoke"):
                docs = retriever.invoke(query)
            elif hasattr(retriever, "get_relevant_documents"):
                docs = retriever.get_relevant_documents(query)
            else:
                raise TypeError("Wrapped Chroma retriever does not expose a retrieval method.")
        finally:
            if original_search_kwargs is not None:
                try:
                    retriever.search_kwargs = original_search_kwargs
                except Exception:
                    pass

        return list(docs)

    def as_tool(self) -> BaseTool:
        source_count = len(self.settings.source_urls)
        return create_retriever_tool(
            self._retriever,
            "retrieve_source_documents",
            (
                "Search and return relevant passages from the configured source "
                f"document set ({source_count} URL(s)). Use this for questions "
                "about the provided article or custom links."
            ),
        )

    def rebuild(self, urls: list[str] | None = None) -> None:
        if urls is not None:
            self.settings = replace(self.settings, source_urls=list(urls))
        self._retriever = self._build_langchain_retriever(rebuild=True)

    def as_langchain_retriever(self) -> Any:
        """Return the legacy LangChain retriever object."""

        return self._retriever

    def _build_langchain_retriever(self, *, rebuild: bool = False) -> Any:
        settings = self.settings

        if not rebuild and _persisted_chroma_exists(settings.chroma_dir):
            rebuild = not _embedding_config_matches(settings)

        if rebuild and settings.chroma_dir.exists():
            _release_chroma_system(settings.chroma_dir)
            _clear_chroma_store(settings.chroma_dir)

        embeddings = self._embeddings or build_embeddings(settings)

        if _persisted_chroma_exists(settings.chroma_dir):
            try:
                vectorstore = self._chroma_cls(
                    collection_name=settings.collection_name,
                    persist_directory=str(settings.chroma_dir),
                    embedding_function=embeddings,
                )
                return vectorstore.as_retriever()
            except Exception as exc:
                logger.warning(
                    "Persisted Chroma store at %s could not be loaded; rebuilding: %s",
                    settings.chroma_dir,
                    exc,
                )
                _release_chroma_system(settings.chroma_dir)
                _clear_chroma_store(settings.chroma_dir)

        doc_splits = load_and_split_documents(
            settings.source_urls,
            page_load_timeout=settings.page_load_timeout,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        logger.info("BUILD CHROMA VECTORSTORE")
        vectorstore = self._chroma_cls.from_documents(
            documents=doc_splits,
            collection_name=settings.collection_name,
            embedding=embeddings,
            persist_directory=str(settings.chroma_dir),
        )
        _write_embedding_config(settings)

        return vectorstore.as_retriever()


def build_retriever(settings: Settings, rebuild: bool = False) -> Any:
    """Build or load the legacy LangChain Chroma retriever."""

    return ChromaRetriever(settings=settings, rebuild=rebuild).as_langchain_retriever()


def build_retriever_tool(
    settings: Settings | Retriever,
    rebuild: bool = False,
) -> BaseTool:
    """Create the retriever tool from settings or any Retriever provider."""

    if isinstance(settings, Retriever):
        if rebuild:
            settings.rebuild()
        return settings.as_tool()

    return ChromaRetriever(settings=settings, rebuild=rebuild).as_tool()
