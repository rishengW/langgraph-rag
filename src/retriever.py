from __future__ import annotations

import json
import shutil
import logging
import time
import os
import gc
import stat
from pathlib import Path

from langchain_core.tools.retriever import create_retriever_tool

os.environ.setdefault("USER_AGENT", "rag-langgraph-local/1.0")

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .embeddings import build_embeddings


logger = logging.getLogger(__name__)

EMBEDDING_CONFIG_FILENAME = "embedding_config.json"


def _persisted_chroma_exists(chroma_dir: Path) -> bool:
    return chroma_dir.exists() and any(chroma_dir.iterdir())


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

    systems_to_stop = []
    with SharedSystemClient._refcount_lock:
        for identifier, system in list(SharedSystemClient._identifier_to_system.items()):
            persist_directory = getattr(system.settings, "persist_directory", None)
            if _paths_match(identifier, chroma_dir) or _paths_match(persist_directory, chroma_dir):
                systems_to_stop.append((identifier, system))
                SharedSystemClient._identifier_to_system.pop(identifier, None)
                SharedSystemClient._identifier_to_refcount.pop(identifier, None)

    for identifier, system in systems_to_stop:
        try:
            system.stop()
            logger.info("Released Chroma system for %s before rebuild", identifier)
        except Exception as exc:
            logger.warning("Failed to stop Chroma system for %s: %s", identifier, exc)


def _rmtree_with_retry(path: Path, max_retries: int = 10, delay: float = 1.0):
    """Remove a directory tree with aggressive retries for Windows file locking.
    
    On Windows, Chroma database files can be locked even after the vectorstore
    is closed. This function uses retries, garbage collection, and per-file
    deletion fallback.
    """
    if not path.exists():
        return
    
    def handle_remove_error(func, fpath, exc_info):
        """Error handler for shutil.rmtree to handle locked files on Windows."""
        try:
            # Make file writable
            if os.path.exists(fpath):
                os.chmod(fpath, stat.S_IWUSR | stat.S_IRUSR)
                os.unlink(fpath)
                logger.debug("Force-removed locked file: %s", fpath)
        except Exception as e:
            logger.warning("Could not force-remove %s: %s", fpath, e)
    
    def remove_tree_manually(dirpath):
        """Manually remove all files and directories in a tree."""
        try:
            for root, dirs, files in os.walk(dirpath, topdown=False):
                for name in files:
                    filepath = os.path.join(root, name)
                    try:
                        os.chmod(filepath, stat.S_IWUSR | stat.S_IRUSR)
                        os.unlink(filepath)
                    except Exception as e:
                        logger.warning("Failed to remove file %s: %s", filepath, e)
                for name in dirs:
                    dirpath_inner = os.path.join(root, name)
                    try:
                        os.rmdir(dirpath_inner)
                    except Exception as e:
                        logger.warning("Failed to remove dir %s: %s", dirpath_inner, e)
            # Try to remove the root directory
            if os.path.exists(dirpath):
                os.rmdir(dirpath)
                return True
        except Exception as e:
            logger.warning("Manual tree removal failed: %s", e)
        return False
    
    for attempt in range(max_retries):
        try:
            # Force garbage collection to release any file handles
            gc.collect()
            time.sleep(0.1)
            
            # Try normal removal with error handler
            shutil.rmtree(str(path), onerror=handle_remove_error)
            
            # Check if it's actually gone
            if not path.exists():
                logger.info("Successfully removed .chroma directory after %d attempt(s)", attempt + 1)
                return
        except (PermissionError, OSError) as e:
            if attempt < max_retries - 1:
                logger.warning(
                    "Failed to remove %s (attempt %d/%d), retrying in %.1f seconds: %s",
                    path,
                    attempt + 1,
                    max_retries,
                    delay,
                    e,
                )
                time.sleep(delay)
            else:
                # Last attempt - try manual removal
                logger.warning("Standard removal failed, attempting manual file-by-file removal")
                if remove_tree_manually(str(path)):
                    logger.info("Manual removal succeeded")
                    return
                logger.error("Failed to remove %s after %d attempts and manual removal", path, max_retries)
                raise

    if path.exists():
        logger.warning("Standard removal did not delete %s; attempting manual file-by-file removal", path)
        if remove_tree_manually(str(path)):
            logger.info("Manual removal succeeded")
            return
        raise OSError(f"Failed to remove {path} after {max_retries} attempts")


def build_retriever(settings: Settings, rebuild: bool = False):
    """Build or load the Chroma retriever used by the RAG tool."""

    if not rebuild and _persisted_chroma_exists(settings.chroma_dir):
        rebuild = not _embedding_config_matches(settings)

    if rebuild and settings.chroma_dir.exists():
        _release_chroma_system(settings.chroma_dir)
        _rmtree_with_retry(settings.chroma_dir)

    embeddings = build_embeddings(settings)

    if _persisted_chroma_exists(settings.chroma_dir):
        vectorstore = Chroma(
            collection_name=settings.collection_name,
            persist_directory=str(settings.chroma_dir),
            embedding_function=embeddings,
        )
        return vectorstore.as_retriever()

    print("---LOAD WEB DOCUMENTS---")
    docs_nested = []
    failed_urls: list[str] = []

    for url in settings.source_urls:
        try:
            docs_nested.append(WebBaseLoader(url).load())
        except Exception as exc:
            failed_urls.append(url)
            logger.warning("Failed to load URL %s: %s", url, exc)

    docs = [doc for sublist in docs_nested for doc in sublist]
    if not docs:
        failed = ", ".join(failed_urls) if failed_urls else "none"
        raise RuntimeError(f"No source documents could be loaded. Failed URLs: {failed}")

    print("---SPLIT DOCUMENTS---")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    doc_splits = text_splitter.split_documents(docs)

    print("---BUILD CHROMA VECTORSTORE---")
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=settings.collection_name,
        embedding=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
    _write_embedding_config(settings)

    return vectorstore.as_retriever()


def build_retriever_tool(settings: Settings, rebuild: bool = False):
    """Create the LangChain retriever tool used by the LangGraph ToolNode."""

    retriever = build_retriever(settings=settings, rebuild=rebuild)
    source_count = len(settings.source_urls)

    return create_retriever_tool(
        retriever,
        "retrieve_source_documents",
        (
            "Search and return relevant passages from the configured source "
            f"document set ({source_count} URL(s)). Use this for questions "
            "about the provided article or custom links."
        ),
    )
