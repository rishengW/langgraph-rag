from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext, suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever as LangChainBaseRetriever
from langchain_core.tools import BaseTool
from langchain_core.tools.retriever import create_retriever_tool

from src.config import Settings
from src.utils.retry import call_with_retry, remove_tree_with_retry

from .chunk_context import (
    CHUNK_CONTEXT_CACHE_FILENAME,
    ChunkContextCache,
    ChunkContextConfig,
)
from .document_loader import default_loader_factory, load_and_split_documents
from .document_quality import DocumentQualityConfig
from .embeddings import build_embeddings
from .llamaindex_adapter import (
    BM25_TOP_K_CAP,
    LLAMA_DOCSTORE_DIRNAME,
    HierarchicalSplitConfig,
    HybridFusionRetriever,
    build_bm25_retriever,
    build_corpus_nodes,
    load_and_split_hierarchical_documents,
    load_docstore,
    nodes_to_documents,
    persist_docstore,
    retrieve_with_auto_merge,
)
from .retriever import Retriever

try:
    from langchain_chroma import Chroma as Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma as Chroma


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
        and child.name != LLAMA_DOCSTORE_DIRNAME
        for child in chroma_dir.iterdir()
    )


def _embedding_config_path(chroma_dir: Path) -> Path:
    return chroma_dir / EMBEDDING_CONFIG_FILENAME


def _embedding_config(settings: Settings) -> dict[str, int | str | bool | None]:
    return {
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        # Contextual prefixes change the embedded content, so toggling the
        # feature must force a rebuild instead of mixing prefixed/unprefixed
        # embeddings in one store.
        "chunk_context_enabled": settings.chunk_context_enabled,
        # Hierarchical chunking changes what is embedded (parent/child chunks
        # with node ids), so toggling or resizing it must rebuild as well.
        "hierarchical_chunking_enabled": settings.hierarchical_chunking_enabled,
        "hierarchical_parent_chunk_size": settings.hierarchical_parent_chunk_size,
        # Structure-aware chunking changes what is embedded (one chunk per
        # heading section), so toggling it must rebuild as well.
        "structure_aware_chunking_enabled": settings.structure_aware_chunking_enabled,
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


def _paths_match(left: str | os.PathLike[str] | None, right: Path) -> bool:
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
    """Clear Chroma files while preserving nested chat/web-search stores.

    The chunk-context prefix cache is preserved too: a rebuild must not re-pay
    the LLM prefix cost for unchanged chunks.
    """

    if not chroma_dir.exists():
        return

    for child in list(chroma_dir.iterdir()):
        if child.name in RESERVED_CHROMA_CHILD_DIRS:
            continue
        if child.name == CHUNK_CONTEXT_CACHE_FILENAME:
            continue
        if child.is_dir():
            _rmtree_with_retry(child)
        else:
            _unlink_with_retry(child)


def _documents_from_vectorstore(vectorstore: Any) -> list[Document]:
    """Read back all chunks (text + metadata) from a loaded vectorstore.

    Used to rebuild the BM25 leg and parent-link corpus from a persisted
    Chroma store without re-fetching source URLs. Corpora here are bounded
    by the configured source URLs, so an unbounded get() stays small.
    """

    get = getattr(vectorstore, "get", None)
    if not callable(get):
        return []
    try:
        data = get()
    except Exception as exc:
        logger.warning("Could not read back chunks from the vectorstore: %s", exc)
        return []
    if not isinstance(data, dict):
        return []
    texts = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    return [
        Document(page_content=text or "", metadata=dict(metadata or {}))
        for text, metadata in zip(texts, metadatas, strict=False)
    ]


def _delete_chunks_by_source(vectorstore: Any, source: str) -> int:
    """Delete every persisted chunk whose metadata source matches ``source``.

    Makes re-indexing one upload idempotent: the old chunks for that file are
    removed before the new ones are added, so nothing accumulates across
    re-uploads of the same filename.
    """

    collection = getattr(vectorstore, "_collection", None)
    get = getattr(collection, "get", None)
    delete = getattr(collection, "delete", None)
    if not callable(get) or not callable(delete):
        return 0
    try:
        data = get(where={"source": source})
    except Exception as exc:
        logger.warning("Could not look up chunks for source %r: %s", source, exc)
        return 0
    ids = list((data or {}).get("ids") or [])
    if not ids:
        return 0
    try:
        delete(ids=ids)
    except Exception as exc:
        logger.warning("Could not delete %d chunk(s) for source %r: %s", len(ids), source, exc)
        return 0
    logger.info("Deleted %d existing chunk(s) for source %r", len(ids), source)
    return len(ids)


class _ProviderBackedLangChainRetriever(LangChainBaseRetriever):
    """LangChain retriever facade over a ChromaRetriever provider.

    Used for the agent tool when hybrid/hierarchical retrieval is active, so
    tool calls go through the same fused/auto-merged path as direct
    ``retrieve()`` calls instead of bypassing it via the raw vectorstore
    retriever.
    """

    provider: Any
    result_k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        return self.provider.retrieve(query, k=self.result_k)


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
        self._vectorstore: Any | None = None
        # Enhanced-retrieval state (llama-index adapter): populated when
        # hybrid and/or hierarchical retrieval is enabled and the corpus
        # artifacts are available. Left empty on the legacy path.
        self._corpus_nodes: dict[str, Any] | None = None
        self._bm25_retriever: Any | None = None
        self._parent_docstore: Any | None = None
        self._retriever = self._build_langchain_retriever(rebuild=rebuild)

    @property
    def _enhanced_retrieval_enabled(self) -> bool:
        settings = self.settings
        return bool(
            getattr(settings, "hybrid_retrieval_enabled", False)
            or getattr(settings, "hierarchical_chunking_enabled", False)
        )

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        if self._enhanced_retrieval_enabled and self._corpus_nodes:
            return self._retrieve_enhanced(query, k)

        # Prefer the vectorstore directly: mutating the shared retriever's
        # ``search_kwargs`` races with concurrent tool calls on the same
        # graph (parallel tool calls, planning Send branches).
        vectorstore = self._vectorstore
        if vectorstore is not None and hasattr(vectorstore, "similarity_search"):
            return list(vectorstore.similarity_search(query, k=k))

        retriever = self._retriever
        original_search_kwargs: dict[str, Any] | None = None

        if hasattr(retriever, "search_kwargs"):
            try:
                original_search_kwargs = dict(retriever.search_kwargs or {})
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
                with suppress(Exception):
                    retriever.search_kwargs = original_search_kwargs

        return list(docs)

    def _retrieve_enhanced(self, query: str, k: int) -> list[Document]:
        """Hybrid (BM25+vector RRF) and/or hierarchical auto-merged retrieval.

        Heavy state (BM25 index, parent docstore) is built once per index and
        shared read-only; the fusion/merger retriever objects are constructed
        per call so concurrent retrieves never mutate shared search kwargs.
        """

        settings = self.settings
        vectorstore = self._vectorstore
        if vectorstore is None or not hasattr(vectorstore, "similarity_search"):
            return []
        if not self._corpus_nodes:
            return []

        fetch_k = max(
            k * max(1, int(getattr(settings, "hybrid_retrieval_fetch_multiplier", 3))),
            k + 2,
        )
        fusion = HybridFusionRetriever(
            vector_search=lambda q, n: list(vectorstore.similarity_search(q, k=n)),
            corpus_nodes=self._corpus_nodes,
            bm25_retriever=(
                self._bm25_retriever
                if getattr(settings, "hybrid_retrieval_enabled", False)
                else None
            ),
            fetch_k=fetch_k,
            rrf_k=max(1, int(getattr(settings, "hybrid_retrieval_rrf_k", 60))),
        )

        if (
            getattr(settings, "hierarchical_chunking_enabled", False)
            and self._parent_docstore is not None
        ):
            merged = retrieve_with_auto_merge(
                fusion,
                self._parent_docstore,
                query=query,
                threshold=float(getattr(settings, "hierarchical_auto_merge_threshold", 0.5)),
            )
        else:
            merged = fusion.retrieve(query)

        return nodes_to_documents(merged)[: max(1, k)]

    def as_tool(self) -> BaseTool:
        source_count = len(self.settings.source_urls)
        retriever: Any = self._retriever
        if self._enhanced_retrieval_enabled and self._corpus_nodes:
            retriever = _ProviderBackedLangChainRetriever(provider=self)
        return create_retriever_tool(
            retriever,
            "retrieve_source_documents",
            (
                "Search and return relevant passages from the configured source "
                f"document set ({source_count} URL(s)). Use this for questions "
                "about the provided article or custom links."
            ),
            # Keep the Document list on the ToolMessage artifact so downstream
            # nodes can recover per-chunk source URLs for citation grounding.
            response_format="content_and_artifact",
        )

    def rebuild(self, urls: list[str] | None = None) -> None:
        if urls is not None:
            self.settings = replace(self.settings, source_urls=list(urls))
        self._retriever = self._build_langchain_retriever(rebuild=True)

    def index_documents(
        self,
        documents: list[Document],
        *,
        parent_docstore_additions: Any | None = None,
    ) -> int:
        """Add parsed upload chunks to the persisted store without a rebuild.

        Idempotent per ``metadata["source"]``: existing chunks with the same
        source are deleted first so re-uploading a file replaces its chunks.
        Returns the number of chunks added. The live graph tool's enhanced
        retrieval state (BM25/corpus nodes) is refreshed from the persisted
        store only on the next provider build — see ``as_tool`` callers.
        """

        vectorstore = self._vectorstore
        if vectorstore is None or not hasattr(vectorstore, "add_documents"):
            raise RuntimeError("no loaded vectorstore; build the provider first.")
        if not documents:
            return 0

        sources = sorted(
            {
                str(document.metadata.get("source") or "")
                for document in documents
                if document.metadata.get("source")
            }
        )
        for source in sources:
            _delete_chunks_by_source(vectorstore, source)

        try:
            vectorstore.add_documents(documents)
        except Exception as exc:
            raise RuntimeError(f"could not add chunks to the vector store: {exc}") from exc

        if (
            parent_docstore_additions is not None
            and self.settings.hierarchical_chunking_enabled
        ):
            # persist_docstore writes the WHOLE store, so merge the additions
            # into the loaded docstore first — persisting the additions alone
            # would drop every previously indexed parent.
            merged = parent_docstore_additions
            existing = self._parent_docstore or load_docstore(self.settings.chroma_dir)
            if existing is not None:
                add_documents = getattr(existing, "add_documents", None)
                if callable(add_documents):
                    added = list(
                        getattr(parent_docstore_additions, "docs", {}).values()
                    )
                    if added:
                        existing.add_documents(added)
                    merged = existing
            persist_docstore(merged, self.settings.chroma_dir)
            self._parent_docstore = merged

        # Refresh hybrid/hierarchical state from the persisted contents so this
        # provider instance (and any later graph build) sees the new chunks.
        self._prepare_enhanced_retrieval(
            vectorstore,
            built_documents=None,
            built_docstore=None,
        )
        return len(documents)

    def delete_documents_by_source(self, source: str) -> int:
        """Remove all chunks for one source from the persisted store."""

        vectorstore = self._vectorstore
        if vectorstore is None or not source:
            return 0
        return _delete_chunks_by_source(vectorstore, source)

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
                self._vectorstore = vectorstore
                if self._prepare_enhanced_retrieval(
                    vectorstore,
                    built_documents=None,
                    built_docstore=None,
                ):
                    return vectorstore.as_retriever()
                raise RuntimeError("enhanced retrieval artifacts missing or empty")
            except Exception as exc:
                logger.warning(
                    "Persisted Chroma store at %s could not be loaded; rebuilding: %s",
                    settings.chroma_dir,
                    exc,
                )
                _release_chroma_system(settings.chroma_dir)
                _clear_chroma_store(settings.chroma_dir)

        chunk_context_config: ChunkContextConfig | None = None
        chunk_context_model: Any | None = None
        chunk_context_cache: ChunkContextCache | None = None
        if settings.chunk_context_enabled:
            from src.backend.llm.provider import build_chat_model

            chunk_context_config = ChunkContextConfig(
                enabled=True,
                document_excerpt_chars=settings.chunk_context_document_excerpt_chars,
                max_prefix_chars=settings.chunk_context_max_prefix_chars,
                max_concurrency=settings.chunk_context_max_concurrency,
                max_retries=settings.dashscope_max_retries,
            )
            chunk_context_model = build_chat_model(settings)
            chunk_context_cache = ChunkContextCache(
                settings.chroma_dir / CHUNK_CONTEXT_CACHE_FILENAME
            )

        structure_aware = bool(getattr(settings, "structure_aware_chunking_enabled", False))
        if structure_aware:
            # Section splitting needs the raw HTML, which only the loader holds;
            # swap in a structure-aware loader factory for this build.
            def _structure_loader_factory(url: str, page_timeout: int) -> Any:
                return default_loader_factory(url, page_timeout, structure_aware=True)

            loader_factory: Any = _structure_loader_factory
        else:
            loader_factory = default_loader_factory

        built_docstore: Any | None = None
        if settings.hierarchical_chunking_enabled:
            # REFACTOR: Parent/child split; children are embedded, parents are
            # persisted to a docstore for auto-merging at query time.
            doc_splits, built_docstore = load_and_split_hierarchical_documents(
                settings.source_urls,
                page_load_timeout=settings.page_load_timeout,
                max_concurrent_loads=settings.page_load_max_concurrency,
                page_load_cache_ttl_seconds=settings.page_load_cache_ttl_seconds,
                loader_factory=loader_factory,
                quality_config=DocumentQualityConfig(
                    enabled=settings.document_quality_filter_enabled,
                    min_text_length=settings.document_quality_min_text_length,
                    min_unique_terms=settings.document_quality_min_unique_terms,
                    relevance_query=settings.document_quality_relevance_query,
                    min_query_term_overlap=settings.document_quality_query_min_overlap,
                    min_similarity=settings.document_quality_min_similarity,
                    recency_bias_days=settings.document_quality_recency_bias_days,
                ),
                embeddings=embeddings,
                chunk_context_config=chunk_context_config,
                chat_model=chunk_context_model,
                chunk_context_cache=chunk_context_cache,
                hierarchical_config=HierarchicalSplitConfig(
                    parent_chunk_size=settings.hierarchical_parent_chunk_size,
                    child_chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                ),
            )
        else:
            doc_splits = load_and_split_documents(
                settings.source_urls,
                page_load_timeout=settings.page_load_timeout,
                max_concurrent_loads=settings.page_load_max_concurrency,
                page_load_cache_ttl_seconds=settings.page_load_cache_ttl_seconds,
                loader_factory=loader_factory,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                structure_aware=structure_aware,
                quality_config=DocumentQualityConfig(
                    enabled=settings.document_quality_filter_enabled,
                    min_text_length=settings.document_quality_min_text_length,
                    min_unique_terms=settings.document_quality_min_unique_terms,
                    relevance_query=settings.document_quality_relevance_query,
                    min_query_term_overlap=settings.document_quality_query_min_overlap,
                    min_similarity=settings.document_quality_min_similarity,
                    recency_bias_days=settings.document_quality_recency_bias_days,
                ),
                embeddings=embeddings,
                chunk_context_config=chunk_context_config,
                chat_model=chunk_context_model,
                chunk_context_cache=chunk_context_cache,
            )

        logger.info("BUILD CHROMA VECTORSTORE")
        vectorstore = self._chroma_cls.from_documents(
            documents=doc_splits,
            collection_name=settings.collection_name,
            embedding=embeddings,
            persist_directory=str(settings.chroma_dir),
        )
        _write_embedding_config(settings)
        if built_docstore is not None:
            persist_docstore(built_docstore, settings.chroma_dir)

        self._vectorstore = vectorstore
        self._prepare_enhanced_retrieval(
            vectorstore,
            built_documents=doc_splits,
            built_docstore=built_docstore,
        )
        return vectorstore.as_retriever()

    def _prepare_enhanced_retrieval(
        self,
        vectorstore: Any,
        *,
        built_documents: list[Document] | None,
        built_docstore: Any | None,
    ) -> bool:
        """Build/refresh the hybrid + auto-merge state from the vectorstore.

        Returns False when a required artifact is missing (caller rebuilds).
        With both features off this is a no-op success so the legacy path is
        untouched. An empty corpus degrades to legacy retrieval rather than
        failing the whole provider.
        """

        self._corpus_nodes = None
        self._bm25_retriever = None
        self._parent_docstore = None
        if not self._enhanced_retrieval_enabled:
            return True

        settings = self.settings
        if built_documents is not None:
            documents = built_documents
        else:
            documents = _documents_from_vectorstore(vectorstore)

        if settings.hierarchical_chunking_enabled:
            docstore = (
                built_docstore
                if built_docstore is not None
                else load_docstore(settings.chroma_dir)
            )
            if docstore is None:
                logger.warning(
                    "Hierarchical chunking is enabled but no parent docstore "
                    "exists under %s; forcing a rebuild",
                    settings.chroma_dir,
                )
                return False
            self._parent_docstore = docstore

        corpus_nodes = build_corpus_nodes(documents)
        if not corpus_nodes:
            logger.warning(
                "Enhanced retrieval found no chunks in the store; "
                "falling back to plain vector retrieval"
            )
            return True
        self._corpus_nodes = corpus_nodes

        if settings.hybrid_retrieval_enabled:
            self._bm25_retriever = build_bm25_retriever(
                list(corpus_nodes.values()),
                similarity_top_k=BM25_TOP_K_CAP,
            )
        return True


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
