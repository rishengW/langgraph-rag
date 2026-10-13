"""LlamaIndex-backed hybrid and hierarchical retrieval, kept behind one adapter.

llama-index types never escape this module: the Chroma vector store stays
LangChain-owned (see ``chroma_retriever.py``), results are converted back to
``langchain_core.documents.Document`` at the boundary, and no llama-index
global ``Settings`` is touched — every component is configured per instance so
multi-session settings never leak across providers.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# On-disk location of the parent-node docstore, relative to the Chroma dir.
# Not part of RESERVED_CHROMA_CHILD_DIRS: a rebuild must wipe it so stale
# parents never outlive the child chunks they merge into.
LLAMA_DOCSTORE_DIRNAME = "llama_docstore"
DOCSTORE_FILENAME = "docstore.json"

# BM25 leg token pattern: latin/digit words plus single CJK ideographs. The
# bm25s default (\b\w\w+\b) treats a whole Chinese sentence as ONE token,
# which would make the keyword leg useless on the mixed zh/en corpora this
# project indexes. CJK unigrams restore keyword recall without a segmenter.
BM25_TOKEN_PATTERN = r"(?u)([A-Za-z0-9_]+|[䀀-䶿一-鿿])"

# Upper bound for the BM25 leg's similarity_top_k. The retriever is built once
# per index (tokenizing the corpus is the expensive part) and per-query slices
# are taken from its ranked output, so this only needs to exceed any fetch_k.
BM25_TOP_K_CAP = 64

VectorSearchFn = Callable[[str, int], list[Document]]


def content_key(text: str) -> str:
    """Stable identity for a chunk across the vector and BM25 legs."""

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _simple_metadata(metadata: Any) -> dict[str, Any]:
    """Keep only Chroma-compatible scalar metadata values."""

    if not isinstance(metadata, dict):
        return {}
    return {
        str(key): value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool))
    }


@dataclass(frozen=True)
class HierarchicalSplitConfig:
    """Two-level parent/child split sizes (llama-index token counts)."""

    parent_chunk_size: int = 512
    child_chunk_size: int = 100
    chunk_overlap: int = 20


def split_documents_hierarchical(
    documents: Sequence[Document],
    config: HierarchicalSplitConfig,
) -> tuple[list[Document], Any]:
    """Split documents into a parent/child hierarchy.

    Returns ``(child_documents, parent_docstore)``. Children carry ``node_id``
    and ``parent_id`` metadata and are the only chunks embedded in Chroma;
    parents live solely in the returned SimpleDocumentStore and surface at
    query time when auto-merging fires.
    """

    if config.parent_chunk_size <= config.child_chunk_size:
        raise ValueError(
            "hierarchical_parent_chunk_size must be larger than chunk_size "
            f"({config.parent_chunk_size} <= {config.child_chunk_size}); a parent "
            "chunk that fits inside a child makes the hierarchy degenerate."
        )

    from llama_index.core.node_parser.relational.hierarchical import (
        HierarchicalNodeParser,
        get_leaf_nodes,
        get_root_nodes,
    )
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore

    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[config.parent_chunk_size, config.child_chunk_size],
        chunk_overlap=config.chunk_overlap,
    )
    llama_docs = [
        LlamaDocument(
            text=document.page_content or "",
            metadata=_simple_metadata(document.metadata),
        )
        for document in documents
    ]
    nodes = parser.get_nodes_from_documents(llama_docs)
    leaves = get_leaf_nodes(nodes)
    roots = get_root_nodes(nodes)

    docstore = SimpleDocumentStore()
    docstore.add_documents(list(roots))

    child_documents: list[Document] = []
    for leaf in leaves:
        metadata = _simple_metadata(leaf.metadata)
        metadata["node_id"] = leaf.node_id
        parent = leaf.parent_node
        metadata["parent_id"] = parent.node_id if parent is not None else ""
        child_documents.append(
            Document(page_content=leaf.get_content(metadata_mode=_metadata_mode_none()), metadata=metadata)
        )

    logger.info(
        "HIERARCHICAL SPLIT: %d documents -> %d parents, %d child chunks",
        len(documents),
        len(roots),
        len(child_documents),
    )
    return child_documents, docstore


def _metadata_mode_none() -> Any:
    from llama_index.core.schema import MetadataMode

    return MetadataMode.NONE


def docstore_persist_path(chroma_dir: Path) -> Path:
    return Path(chroma_dir) / LLAMA_DOCSTORE_DIRNAME / DOCSTORE_FILENAME


def persist_docstore(docstore: Any, chroma_dir: Path) -> None:
    path = docstore_persist_path(chroma_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    docstore.persist(str(path))


def load_docstore(chroma_dir: Path) -> Any | None:
    """Load the persisted parent docstore; None when absent or unreadable."""

    path = docstore_persist_path(chroma_dir)
    if not path.exists():
        return None
    try:
        from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore

        return SimpleDocumentStore.from_persist_path(str(path))
    except Exception as exc:
        logger.warning("Could not load parent docstore at %s: %s", path, exc)
        return None


def build_corpus_nodes(documents: Sequence[Document]) -> dict[str, Any]:
    """Build canonical llama-index TextNodes keyed by content hash.

    Both retrieval legs re-resolve through this map so fused/merged results
    always reference nodes with intact parent relationships — the BM25 leg
    returns nodes reconstructed from serialized metadata, which would
    otherwise lose the parent link AutoMergingRetriever depends on.
    """

    from llama_index.core.schema import (
        NodeRelationship,
        RelatedNodeInfo,
        TextNode,
    )

    nodes: dict[str, Any] = {}
    for document in documents:
        key = content_key(document.page_content or "")
        if key in nodes:
            continue
        metadata = _simple_metadata(document.metadata)
        node_id = str(metadata.get("node_id") or key)
        node = TextNode(
            text=document.page_content or "",
            id_=node_id,
            metadata=metadata,
        )
        parent_id = str(metadata.get("parent_id") or "")
        if parent_id:
            node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent_id)
        nodes[key] = node
    return nodes


def build_bm25_retriever(corpus_nodes: Sequence[Any], *, similarity_top_k: int) -> Any:
    """Build the BM25 keyword leg over the same chunks Chroma embeds."""

    try:
        from llama_index.retrievers.bm25 import BM25Retriever
    except ImportError as exc:
        raise RuntimeError(
            "hybrid_retrieval_enabled requires the llama-index-retrievers-bm25 "
            "package; install it with: pip install llama-index-retrievers-bm25"
        ) from exc

    return BM25Retriever(
        nodes=list(corpus_nodes),
        similarity_top_k=max(1, similarity_top_k),
        # Deterministic multilingual behavior: no English stemmer on a mixed
        # zh/en corpus, CJK-aware token pattern instead of \w+ runs.
        skip_stemming=True,
        token_pattern=BM25_TOKEN_PATTERN,
    )


class HybridFusionRetriever:
    """Vector + optional BM25 legs fused with reciprocal rank fusion (RRF).

    Duck-types llama-index's BaseRetriever contract (a ``retrieve`` method
    accepting a query string or QueryBundle) so AutoMergingRetriever can wrap
    it without this class inheriting llama-index pydantic plumbing.
    """

    def __init__(
        self,
        *,
        vector_search: VectorSearchFn,
        corpus_nodes: dict[str, Any],
        bm25_retriever: Any | None = None,
        fetch_k: int = 8,
        rrf_k: int = 60,
    ) -> None:
        self._vector_search = vector_search
        self._corpus_nodes = corpus_nodes
        self._bm25_retriever = bm25_retriever
        self._fetch_k = max(1, int(fetch_k))
        self._rrf_k = max(1, int(rrf_k))

    def retrieve(self, query: Any) -> list[Any]:
        from llama_index.core.schema import NodeWithScore

        query_str = getattr(query, "query_str", None) or str(query)

        legs: list[list[str]] = []
        vector_docs = self._vector_search(query_str, self._fetch_k) or []
        legs.append([content_key(document.page_content or "") for document in vector_docs])

        if self._bm25_retriever is not None:
            bm25_hits = self._bm25_retriever.retrieve(query_str)[: self._fetch_k]
            legs.append(
                [
                    content_key(hit.node.get_content(metadata_mode=_metadata_mode_none()))
                    for hit in bm25_hits
                    # bm25s pads the leg with zero-score (no keyword overlap)
                    # chunks; those must not earn fusion credit.
                    if (hit.score or 0.0) > 0.0
                ]
            )

        # Reciprocal rank fusion: 1 / (rrf_k + rank) per leg, summed per chunk.
        scores: dict[str, float] = {}
        for leg in legs:
            seen_in_leg: set[str] = set()
            for key in leg:
                if key in seen_in_leg or key not in self._corpus_nodes:
                    continue
                seen_in_leg.add(key)
                rank = len(seen_in_leg)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank)

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            NodeWithScore(node=self._corpus_nodes[key], score=score)
            for key, score in ordered[: self._fetch_k]
        ]


def retrieve_with_auto_merge(
    fusion_retriever: HybridFusionRetriever,
    docstore: Any,
    *,
    query: str,
    threshold: float,
) -> list[Any]:
    """Run the fusion leg, then merge dense child groups into parent chunks."""

    from llama_index.core.retrievers import AutoMergingRetriever
    from llama_index.core.storage.storage_context import StorageContext

    merger = AutoMergingRetriever(
        fusion_retriever,  # type: ignore[arg-type] - duck-typed retrieve() contract
        StorageContext.from_defaults(docstore=docstore),
        simple_ratio_thresh=threshold,
    )
    return merger.retrieve(query)


def nodes_to_documents(nodes_with_score: Sequence[Any]) -> list[Document]:
    """Convert llama-index results back to LangChain Documents at the boundary."""

    documents: list[Document] = []
    for node_with_score in nodes_with_score:
        node = node_with_score.node
        documents.append(
            Document(
                page_content=node.get_content(metadata_mode=_metadata_mode_none()),
                metadata=_simple_metadata(getattr(node, "metadata", None)),
            )
        )
    return documents


def load_and_split_hierarchical_documents(
    source_urls: Sequence[str],
    *,
    page_load_timeout: int,
    max_concurrent_loads: int = 4,
    page_load_cache_ttl_seconds: int = 0,
    document_cache: Any | None = None,
    loader_factory: Any | None = None,
    quality_config: Any | None = None,
    embeddings: Any | None = None,
    chunk_context_config: Any | None = None,
    chat_model: Any | None = None,
    chunk_context_cache: Any | None = None,
    hierarchical_config: HierarchicalSplitConfig,
) -> tuple[list[Document], Any]:
    """Hierarchical variant of ``load_and_split_documents``.

    Same load -> quality-filter pipeline, but splits parent/child and returns
    the parent docstore alongside the child chunks. Contextual prefixes apply
    to children only: they are the embedded chunks, while parents serve as
    merged output context.
    """

    from .chunk_context import apply_chunk_context
    from .document_loader import default_loader_factory, load_source_documents
    from .document_quality import filter_quality_documents

    docs = load_source_documents(
        source_urls,
        page_load_timeout=page_load_timeout,
        max_concurrent_loads=max_concurrent_loads,
        page_load_cache_ttl_seconds=page_load_cache_ttl_seconds,
        document_cache=document_cache,
        loader_factory=loader_factory or default_loader_factory,
    )
    docs = filter_quality_documents(docs, quality_config, embeddings=embeddings)
    if not docs:
        raise RuntimeError(
            "All loaded source documents were filtered out before indexing. "
            "No documents met the configured content quality thresholds."
        )

    children, docstore = split_documents_hierarchical(docs, hierarchical_config)
    if (
        chunk_context_config is not None
        and chunk_context_config.enabled
        and chat_model is not None
    ):
        children = apply_chunk_context(
            children,
            config=chunk_context_config,
            chat_model=chat_model,
            cache=chunk_context_cache,
        )
    return children, docstore


__all__ = [
    "BM25_TOKEN_PATTERN",
    "HierarchicalSplitConfig",
    "HybridFusionRetriever",
    "LLAMA_DOCSTORE_DIRNAME",
    "build_bm25_retriever",
    "build_corpus_nodes",
    "content_key",
    "docstore_persist_path",
    "load_and_split_hierarchical_documents",
    "load_docstore",
    "nodes_to_documents",
    "persist_docstore",
    "retrieve_with_auto_merge",
    "split_documents_hierarchical",
]
