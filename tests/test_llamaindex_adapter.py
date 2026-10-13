from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.backend.rag import chroma_retriever as chroma_module
from src.backend.rag.chroma_retriever import ChromaRetriever
from src.backend.rag.llamaindex_adapter import (
    HierarchicalSplitConfig,
    HybridFusionRetriever,
    build_bm25_retriever,
    build_corpus_nodes,
    content_key,
    docstore_persist_path,
    load_docstore,
    nodes_to_documents,
    persist_docstore,
    retrieve_with_auto_merge,
    split_documents_hierarchical,
)


def _long_text(label: str, sentences: int = 60) -> str:
    # Enough prose for the hierarchical splitter to produce multiple children
    # per parent at the tiny test chunk sizes.
    return " ".join(f"{label} sentence {index} about retrieval and indexing." for index in range(sentences))


def test_split_documents_hierarchical_builds_parents_and_linked_children():
    docs = [
        Document(page_content=_long_text("alpha"), metadata={"source": "https://a.test"}),
        Document(page_content=_long_text("beta"), metadata={"source": "https://b.test"}),
    ]
    config = HierarchicalSplitConfig(
        parent_chunk_size=40,
        child_chunk_size=10,
        chunk_overlap=2,
    )

    children, docstore = split_documents_hierarchical(docs, config)

    assert children, "expected child chunks"
    parent_ids = set()
    for child in children:
        assert child.metadata["node_id"]
        parent_id = child.metadata["parent_id"]
        assert parent_id
        parent_ids.add(parent_id)
        assert child.metadata["source"] in {"https://a.test", "https://b.test"}

    # Every referenced parent exists in the docstore and knows its children.
    for parent_id in parent_ids:
        parent = docstore.get_document(parent_id)
        assert parent.child_nodes, "parent must reference its children"


def test_split_documents_hierarchical_rejects_degenerate_sizes():
    with pytest.raises(ValueError, match="larger than chunk_size"):
        split_documents_hierarchical(
            [Document(page_content="text")],
            HierarchicalSplitConfig(parent_chunk_size=10, child_chunk_size=10, chunk_overlap=2),
        )


def test_hybrid_fusion_combines_vector_and_bm25_legs():
    docs = [
        Document(page_content="vector only chunk about graphs", metadata={"source": "s"}),
        Document(page_content="keyword needle chunk about bm25", metadata={"source": "s"}),
        Document(page_content="shared chunk about retrieval", metadata={"source": "s"}),
    ]
    corpus = build_corpus_nodes(docs)
    bm25 = build_bm25_retriever(list(corpus.values()), similarity_top_k=8)

    # Vector leg ranks the shared chunk first; the BM25 leg additionally
    # surfaces the keyword-only chunk.
    vector_docs = [docs[2], docs[0]]
    fusion = HybridFusionRetriever(
        vector_search=lambda _query, _n: list(vector_docs),
        corpus_nodes=corpus,
        bm25_retriever=bm25,
        fetch_k=3,
        rrf_k=60,
    )

    results = fusion.retrieve("needle bm25 retrieval")

    texts = [node.node.get_content() for node in results]
    assert texts, "fusion returned nothing"
    # The shared chunk is in both legs and must outrank single-leg chunks.
    assert texts[0] == docs[2].page_content
    # The keyword-only chunk appears thanks to the BM25 leg.
    assert docs[1].page_content in texts


def test_hybrid_fusion_vector_only_without_bm25_leg():
    docs = [Document(page_content="plain chunk"), Document(page_content="other chunk")]
    corpus = build_corpus_nodes(docs)
    fusion = HybridFusionRetriever(
        vector_search=lambda _query, n: docs[:n],
        corpus_nodes=corpus,
        bm25_retriever=None,
        fetch_k=2,
    )

    results = fusion.retrieve("anything")

    assert [node.node.get_content() for node in results] == ["plain chunk", "other chunk"]


def test_auto_merge_replaces_dense_child_groups_with_parent():
    children, docstore = split_documents_hierarchical(
        [Document(page_content=_long_text("gamma", sentences=80), metadata={"source": "s"})],
        HierarchicalSplitConfig(parent_chunk_size=30, child_chunk_size=10, chunk_overlap=2),
    )
    assert len(children) >= 4, "test needs several children per parent"
    corpus = build_corpus_nodes(children)

    first_parent_id = children[0].metadata["parent_id"]
    siblings = [child for child in children if child.metadata["parent_id"] == first_parent_id]
    assert len(siblings) >= 2

    class _StubFusion:
        def retrieve(self, _query):
            from llama_index.core.schema import NodeWithScore

            # Two of the parent's children, as if retrieval surfaced both.
            keys = [content_key(child.page_content) for child in siblings[:2]]
            return [NodeWithScore(node=corpus[key], score=0.5) for key in keys]

    merged = retrieve_with_auto_merge(
        _StubFusion(),
        docstore,
        query="gamma retrieval",
        threshold=0.4,  # 2 of >=3 children exceeds 0.4
    )
    merged_ids = {node.node.node_id for node in merged}
    assert first_parent_id in merged_ids
    assert not {child.metadata["node_id"] for child in siblings[:2]} & merged_ids

    unmerged = retrieve_with_auto_merge(
        _StubFusion(),
        docstore,
        query="gamma retrieval",
        threshold=0.99,  # ratio can never exceed 1.0 at this threshold
    )
    unmerged_ids = {node.node.node_id for node in unmerged}
    assert first_parent_id not in unmerged_ids


def test_docstore_persist_roundtrip(tmp_path):
    children, docstore = split_documents_hierarchical(
        [Document(page_content=_long_text("delta"), metadata={"source": "s"})],
        HierarchicalSplitConfig(parent_chunk_size=30, child_chunk_size=10, chunk_overlap=2),
    )
    assert children

    persist_docstore(docstore, tmp_path)
    assert docstore_persist_path(tmp_path).exists()

    loaded = load_docstore(tmp_path)
    assert loaded is not None
    parent_id = children[0].metadata["parent_id"]
    assert loaded.get_document(parent_id).node_id == parent_id


def test_nodes_to_documents_strips_llamaindex_types():
    docs = [Document(page_content="round trip chunk", metadata={"source": "s", "node_id": "n1"})]
    corpus = build_corpus_nodes(docs)
    fusion = HybridFusionRetriever(
        vector_search=lambda _query, _n: docs,
        corpus_nodes=corpus,
        fetch_k=1,
    )

    converted = nodes_to_documents(fusion.retrieve("round"))

    assert len(converted) == 1
    assert isinstance(converted[0], Document)
    assert converted[0].page_content == "round trip chunk"
    assert converted[0].metadata["source"] == "s"


class _FakeVectorstore:
    def __init__(self, documents: list[Document]):
        self._documents = documents

    @classmethod
    def from_documents(cls, documents, **_kwargs):
        return cls(list(documents))

    def similarity_search(self, _query: str, k: int = 4):
        return self._documents[:k]

    def as_retriever(self):
        return "fake-retriever"


def test_chroma_retriever_hybrid_build_and_retrieve(monkeypatch, isolated_settings):
    settings = isolated_settings(hybrid_retrieval_enabled=True)

    def fake_load_and_split(*_args, **_kwargs):
        return [
            Document(page_content="vector leg chunk about graphs"),
            Document(page_content="keyword needle chunk about bm25"),
        ]

    monkeypatch.setattr(chroma_module, "load_and_split_documents", fake_load_and_split)

    provider = ChromaRetriever(settings, chroma_cls=_FakeVectorstore)

    results = provider.retrieve("bm25 needle", k=2)

    texts = [document.page_content for document in results]
    assert len(texts) == 2
    # Both chunks come back; the keyword chunk is present via the BM25 leg.
    assert "keyword needle chunk about bm25" in texts


def test_chroma_retriever_hierarchical_persists_docstore(monkeypatch, isolated_settings, tmp_path):
    settings = isolated_settings(hierarchical_chunking_enabled=True)

    def fake_load_and_split(*_args, **_kwargs):
        pytest.fail("flat split must not run when hierarchical chunking is enabled")

    children, docstore = split_documents_hierarchical(
        [Document(page_content=_long_text("epsilon", sentences=80), metadata={"source": "s"})],
        HierarchicalSplitConfig(parent_chunk_size=30, child_chunk_size=10, chunk_overlap=2),
    )

    def fake_hierarchical(*_args, **kwargs):
        config = kwargs["hierarchical_config"]
        assert config.child_chunk_size == settings.chunk_size
        return children, docstore

    monkeypatch.setattr(chroma_module, "load_and_split_documents", fake_load_and_split)
    monkeypatch.setattr(
        chroma_module,
        "load_and_split_hierarchical_documents",
        fake_hierarchical,
    )

    provider = ChromaRetriever(settings, chroma_cls=_FakeVectorstore)

    assert docstore_persist_path(settings.chroma_dir).exists()
    assert load_docstore(settings.chroma_dir) is not None

    results = provider.retrieve("epsilon retrieval", k=3)
    assert results, "hierarchical retrieval returned nothing"


def test_chroma_retriever_enhanced_tool_uses_provider_path(monkeypatch, isolated_settings):
    settings = isolated_settings(hybrid_retrieval_enabled=True)

    def fake_load_and_split(*_args, **_kwargs):
        return [Document(page_content="chunk one"), Document(page_content="chunk two")]

    monkeypatch.setattr(chroma_module, "load_and_split_documents", fake_load_and_split)
    captured: dict[str, object] = {}

    def fake_create_retriever_tool(retriever, name, description):
        captured["retriever"] = retriever
        captured["name"] = name
        return "tool"

    monkeypatch.setattr(chroma_module, "create_retriever_tool", fake_create_retriever_tool)

    provider = ChromaRetriever(settings, chroma_cls=_FakeVectorstore)
    assert provider.as_tool() == "tool"
    assert captured["name"] == "retrieve_source_documents"
    # The facade routes through provider.retrieve, not the raw vectorstore.
    facade = captured["retriever"]
    assert facade._get_relevant_documents("chunk")


def test_chroma_retriever_legacy_path_unchanged_without_flags(monkeypatch, isolated_settings):
    settings = isolated_settings()

    def fake_load_and_split(*_args, **_kwargs):
        return [Document(page_content="legacy chunk")]

    monkeypatch.setattr(chroma_module, "load_and_split_documents", fake_load_and_split)

    provider = ChromaRetriever(settings, chroma_cls=_FakeVectorstore)

    assert provider._corpus_nodes is None
    assert provider.retrieve("legacy", k=1) == [Document(page_content="legacy chunk")]
