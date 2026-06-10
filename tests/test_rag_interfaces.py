from __future__ import annotations

from datetime import date, timedelta
import sys
import threading
import time
from types import ModuleType

import pytest
from langchain_core.documents import Document

from src.rag import chroma_retriever as chroma_module
from src.rag.chroma_retriever import ChromaRetriever, build_retriever, build_retriever_tool
from src.rag.dashscope_embeddings import DashScopeEmbeddings
from src.rag.document_loader import (
    SourceDocumentCache,
    load_and_split_documents,
    load_source_documents,
)
from src.rag.document_quality import DocumentQualityConfig, filter_quality_documents
from src.rag.embeddings import EmbeddingModel, is_dashscope_embedding_model
from src.rag.hf_embeddings import HuggingFaceEmbeddingModel
from src.rag.retriever import Retriever


def test_dashscope_embedding_model_satisfies_protocol(monkeypatch):
    dashscope_module = ModuleType("dashscope")
    dashscope_module.base_http_api_url = ""
    monkeypatch.setitem(sys.modules, "dashscope", dashscope_module)

    embeddings = DashScopeEmbeddings(
        api_key="test-key",
        model="text-embedding-v4",
        dimension=1024,
        batch_size=10,
        request_timeout=30,
        max_retries=1,
    )

    assert isinstance(embeddings, EmbeddingModel)
    assert embeddings.model_name == "text-embedding-v4"
    assert is_dashscope_embedding_model(" text-embedding-v4 ")
    assert not is_dashscope_embedding_model("sentence-transformers/all-MiniLM-L6-v2")


def test_huggingface_adapter_satisfies_protocol_and_delegates(monkeypatch):
    parent_module = ModuleType("langchain_huggingface")
    embeddings_module = ModuleType("langchain_huggingface.embeddings")

    class FakeHuggingFaceEmbeddings:
        def __init__(self, model_name: str, **kwargs):
            self.model_name = model_name
            self.kwargs = kwargs

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text))] for text in texts]

        def embed_query(self, text: str) -> list[float]:
            return [float(len(text))]

    embeddings_module.HuggingFaceEmbeddings = FakeHuggingFaceEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_huggingface", parent_module)
    monkeypatch.setitem(sys.modules, "langchain_huggingface.embeddings", embeddings_module)

    embeddings = HuggingFaceEmbeddingModel("local-model", device="cpu")

    assert isinstance(embeddings, EmbeddingModel)
    assert embeddings.model_name == "local-model"
    assert embeddings.embed_documents(["a", "abcd"]) == [[1.0], [4.0]]
    assert embeddings.embed_query("abc") == [3.0]


def test_load_and_split_documents_skips_failed_sources():
    calls: list[tuple[str, int]] = []
    good_text = (
        "This source document has useful article content about retrieval, "
        "indexing, embeddings, and source loading. It is long enough to pass "
        "the conservative pre-index quality filter."
    )

    class FakeLoader:
        def __init__(self, url: str):
            self.url = url

        def load(self) -> list[Document]:
            if self.url.endswith("/bad"):
                raise RuntimeError("unreachable")
            return [Document(page_content=f"{good_text} URL: {self.url}")]

    class FakeSplitter:
        def split_documents(self, documents: list[Document]) -> list[Document]:
            return [
                Document(page_content=f"chunk:{document.page_content}")
                for document in documents
            ]

    def loader_factory(url: str, timeout: int) -> FakeLoader:
        calls.append((url, timeout))
        return FakeLoader(url)

    def splitter_factory(chunk_size: int, chunk_overlap: int) -> FakeSplitter:
        assert chunk_size == 10
        assert chunk_overlap == 2
        return FakeSplitter()

    chunks = load_and_split_documents(
        ["https://example.test/good", "https://example.test/bad"],
        page_load_timeout=0,
        chunk_size=10,
        chunk_overlap=2,
        loader_factory=loader_factory,
        splitter_factory=splitter_factory,
    )

    assert sorted(calls) == [
        ("https://example.test/bad", 1),
        ("https://example.test/good", 1),
    ]
    assert [document.page_content for document in chunks] == [
        f"chunk:{good_text} URL: https://example.test/good"
    ]


def test_document_quality_filter_keeps_good_documents():
    document = Document(
        page_content=(
            "LangGraph retrieval systems load source documents before splitting "
            "them into chunks for embeddings. Useful documents contain enough "
            "specific terms for downstream indexing and question answering."
        )
    )

    kept = filter_quality_documents([document])

    assert kept == [document]


def test_document_quality_filter_skips_normalized_duplicate_documents():
    text = (
        "LangGraph retrieval systems load source documents before splitting "
        "them into chunks for embeddings. Useful documents contain enough "
        "specific terms for downstream indexing and question answering."
    )
    original = Document(page_content=text)
    duplicate = Document(page_content=f"  {text.upper()}  ")

    kept = filter_quality_documents([original, duplicate])

    assert kept == [original]


def test_load_and_split_documents_raises_when_all_documents_filtered():
    class EmptyLoader:
        def load(self) -> list[Document]:
            return [
                Document(page_content=""),
                Document(page_content="cookie privacy terms login menu " * 5),
            ]

    with pytest.raises(
        RuntimeError,
        match="All loaded source documents were filtered out before indexing",
    ):
        load_and_split_documents(
            ["https://example.test/boilerplate"],
            page_load_timeout=5,
            chunk_size=10,
            chunk_overlap=2,
            loader_factory=lambda _url, _timeout: EmptyLoader(),
        )


def test_document_quality_filter_can_apply_query_overlap():
    config = DocumentQualityConfig(relevance_query="langgraph embeddings retrieval")
    relevant = Document(
        page_content=(
            "A detailed LangGraph retrieval article explains embeddings, indexes, "
            "source loading, and document splitting for RAG applications."
        )
    )
    irrelevant = Document(
        page_content=(
            "A detailed gardening article explains compost, irrigation, seedlings, "
            "soil preparation, seasonal pruning, and greenhouse planning."
        )
    )

    kept = filter_quality_documents([relevant, irrelevant], config)

    assert kept == [relevant]


def test_document_quality_filter_uses_embedding_similarity_gate():
    class FakeEmbeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [self.embed_query(text) for text in texts]

        def embed_query(self, text: str) -> list[float]:
            lowered = text.lower()
            if lowered == "langgraph retrieval":
                return [1.0, 0.0]
            if "state graph orchestration" in lowered:
                return [0.92, 0.08]
            return [0.0, 1.0]

    config = DocumentQualityConfig(
        relevance_query="langgraph retrieval",
        min_similarity=0.8,
    )
    relevant = Document(
        page_content=(
            "State graph orchestration coordinates retrieval workflows with "
            "source indexing, embeddings, chunk storage, and grounded answers."
        )
    )
    irrelevant = Document(
        page_content=(
            "Seasonal greenhouse planning explains compost, irrigation, seed "
            "rotation, pruning schedules, and vegetable harvest preparation."
        )
    )

    kept = filter_quality_documents([relevant, irrelevant], config, embeddings=FakeEmbeddings())

    assert kept == [relevant]


def test_document_quality_filter_biases_recent_publication_dates():
    recent_date = date.today() - timedelta(days=30)
    old_date = date.today() - timedelta(days=800)
    old = Document(
        page_content=(
            "Older LangGraph retrieval notes describe source document loading, "
            "embedding indexes, retriever tools, chunk storage, and answers."
        ),
        metadata={"publication_date": old_date.isoformat()},
    )
    recent = Document(
        page_content=(
            "Recent LangGraph retrieval notes describe source document loading, "
            "embedding indexes, retriever tools, chunk storage, and answers."
        ),
        metadata={"publication_date": recent_date.isoformat()},
    )

    kept = filter_quality_documents([old, recent])

    assert kept == [recent, old]
    assert recent.metadata["document_quality_recency_score"] > 0
    assert old.metadata["document_quality_recency_score"] == 0


def test_document_quality_filter_extracts_publication_date_from_html():
    document = Document(
        page_content=(
            "<html><head><meta property='article:published_time' "
            "content='2026-02-03T10:15:00Z'></head><body>"
            "LangGraph retrieval systems load source documents before splitting "
            "them into chunks for embeddings and grounded question answering."
            "</body></html>"
        )
    )

    kept = filter_quality_documents([document])

    assert kept == [document]
    assert document.metadata["publication_date"] == "2026-02-03"


def test_load_source_documents_raises_when_all_sources_fail():
    class FailingLoader:
        def load(self) -> list[Document]:
            raise RuntimeError("offline")

    with pytest.raises(
        RuntimeError,
        match="No source documents could be loaded. Failed URLs: https://example.test/bad",
    ):
        load_source_documents(
            ["https://example.test/bad"],
            page_load_timeout=15,
            loader_factory=lambda _url, _timeout: FailingLoader(),
        )


def test_load_source_documents_preserves_url_order_after_concurrent_loads():
    delays = {
        "https://example.test/slow": 0.05,
        "https://example.test/fast": 0.0,
        "https://example.test/bad": 0.01,
        "https://example.test/medium": 0.02,
    }

    class TimedLoader:
        def __init__(self, url: str) -> None:
            self.url = url

        def load(self) -> list[Document]:
            time.sleep(delays[self.url])
            if self.url.endswith("/bad"):
                raise RuntimeError("offline")
            return [Document(page_content=f"loaded:{self.url}")]

    docs = load_source_documents(
        list(delays),
        page_load_timeout=5,
        max_concurrent_loads=4,
        loader_factory=lambda url, _timeout: TimedLoader(url),
    )

    assert [document.page_content for document in docs] == [
        "loaded:https://example.test/slow",
        "loaded:https://example.test/fast",
        "loaded:https://example.test/medium",
    ]


def test_load_source_documents_bounds_concurrent_url_loads():
    lock = threading.Lock()
    release_gate = threading.Event()
    active_loads = 0
    peak_loads = 0

    class BlockingLoader:
        def load(self) -> list[Document]:
            nonlocal active_loads, peak_loads
            with lock:
                active_loads += 1
                peak_loads = max(peak_loads, active_loads)
                if active_loads == 2:
                    release_gate.set()
            release_gate.wait(timeout=0.2)
            time.sleep(0.01)
            with lock:
                active_loads -= 1
            return [Document(page_content="loaded")]

    docs = load_source_documents(
        [f"https://example.test/{index}" for index in range(6)],
        page_load_timeout=5,
        max_concurrent_loads=2,
        loader_factory=lambda _url, _timeout: BlockingLoader(),
    )

    assert len(docs) == 6
    assert peak_loads == 2


def test_load_source_documents_uses_cache_for_repeated_url_loads():
    calls = 0
    cache = SourceDocumentCache()

    class CountingLoader:
        def load(self) -> list[Document]:
            nonlocal calls
            calls += 1
            return [
                Document(
                    page_content=f"loaded:{calls}",
                    metadata={"call": calls},
                )
            ]

    first_docs = load_source_documents(
        ["https://example.test/cache"],
        page_load_timeout=5,
        page_load_cache_ttl_seconds=30,
        document_cache=cache,
        loader_factory=lambda _url, _timeout: CountingLoader(),
    )
    first_docs[0].metadata["mutated"] = True

    second_docs = load_source_documents(
        ["https://example.test/cache"],
        page_load_timeout=5,
        page_load_cache_ttl_seconds=30,
        document_cache=cache,
        loader_factory=lambda _url, _timeout: CountingLoader(),
    )

    assert calls == 1
    assert [document.page_content for document in second_docs] == ["loaded:1"]
    assert second_docs[0].metadata == {"call": 1}


def test_load_source_documents_deduplicates_concurrent_cache_misses():
    calls = 0
    cache = SourceDocumentCache()

    class SlowLoader:
        def load(self) -> list[Document]:
            nonlocal calls
            calls += 1
            call_number = calls
            time.sleep(0.02)
            return [Document(page_content=f"loaded:{call_number}")]

    docs = load_source_documents(
        ["https://example.test/cache", "https://example.test/cache"],
        page_load_timeout=5,
        max_concurrent_loads=2,
        page_load_cache_ttl_seconds=30,
        document_cache=cache,
        loader_factory=lambda _url, _timeout: SlowLoader(),
    )

    assert calls == 1
    assert [document.page_content for document in docs] == [
        "loaded:1",
        "loaded:1",
    ]


def test_load_source_documents_cache_ttl_expiry_refetches():
    calls = 0
    now = 100.0
    cache = SourceDocumentCache(clock=lambda: now)

    class CountingLoader:
        def load(self) -> list[Document]:
            nonlocal calls
            calls += 1
            return [Document(page_content=f"loaded:{calls}")]

    load_source_documents(
        ["https://example.test/cache"],
        page_load_timeout=5,
        page_load_cache_ttl_seconds=10,
        document_cache=cache,
        loader_factory=lambda _url, _timeout: CountingLoader(),
    )
    now = 111.0
    docs = load_source_documents(
        ["https://example.test/cache"],
        page_load_timeout=5,
        page_load_cache_ttl_seconds=10,
        document_cache=cache,
        loader_factory=lambda _url, _timeout: CountingLoader(),
    )

    assert calls == 2
    assert [document.page_content for document in docs] == ["loaded:2"]


def test_load_source_documents_cache_ttl_zero_disables_cache():
    calls = 0
    cache = SourceDocumentCache()

    class CountingLoader:
        def load(self) -> list[Document]:
            nonlocal calls
            calls += 1
            return [Document(page_content=f"loaded:{calls}")]

    for _ in range(2):
        load_source_documents(
            ["https://example.test/cache"],
            page_load_timeout=5,
            page_load_cache_ttl_seconds=0,
            document_cache=cache,
            loader_factory=lambda _url, _timeout: CountingLoader(),
        )

    assert calls == 2


def test_load_source_documents_does_not_cache_failed_loads():
    calls = 0
    cache = SourceDocumentCache()

    class FlakyLoader:
        def load(self) -> list[Document]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("offline")
            return [Document(page_content="loaded")]

    with pytest.raises(RuntimeError, match="No source documents could be loaded"):
        load_source_documents(
            ["https://example.test/cache"],
            page_load_timeout=5,
            page_load_cache_ttl_seconds=30,
            document_cache=cache,
            loader_factory=lambda _url, _timeout: FlakyLoader(),
        )

    docs = load_source_documents(
        ["https://example.test/cache"],
        page_load_timeout=5,
        page_load_cache_ttl_seconds=30,
        document_cache=cache,
        loader_factory=lambda _url, _timeout: FlakyLoader(),
    )

    assert calls == 2
    assert [document.page_content for document in docs] == ["loaded"]


def test_chroma_retriever_satisfies_protocol_and_rebuilds(monkeypatch, mock_settings):
    builds: list[tuple[bool, list[str]]] = []

    class FakeLangChainRetriever:
        def __init__(self) -> None:
            self.search_kwargs = {}
            self.calls: list[tuple[str, dict[str, int]]] = []

        def invoke(self, query: str) -> list[Document]:
            self.calls.append((query, dict(self.search_kwargs)))
            return [Document(page_content=f"result:{query}")]

    fake_retriever = FakeLangChainRetriever()

    def fake_build(self: ChromaRetriever, *, rebuild: bool = False):
        builds.append((rebuild, list(self.settings.source_urls)))
        return fake_retriever

    monkeypatch.setattr(ChromaRetriever, "_build_langchain_retriever", fake_build)

    provider = ChromaRetriever(mock_settings)

    assert isinstance(provider, Retriever)
    assert build_retriever(mock_settings) is fake_retriever
    assert provider.retrieve("needle", k=7) == [Document(page_content="result:needle")]
    assert fake_retriever.calls == [("needle", {"k": 7})]
    assert fake_retriever.search_kwargs == {}

    provider.rebuild(["https://example.test/new"])
    assert builds[-1] == (True, ["https://example.test/new"])


def test_chroma_retriever_tool_uses_legacy_name_and_description(monkeypatch, mock_settings):
    class FakeLangChainRetriever:
        pass

    fake_retriever = FakeLangChainRetriever()
    created: dict[str, object] = {}

    def fake_build(self: ChromaRetriever, *, rebuild: bool = False):
        return fake_retriever

    def fake_create_retriever_tool(retriever, name: str, description: str):
        created.update(
            retriever=retriever,
            name=name,
            description=description,
        )
        return "tool"

    monkeypatch.setattr(ChromaRetriever, "_build_langchain_retriever", fake_build)
    monkeypatch.setattr(chroma_module, "create_retriever_tool", fake_create_retriever_tool)

    tool = ChromaRetriever(mock_settings).as_tool()

    assert tool == "tool"
    assert created["retriever"] is fake_retriever
    assert created["name"] == "retrieve_source_documents"
    assert "1 URL(s)" in str(created["description"])


def test_release_chroma_system_handles_cache_without_refcount_lock(monkeypatch, tmp_path):
    shared_module = ModuleType("chromadb.api.shared_system_client")
    stopped: list[str] = []
    chroma_dir = tmp_path / "chroma"

    class FakeSettings:
        persist_directory = str(chroma_dir)

    class FakeSystem:
        settings = FakeSettings()

        def stop(self) -> None:
            stopped.append("system")

    class FakeSharedSystemClient:
        _identifier_to_system = {
            str(chroma_dir): FakeSystem(),
        }

    shared_module.SharedSystemClient = FakeSharedSystemClient
    monkeypatch.setitem(sys.modules, "chromadb.api.shared_system_client", shared_module)

    chroma_module._release_chroma_system(chroma_dir)

    assert stopped == ["system"]
    assert FakeSharedSystemClient._identifier_to_system == {}


def test_chroma_retriever_rebuilds_incompatible_persisted_store(
    monkeypatch,
    isolated_settings,
    tmp_path,
):
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    (chroma_dir / "chroma.sqlite3").write_text("old schema", encoding="utf-8")
    (chroma_dir / "embedding_config.json").write_text(
        '{"embedding_dimension": 1024, "embedding_model": "text-embedding-v4"}',
        encoding="utf-8",
    )
    (chroma_dir / "chat").mkdir()
    (chroma_dir / "chat" / "sessions.sqlite3").write_text("keep", encoding="utf-8")

    settings = isolated_settings(
        chroma_dir=chroma_dir,
        page_load_max_concurrency=3,
    )
    loaded_documents: list[list[str]] = []
    loader_calls: list[dict[str, object]] = []
    embedding_model = object()

    class FreshVectorstore:
        def as_retriever(self):
            return "fresh-retriever"

    class IncompatibleChroma:
        def __init__(self, **kwargs):
            raise KeyError("_type")

        @classmethod
        def from_documents(cls, documents, **kwargs):
            loaded_documents.append([doc.page_content for doc in documents])
            return FreshVectorstore()

    def fake_load_and_split_documents(*args, **kwargs):
        loader_calls.append(dict(kwargs))
        return [Document(page_content="fresh document")]

    monkeypatch.setattr(
        chroma_module,
        "load_and_split_documents",
        fake_load_and_split_documents,
    )

    provider = ChromaRetriever(
        settings,
        embeddings=embedding_model,
        chroma_cls=IncompatibleChroma,
    )

    assert provider.as_langchain_retriever() == "fresh-retriever"
    assert loaded_documents == [["fresh document"]]
    assert loader_calls == [
        {
            "page_load_timeout": settings.page_load_timeout,
            "max_concurrent_loads": 3,
            "page_load_cache_ttl_seconds": settings.page_load_cache_ttl_seconds,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "quality_config": DocumentQualityConfig(
                enabled=settings.document_quality_filter_enabled,
                min_text_length=settings.document_quality_min_text_length,
                min_unique_terms=settings.document_quality_min_unique_terms,
                relevance_query=settings.document_quality_relevance_query,
                min_query_term_overlap=settings.document_quality_query_min_overlap,
                min_similarity=settings.document_quality_min_similarity,
                recency_bias_days=settings.document_quality_recency_bias_days,
            ),
            "embeddings": embedding_model,
        }
    ]
    assert not (chroma_dir / "chroma.sqlite3").exists()
    assert (chroma_dir / "chat" / "sessions.sqlite3").read_text(encoding="utf-8") == "keep"


def test_build_retriever_tool_accepts_retriever_protocol():
    class FakeProvider:
        def __init__(self) -> None:
            self.rebuild_calls = 0
            self.tool = object()

        def retrieve(self, query: str, k: int = 4) -> list[Document]:
            return [Document(page_content=query)]

        def as_tool(self):
            return self.tool

        def rebuild(self, urls: list[str] | None = None) -> None:
            self.rebuild_calls += 1

    provider = FakeProvider()

    assert build_retriever_tool(provider, rebuild=True) is provider.tool
    assert provider.rebuild_calls == 1
