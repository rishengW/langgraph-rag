from __future__ import annotations

import sys
from types import ModuleType

import pytest
from langchain_core.documents import Document

from src.rag import chroma_retriever as chroma_module
from src.rag.chroma_retriever import ChromaRetriever, build_retriever, build_retriever_tool
from src.rag.dashscope_embeddings import DashScopeEmbeddings
from src.rag.document_loader import load_and_split_documents, load_source_documents
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

    class FakeLoader:
        def __init__(self, url: str):
            self.url = url

        def load(self) -> list[Document]:
            if self.url.endswith("/bad"):
                raise RuntimeError("unreachable")
            return [Document(page_content=f"loaded:{self.url}")]

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

    assert calls == [
        ("https://example.test/good", 1),
        ("https://example.test/bad", 1),
    ]
    assert [document.page_content for document in chunks] == [
        "chunk:loaded:https://example.test/good"
    ]


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
