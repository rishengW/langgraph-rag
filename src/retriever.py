\
from __future__ import annotations

import shutil
from pathlib import Path

from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings


def _persisted_chroma_exists(chroma_dir: Path) -> bool:
    return chroma_dir.exists() and any(chroma_dir.iterdir())


def build_retriever(settings: Settings, rebuild: bool = False):
    """Build or load the Chroma retriever used by the RAG tool."""

    if rebuild and settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    if _persisted_chroma_exists(settings.chroma_dir):
        vectorstore = Chroma(
            collection_name=settings.collection_name,
            persist_directory=str(settings.chroma_dir),
            embedding_function=embeddings,
        )
        return vectorstore.as_retriever()

    print("---LOAD WEB DOCUMENTS---")
    docs_nested = [WebBaseLoader(url).load() for url in settings.source_urls]
    docs = [doc for sublist in docs_nested for doc in sublist]

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

    return vectorstore.as_retriever()


def build_retriever_tool(settings: Settings, rebuild: bool = False):
    """Create the LangChain retriever tool used by the LangGraph ToolNode."""

    retriever = build_retriever(settings=settings, rebuild=rebuild)

    return create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        (
            "Search and return information about Lilian Weng blog posts on "
            "LLM agents, prompt engineering, and adversarial attacks on LLMs."
        ),
    )
