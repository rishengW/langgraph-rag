from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.documents import Document
from langchain_core.tools import BaseTool


@runtime_checkable
class Retriever(Protocol):
    """Provider boundary for document retrieval implementations."""

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        """Return documents relevant to a query."""

    def as_tool(self) -> BaseTool:
        """Expose the retriever as a LangChain tool."""

    def rebuild(self, urls: list[str] | None = None) -> None:
        """Rebuild the backing index, optionally with replacement source URLs."""
