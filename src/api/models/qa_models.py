"""Pydantic models for the single-shot QA API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    question: str = Field(..., min_length=1, description="The question to ask")
    urls: str | list[str] | None = Field(
        None,
        description=(
            "Comma-separated URLs or a URL list for RAG sources. "
            "If not provided or empty, web search can discover sources."
        ),
    )
    rebuild: bool = Field(
        False,
        description="Whether to rebuild the vector database",
    )
    web_search: bool = Field(
        True,
        description="If true and URLs are empty, discover source URLs from web search",
    )
    debug: bool = Field(
        False,
        description="If true, include intermediate messages for debugging",
    )


class QueryResponse(BaseModel):
    """Response model for RAG queries."""

    answer: str | None = Field(None, description="The generated answer")
    error: str | None = Field(None, description="Error message if query failed")
    success: bool = Field(True, description="Whether the query was successful")
    messages: list[str] | None = Field(None, description="Intermediate messages (debug)")
    source_urls: list[str] | None = Field(None, description="URLs used for retrieval")
    source_mode: str | None = Field(None, description="Source mode used (explicit, web_search, defaults)")
    source_note: str | None = Field(None, description="Additional detail about source selection")


__all__ = ["QueryRequest", "QueryResponse"]

