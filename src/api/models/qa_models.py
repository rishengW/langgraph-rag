"""Pydantic models for the single-shot QA API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    question: str = Field(..., min_length=1, description="The question to ask")
    urls: Optional[str | list[str]] = Field(
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

    answer: Optional[str] = Field(None, description="The generated answer")
    error: Optional[str] = Field(None, description="Error message if query failed")
    success: bool = Field(True, description="Whether the query was successful")
    messages: Optional[list[str]] = Field(None, description="Intermediate messages (debug)")
    source_urls: Optional[list[str]] = Field(None, description="URLs used for retrieval")
    source_mode: Optional[str] = Field(None, description="Source mode used (explicit, web_search, defaults)")
    source_note: Optional[str] = Field(None, description="Additional detail about source selection")


__all__ = ["QueryRequest", "QueryResponse"]

