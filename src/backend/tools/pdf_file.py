from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._files import FileAccessError, resolve_safe_path

if TYPE_CHECKING:
    from src.config import Settings

_SUFFIXES = (".pdf",)
_MAX_CHARS = 20_000
# Bound how many pages we extract so a huge PDF cannot blow up the context.
_MAX_PAGES = 50


class PdfFileInput(BaseModel):
    """Input schema for the PDF reader."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .pdf file, relative to the configured file-read root "
            "directory."
        ),
    )
    max_chars: int = Field(
        default=_MAX_CHARS,
        ge=1,
        le=200_000,
        description="Maximum characters of extracted text to return.",
    )
    max_pages: int = Field(
        default=_MAX_PAGES,
        ge=1,
        le=500,
        description="Maximum number of pages to extract text from.",
    )


def build_pdf_tool(
    settings: Settings,
) -> BaseTool:
    """Create a PDF text-extraction tool confined to the file-read root."""

    root = Path(settings.file_read_root)
    max_bytes = settings.file_read_max_bytes

    def _run_pdf(
        path: str,
        max_chars: int = _MAX_CHARS,
        max_pages: int = _MAX_PAGES,
    ) -> str:
        return read_pdf(
            path,
            root=root,
            max_bytes=max_bytes,
            max_chars=max_chars,
            max_pages=max_pages,
        )

    return StructuredTool.from_function(
        func=_run_pdf,
        name="read_pdf",
        description=(
            "Extract and read the text of a PDF document from the local "
            "document directory. Use when the user references a PDF and wants "
            "its text read, summarized, or searched. Returns the document's "
            "extracted text (possibly truncated). Scanned/image-only PDFs "
            "without a text layer cannot be read."
        ),
        args_schema=PdfFileInput,
    )


def read_pdf(
    path: str,
    *,
    root: Path,
    max_bytes: int,
    max_chars: int = _MAX_CHARS,
    max_pages: int = _MAX_PAGES,
) -> str:
    try:
        resolved = resolve_safe_path(
            path,
            root=root,
            expected_suffixes=_SUFFIXES,
            max_bytes=max_bytes,
        )
    except FileAccessError as exc:
        return f"Could not read PDF: {exc}"

    try:
        import pypdf
    except ImportError as exc:
        return (
            "pypdf is not installed. Run `python -m pip install pypdf` to "
            f"enable the PDF tool: {exc}"
        )

    page_limit = max(1, int(max_pages))
    try:
        from ._file_cache import PARSED_FILE_CACHE

        page_texts, total_pages = PARSED_FILE_CACHE.get_or_compute(
            resolved,
            parser_key=f"pdf-text-v1:pages={page_limit}",
            loader=lambda: _extract_pdf_text(pypdf, resolved, page_limit),
        )
    except Exception as exc:
        return f"Could not parse PDF {resolved.name!r}: {exc}"

    full_text = "\n\n".join(page_texts).strip()
    if not full_text:
        return (
            f"{resolved.name} has {total_pages} page(s) but no extractable text "
            "(it may be a scanned/image-only PDF)."
        )

    limit = max(1, int(max_chars))
    truncated = len(full_text) > limit
    body = full_text[:limit]
    read_pages = len(page_texts)
    if truncated:
        header = (
            f"Text of {resolved.name} (showing first {limit:,} of "
            f"{len(full_text):,} chars, read {read_pages} of {total_pages} pages):"
        )
    else:
        header = (
            f"Text of {resolved.name} (read {read_pages} of {total_pages} pages):"
        )
    return f"{header}\n\n{body}"


def _extract_pdf_text(pypdf: Any, path: Path, max_pages: int) -> tuple[list[str], int]:
    reader = pypdf.PdfReader(str(path))
    total_pages = len(reader.pages)
    limit = min(max(1, int(max_pages)), total_pages)
    texts: list[str] = []
    for index in range(limit):
        page = reader.pages[index]
        extracted = (page.extract_text() or "").strip()
        if extracted:
            texts.append(extracted)
    return texts, total_pages


__all__ = [
    "PdfFileInput",
    "build_pdf_tool",
    "read_pdf",
]
