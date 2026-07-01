from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._files import FileAccessError, resolve_safe_path

if TYPE_CHECKING:
    from ..config import Settings

_SUFFIXES = (".docx",)
_MAX_CHARS = 20_000

# WordprocessingML namespace; paragraphs are <w:p>, text runs are <w:t>.
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_DOCUMENT_PART = "word/document.xml"


class WordFileInput(BaseModel):
    """Input schema for the Word (.docx) reader."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .docx file, relative to the configured file-read root "
            "directory. Legacy .doc (binary) is not supported."
        ),
    )
    max_chars: int = Field(
        default=_MAX_CHARS,
        ge=1,
        le=200_000,
        description="Maximum characters of extracted text to return.",
    )


def build_word_tool(
    settings: Settings,
) -> BaseTool:
    """Create a Word .docx text-extraction tool confined to the file-read root."""

    root = Path(settings.file_read_root)
    max_bytes = settings.file_read_max_bytes

    def _run_word(path: str, max_chars: int = _MAX_CHARS) -> str:
        return read_word_document(
            path,
            root=root,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    return StructuredTool.from_function(
        func=_run_word,
        name="read_word_document",
        description=(
            "Extract and read the text of a Microsoft Word .docx document from "
            "the local document directory. Use when the user references a Word "
            "document and wants its text read, summarized, or searched. Returns "
            "the document's paragraphs as plain text. Legacy .doc files are not "
            "supported."
        ),
        args_schema=WordFileInput,
    )


def read_word_document(
    path: str,
    *,
    root: Path,
    max_bytes: int,
    max_chars: int = _MAX_CHARS,
) -> str:
    text = (path or "").strip().strip('"').strip("'")
    if text.lower().endswith(".doc"):
        return (
            "Legacy .doc (binary Word) files are not supported. Please convert "
            "the file to .docx and try again."
        )

    try:
        resolved = resolve_safe_path(
            path,
            root=root,
            expected_suffixes=_SUFFIXES,
            max_bytes=max_bytes,
        )
    except FileAccessError as exc:
        return f"Could not read Word document: {exc}"

    try:
        paragraphs = _extract_docx_paragraphs(resolved)
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        return f"Could not parse Word document {resolved.name!r}: {exc}"
    except OSError as exc:
        return f"Could not read Word document {resolved.name!r}: {exc}"

    if not paragraphs:
        return f"{resolved.name} contains no readable text."

    full_text = "\n".join(paragraphs)
    limit = max(1, int(max_chars))
    truncated = len(full_text) > limit
    body = full_text[:limit]
    if truncated:
        header = (
            f"Text of {resolved.name} (showing first {limit:,} of "
            f"{len(full_text):,} chars, {len(paragraphs)} paragraphs):"
        )
    else:
        header = f"Text of {resolved.name} ({len(paragraphs)} paragraphs):"
    return f"{header}\n\n{body}"


def _extract_docx_paragraphs(path: Path) -> list[str]:
    """Extract paragraph text from a .docx without external dependencies."""

    with zipfile.ZipFile(path) as archive, archive.open(_DOCUMENT_PART) as document:
        tree = ET.parse(document)

    paragraphs: list[str] = []
    for paragraph in tree.iter(f"{{{_W_NS}}}p"):
        texts = [
            node.text
            for node in paragraph.iter(f"{{{_W_NS}}}t")
            if node.text
        ]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)
    return paragraphs


__all__ = [
    "WordFileInput",
    "build_word_tool",
    "read_word_document",
]
