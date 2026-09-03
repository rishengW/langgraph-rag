from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._files import FileAccessError, resolve_safe_path

if TYPE_CHECKING:
    from src.config import Settings

_SUFFIXES = (".txt", ".md", ".log", ".csv")
# Cap returned characters so a large file does not blow up the LLM context.
_MAX_CHARS = 20_000


class TextFileInput(BaseModel):
    """Input schema for the plain-text file reader."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .txt, .md, .log, or .csv file, relative to the "
            "configured file-read root directory."
        ),
    )
    max_chars: int = Field(
        default=_MAX_CHARS,
        ge=1,
        le=200_000,
        description="Maximum characters of file content to return.",
    )


def build_text_file_tool(
    settings: Settings,
) -> BaseTool:
    """Create a plain-text file reader confined to the file-read root."""

    root = Path(settings.file_read_root)
    max_bytes = settings.file_read_max_bytes

    def _run_text_file(path: str, max_chars: int = _MAX_CHARS) -> str:
        return read_text_file(
            path,
            root=root,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    return StructuredTool.from_function(
        func=_run_text_file,
        name="read_text_file",
        description=(
            "Read the contents of a plain-text file (.txt, .md, .log, .csv) "
            "from the local document directory. Use when the user references a "
            "text file by name and wants its contents read, summarized, or "
            "searched. Returns the file text (possibly truncated)."
        ),
        args_schema=TextFileInput,
    )


def read_text_file(
    path: str,
    *,
    root: Path,
    max_bytes: int,
    max_chars: int = _MAX_CHARS,
) -> str:
    try:
        resolved = resolve_safe_path(
            path,
            root=root,
            expected_suffixes=_SUFFIXES,
            max_bytes=max_bytes,
        )
    except FileAccessError as exc:
        return f"Could not read text file: {exc}"

    try:
        from ._file_cache import PARSED_FILE_CACHE

        # Cache the complete decoded text; callers can request different
        # truncation limits without rereading the file.
        text = PARSED_FILE_CACHE.get_or_compute(
            resolved,
            parser_key="utf8-text-v1",
            loader=lambda: resolved.read_text(encoding="utf-8", errors="replace"),
        )
    except OSError as exc:
        return f"Could not read text file {resolved.name!r}: {exc}"

    limit = max(1, int(max_chars))
    truncated = len(text) > limit
    body = text[:limit]
    header = f"Contents of {resolved.name} ({len(text):,} chars):"
    if truncated:
        header = (
            f"Contents of {resolved.name} (showing first {limit:,} of "
            f"{len(text):,} chars):"
        )
    return f"{header}\n\n{body}"


__all__ = [
    "TextFileInput",
    "build_text_file_tool",
    "read_text_file",
]
