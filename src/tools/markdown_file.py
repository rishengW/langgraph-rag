from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._files import FileAccessError, resolve_safe_path

if TYPE_CHECKING:
    from ..config import Settings

_SUFFIXES = (".md",)
# Cap returned characters so a large file does not blow up the LLM context.
_MAX_CHARS = 20_000


class MarkdownFileInput(BaseModel):
    """Input schema for the Markdown file reader."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .md Markdown file, relative to the configured "
            "file-read root directory. This also covers .md files uploaded "
            "to the current chat session (under chat_uploads/<thread-id>/)."
        ),
    )
    max_chars: int = Field(
        default=_MAX_CHARS,
        ge=1,
        le=200_000,
        description="Maximum characters of file content to return.",
    )


def build_markdown_file_tool(
    settings: Settings,
) -> BaseTool:
    """Create a Markdown file reader confined to the file-read root."""

    root = Path(settings.file_read_root)
    max_bytes = settings.file_read_max_bytes

    def _run_markdown_file(path: str, max_chars: int = _MAX_CHARS) -> str:
        return read_markdown_file(
            path,
            root=root,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    return StructuredTool.from_function(
        func=_run_markdown_file,
        name="read_markdown_file",
        description=(
            "Read the contents of a Markdown .md file from the local document "
            "directory (including files uploaded to the current chat session). "
            "Use when the user references a .md file by name and wants its "
            "contents read, summarized, or searched. Returns the raw Markdown "
            "text (possibly truncated)."
        ),
        args_schema=MarkdownFileInput,
    )


def read_markdown_file(
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
        return f"Could not read Markdown file: {exc}"

    try:
        # errors="replace" keeps the tool resilient to mixed/unknown encodings
        # instead of raising on a single bad byte.
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Could not read Markdown file {resolved.name!r}: {exc}"

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
    "MarkdownFileInput",
    "build_markdown_file_tool",
    "read_markdown_file",
]
