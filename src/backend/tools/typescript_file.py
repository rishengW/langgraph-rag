from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._files import FileAccessError, resolve_safe_path

if TYPE_CHECKING:
    from src.config import Settings

_SUFFIXES = (".ts", ".tsx")
# Cap returned characters so a large file does not blow up the LLM context.
_MAX_CHARS = 20_000


class TypeScriptFileInput(BaseModel):
    """Input schema for the TypeScript file reader."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .ts or .tsx TypeScript file, relative to the "
            "configured file-read root directory. This also covers TypeScript "
            "files uploaded to the current chat session "
            "(under chat_uploads/<thread-id>/)."
        ),
    )
    max_chars: int = Field(
        default=_MAX_CHARS,
        ge=1,
        le=200_000,
        description="Maximum characters of file content to return.",
    )


def build_typescript_file_tool(
    settings: Settings,
) -> BaseTool:
    """Create a TypeScript file reader confined to the file-read root."""

    root = Path(settings.file_read_root)
    max_bytes = settings.file_read_max_bytes

    def _run_typescript_file(path: str, max_chars: int = _MAX_CHARS) -> str:
        return read_typescript_file(
            path,
            root=root,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    return StructuredTool.from_function(
        func=_run_typescript_file,
        name="read_typescript_file",
        description=(
            "Read the contents of a TypeScript .ts or .tsx file from the "
            "local document directory (including files uploaded to the "
            "current chat session). Use when the user references a TypeScript "
            "file by name and wants its contents read, summarized, or "
            "searched. Returns the raw source text (possibly truncated)."
        ),
        args_schema=TypeScriptFileInput,
    )


def read_typescript_file(
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
        return f"Could not read TypeScript file: {exc}"

    try:
        from ._file_cache import PARSED_FILE_CACHE

        # TypeScript and plain text share a parser key, so identical uploaded
        # content is decoded only once even when reached through either tool.
        text = PARSED_FILE_CACHE.get_or_compute(
            resolved,
            parser_key="utf8-text-v1",
            loader=lambda: resolved.read_text(encoding="utf-8", errors="replace"),
        )
    except OSError as exc:
        return f"Could not read TypeScript file {resolved.name!r}: {exc}"

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
    "TypeScriptFileInput",
    "build_typescript_file_tool",
    "read_typescript_file",
]
