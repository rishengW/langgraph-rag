from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._files import FileAccessError, resolve_safe_path

if TYPE_CHECKING:
    from src.config import Settings

_SUFFIXES = (".toml",)
# Cap returned characters so a large file does not blow up the LLM context.
_MAX_CHARS = 20_000


class TomlFileInput(BaseModel):
    """Input schema for the TOML file reader."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .toml file, relative to the configured file-read "
            "root directory. This also covers files uploaded to the "
            "current chat session (under chat_uploads/<thread-id>/)."
        ),
    )
    max_chars: int = Field(
        default=_MAX_CHARS,
        ge=1,
        le=200_000,
        description="Maximum characters of file content to return.",
    )


def build_toml_file_tool(
    settings: Settings,
) -> BaseTool:
    """Create a TOML file reader confined to the file-read root."""

    root = Path(settings.file_read_root)
    max_bytes = settings.file_read_max_bytes

    def _run_toml_file(path: str, max_chars: int = _MAX_CHARS) -> str:
        return read_toml_file(
            path,
            root=root,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    return StructuredTool.from_function(
        func=_run_toml_file,
        name="read_toml_file",
        description=(
            "Read the raw contents of a .toml file from the local "
            "document directory (including files uploaded to the current "
            "chat session). Use when the user references a TOML file by "
            "name and wants its contents read, summarized, or searched. "
            "Returns the raw text (possibly truncated)."
        ),
        args_schema=TomlFileInput,
    )


def read_toml_file(
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
        return f"Could not read TOML file: {exc}"

    try:
        from ._file_cache import PARSED_FILE_CACHE

        # TOML shares the UTF-8 text parser key, so identical uploaded
        # content is decoded only once even when reached through either tool.
        text = PARSED_FILE_CACHE.get_or_compute(
            resolved,
            parser_key="utf8-text-v1",
            loader=lambda: resolved.read_text(encoding="utf-8", errors="replace"),
        )
    except OSError as exc:
        return f"Could not read TOML file {resolved.name!r}: {exc}"

    limit = max(1, int(max_chars))
    truncated = len(text) > limit
    body = text[:limit]
    header = f"Contents of {resolved.name} ({len(text):,} chars):"
    if truncated:
        header = f"Contents of {resolved.name} (showing first {limit:,} of {len(text):,} chars):"
    return f"{header}\n\n{body}"


__all__ = [
    "TomlFileInput",
    "build_toml_file_tool",
    "read_toml_file",
]
