"""Go (.go) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

GO_CONFIG = SourceEditConfig(
    label="Go",
    suffixes=(".go",),
    creation_suffix=".go",
    tool_prefix="go",
    mime_type="text/x-go",
    flag_name="go_edit_enabled",
    temp_prefix=".go_write-",
)


def build_go_file_tool(settings: Settings) -> BaseTool:
    """Create a Go .go file reader confined to the file-read root."""

    return build_source_file_tool(settings, GO_CONFIG)


__all__ = [
    "GO_CONFIG",
    "build_go_file_tool",
]
