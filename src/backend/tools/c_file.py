"""C (.c, .h) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

C_CONFIG = SourceEditConfig(
    label="C",
    suffixes=(".c", ".h"),
    creation_suffix=".c",
    tool_prefix="c",
    mime_type="text/x-c",
    flag_name="c_edit_enabled",
    temp_prefix=".c_write-",
)


def build_c_file_tool(settings: Settings) -> BaseTool:
    """Create a C .c/.h file reader confined to the file-read root."""

    return build_source_file_tool(settings, C_CONFIG)


__all__ = [
    "C_CONFIG",
    "build_c_file_tool",
]
