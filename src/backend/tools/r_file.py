"""R (.r) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

R_CONFIG = SourceEditConfig(
    label="R",
    suffixes=(".r",),
    creation_suffix=".r",
    tool_prefix="r",
    mime_type="text/x-r",
    flag_name="r_edit_enabled",
    temp_prefix=".r_write-",
)


def build_r_file_tool(settings: Settings) -> BaseTool:
    """Create a R .r file reader confined to the file-read root."""

    return build_source_file_tool(settings, R_CONFIG)


__all__ = [
    "R_CONFIG",
    "build_r_file_tool",
]
