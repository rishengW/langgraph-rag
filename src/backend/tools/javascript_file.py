"""JavaScript (.js, .mjs, .cjs) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

JAVASCRIPT_CONFIG = SourceEditConfig(
    label="JavaScript",
    suffixes=(".js", ".mjs", ".cjs"),
    creation_suffix=".js",
    tool_prefix="javascript",
    mime_type="text/javascript",
    flag_name="javascript_edit_enabled",
    temp_prefix=".javascript_write-",
)


def build_javascript_file_tool(settings: Settings) -> BaseTool:
    """Create a JavaScript file reader confined to the file-read root."""

    return build_source_file_tool(settings, JAVASCRIPT_CONFIG)


__all__ = [
    "JAVASCRIPT_CONFIG",
    "build_javascript_file_tool",
]
