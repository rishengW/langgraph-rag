"""HTML (.html, .htm) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

HTML_CONFIG = SourceEditConfig(
    label="HTML",
    suffixes=(".html", ".htm"),
    creation_suffix=".html",
    tool_prefix="html",
    mime_type="text/html",
    flag_name="html_edit_enabled",
    temp_prefix=".html_write-",
)


def build_html_file_tool(settings: Settings) -> BaseTool:
    """Create an HTML .html/.htm file reader confined to the file-read root."""

    return build_source_file_tool(settings, HTML_CONFIG)


__all__ = [
    "HTML_CONFIG",
    "build_html_file_tool",
]
