"""LaTeX (.tex) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

LATEX_CONFIG = SourceEditConfig(
    label="LaTeX",
    suffixes=(".tex",),
    creation_suffix=".tex",
    tool_prefix="latex",
    mime_type="text/x-tex",
    flag_name="latex_edit_enabled",
    temp_prefix=".latex_write-",
)


def build_latex_file_tool(settings: Settings) -> BaseTool:
    """Create a LaTeX .tex file reader confined to the file-read root."""

    return build_source_file_tool(settings, LATEX_CONFIG)


__all__ = [
    "LATEX_CONFIG",
    "build_latex_file_tool",
]
