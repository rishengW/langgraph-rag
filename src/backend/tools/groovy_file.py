"""Groovy (.groovy) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

GROOVY_CONFIG = SourceEditConfig(
    label="Groovy",
    suffixes=(".groovy",),
    creation_suffix=".groovy",
    tool_prefix="groovy",
    mime_type="text/x-groovy",
    flag_name="groovy_edit_enabled",
    temp_prefix=".groovy_write-",
)


def build_groovy_file_tool(settings: Settings) -> BaseTool:
    """Create a Groovy .groovy file reader confined to the file-read root."""

    return build_source_file_tool(settings, GROOVY_CONFIG)


__all__ = [
    "GROOVY_CONFIG",
    "build_groovy_file_tool",
]
