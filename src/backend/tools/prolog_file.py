"""Prolog (.pl) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

PROLOG_CONFIG = SourceEditConfig(
    label="Prolog",
    suffixes=(".pl",),
    creation_suffix=".pl",
    tool_prefix="prolog",
    mime_type="text/x-prolog",
    flag_name="prolog_edit_enabled",
    temp_prefix=".prolog_write-",
)


def build_prolog_file_tool(settings: Settings) -> BaseTool:
    """Create a Prolog .pl file reader confined to the file-read root."""

    return build_source_file_tool(settings, PROLOG_CONFIG)


__all__ = [
    "PROLOG_CONFIG",
    "build_prolog_file_tool",
]
