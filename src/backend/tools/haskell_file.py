"""Haskell (.hs) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

HASKELL_CONFIG = SourceEditConfig(
    label="Haskell",
    suffixes=(".hs",),
    creation_suffix=".hs",
    tool_prefix="haskell",
    mime_type="text/x-haskell",
    flag_name="haskell_edit_enabled",
    temp_prefix=".haskell_write-",
)


def build_haskell_file_tool(settings: Settings) -> BaseTool:
    """Create a Haskell .hs file reader confined to the file-read root."""

    return build_source_file_tool(settings, HASKELL_CONFIG)


__all__ = [
    "HASKELL_CONFIG",
    "build_haskell_file_tool",
]
