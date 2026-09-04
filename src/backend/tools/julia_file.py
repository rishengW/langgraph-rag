"""Julia (.jl) file reader for the configured file-read root."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

JULIA_CONFIG = SourceEditConfig(
    label="Julia",
    suffixes=(".jl",),
    creation_suffix=".jl",
    tool_prefix="julia",
    mime_type="text/x-julia",
    flag_name="julia_edit_enabled",
    temp_prefix=".julia_write-",
)


def build_julia_file_tool(settings: Settings) -> BaseTool:
    """Create a Julia .jl file reader confined to the file-read root."""

    return build_source_file_tool(settings, JULIA_CONFIG)


__all__ = [
    "JULIA_CONFIG",
    "build_julia_file_tool",
]
