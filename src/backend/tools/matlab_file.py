"""MATLAB (.m) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

MATLAB_CONFIG = SourceEditConfig(
    label="MATLAB",
    suffixes=(".m",),
    creation_suffix=".m",
    tool_prefix="matlab",
    mime_type="text/x-matlab",
    flag_name="matlab_edit_enabled",
    temp_prefix=".matlab_write-",
)


def build_matlab_file_tool(settings: Settings) -> BaseTool:
    """Create a MATLAB .m file reader confined to the file-read root."""

    return build_source_file_tool(settings, MATLAB_CONFIG)


__all__ = [
    "MATLAB_CONFIG",
    "build_matlab_file_tool",
]
