"""Log (.log) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

LOG_CONFIG = SourceEditConfig(
    label="Log",
    suffixes=(".log",),
    creation_suffix=".log",
    tool_prefix="log",
    mime_type="text/plain",
    flag_name="log_edit_enabled",
    temp_prefix=".log_write-",
)


def build_log_file_tool(settings: Settings) -> BaseTool:
    """Create a .log file reader confined to the file-read root."""

    return build_source_file_tool(settings, LOG_CONFIG)


__all__ = [
    "LOG_CONFIG",
    "build_log_file_tool",
]
