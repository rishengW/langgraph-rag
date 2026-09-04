"""Swift (.swift) file reader for the configured file-read root."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

SWIFT_CONFIG = SourceEditConfig(
    label="Swift",
    suffixes=(".swift",),
    creation_suffix=".swift",
    tool_prefix="swift",
    mime_type="text/x-swift",
    flag_name="swift_edit_enabled",
    temp_prefix=".swift_write-",
)


def build_swift_file_tool(settings: Settings) -> BaseTool:
    """Create a Swift .swift file reader confined to the file-read root."""

    return build_source_file_tool(settings, SWIFT_CONFIG)


__all__ = [
    "SWIFT_CONFIG",
    "build_swift_file_tool",
]
