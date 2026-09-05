"""Java (.java) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

JAVA_CONFIG = SourceEditConfig(
    label="Java",
    suffixes=(".java",),
    creation_suffix=".java",
    tool_prefix="java",
    mime_type="text/x-java",
    flag_name="java_edit_enabled",
    temp_prefix=".java_write-",
)


def build_java_file_tool(settings: Settings) -> BaseTool:
    """Create a Java .java file reader confined to the file-read root."""

    return build_source_file_tool(settings, JAVA_CONFIG)


__all__ = [
    "JAVA_CONFIG",
    "build_java_file_tool",
]
