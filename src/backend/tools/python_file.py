"""Python (.py) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

PYTHON_CONFIG = SourceEditConfig(
    label="Python",
    suffixes=(".py",),
    creation_suffix=".py",
    tool_prefix="python",
    mime_type="text/x-python",
    flag_name="python_edit_enabled",
    temp_prefix=".python_write-",
)


def build_python_file_tool(settings: Settings) -> BaseTool:
    """Create a Python .py file reader confined to the file-read root."""

    return build_source_file_tool(settings, PYTHON_CONFIG)


__all__ = [
    "PYTHON_CONFIG",
    "build_python_file_tool",
]
