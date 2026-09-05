"""Shell-script (.sh/.bash) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

SHELL_CONFIG = SourceEditConfig(
    label="Shell",
    suffixes=(".sh", ".bash"),
    creation_suffix=".sh",
    tool_prefix="shell",
    mime_type="text/x-shellscript",
    flag_name="shell_edit_enabled",
    temp_prefix=".shell_write-",
)


def build_shell_file_tool(settings: Settings) -> BaseTool:
    """Create a shell-script (.sh/.bash) file reader confined to the file-read root."""

    return build_source_file_tool(settings, SHELL_CONFIG)


__all__ = [
    "SHELL_CONFIG",
    "build_shell_file_tool",
]
