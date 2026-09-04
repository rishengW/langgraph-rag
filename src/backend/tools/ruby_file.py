"""Ruby (.rb) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

RUBY_CONFIG = SourceEditConfig(
    label="Ruby",
    suffixes=(".rb",),
    creation_suffix=".rb",
    tool_prefix="ruby",
    mime_type="text/x-ruby",
    flag_name="ruby_edit_enabled",
    temp_prefix=".ruby_write-",
)


def build_ruby_file_tool(settings: Settings) -> BaseTool:
    """Create a Ruby .rb file reader confined to the file-read root."""

    return build_source_file_tool(settings, RUBY_CONFIG)


__all__ = [
    "RUBY_CONFIG",
    "build_ruby_file_tool",
]
