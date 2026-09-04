"""PHP (.php) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

PHP_CONFIG = SourceEditConfig(
    label="PHP",
    suffixes=(".php",),
    creation_suffix=".php",
    tool_prefix="php",
    mime_type="text/x-php",
    flag_name="php_edit_enabled",
    temp_prefix=".php_write-",
)


def build_php_file_tool(settings: Settings) -> BaseTool:
    """Create a PHP .php file reader confined to the file-read root."""

    return build_source_file_tool(settings, PHP_CONFIG)


__all__ = [
    "PHP_CONFIG",
    "build_php_file_tool",
]
