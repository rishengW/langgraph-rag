"""SQL (.sql) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

SQL_CONFIG = SourceEditConfig(
    label="SQL",
    suffixes=(".sql",),
    creation_suffix=".sql",
    tool_prefix="sql",
    mime_type="application/sql",
    flag_name="sql_edit_enabled",
    temp_prefix=".sql_write-",
)


def build_sql_file_tool(settings: Settings) -> BaseTool:
    """Create a SQL .sql file reader confined to the file-read root."""

    return build_source_file_tool(settings, SQL_CONFIG)


__all__ = [
    "SQL_CONFIG",
    "build_sql_file_tool",
]
