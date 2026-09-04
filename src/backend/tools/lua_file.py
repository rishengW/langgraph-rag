"""Lua (.lua) file reader for the configured file-read root."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

LUA_CONFIG = SourceEditConfig(
    label="Lua",
    suffixes=(".lua",),
    creation_suffix=".lua",
    tool_prefix="lua",
    mime_type="text/x-lua",
    flag_name="lua_edit_enabled",
    temp_prefix=".lua_write-",
)


def build_lua_file_tool(settings: Settings) -> BaseTool:
    """Create a Lua .lua file reader confined to the file-read root."""

    return build_source_file_tool(settings, LUA_CONFIG)


__all__ = [
    "LUA_CONFIG",
    "build_lua_file_tool",
]
