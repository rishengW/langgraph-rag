"""Rust (.rs) file reader for the configured file-read root."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_file_tool

if TYPE_CHECKING:
    from src.config import Settings

RUST_CONFIG = SourceEditConfig(
    label="Rust",
    suffixes=(".rs",),
    creation_suffix=".rs",
    tool_prefix="rust",
    mime_type="text/rust",
    flag_name="rust_edit_enabled",
    temp_prefix=".rust_write-",
)


def build_rust_file_tool(settings: Settings) -> BaseTool:
    """Create a Rust .rs file reader confined to the file-read root."""

    return build_source_file_tool(settings, RUST_CONFIG)


__all__ = [
    "RUST_CONFIG",
    "build_rust_file_tool",
]
