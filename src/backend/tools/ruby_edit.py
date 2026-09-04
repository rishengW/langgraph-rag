"""Session-scoped Ruby (.rb) creation, inspection, and editing tools.

Like the other source editors, this module is a thin per-language adapter:
the shared line-oriented engine lives in :mod:`._source_edit`. The tools
accept only UTF-8 files in the current chat session, never overwrite an
existing file, and publish edited copies as new downloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_edit_tools

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


def build_ruby_edit_tools(
    settings: Settings,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> list[BaseTool]:
    """Create Ruby tools only when enabled and safely scoped to a chat."""

    return build_source_edit_tools(
        settings, RUBY_CONFIG, session_root=session_root, thread_id=thread_id
    )


__all__ = [
    "RUBY_CONFIG",
    "build_ruby_edit_tools",
]
