"""Session-scoped Go (.go) creation, inspection, and editing tools.

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

GO_CONFIG = SourceEditConfig(
    label="Go",
    suffixes=(".go",),
    creation_suffix=".go",
    tool_prefix="go",
    mime_type="text/x-go",
    flag_name="go_edit_enabled",
    temp_prefix=".go_write-",
)


def build_go_edit_tools(
    settings: Settings,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> list[BaseTool]:
    """Create Go tools only when enabled and safely scoped to a chat."""

    return build_source_edit_tools(
        settings, GO_CONFIG, session_root=session_root, thread_id=thread_id
    )


__all__ = [
    "GO_CONFIG",
    "build_go_edit_tools",
]
