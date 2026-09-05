"""Session-scoped Java (.java) creation, inspection, and editing tools.

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

JAVA_CONFIG = SourceEditConfig(
    label="Java",
    suffixes=(".java",),
    creation_suffix=".java",
    tool_prefix="java",
    mime_type="text/x-java",
    flag_name="java_edit_enabled",
    temp_prefix=".java_write-",
)


def build_java_edit_tools(
    settings: Settings,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> list[BaseTool]:
    """Create Java tools only when enabled and safely scoped to a chat."""

    return build_source_edit_tools(
        settings, JAVA_CONFIG, session_root=session_root, thread_id=thread_id
    )


__all__ = [
    "JAVA_CONFIG",
    "build_java_edit_tools",
]
