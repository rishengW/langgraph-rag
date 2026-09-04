"""Session-scoped PHP (.php) creation, inspection, and editing tools.

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

PHP_CONFIG = SourceEditConfig(
    label="PHP",
    suffixes=(".php",),
    creation_suffix=".php",
    tool_prefix="php",
    mime_type="text/x-php",
    flag_name="php_edit_enabled",
    temp_prefix=".php_write-",
)


def build_php_edit_tools(
    settings: Settings,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> list[BaseTool]:
    """Create PHP tools only when enabled and safely scoped to a chat."""

    return build_source_edit_tools(
        settings, PHP_CONFIG, session_root=session_root, thread_id=thread_id
    )


__all__ = [
    "PHP_CONFIG",
    "build_php_edit_tools",
]
