"""Session-scoped Julia (.jl) creation, inspection, and editing tools.

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

JULIA_CONFIG = SourceEditConfig(
    label="Julia",
    suffixes=(".jl",),
    creation_suffix=".jl",
    tool_prefix="julia",
    mime_type="text/x-julia",
    flag_name="julia_edit_enabled",
    temp_prefix=".julia_write-",
)


def build_julia_edit_tools(
    settings: Settings,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> list[BaseTool]:
    """Create Julia tools only when enabled and safely scoped to a chat."""

    return build_source_edit_tools(
        settings, JULIA_CONFIG, session_root=session_root, thread_id=thread_id
    )


__all__ = [
    "JULIA_CONFIG",
    "build_julia_edit_tools",
]
