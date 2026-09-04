"""Session-scoped JSONL (.jsonl) creation, inspection, and editing tools.

JSONL files are newline-delimited JSON, so the shared line-oriented engine in
:mod:`._source_edit` drives these tools with one extra rule: every written
line must parse as a JSON value. Existing files are never overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ._source_edit import SourceEditConfig, build_source_edit_tools

if TYPE_CHECKING:
    from src.config import Settings

JSONL_MIME_TYPE = "application/x-ndjson"


def _validate_json_line(line: str, index: int) -> str | None:
    """Return an error message when the line is not a valid JSON value."""

    if not line.strip():
        return f"line {index} is empty; every .jsonl line must be a JSON value."
    try:
        json.loads(line)
    except (ValueError, TypeError) as exc:
        return f"line {index} is not valid JSON ({exc})."
    return None


JSONL_CONFIG = SourceEditConfig(
    label="JSONL",
    suffixes=(".jsonl",),
    creation_suffix=".jsonl",
    tool_prefix="jsonl",
    mime_type=JSONL_MIME_TYPE,
    flag_name="jsonl_edit_enabled",
    temp_prefix=".jsonl_write-",
    line_validator=_validate_json_line,
)


def build_jsonl_edit_tools(
    settings: Settings,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> list[BaseTool]:
    """Create JSONL tools only when enabled and safely scoped to a chat."""

    return build_source_edit_tools(
        settings, JSONL_CONFIG, session_root=session_root, thread_id=thread_id
    )


__all__ = [
    "JSONL_CONFIG",
    "JSONL_MIME_TYPE",
    "build_jsonl_edit_tools",
]
