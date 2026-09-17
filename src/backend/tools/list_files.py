from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._files import FileAccessError, list_safe_directory

if TYPE_CHECKING:
    from src.config import Settings

# Cap returned entries so a large directory does not blow up the LLM context.
_MAX_ENTRIES = 500


class ListFilesInput(BaseModel):
    """Input schema for the directory listing tool."""

    path: str = Field(
        default="",
        description=(
            "Directory to list, relative to the configured file-read root "
            "directory. Empty or '.' lists the root itself."
        ),
    )
    max_entries: int = Field(
        default=_MAX_ENTRIES,
        ge=1,
        le=2000,
        description="Maximum number of entries to return.",
    )


def build_list_files_tool(
    settings: Settings,
) -> BaseTool:
    """Create a directory-listing tool confined to the file-read root."""

    root = Path(settings.file_read_root)

    def _run_list_files(path: str = "", max_entries: int = _MAX_ENTRIES) -> str:
        try:
            return list_safe_directory(path, root=root, max_entries=max_entries)
        except FileAccessError as exc:
            return f"Could not list directory: {exc}"

    return StructuredTool.from_function(
        func=_run_list_files,
        name="list_files",
        description=(
            "List the files and subdirectories of the local document "
            "directory. Use this first when the user references a file "
            "without giving its exact name, then read the matching file "
            "with the corresponding reader tool. Hidden entries and "
            "symbolic links are not shown."
        ),
        args_schema=ListFilesInput,
    )


__all__ = [
    "ListFilesInput",
    "build_list_files_tool",
]
