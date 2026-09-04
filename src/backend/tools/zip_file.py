"""Zip-archive reading tools: list entries and read one member's text.

Archives are only ever opened in memory — entries are never extracted to
disk, so archive path-traversal names cannot touch the filesystem. Per-entry
uncompressed sizes are checked before reading so a zip bomb cannot blow
through the configured byte limit, and binary members are described rather
than dumped into the model context.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._files import FileAccessError, resolve_safe_path

if TYPE_CHECKING:
    from src.config import Settings

_SUFFIXES = (".zip",)
# Cap returned characters so a large entry does not blow up the LLM context.
_MAX_CHARS = 20_000
# Upper bound on the entry listing so a huge archive stays bounded.
_MAX_LISTED_ENTRIES = 500
# Bytes inspected when deciding whether a member is binary.
_BINARY_SNIFF_BYTES = 8_192


class ZipFileInput(BaseModel):
    """Input schema for the zip listing tool."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .zip archive, relative to the configured file-read "
            "root directory. This also covers archives uploaded to the "
            "current chat session (under chat_uploads/<thread-id>/)."
        ),
    )
    max_chars: int = Field(
        default=20_000,
        ge=1,
        le=200_000,
        description="Maximum characters of the listing to return.",
    )


class ZipEntryInput(BaseModel):
    """Input schema for reading one zip member."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .zip archive, relative to the configured file-read "
            "root directory (including files uploaded to the current chat "
            "session)."
        ),
    )
    entry: str = Field(
        ...,
        min_length=1,
        description=(
            "Exact member name as shown by inspect_zip_file, e.g. "
            "'docs/readme.txt'."
        ),
    )
    max_chars: int = Field(
        default=_MAX_CHARS,
        ge=1,
        le=200_000,
        description="Maximum characters of the entry's text to return.",
    )


def build_zip_tools(
    settings: Settings,
) -> list[BaseTool]:
    """Create the zip lister and entry reader confined to the file-read root."""

    if not settings.file_read_enabled:
        return []
    root = Path(settings.file_read_root)
    max_bytes = settings.file_read_max_bytes

    def _run_inspect(path: str, max_chars: int = _MAX_CHARS) -> str:
        return inspect_zip_file(
            path,
            root=root,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    def _run_read(path: str, entry: str, max_chars: int = _MAX_CHARS) -> str:
        return read_zip_entry(
            path,
            entry,
            root=root,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    inspect_tool = StructuredTool.from_function(
        func=_run_inspect,
        name="inspect_zip_file",
        description=(
            "List the entries of a .zip archive from the local document "
            "directory (including files uploaded to the current chat "
            "session), with sizes. Call this before read_zip_entry to get "
            "the exact entry names."
        ),
        args_schema=ZipFileInput,
    )
    read_tool = StructuredTool.from_function(
        func=_run_read,
        name="read_zip_entry",
        description=(
            "Read one text entry from inside a .zip archive without "
            "extracting it. Pass the exact entry name shown by "
            "inspect_zip_file. Binary entries (images, executables, "
            "office documents) are described but not shown. Returns the "
            "entry text (possibly truncated)."
        ),
        args_schema=ZipEntryInput,
    )
    return [inspect_tool, read_tool]


def inspect_zip_file(
    path: str,
    *,
    root: Path,
    max_bytes: int,
    max_chars: int = _MAX_CHARS,
) -> str:
    """List an archive's entries with sizes, without extracting anything."""

    try:
        resolved = resolve_safe_path(
            path,
            root=root,
            expected_suffixes=_SUFFIXES,
            max_bytes=max_bytes,
        )
    except FileAccessError as exc:
        return f"Could not read zip file: {exc}"

    try:
        with zipfile.ZipFile(resolved) as archive:
            infos = archive.infolist()
    except zipfile.BadZipFile as exc:
        return f"Could not read zip file {resolved.name!r}: not a valid zip archive ({exc})."
    except OSError as exc:
        return f"Could not read zip file {resolved.name!r}: {exc}"

    files = sum(1 for info in infos if not info.is_dir())
    directories = len(infos) - files
    lines = [
        f"ZIP archive {resolved.name}: {len(infos)} entries "
        f"({files} files, {directories} directories)"
    ]
    truncated = False
    for info in infos[:_MAX_LISTED_ENTRIES]:
        name = info.filename
        if info.is_dir():
            lines.append(f"- {name} (directory)")
            continue
        flags = f" ({_size_line(info)})"
        if info.flag_bits & 0x1:
            flags += " [encrypted]"
        lines.append(f"- {name}{flags}")
    if len(infos) > _MAX_LISTED_ENTRIES:
        truncated = True
        lines.append(
            f"[… showing {_MAX_LISTED_ENTRIES} of {len(infos):,} entries]"
        )

    body = "\n".join(lines)
    limit = max(1, int(max_chars))
    if len(body) > limit or truncated:
        suffix = "" if len(body) <= limit else (
            f"\n\n[listing truncated to {limit:,} of {len(body):,} chars]"
        )
        return body[:limit] + suffix
    return body


def read_zip_entry(
    path: str,
    entry: str,
    *,
    root: Path,
    max_bytes: int,
    max_chars: int = _MAX_CHARS,
) -> str:
    """Return one member's decoded text, guarded against zip bombs."""

    try:
        resolved = resolve_safe_path(
            path,
            root=root,
            expected_suffixes=_SUFFIXES,
            max_bytes=max_bytes,
        )
    except FileAccessError as exc:
        return f"Could not read zip file: {exc}"

    try:
        with zipfile.ZipFile(resolved) as archive:
            names = archive.namelist()
            target = entry.strip().strip('"').strip("'")
            if target not in names:
                return (
                    f"Could not read zip entry: {entry!r} is not in "
                    f"{resolved.name!r}. Call inspect_zip_file for the exact "
                    "entry names."
                )
            info = archive.getinfo(target)
            if info.is_dir():
                return f"Could not read zip entry: {entry!r} is a directory."
            if info.flag_bits & 0x1:
                return (
                    f"Could not read zip entry: {entry!r} is password "
                    "protected; reading it is not supported."
                )
            # Check the uncompressed size BEFORE reading so a zip bomb
            # cannot push gigabytes through the archive reader.
            if info.file_size > max_bytes:
                return (
                    f"Could not read zip entry: {entry!r} decompresses to "
                    f"{info.file_size:,} bytes; limit {max_bytes:,} bytes."
                )
            try:
                data = archive.read(target)
            except RuntimeError as exc:
                # zipfile raises RuntimeError for unsupported compression
                # and encryption rather than a parsed error type.
                return f"Could not read zip entry {entry!r}: {exc}"
    except zipfile.BadZipFile as exc:
        return f"Could not read zip file {resolved.name!r}: not a valid zip archive ({exc})."
    except OSError as exc:
        return f"Could not read zip file {resolved.name!r}: {exc}"

    if b"\x00" in data[:_BINARY_SNIFF_BYTES]:
        return (
            f"Entry {target!r} in {resolved.name} appears to be binary "
            f"({len(data):,} bytes); its contents are not shown."
        )

    text = data.decode("utf-8", errors="replace")
    limit = max(1, int(max_chars))
    truncated = len(text) > limit
    header = f"Contents of {target} in {resolved.name} ({len(text):,} chars):"
    if truncated:
        header = (
            f"Contents of {target} in {resolved.name} (showing first "
            f"{limit:,} of {len(text):,} chars):"
        )
    return f"{header}\n\n{text[:limit]}"


def _size_line(info: zipfile.ZipInfo) -> str:
    return f"{info.file_size:,} bytes, compressed {info.compress_size:,} bytes"


__all__ = [
    "ZipEntryInput",
    "ZipFileInput",
    "build_zip_tools",
    "inspect_zip_file",
    "read_zip_entry",
]
