"""Per-session file upload handling for chat mode.

Uploaded files are saved inside the configured file-read root so the existing
file-reading tools (read_text_file / read_word_document / read_excel_spreadsheet)
can read them back. Uploads are confined to a per-thread subdirectory, the
filename is sanitized to its basename, the suffix is allowlisted, and the size
is capped using the same limit the tools enforce.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from src.config import Settings

# Suffixes accepted for upload — the union of what the file tools read.
ALLOWED_UPLOAD_SUFFIXES = (".txt", ".md", ".log", ".csv", ".docx", ".xlsx", ".pptx", ".pdf")

UPLOAD_SUBDIR = "chat_uploads"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class UploadError(Exception):
    """Raised when an uploaded file is rejected before or during save."""


@dataclass(frozen=True)
class SavedUpload:
    """Result of a successful upload save."""

    filename: str
    # Path relative to the file-read root, usable directly as a tool argument.
    relative_path: str
    size_bytes: int


def session_upload_dir(settings: Settings, thread_id: str) -> Path:
    """Return the per-session upload directory under the file-read root."""

    root = Path(settings.file_read_root).expanduser()
    safe_thread = sanitize_filename(thread_id) or "session"
    return root / UPLOAD_SUBDIR / safe_thread


def list_session_uploads(settings: Settings, thread_id: str) -> list[str]:
    """Return tool-ready relative paths of files uploaded for a session.

    Paths are relative to the file-read root so they can be passed directly to
    the file-reading tools. The filesystem is the source of truth, so this
    survives a server restart even though the in-memory session does not.
    """

    target_dir = session_upload_dir(settings, thread_id)
    if not target_dir.exists():
        return []

    try:
        root = Path(settings.file_read_root).expanduser().resolve()
    except OSError:
        return []

    paths: list[str] = []
    for entry in sorted(target_dir.iterdir()):
        if not entry.is_file():
            continue
        try:
            relative = entry.resolve().relative_to(root).as_posix()
        except (ValueError, OSError):
            continue
        paths.append(relative)
    return paths


def build_upload_context_note(
    relative_paths: list[str],
    *,
    word_edit_enabled: bool = False,
    text_edit_enabled: bool = False,
    markdown_edit_enabled: bool = False,
    powerpoint_edit_enabled: bool = False,
    excel_edit_enabled: bool = False,
) -> str:
    """Build a system-message note telling the LLM which uploads it can read.

    When an edit feature is enabled, the note advertises its session-scoped
    tools and states the inspect-before-edit and explicit-request rules, so the
    model does not modify a document on its own initiative.
    """

    listing = "\n".join(f"- {path}" for path in relative_paths)
    note = (
        "The following file(s) are available in this chat session. "
        "You can read them with the file tools (read_text_file, "
        "read_word_document, read_excel_spreadsheet, read_pdf) by passing the "
        "exact path shown below:\n"
        f"{listing}\n"
        "When the user refers to an uploaded file by name, call the matching "
        "file tool with its full path from this list."
    )
    edit_notes: list[str] = []
    if word_edit_enabled:
        edit_notes.append(
            "For .docx files you may also call inspect_word_document to list their "
            "numbered paragraphs and table cells, and edit_word_document to apply "
            "edits. Only edit when the user explicitly asks for a change; always "
            "call inspect_word_document first and pass the exact current text as "
            "expected_text. Editing writes a new file and leaves the upload intact."
        )
    if text_edit_enabled:
        edit_notes.append(
            "For .txt files you may call create_text_file to create a new text "
            "file, inspect_text_file to list numbered lines, and edit_text_file "
            "to apply structured line edits. Only create or edit when the user "
            "explicitly asks; always inspect before editing and pass the exact "
            "current text as expected_text. Writes create a new file and never "
            "overwrite an existing file."
        )
    if markdown_edit_enabled:
        edit_notes.append(
            "For .md files you may call create_markdown_file to create a new "
            "Markdown file, inspect_markdown_file to list numbered lines, and "
            "edit_markdown_file to apply structured line edits. Only create or "
            "edit when the user explicitly asks; always inspect before editing "
            "and pass the exact current text as expected_text. Writes create a "
            "new file and never overwrite an existing file."
        )
    if powerpoint_edit_enabled:
        edit_notes.append(
            "For .pptx files you may call inspect_powerpoint to list slides and "
            "shapes, then edit_powerpoint to apply text or table-cell edits. "
            "Only edit when explicitly asked; pass exact expected_text. The "
            "original upload is never overwritten."
        )
    if excel_edit_enabled:
        edit_notes.append(
            "For .xlsx files you may call inspect_excel_spreadsheet to list "
            "worksheets, cells, values, and formulas, then "
            "edit_excel_spreadsheet to apply cell, formula, row/column, "
            "worksheet, column-width, and number-format edits. Only edit when "
            "explicitly asked; always inspect first and pass the exact current "
            "value as expected_value. The original upload is never overwritten."
        )
    return "\n".join([note, *edit_notes])


def sanitize_filename(name: str) -> str:
    """Reduce an arbitrary client filename to a safe basename.

    Strips any directory components and replaces unsafe characters so the
    saved name cannot escape the per-session directory or collide with shell
    metacharacters.
    """

    base = Path(str(name or "")).name
    base = base.strip().strip(".")
    cleaned = _SAFE_NAME.sub("_", base)
    return cleaned[:200]


def validate_suffix(filename: str) -> str:
    """Validate and return the lowercase suffix for an allowed upload type."""

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        allowed = ", ".join(ALLOWED_UPLOAD_SUFFIXES)
        raise UploadError(
            f"unsupported file type {suffix or '(none)'!r}; allowed: {allowed}."
        )
    return suffix


def _upload_target(
    settings: Settings,
    thread_id: str,
    filename: str,
) -> tuple[str, Path]:
    if not settings.file_read_enabled:
        raise UploadError(
            "file reading is disabled; set FILE_READ_ENABLED=true to accept uploads."
        )

    safe_name = sanitize_filename(filename)
    if not safe_name:
        raise UploadError("the uploaded file has no usable name.")
    suffix = validate_suffix(safe_name)
    if suffix == ".pptx" and not settings.powerpoint_edit_enabled:
        raise UploadError(
            "PowerPoint uploads are disabled; set "
            "POWERPOINT_EDIT_ENABLED=true to inspect or edit .pptx files."
        )

    target_dir = session_upload_dir(settings, thread_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    return safe_name, target_dir / safe_name


def _saved_upload(
    settings: Settings,
    safe_name: str,
    target_path: Path,
    size_bytes: int,
) -> SavedUpload:
    root = Path(settings.file_read_root).expanduser().resolve()
    try:
        relative = target_path.resolve().relative_to(root).as_posix()
    except ValueError:
        raise UploadError(
            "internal error: upload path escaped the file-read root."
        ) from None
    return SavedUpload(safe_name, relative, size_bytes)


def save_upload(
    *,
    settings: Settings,
    thread_id: str,
    filename: str,
    content: bytes,
) -> SavedUpload:
    """Validate and persist an in-memory upload (compatibility API)."""

    safe_name, target_path = _upload_target(settings, thread_id, filename)
    max_bytes = settings.file_read_max_bytes
    if not content:
        raise UploadError("the uploaded file is empty.")
    if len(content) > max_bytes:
        raise UploadError(
            f"file is too large ({len(content):,} bytes; limit {max_bytes:,} bytes)."
        )

    try:
        target_path.write_bytes(content)
    except OSError as exc:
        raise UploadError(f"could not save the uploaded file: {exc}") from exc
    return _saved_upload(settings, safe_name, target_path, len(content))


def save_upload_stream(
    *,
    settings: Settings,
    thread_id: str,
    filename: str,
    stream: BinaryIO,
    chunk_bytes: int = 1024 * 1024,
) -> SavedUpload:
    """Stream an upload to disk with bounded memory and atomic publication."""

    safe_name, target_path = _upload_target(settings, thread_id, filename)
    max_bytes = settings.file_read_max_bytes
    size_bytes = 0
    temp_path: Path | None = None

    try:
        if stream.seekable():
            stream.seek(0)
        # Keep the temporary file outside the session directory so turn-time
        # upload discovery never announces a partially written file.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target_path.parent.parent,
            prefix=".chat-upload-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
            while chunk := stream.read(max(1, int(chunk_bytes))):
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise UploadError(
                        f"file is too large ({size_bytes:,}+ bytes; "
                        f"limit {max_bytes:,} bytes)."
                    )
                temporary.write(chunk)

        if size_bytes == 0:
            raise UploadError("the uploaded file is empty.")
        temp_path.replace(target_path)
    except UploadError:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise UploadError(f"could not save the uploaded file: {exc}") from exc

    return _saved_upload(settings, safe_name, target_path, size_bytes)


__all__ = [
    "ALLOWED_UPLOAD_SUFFIXES",
    "SavedUpload",
    "UploadError",
    "build_upload_context_note",
    "list_session_uploads",
    "save_upload",
    "save_upload_stream",
    "sanitize_filename",
    "session_upload_dir",
    "validate_suffix",
]
