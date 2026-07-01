"""Per-session file upload handling for chat mode.

Uploaded files are saved inside the configured file-read root so the existing
file-reading tools (read_text_file / read_word_document / read_excel_spreadsheet)
can read them back. Uploads are confined to a per-thread subdirectory, the
filename is sanitized to its basename, the suffix is allowlisted, and the size
is capped using the same limit the tools enforce.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings

# Suffixes accepted for upload — the union of what the file tools read.
ALLOWED_UPLOAD_SUFFIXES = (".txt", ".md", ".log", ".csv", ".docx", ".xlsx", ".pdf")

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


def build_upload_context_note(relative_paths: list[str]) -> str:
    """Build a system-message note telling the LLM which uploads it can read."""

    listing = "\n".join(f"- {path}" for path in relative_paths)
    return (
        "The user has uploaded the following file(s) to this chat session. "
        "You can read them with the file tools (read_text_file, "
        "read_word_document, read_excel_spreadsheet, read_pdf) by passing the "
        "exact path shown below:\n"
        f"{listing}\n"
        "When the user refers to an uploaded file by name, call the matching "
        "file tool with its full path from this list."
    )


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


def save_upload(
    *,
    settings: Settings,
    thread_id: str,
    filename: str,
    content: bytes,
) -> SavedUpload:
    """Validate and persist one uploaded file for a chat session.

    Raises ``UploadError`` on an empty/oversized payload or disallowed type.
    """

    if not settings.file_read_enabled:
        raise UploadError(
            "file reading is disabled; set FILE_READ_ENABLED=true to accept uploads."
        )

    safe_name = sanitize_filename(filename)
    if not safe_name:
        raise UploadError("the uploaded file has no usable name.")
    validate_suffix(safe_name)

    if not content:
        raise UploadError("the uploaded file is empty.")
    max_bytes = settings.file_read_max_bytes
    if len(content) > max_bytes:
        raise UploadError(
            f"file is too large ({len(content):,} bytes; limit {max_bytes:,} bytes)."
        )

    target_dir = session_upload_dir(settings, thread_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / safe_name

    try:
        target_path.write_bytes(content)
    except OSError as exc:
        raise UploadError(f"could not save the uploaded file: {exc}") from exc

    root = Path(settings.file_read_root).expanduser().resolve()
    try:
        relative = target_path.resolve().relative_to(root).as_posix()
    except ValueError:
        # Should not happen (target is built under root), but never return a
        # path the tools would refuse as outside the root.
        raise UploadError(
            "internal error: upload path escaped the file-read root."
        ) from None

    return SavedUpload(
        filename=safe_name,
        relative_path=relative,
        size_bytes=len(content),
    )


__all__ = [
    "ALLOWED_UPLOAD_SUFFIXES",
    "SavedUpload",
    "UploadError",
    "build_upload_context_note",
    "list_session_uploads",
    "save_upload",
    "sanitize_filename",
    "session_upload_dir",
    "validate_suffix",
]
