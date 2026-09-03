"""Shared safety helpers for file-reading tools.

File-reading tools accept an LLM-chosen path, so they must guard against
arbitrary-file-read and path-traversal. All readers resolve the requested
path against a configurable root directory, refuse paths that escape it,
refuse known-sensitive file names, and enforce a maximum file size.
"""

from __future__ import annotations

from pathlib import Path

# File names/suffixes that must never be read regardless of the root, since a
# tool path is chosen by the model and the project ingests untrusted content.
_BLOCKED_NAMES = {
    ".env",
    ".env.local",
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
    ".netrc",
    ".pgpass",
    ".htpasswd",
}
_BLOCKED_SUFFIXES = {".pem", ".key", ".pfx", ".p12"}


class FileAccessError(Exception):
    """Raised when a requested file path is not allowed or not readable."""


def resolve_safe_path(
    raw_path: str,
    *,
    root: Path,
    expected_suffixes: tuple[str, ...],
    max_bytes: int,
) -> Path:
    """Resolve ``raw_path`` under ``root`` with safety checks.

    Raises ``FileAccessError`` with a human-readable message on any violation.
    Returns the resolved absolute path when the file is safe to read.
    """

    text = (raw_path or "").strip().strip('"').strip("'")
    if not text:
        raise FileAccessError("a non-empty file path is required.")

    root_resolved = root.expanduser().resolve()
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = root_resolved / candidate

    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise FileAccessError(f"could not resolve path {text!r}: {exc}") from exc

    if not _is_within(resolved, root_resolved):
        raise FileAccessError(
            f"access denied: {text!r} is outside the allowed directory "
            f"({root_resolved})."
        )

    if resolved.name.lower() in _BLOCKED_NAMES or resolved.suffix.lower() in _BLOCKED_SUFFIXES:
        raise FileAccessError(f"access denied: {resolved.name!r} is a protected file.")

    if not resolved.exists():
        raise FileAccessError(f"file not found: {text!r}.")
    if not resolved.is_file():
        raise FileAccessError(f"not a file: {text!r}.")

    if expected_suffixes and resolved.suffix.lower() not in expected_suffixes:
        allowed = ", ".join(expected_suffixes)
        raise FileAccessError(
            f"unsupported file type {resolved.suffix!r}; expected one of: {allowed}."
        )

    size = resolved.stat().st_size
    if size > max_bytes:
        raise FileAccessError(
            f"file is too large ({size:,} bytes; limit {max_bytes:,} bytes)."
        )

    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


__all__ = ["FileAccessError", "resolve_safe_path"]
