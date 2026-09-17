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
            f"access denied: {text!r} is outside the allowed directory ({root_resolved})."
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
        raise FileAccessError(f"file is too large ({size:,} bytes; limit {max_bytes:,} bytes).")

    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_safe_directory(raw_path: str, *, root: Path) -> Path:
    """Resolve ``raw_path`` under ``root`` for directory listing.

    Same boundary rules as :func:`resolve_safe_path` — refuse traversal,
    refuse protected names — but the target must be a directory and the
    file-suffix/size checks do not apply.
    """

    text = (raw_path or "").strip().strip('"').strip("'")
    root_resolved = root.expanduser().resolve()

    candidate = root_resolved
    if text and text != ".":
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = root_resolved / candidate

    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise FileAccessError(f"could not resolve path {text!r}: {exc}") from exc

    if not _is_within(resolved, root_resolved):
        raise FileAccessError(
            f"access denied: {text!r} is outside the allowed directory ({root_resolved})."
        )

    name_hit = resolved.name.lower() in _BLOCKED_NAMES
    suffix_hit = resolved.suffix.lower() in _BLOCKED_SUFFIXES
    if resolved != root_resolved and (name_hit or suffix_hit):
        raise FileAccessError(f"access denied: {resolved.name!r} is a protected path.")

    if not resolved.exists():
        raise FileAccessError(f"directory not found: {text!r}.")
    if not resolved.is_dir():
        raise FileAccessError(f"not a directory: {text!r}.")

    return resolved


def list_safe_directory(raw_path: str, *, root: Path, max_entries: int) -> str:
    """Render a text listing of one directory under ``root``.

    Entries are sorted (directories first), annotated with type and size,
    and capped at ``max_entries``. Symbolic links and dot-prefixed hidden
    entries are skipped so the listing never reveals out-of-root targets.
    """

    resolved = resolve_safe_directory(raw_path, root=root)
    root_resolved = root.expanduser().resolve()

    try:
        children = sorted(resolved.iterdir())
    except OSError as exc:
        return f"Could not list directory: {exc}"

    lines: list[str] = []
    hidden_count = 0
    for child in children:
        if child.name.startswith("."):
            hidden_count += 1
            continue
        try:
            if child.is_symlink():
                continue
            is_dir = child.is_dir()
            size = None if is_dir else child.stat().st_size
        except OSError:
            continue
        if is_dir:
            lines.append(f"[dir]  {child.name}/")
        else:
            lines.append(f"[file] {child.name} ({size:,} bytes)")

    limit = max(1, int(max_entries))
    truncated = len(lines) > limit
    body = lines[:limit]

    scope = "the file-read root directory" if resolved == root_resolved else resolved.name
    header = f"Listing of {scope}:"
    if truncated:
        header += f" (showing first {limit:,} of {len(lines):,} entries"
        if hidden_count:
            header += f"; {hidden_count:,} hidden entries skipped"
        header += ")"
    elif hidden_count:
        header += f" ({hidden_count:,} hidden entries skipped)"
    if not body:
        return f"{header}\n\n(no entries)"
    return header + "\n\n" + "\n".join(body)


__all__ = [
    "FileAccessError",
    "list_safe_directory",
    "resolve_safe_directory",
    "resolve_safe_path",
]
