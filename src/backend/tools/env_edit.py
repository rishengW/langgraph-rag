"""Session-scoped .env-style (.env) creation, inspection, and editing tools.

.env files are line-oriented KEY=VALUE documents, so the editor is
addressed by key: the inspector lists every key with its value, and the
editor applies key-based operations (set_value, delete_key) whose
expectations are checked against the parsed file before anything is
written. Unlike the XML/YAML editors, formatting is preserved: only the
changed lines are rewritten, so comments and blank lines survive edits.
Parsing uses a simple dotenv grammar (KEY=VALUE lines, # comments,
export prefixes, optional quotes); values are never interpolated. The
uploaded source is never modified.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, model_validator

from ._source_edit import (
    _INSPECT_MAX_CHARS,
    _MAX_OPERATIONS,
    _THREAD_ID_PATTERN,
    SourceEditConfig,
    SourceEditError,
    SourceEditResult,
    _build_file_artifact,
    _creation_stem,
    _prepare_session_directory,
    _read_source_document,
    _reserve_created_output_path,
    _reserve_output_path,
    _resolve_session_source,
    _SourceDocument,
    _write_source_atomically,
    decode_source,
)

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)

ENV_MIME_TYPE = "text/plain"
_MAX_VALUE_CHARS = 100_000
_MAX_INSPECT_ENTRIES = 1_000

# KEY=VALUE grammar: optional "export ", bare key, optional surrounding
# whitespace, '=', then the value up to end of line. Inline comments only
# count when unquoted.
_LINE_PATTERN = re.compile(
    r"^(?P<export>export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_.]*)"
    r"(?P<assign>\s*=(?P<value>.*))?$"
)
_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

ENV_CONFIG = SourceEditConfig(
    label=".env",
    suffixes=(".env", ".env.local", ".env.example"),
    creation_suffix=".env",
    tool_prefix="env",
    mime_type=ENV_MIME_TYPE,
    flag_name="env_edit_enabled",
    temp_prefix=".env_write-",
)


class EnvEditError(Exception):
    """Raised when a .env file cannot be inspected or edited safely."""


def _is_valid_key(key: str) -> bool:
    return bool(_KEY_PATTERN.fullmatch(key))


class EnvEditOperation(BaseModel):
    """One key-addressed edit, validated against the current file."""

    action: Literal["set_value", "delete_key"] = Field(
        ..., description="The kind of .env edit to apply."
    )
    key: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Variable name to edit, e.g. WEB_SEARCH_ENABLED.",
    )
    value: str | None = Field(
        default=None,
        max_length=_MAX_VALUE_CHARS,
        description="New raw value for set_value, written verbatim after '='.",
    )
    expected_value: str | None = Field(
        default=None,
        max_length=_MAX_VALUE_CHARS,
        description=(
            "Current raw value at the key, required by set_value on an "
            "existing key and by every delete_key."
        ),
    )
    expected_missing: bool = Field(
        default=False,
        description=(
            "set_value only: declare the key does not exist yet so it can "
            "be created without an expected_value."
        ),
    )

    @model_validator(mode="after")
    def _check_fields(self) -> EnvEditOperation:
        if not _is_valid_key(self.key):
            raise ValueError(
                f"invalid key {self.key!r}: use letters, digits, '_' and '.', "
                "starting with a letter or '_'."
            )
        if self.action == "set_value":
            if self.value is None:
                raise ValueError("set_value requires: value.")
            guards = [
                name
                for name, given in (
                    ("expected_value", self.expected_value is not None),
                    ("expected_missing", self.expected_missing),
                )
                if given
            ]
            if len(guards) != 1:
                raise ValueError(
                    "set_value requires exactly one of expected_value "
                    "(key exists) or expected_missing=true (create it)."
                )
        else:  # delete_key
            if self.expected_value is None:
                raise ValueError("delete_key requires: expected_value.")
            if self.expected_missing:
                raise ValueError("delete_key does not accept expected_missing.")
            if self.value is not None:
                raise ValueError("delete_key does not accept value.")
        return self


class EnvInspectInput(BaseModel):
    """Input schema for the .env inspector."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .env-style file uploaded to this chat session, "
            "exactly as listed in the upload note."
        ),
    )
    max_chars: int = Field(
        default=_INSPECT_MAX_CHARS,
        ge=1,
        le=200_000,
        description="Maximum characters of the key listing to return.",
    )


class EnvCreateInput(BaseModel):
    """Input schema for creating a .env file."""

    filename: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "Bare filename for the new .env-style file. An .env suffix is "
            "added when needed. Existing files are never overwritten."
        ),
    )
    content: str = Field(
        ...,
        description=(
            "Complete UTF-8 .env text to write. Every non-comment, "
            "non-empty line must parse as KEY=VALUE; formatting is "
            "preserved as provided."
        ),
    )


class EnvEditInput(BaseModel):
    """Input schema for the .env editor."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .env-style file uploaded to this chat session. The source is never modified."
        ),
    )
    operations: list[EnvEditOperation] = Field(
        ...,
        min_length=1,
        max_length=_MAX_OPERATIONS,
        description=(
            "Key-based edits to validate against the original file and "
            "then apply as one transaction."
        ),
    )
    output_name: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Optional name for the edited copy. Defaults to "
            "'<original>.edited.env'; an .env suffix is enforced."
        ),
    )


def build_env_edit_tools(
    settings: Settings,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> list[BaseTool]:
    """Create .env tools only when enabled and safely scoped to a chat."""

    if not (settings.file_read_enabled and settings.env_edit_enabled):
        return []
    if session_root is None or not _THREAD_ID_PATTERN.fullmatch(thread_id):
        return []

    file_root = Path(settings.file_read_root)
    max_bytes = settings.file_read_max_bytes

    def _run_create(
        filename: str,
        content: str,
    ) -> tuple[str, dict[str, object] | None]:
        result = create_env_file(
            filename,
            content,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
            thread_id=thread_id,
        )
        return result.content, result.artifact

    def _run_inspect(path: str, max_chars: int = _INSPECT_MAX_CHARS) -> str:
        return inspect_env_file(
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    def _run_edit(
        path: str,
        operations: list[EnvEditOperation],
        output_name: str | None = None,
    ) -> tuple[str, dict[str, object] | None]:
        result = edit_env_file(
            path,
            operations=operations,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
            thread_id=thread_id,
            output_name=output_name,
        )
        return result.content, result.artifact

    create_tool = StructuredTool.from_function(
        func=_run_create,
        name="create_env_file",
        description=(
            "Create a new UTF-8 .env-style file in the current chat "
            "session and return it as a download. Every non-comment line "
            "must parse as KEY=VALUE. Existing files are never "
            "overwritten; a numbered filename is chosen on collision. Use "
            "only when the user explicitly asks to create a .env file."
        ),
        args_schema=EnvCreateInput,
        response_format="content_and_artifact",
    )
    inspect_tool = StructuredTool.from_function(
        func=_run_inspect,
        name="inspect_env_file",
        description=(
            "List every KEY and its current value in a .env-style file "
            "uploaded to this chat session. Call this before edit_env_file "
            "to obtain the exact keys and current values required by edit "
            "operations."
        ),
        args_schema=EnvInspectInput,
    )
    edit_tool = StructuredTool.from_function(
        func=_run_edit,
        name="edit_env_file",
        description=(
            "Apply key-based edits (set_value, delete_key) to an uploaded "
            ".env-style file and create a new downloadable copy without "
            "changing the original. Call inspect_env_file first; pass "
            "expected_value for existing keys or expected_missing=true to "
            "create a key. Comments and unchanged lines are preserved. Use "
            "only when the user explicitly asks to change the file."
        ),
        args_schema=EnvEditInput,
        response_format="content_and_artifact",
    )
    return [create_tool, inspect_tool, edit_tool]


def create_env_file(
    filename: str,
    content: str,
    *,
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    thread_id: str = "",
) -> SourceEditResult:
    """Create a session-scoped UTF-8 .env file and downloadable artifact."""

    output_name = "(unresolved)"
    try:
        if not _THREAD_ID_PATTERN.fullmatch(thread_id):
            raise EnvEditError("a valid chat thread is required to create a .env file.")
        directory = _prepare_session_directory(
            session_root=session_root,
            file_root=file_root,
        )
        stem, suffix = _creation_stem(ENV_CONFIG, filename)
        document = decode_source(content.encode("utf-8"), name=f"{stem}{suffix}")
        _parse_document_text(document, f"{stem}{suffix}")
        target = _reserve_created_output_path(
            directory=directory,
            stem=stem,
            suffix=suffix,
        )
        output_name = target.name
        published = _write_source_atomically(
            ENV_CONFIG,
            document,
            target=target,
            max_bytes=max_bytes,
            size_label="new",
        )
        size_bytes = published.stat().st_size
    except (OSError, UnicodeError, SourceEditError, EnvEditError) as exc:
        logger.info(
            "env_create failed: thread=%s output=%s reason=%s",
            thread_id or "(none)",
            output_name,
            type(exc).__name__,
        )
        return SourceEditResult(content=f"Could not create .env file: {exc}")

    logger.info(
        "env_create succeeded: thread=%s output=%s bytes=%d",
        thread_id,
        published.name,
        size_bytes,
    )
    return SourceEditResult(
        content=(
            f"Created {published.name} in this chat session. The file is available to download."
        ),
        artifact=_build_file_artifact(
            ENV_CONFIG,
            thread_id=thread_id,
            filename=published.name,
            size_bytes=size_bytes,
        ),
    )


def inspect_env_file(
    path: str,
    *,
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    max_chars: int = _INSPECT_MAX_CHARS,
) -> str:
    """Return format metadata and a listing of every KEY=value pair."""

    try:
        resolved = _resolve_session_source(
            ENV_CONFIG,
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
        )
        document = _read_source_document(resolved)
        entries = _parse_document_text(document, resolved.name)
    except SourceEditError as exc:
        return f"Could not inspect .env file: {exc}"
    except EnvEditError as exc:
        return f"Could not inspect .env file: {exc}"

    encoding = "UTF-8 with BOM" if document.encoding == "utf-8-sig" else "UTF-8"
    final_newline = "yes" if document.has_final_newline else "no"
    lines = [
        f".env file {resolved.name}:",
        f"Encoding: {encoding}",
        f"Newline: {document.newline}",
        f"Final newline: {final_newline}",
        "KEYS:",
    ]
    keys = list(entries)
    truncated = len(keys) > _MAX_INSPECT_ENTRIES
    for key in keys[:_MAX_INSPECT_ENTRIES]:
        lines.append(f"{key} = {entries[key]}")
    if truncated:
        lines.append(
            f"[… {_MAX_INSPECT_ENTRIES:,} of {len(keys):,} keys shown]"
        )
    body = "\n".join(lines)
    limit = max(1, int(max_chars))
    if len(body) > limit:
        return f"{body[:limit]}\n\n[listing truncated to {limit:,} of {len(body):,} chars]"
    return body


def edit_env_file(
    path: str,
    *,
    operations: list[EnvEditOperation],
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    thread_id: str = "",
    output_name: str | None = None,
) -> SourceEditResult:
    """Validate every operation, apply them, then publish a copy-on-write copy."""

    source_name = "(unresolved)"
    try:
        resolved = _resolve_session_source(
            ENV_CONFIG,
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
        )
        source_name = resolved.name
        document = _read_source_document(resolved)
        entries = _parse_document_text(document, resolved.name)

        problems: list[str] = []
        for position, operation in enumerate(operations):
            problem = _check_operation(entries, operation)
            if problem:
                problems.append(f"operation {position} ({operation.action}): {problem}")
        if problems:
            raise EnvEditError(
                f"no changes were made because {len(problems)} operation(s) "
                f"did not match the file: {' '.join(problems)} Re-run "
                "inspect_env_file and retry."
            )

        working_lines = list(document.lines)
        for operation in operations:
            working_lines = _apply_operation(working_lines, operation)

        edited = _document_from_lines(
            working_lines,
            encoding=document.encoding,
            newline=document.newline,
            has_final_newline=True,
        )
        published = _publish_edited(
            ENV_CONFIG,
            edited,
            session_root=session_root,
            source_path=resolved,
            output_name=output_name,
            max_bytes=max_bytes,
        )
    except SourceEditError as exc:
        logger.info(
            "env_edit failed: thread=%s source=%s operations=%d reason=%s",
            thread_id or "(none)",
            source_name,
            len(operations),
            type(exc).__name__,
        )
        return SourceEditResult(content=f"Could not edit .env file: {exc}")
    except EnvEditError as exc:
        logger.info(
            "env_edit failed: thread=%s source=%s operations=%d reason=%s",
            thread_id or "(none)",
            source_name,
            len(operations),
            type(exc).__name__,
        )
        return SourceEditResult(content=f"Could not edit .env file: {exc}")

    size_bytes = published.stat().st_size
    logger.info(
        "env_edit succeeded: thread=%s source=%s output=%s operations=%d bytes=%d",
        thread_id or "(none)",
        source_name,
        published.name,
        len(operations),
        size_bytes,
    )
    summary = ", ".join(operation.action for operation in operations)
    content = (
        f"Applied {len(operations)} edit(s) ({summary}) to {resolved.name}. "
        f"The original file is unchanged; the edited copy was saved as "
        f"{published.name} and is available to download."
    )
    return SourceEditResult(
        content=content,
        artifact=_build_file_artifact(
            ENV_CONFIG,
            thread_id=thread_id,
            filename=published.name,
            size_bytes=size_bytes,
        ),
    )


def _parse_document_text(document: _SourceDocument, name: str) -> dict[str, str]:
    """Parse KEY=VALUE lines, raising EnvEditError on a malformed line."""

    entries: dict[str, str] = {}
    text = document.newline_text.join(document.lines)
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE_PATTERN.fullmatch(line)
        if not match:
            raise EnvEditError(
                f"{name!r} line {line_number} is not KEY=VALUE: {raw_line[:80]!r}"
            )
        value = match.group("value")
        entries[match.group("key")] = "" if value is None else value.strip()
    return entries


def _parse_line(raw_line: str) -> tuple[str, str] | None:
    """Parse one raw line into (key, value), or None for blank/comment."""

    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    match = _LINE_PATTERN.fullmatch(line)
    if not match:
        raise EnvEditError(f"line is not KEY=VALUE: {raw_line[:80]!r}")
    value = match.group("value")
    return match.group("key"), ("" if value is None else value.strip())


def _document_from_lines(
    lines: list[str],
    *,
    encoding: Literal["utf-8", "utf-8-sig"],
    newline: Literal["LF", "CRLF"],
    has_final_newline: bool,
) -> _SourceDocument:
    return _SourceDocument(
        lines=tuple(lines),
        encoding=encoding,
        newline=newline,
        has_final_newline=has_final_newline,
    )


def _publish_edited(
    config: SourceEditConfig,
    document: _SourceDocument,
    *,
    session_root: Path,
    source_path: Path,
    output_name: str | None,
    max_bytes: int,
) -> Path:
    from ._source_edit import _write_source_atomically

    target = _reserve_output_path(
        session_root=session_root,
        source_path=source_path,
        output_name=output_name,
    )
    return _write_source_atomically(
        config,
        document,
        target=target,
        max_bytes=max_bytes,
        size_label="edited",
    )


def _check_operation(entries: dict[str, str], operation: EnvEditOperation) -> str | None:
    """Return a problem description, or None when the operation may apply."""

    if operation.action == "set_value":
        if operation.key in entries:
            if operation.expected_missing:
                return f"key {operation.key} already exists but expected_missing was set."
            if entries[operation.key] != operation.expected_value:
                return "expected_value does not match the current value."
        elif not operation.expected_missing:
            return f"key {operation.key} does not exist; set expected_missing=true to create it."
    else:  # delete_key
        if operation.key not in entries:
            return f"key {operation.key} does not exist."
        if entries[operation.key] != operation.expected_value:
            return "expected_value does not match the current value."
    return None


def _apply_operation(
    lines: list[str],
    operation: EnvEditOperation,
) -> list[str]:
    """Apply one already-validated operation, returning the new line list."""

    problem = _check_operation(
        dict(_entries_from_lines(lines)), operation
    )
    if problem:
        raise EnvEditError(f"{operation.action} {operation.key}: {problem}")

    if operation.action == "set_value":
        replacement: str | None = None
        new_lines: list[str] = []
        replaced = False
        for raw_line in lines:
            parsed = _safe_parse(raw_line)
            if parsed is not None and parsed[0] == operation.key:
                # Keep any 'export ' prefix and inline formatting width.
                match = _LINE_PATTERN.fullmatch(raw_line.strip())
                prefix = "export " if match and match.group("export") else ""
                replacement = f"{prefix}{operation.key}={operation.value}"
                new_lines.append(replacement)
                replaced = True
            else:
                new_lines.append(raw_line)
        if not replaced:
            new_lines.append(f"{operation.key}={operation.value}")
        return new_lines

    # delete_key: drop the line entirely.
    kept: list[str] = []
    for raw_line in lines:
        parsed = _safe_parse(raw_line)
        if parsed is not None and parsed[0] == operation.key:
            continue
        kept.append(raw_line)
    return kept


def _safe_parse(raw_line: str) -> tuple[str, str] | None:
    """Parse a line, treating malformed lines as non-entries."""

    try:
        return _parse_line(raw_line)
    except EnvEditError:
        return None


def _entries_from_lines(lines: list[str]) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw_line in lines:
        parsed = _safe_parse(raw_line)
        if parsed is not None:
            entries[parsed[0]] = parsed[1]
    return entries


__all__ = [
    "ENV_MIME_TYPE",
    "EnvCreateInput",
    "EnvEditError",
    "EnvEditInput",
    "EnvEditOperation",
    "EnvInspectInput",
    "build_env_edit_tools",
    "create_env_file",
    "edit_env_file",
    "inspect_env_file",
]
