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

INI_MIME_TYPE = "text/plain"
_MAX_VALUE_CHARS = 100_000
_MAX_INSPECT_ENTRIES = 1_000

# INI line grammar: [section] headers, or key=value / key:value pairs.
# Comments start with '#' or ';'. Keys must not contain '=', ':', '[',
# or whitespace so dotted section.key addressing stays unambiguous.
_SECTION_PATTERN = re.compile(r"^\[(?P<name>[^\]]*)\]$")
_LINE_PATTERN = re.compile(
    r"^(?P<key>[^=:\[\s][^=:]*?)\s*(?P<sep>[=:])\s*(?P<value>.*)$"
)
_KEY_PATTERN = re.compile(r"^[^=:\[\]\s]+(\.[^=:\[\]\s]+)*$")

INI_CONFIG = SourceEditConfig(
    label=".ini",
    suffixes=(".ini", ".cfg", ".conf"),
    creation_suffix=".ini",
    tool_prefix="ini",
    mime_type=INI_MIME_TYPE,
    flag_name="ini_edit_enabled",
    temp_prefix=".ini_write-",
)

# Prefix used in listings for keys that appear before any [section] header.
ROOT_SECTION = "(root)"


class IniEditError(Exception):
    """Raised when an .ini file cannot be inspected or edited safely."""


def _is_valid_key(key: str) -> bool:
    return bool(_KEY_PATTERN.fullmatch(key))


class IniEditOperation(BaseModel):
    """One section.key-addressed edit, validated against the current file."""

    action: Literal["set_value", "delete_key"] = Field(
        ..., description="The kind of .ini edit to apply."
    )
    key: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description=(
            "Property to edit in 'section.name' form, e.g. database.host. "
            "Keys before any [section] header are addressed without a "
            "prefix, e.g. runtime_mode."
        ),
    )
    value: str | None = Field(
        default=None,
        max_length=_MAX_VALUE_CHARS,
        description="New raw value for set_value, written verbatim after the separator.",
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
            "be created without an expected_value. A missing section is "
            "created as a new [section] header appended at the end."
        ),
    )

    @model_validator(mode="after")
    def _check_fields(self) -> IniEditOperation:
        if not _is_valid_key(self.key):
            raise ValueError(
                f"invalid key {self.key!r}: use 'section.name' dotted form; "
                "keys must not contain '=', ':', '[', ']', or whitespace."
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


class IniInspectInput(BaseModel):
    """Input schema for the .ini inspector."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to an INI-style file uploaded to this chat session, "
            "exactly as listed in the upload note."
        ),
    )
    max_chars: int = Field(
        default=_INSPECT_MAX_CHARS,
        ge=1,
        le=200_000,
        description="Maximum characters of the key listing to return.",
    )


class IniCreateInput(BaseModel):
    """Input schema for creating an .ini file."""

    filename: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "Bare filename for the new INI-style file. An .ini suffix is "
            "added when needed. Existing files are never overwritten."
        ),
    )
    content: str = Field(
        ...,
        description=(
            "Complete UTF-8 INI text to write. Every non-comment, "
            "non-empty line must parse as a [section] header or a "
            "key=value / key:value pair; formatting is preserved as "
            "provided."
        ),
    )


class IniEditInput(BaseModel):
    """Input schema for the .ini editor."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to an INI-style file uploaded to this chat session. The source is never modified."
        ),
    )
    operations: list[IniEditOperation] = Field(
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
            "'<original>.edited.ini'; an .ini suffix is enforced."
        ),
    )


def build_ini_edit_tools(
    settings: Settings,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> list[BaseTool]:
    """Create .ini tools only when enabled and safely scoped to a chat."""

    if not (settings.file_read_enabled and settings.ini_edit_enabled):
        return []
    if session_root is None or not _THREAD_ID_PATTERN.fullmatch(thread_id):
        return []

    file_root = Path(settings.file_read_root)
    max_bytes = settings.file_read_max_bytes

    def _run_create(
        filename: str,
        content: str,
    ) -> tuple[str, dict[str, object] | None]:
        result = create_ini_file(
            filename,
            content,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
            thread_id=thread_id,
        )
        return result.content, result.artifact

    def _run_inspect(path: str, max_chars: int = _INSPECT_MAX_CHARS) -> str:
        return inspect_ini_file(
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    def _run_edit(
        path: str,
        operations: list[IniEditOperation],
        output_name: str | None = None,
    ) -> tuple[str, dict[str, object] | None]:
        result = edit_ini_file(
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
        name="create_ini_file",
        description=(
            "Create a new UTF-8 INI-style file (.ini/.cfg/.conf) in the "
            "current chat session and return it as a download. Every "
            "non-comment line must be a [section] header or a "
            "key=value pair. Existing files are never overwritten; a "
            "numbered filename is chosen on collision. Use only when the "
            "user explicitly asks to create an .ini file."
        ),
        args_schema=IniCreateInput,
        response_format="content_and_artifact",
    )
    inspect_tool = StructuredTool.from_function(
        func=_run_inspect,
        name="inspect_ini_file",
        description=(
            "List every section.key and its current value in an INI-style "
            "file uploaded to this chat session. Call this before "
            "edit_ini_file to obtain the exact keys and current values "
            "required by edit operations."
        ),
        args_schema=IniInspectInput,
    )
    edit_tool = StructuredTool.from_function(
        func=_run_edit,
        name="edit_ini_file",
        description=(
            "Apply section.key-based edits (set_value, delete_key) to an "
            "uploaded INI-style file and create a new downloadable copy "
            "without changing the original. Call inspect_ini_file first; "
            "pass expected_value for existing keys or expected_missing=true "
            "to create a key (a missing section is appended as a new "
            "[section] header). Comments, sections, and unchanged lines "
            "are preserved. Use only when the user explicitly asks to "
            "change the file."
        ),
        args_schema=IniEditInput,
        response_format="content_and_artifact",
    )
    return [create_tool, inspect_tool, edit_tool]


def create_ini_file(
    filename: str,
    content: str,
    *,
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    thread_id: str = "",
) -> SourceEditResult:
    """Create a session-scoped UTF-8 .ini file and downloadable artifact."""

    output_name = "(unresolved)"
    try:
        if not _THREAD_ID_PATTERN.fullmatch(thread_id):
            raise IniEditError("a valid chat thread is required to create an .ini file.")
        directory = _prepare_session_directory(
            session_root=session_root,
            file_root=file_root,
        )
        stem, suffix = _creation_stem(INI_CONFIG, filename)
        document = decode_source(content.encode("utf-8"), name=f"{stem}{suffix}")
        _parse_document_text(document, f"{stem}{suffix}")
        target = _reserve_created_output_path(
            directory=directory,
            stem=stem,
            suffix=suffix,
        )
        output_name = target.name
        published = _write_source_atomically(
            INI_CONFIG,
            document,
            target=target,
            max_bytes=max_bytes,
            size_label="new",
        )
        size_bytes = published.stat().st_size
    except (OSError, UnicodeError, SourceEditError, IniEditError) as exc:
        logger.info(
            "ini_create failed: thread=%s output=%s reason=%s",
            thread_id or "(none)",
            output_name,
            type(exc).__name__,
        )
        return SourceEditResult(content=f"Could not create .ini file: {exc}")

    logger.info(
        "ini_create succeeded: thread=%s output=%s bytes=%d",
        thread_id,
        published.name,
        size_bytes,
    )
    return SourceEditResult(
        content=(
            f"Created {published.name} in this chat session. The file is available to download."
        ),
        artifact=_build_file_artifact(
            INI_CONFIG,
            thread_id=thread_id,
            filename=published.name,
            size_bytes=size_bytes,
        ),
    )


def inspect_ini_file(
    path: str,
    *,
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    max_chars: int = _INSPECT_MAX_CHARS,
) -> str:
    """Return format metadata and a listing of every section.key=value pair."""

    try:
        resolved = _resolve_session_source(
            INI_CONFIG,
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
        )
        document = _read_source_document(resolved)
        entries = _parse_document_text(document, resolved.name)
    except SourceEditError as exc:
        return f"Could not inspect .ini file: {exc}"
    except IniEditError as exc:
        return f"Could not inspect .ini file: {exc}"

    encoding = "UTF-8 with BOM" if document.encoding == "utf-8-sig" else "UTF-8"
    final_newline = "yes" if document.has_final_newline else "no"
    lines = [
        f".ini file {resolved.name}:",
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


def edit_ini_file(
    path: str,
    *,
    operations: list[IniEditOperation],
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
            INI_CONFIG,
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
            raise IniEditError(
                f"no changes were made because {len(problems)} operation(s) "
                f"did not match the file: {' '.join(problems)} Re-run "
                "inspect_ini_file and retry."
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
            INI_CONFIG,
            edited,
            session_root=session_root,
            source_path=resolved,
            output_name=output_name,
            max_bytes=max_bytes,
        )
    except SourceEditError as exc:
        logger.info(
            "ini_edit failed: thread=%s source=%s operations=%d reason=%s",
            thread_id or "(none)",
            source_name,
            len(operations),
            type(exc).__name__,
        )
        return SourceEditResult(content=f"Could not edit .ini file: {exc}")
    except IniEditError as exc:
        logger.info(
            "ini_edit failed: thread=%s source=%s operations=%d reason=%s",
            thread_id or "(none)",
            source_name,
            len(operations),
            type(exc).__name__,
        )
        return SourceEditResult(content=f"Could not edit .ini file: {exc}")

    size_bytes = published.stat().st_size
    logger.info(
        "ini_edit succeeded: thread=%s source=%s output=%s operations=%d bytes=%d",
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
            INI_CONFIG,
            thread_id=thread_id,
            filename=published.name,
            size_bytes=size_bytes,
        ),
    )


def _is_comment(line: str) -> bool:
    return line.startswith("#") or line.startswith(";")


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


def _parse_document_text(document: _SourceDocument, name: str) -> dict[str, str]:
    """Parse [section]-grouped key=value lines into 'section.key' entries.

    Keys before any [section] header are stored under their bare name.
    Duplicate keys within one section raise; duplicate section headers
    merge.
    """

    entries: dict[str, str] = {}
    section = ""
    seen_sections: set[str] = set()
    text = document.newline_text.join(document.lines)
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or _is_comment(line):
            continue
        section_match = _SECTION_PATTERN.fullmatch(line)
        if section_match:
            section = section_match.group("name").strip()
            if section in seen_sections:
                raise IniEditError(
                    f"{name!r} line {line_number} repeats the "
                    f"[{section}] section header."
                )
            seen_sections.add(section)
            continue
        match = _LINE_PATTERN.fullmatch(line)
        if not match:
            raise IniEditError(
                f"{name!r} line {line_number} is not a [section] header "
                f"or key=value pair: {raw_line[:80]!r}"
            )
        key = match.group("key").strip()
        if section:
            full_key = f"{section}.{key}"
        else:
            full_key = key
        if full_key in entries:
            raise IniEditError(
                f"{name!r} line {line_number} repeats key {full_key!r}."
            )
        value = match.group("value")
        entries[full_key] = "" if value is None else value.strip()
    return entries


def _parse_line(
    raw_line: str, section: str
) -> tuple[str, str] | None:
    """Parse one raw line into (addressed_key, value), or None for blank/comment."""

    line = raw_line.strip()
    if not line or _is_comment(line):
        return None
    section_match = _SECTION_PATTERN.fullmatch(line)
    if section_match:
        # Section headers are structural; treat them as non-entries here.
        return None
    match = _LINE_PATTERN.fullmatch(line)
    if not match:
        raise IniEditError(
            f"line is not a [section] header or key=value pair: {raw_line[:80]!r}"
        )
    key = match.group("key").strip()
    if section:
        key = f"{section}.{key}"
    value = match.group("value")
    return key, ("" if value is None else value.strip())


def _check_operation(
    entries: dict[str, str], operation: IniEditOperation
) -> str | None:
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
    operation: IniEditOperation,
) -> list[str]:
    """Apply one already-validated operation, returning the new line list."""

    # Re-validate against the working lines so sequential operations see
    # the effect of earlier ones.
    working_entries: dict[str, str] = {}
    section = ""
    for raw_line in lines:
        parsed = _safe_parse(raw_line, section)
        if parsed is not None:
            working_entries[parsed[0]] = parsed[1]
        section_match = _SECTION_PATTERN.fullmatch(raw_line.strip())
        if section_match:
            section = section_match.group("name").strip()
    problem = _check_operation(working_entries, operation)
    if problem:
        raise IniEditError(f"{operation.action} {operation.key}: {problem}")

    if operation.action == "set_value":
        bare_key = operation.key.rsplit(".", 1)[-1] if "." in operation.key else operation.key
        new_lines: list[str] = []
        replaced = False
        section = ""
        for raw_line in lines:
            section_match = _SECTION_PATTERN.fullmatch(raw_line.strip())
            if section_match:
                section = section_match.group("name").strip()
                new_lines.append(raw_line)
                continue
            parsed = _safe_parse(raw_line, section)
            if parsed is not None and parsed[0] == operation.key:
                # Keep the original separator style when replacing in place.
                match = _LINE_PATTERN.fullmatch(raw_line.strip())
                sep = match.group("sep") if match and match.group("sep") else "="
                new_lines.append(f"{bare_key}{sep}{operation.value}")
                replaced = True
            else:
                new_lines.append(raw_line)
        if not replaced:
            new_line = f"{bare_key}={operation.value}"
            if "." in operation.key:
                # Create the key inside its section: an existing section
                # receives the key after its last entry; a missing section
                # is appended as a new [section] header at the end.
                section_name = operation.key.rsplit(".", 1)[0]
                if section_name in _working_sections(lines):
                    insert_at = _section_insert_index(lines, section_name)
                    new_lines.insert(insert_at, new_line)
                else:
                    if new_lines and new_lines[-1].strip():
                        new_lines.append("")
                    new_lines.append(f"[{section_name}]")
                    new_lines.append(new_line)
            else:
                # Root-level key: keep it before the first [section] header.
                first_header = _first_section_index(lines)
                if first_header is None:
                    new_lines.append(new_line)
                else:
                    new_lines.insert(first_header, new_line)
        return new_lines

    # delete_key: drop the line entirely.
    kept: list[str] = []
    section = ""
    for raw_line in lines:
        section_match = _SECTION_PATTERN.fullmatch(raw_line.strip())
        if section_match:
            section = section_match.group("name").strip()
            kept.append(raw_line)
            continue
        parsed = _safe_parse(raw_line, section)
        if parsed is not None and parsed[0] == operation.key:
            continue
        kept.append(raw_line)
    return kept


def _working_sections(lines: list[str]) -> set[str]:
    """Return the set of section names declared by [section] headers."""

    sections: set[str] = set()
    for raw_line in lines:
        match = _SECTION_PATTERN.fullmatch(raw_line.strip())
        if match:
            sections.add(match.group("name").strip())
    return sections


def _first_section_index(lines: list[str]) -> int | None:
    """Index of the first [section] header line, or None when absent."""

    for index, raw_line in enumerate(lines):
        if _SECTION_PATTERN.fullmatch(raw_line.strip()):
            return index
    return None


def _section_insert_index(lines: list[str], section_name: str) -> int:
    """Index just past the last entry of the section, before the next header."""

    in_section = False
    insert_at = len(lines)
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        match = _SECTION_PATTERN.fullmatch(stripped)
        if match:
            if in_section:
                return index
            in_section = match.group("name").strip() == section_name
            if in_section:
                insert_at = index + 1
            continue
        if in_section and stripped and not _is_comment(stripped):
            insert_at = index + 1
    return insert_at


def _safe_parse(raw_line: str, section: str) -> tuple[str, str] | None:
    """Parse a line, treating malformed lines as non-entries."""

    try:
        return _parse_line(raw_line, section)
    except IniEditError:
        return None
