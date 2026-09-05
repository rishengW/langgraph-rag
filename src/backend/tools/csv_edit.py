"""Session-scoped CSV creation, inspection, and editing tools.

The tools accept only UTF-8 ``.csv`` files in the current chat session. New and
edited files are written to a temporary file, parsed again, and atomically
published as downloads. Existing files are never overwritten.

CSV parsing and writing use the standard-library :mod:`csv` module so that
quoting, embedded commas, and embedded newlines are handled correctly. A
best-effort delimiter sniff (comma/semicolon/tab/pipe) preserves the original
separator of an uploaded file; created files always use RFC 4180 commas.
"""

from __future__ import annotations

import codecs
import csv
import io
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import quote

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, model_validator

from ._files import FileAccessError, resolve_safe_path

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)

CSV_MIME_TYPE = "text/csv"
FILE_ARTIFACT_TYPE = "file"
FILE_ARTIFACT_VERSION = 1
FILE_ARTIFACT_KIND = "download"
FILE_ARTIFACT_PROVIDER = "chat_upload"

_SUFFIXES = (".csv",)
_DELIMITERS = (",", ";", "\t", "|")
_INSPECT_MAX_CHARS = 20_000
_MAX_OPERATIONS = 100
_MAX_ROWS = 20_000
_MAX_COLUMNS = 256
_MAX_CELL_CHARS = 10_000
_MAX_OUTPUT_VARIANTS = 100
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class CsvEditError(Exception):
    """Raised when a CSV file cannot be inspected or edited safely."""


@dataclass(frozen=True)
class CsvEditResult:
    """Internal content/artifact result for a CSV-file write."""

    content: str
    artifact: dict[str, object] | None = None


@dataclass(frozen=True)
class _CsvDocument:
    """Parsed rows plus the byte-level characteristics that edits preserve."""

    rows: tuple[tuple[str, ...], ...]
    encoding: Literal["utf-8", "utf-8-sig"]
    newline: Literal["LF", "CRLF"]
    delimiter: str
    has_final_newline: bool

    @property
    def newline_text(self) -> str:
        return "\r\n" if self.newline == "CRLF" else "\n"


def _validate_row(row: list[str], *, label: str) -> None:
    """Enforce column and per-cell length limits on a user-supplied row."""

    if len(row) > _MAX_COLUMNS:
        raise ValueError(f"{label} has {len(row)} columns; the limit is {_MAX_COLUMNS}.")
    for cell in row:
        if len(cell) > _MAX_CELL_CHARS:
            raise ValueError(f"{label} has a cell longer than {_MAX_CELL_CHARS} chars.")


class CsvEditOperation(BaseModel):
    """One row- or cell-oriented edit, addressed by the inspector's indexes."""

    action: Literal[
        "replace_row",
        "delete_row",
        "insert_before_row",
        "append_row",
        "update_cell",
    ] = Field(..., description="The kind of CSV edit to apply.")
    row_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Zero-based row number from inspect_csv_file. Required for every "
            "action except append_row."
        ),
    )
    column: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Zero-based column number from inspect_csv_file's COLUMNS listing. "
            "Required for update_cell."
        ),
    )
    expected_row: list[str] | None = Field(
        default=None,
        description=(
            "The exact current contents of the indexed row. Required for "
            "replace_row, delete_row, and insert_before_row."
        ),
    )
    expected_value: str | None = Field(
        default=None,
        max_length=_MAX_CELL_CHARS,
        description=("The exact current value of the targeted cell. Required for update_cell."),
    )
    new_row: list[str] | None = Field(
        default=None,
        description=(
            "The new complete row. Required for replace_row, insert_before_row, and append_row."
        ),
    )
    new_value: str | None = Field(
        default=None,
        max_length=_MAX_CELL_CHARS,
        description="The new cell value. Required for update_cell.",
    )

    @model_validator(mode="after")
    def _check_fields(self) -> CsvEditOperation:
        needs_index = self.action != "append_row"
        needs_new_row = self.action in ("replace_row", "insert_before_row", "append_row")
        missing: list[str] = []
        if needs_index and self.row_index is None:
            missing.append("row_index")
        if self.action == "update_cell" and self.column is None:
            missing.append("column")
        if (
            self.action != "append_row"
            and self.action != "update_cell"
            and self.expected_row is None
        ):
            missing.append("expected_row")
        if self.action == "update_cell" and self.expected_value is None:
            missing.append("expected_value")
        if needs_new_row and self.new_row is None:
            missing.append("new_row")
        if self.action == "update_cell" and self.new_value is None:
            missing.append("new_value")
        if missing:
            raise ValueError(f"{self.action} requires: {', '.join(missing)}.")

        if self.expected_row is not None:
            _validate_row(self.expected_row, label="expected_row")
        if self.new_row is not None:
            _validate_row(self.new_row, label="new_row")
        return self


class CsvInspectInput(BaseModel):
    """Input schema for the CSV-file inspector."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .csv file uploaded to this chat session, exactly as "
            "listed in the upload note."
        ),
    )
    max_chars: int = Field(
        default=_INSPECT_MAX_CHARS,
        ge=1,
        le=200_000,
        description="Maximum characters of the numbered listing to return.",
    )


class CsvCreateInput(BaseModel):
    """Input schema for creating a CSV file."""

    filename: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "Bare filename for the new CSV file. A .csv suffix is added when "
            "needed. Existing files are never overwritten."
        ),
    )
    headers: list[str] = Field(
        ...,
        min_length=1,
        max_length=_MAX_COLUMNS,
        description="Header row written as the first CSV record.",
    )
    rows: list[list[str]] = Field(
        default_factory=list,
        max_length=_MAX_ROWS,
        description="Data rows, each a list of cell strings aligned with the headers.",
    )

    @model_validator(mode="after")
    def _check_dimensions(self) -> CsvCreateInput:
        _validate_row(self.headers, label="headers")
        for position, row in enumerate(self.rows):
            _validate_row(row, label=f"rows[{position}]")
        return self


class CsvEditInput(BaseModel):
    """Input schema for the CSV-file editor."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .csv file uploaded to this chat session. The source is never modified."
        ),
    )
    operations: list[CsvEditOperation] = Field(
        ...,
        min_length=1,
        max_length=_MAX_OPERATIONS,
        description=(
            "Row and cell edits to validate against the original file and then "
            "apply as one transaction."
        ),
    )
    output_name: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Optional name for the edited copy. Defaults to "
            "'<original>.edited.csv'; a .csv suffix is enforced."
        ),
    )


def build_csv_edit_tools(
    settings: Settings,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> list[BaseTool]:
    """Create CSV tools only when enabled and safely scoped to a chat."""

    if not (settings.file_read_enabled and settings.csv_edit_enabled):
        return []
    if session_root is None or not _THREAD_ID_PATTERN.fullmatch(thread_id):
        return []

    file_root = Path(settings.file_read_root)
    max_bytes = settings.file_read_max_bytes

    def _run_create(
        filename: str,
        headers: list[str],
        rows: list[list[str]],
    ) -> tuple[str, dict[str, object] | None]:
        result = create_csv_file(
            filename,
            headers=headers,
            rows=rows,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
            thread_id=thread_id,
        )
        return result.content, result.artifact

    def _run_inspect(path: str, max_chars: int = _INSPECT_MAX_CHARS) -> str:
        return inspect_csv_file(
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    def _run_edit(
        path: str,
        operations: list[CsvEditOperation],
        output_name: str | None = None,
    ) -> tuple[str, dict[str, object] | None]:
        result = edit_csv_file(
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
        name="create_csv_file",
        description=(
            "Create a new UTF-8 .csv file in the current chat session and "
            "return it as a download. The first record is the header row. "
            "Existing files are never overwritten; a numbered filename is "
            "chosen on collision. Use only when the user explicitly asks to "
            "create a CSV file."
        ),
        args_schema=CsvCreateInput,
        response_format="content_and_artifact",
    )
    inspect_tool = StructuredTool.from_function(
        func=_run_inspect,
        name="inspect_csv_file",
        description=(
            "List the columns and numbered rows of a .csv file uploaded to "
            "this chat session. Call this before edit_csv_file to obtain the "
            "row indexes, column indexes, and exact current cell values "
            "required by edit operations."
        ),
        args_schema=CsvInspectInput,
    )
    edit_tool = StructuredTool.from_function(
        func=_run_edit,
        name="edit_csv_file",
        description=(
            "Apply structured row and cell edits to an uploaded .csv file and "
            "create a new downloadable copy without changing the original. "
            "Supports replace_row, delete_row, insert_before_row, append_row, "
            "and update_cell. Call inspect_csv_file first and supply exact "
            "expected_row / expected_value for targeted operations. Use only "
            "when the user explicitly asks to change the file."
        ),
        args_schema=CsvEditInput,
        response_format="content_and_artifact",
    )
    return [create_tool, inspect_tool, edit_tool]


def create_csv_file(
    filename: str,
    *,
    headers: list[str],
    rows: list[list[str]],
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    thread_id: str = "",
) -> CsvEditResult:
    """Create a session-scoped UTF-8 CSV file and downloadable artifact."""

    output_name = "(unresolved)"
    try:
        if not _THREAD_ID_PATTERN.fullmatch(thread_id):
            raise CsvEditError("a valid chat thread is required to create a CSV file.")
        if len(rows) > _MAX_ROWS:
            raise CsvEditError(f"the CSV has {len(rows)} rows; the limit is {_MAX_ROWS}.")
        _validate_row(headers, label="headers")
        for position, row in enumerate(rows):
            _validate_row(row, label=f"rows[{position}]")
        directory = _prepare_session_directory(
            session_root=session_root,
            file_root=file_root,
        )
        stem = _creation_stem(filename)
        document = _CsvDocument(
            rows=(tuple(headers),) + tuple(tuple(row) for row in rows),
            encoding="utf-8",
            newline="LF",
            delimiter=",",
            has_final_newline=True,
        )
        target = _reserve_created_output_path(directory=directory, stem=stem)
        output_name = target.name
        published = _write_csv_atomically(
            document,
            target=target,
            max_bytes=max_bytes,
            size_label="new",
        )
        size_bytes = published.stat().st_size
    except (OSError, UnicodeError, CsvEditError) as exc:
        logger.info(
            "csv_create failed: thread=%s output=%s reason=%s",
            thread_id or "(none)",
            output_name,
            type(exc).__name__,
        )
        return CsvEditResult(content=f"Could not create CSV file: {exc}")

    logger.info(
        "csv_create succeeded: thread=%s output=%s rows=%d bytes=%d",
        thread_id,
        published.name,
        len(document.rows),
        size_bytes,
    )
    return CsvEditResult(
        content=(
            f"Created {published.name} in this chat session "
            f"({len(document.rows)} records including the header). "
            "The file is available to download."
        ),
        artifact=_build_file_artifact(
            thread_id=thread_id,
            filename=published.name,
            size_bytes=size_bytes,
        ),
    )


def inspect_csv_file(
    path: str,
    *,
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    max_chars: int = _INSPECT_MAX_CHARS,
) -> str:
    """Return format metadata, columns, and a zero-based listing of rows."""

    try:
        resolved = _resolve_session_csv(
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
        )
        document = _read_csv_document(resolved)
    except CsvEditError as exc:
        return f"Could not inspect CSV file: {exc}"

    encoding = "UTF-8 with BOM" if document.encoding == "utf-8-sig" else "UTF-8"
    final_newline = "yes" if document.has_final_newline else "no"
    column_count = len(document.rows[0]) if document.rows else 0
    columns = _column_listing(document)
    lines = [
        f"CSV file {resolved.name}:",
        f"Encoding: {encoding}",
        f"Newline: {document.newline}",
        f"Delimiter: {document.delimiter!r}",
        f"Final newline: {final_newline}",
        f"COLUMNS ({column_count}): {columns}",
        f"ROWS ({len(document.rows)}):",
    ]
    lines.extend(
        f"[{index}] {json.dumps(list(row), ensure_ascii=False)}"
        for index, row in enumerate(document.rows)
    )
    body = "\n".join(lines)
    limit = max(1, int(max_chars))
    if len(body) > limit:
        return f"{body[:limit]}\n\n[listing truncated to {limit:,} of {len(body):,} chars]"
    return body


def edit_csv_file(
    path: str,
    *,
    operations: list[CsvEditOperation],
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    thread_id: str = "",
    output_name: str | None = None,
) -> CsvEditResult:
    """Validate and apply a batch, then publish a copy-on-write CSV artifact."""

    source_name = "(unresolved)"
    try:
        resolved = _resolve_session_csv(
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
        )
        source_name = resolved.name
        document = _read_csv_document(resolved)
        plan = _plan_operations(document, operations)
        edited = _apply_operations(document, plan=plan)
        published = _publish_csv(
            edited,
            session_root=session_root,
            source_path=resolved,
            output_name=output_name,
            max_bytes=max_bytes,
        )
    except CsvEditError as exc:
        logger.info(
            "csv_edit failed: thread=%s source=%s operations=%d reason=%s",
            thread_id or "(none)",
            source_name,
            len(operations),
            type(exc).__name__,
        )
        return CsvEditResult(content=f"Could not edit CSV file: {exc}")

    size_bytes = published.stat().st_size
    logger.info(
        "csv_edit succeeded: thread=%s source=%s output=%s operations=%d bytes=%d",
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
    return CsvEditResult(
        content=content,
        artifact=_build_file_artifact(
            thread_id=thread_id,
            filename=published.name,
            size_bytes=size_bytes,
        ),
    )


def _column_listing(document: _CsvDocument) -> str:
    """Render the header row as a numbered column list for the inspector."""

    if not document.rows:
        return "(none)"
    header = document.rows[0]
    return ", ".join(
        f"[{i}] {json.dumps(cell, ensure_ascii=False)}" for i, cell in enumerate(header)
    )


def _resolve_session_csv(
    raw_path: str,
    *,
    session_root: Path,
    file_root: Path,
    max_bytes: int,
) -> Path:
    """Resolve a bare or upload-note path within this session directory."""

    text = (raw_path or "").strip().strip('"').strip("'")
    if not text:
        raise CsvEditError("a non-empty file path is required.")

    try:
        session_resolved = session_root.expanduser().resolve()
    except OSError as exc:
        raise CsvEditError(f"could not resolve the session directory: {exc}") from exc

    candidate = Path(text).expanduser()
    candidates = (
        [candidate]
        if candidate.is_absolute()
        else [session_resolved / candidate, file_root.expanduser() / candidate]
    )
    access_errors: list[FileAccessError] = []
    found_in_scope = False
    for option in candidates:
        try:
            resolved = option.resolve()
        except OSError:
            continue
        if not _is_within(resolved, session_resolved):
            continue
        found_in_scope = True
        try:
            return resolve_safe_path(
                str(resolved),
                root=session_resolved,
                expected_suffixes=_SUFFIXES,
                max_bytes=max_bytes,
            )
        except FileAccessError as exc:
            access_errors.append(exc)

    if not found_in_scope:
        raise CsvEditError(
            f"access denied: {text!r} is outside this chat session's CSV-file "
            "directory. Only files uploaded to this session can be edited."
        )
    if access_errors:
        raise CsvEditError(str(access_errors[-1])) from access_errors[-1]
    raise CsvEditError(f"could not resolve {text!r} inside this chat session.")


def _read_csv_document(path: Path, *, delimiter: str | None = None) -> _CsvDocument:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CsvEditError(f"could not read {path.name!r}: {exc}") from exc
    return _decode_csv(data, name=path.name, delimiter=delimiter)


def _decode_csv(data: bytes, *, name: str, delimiter: str | None = None) -> _CsvDocument:
    """Strictly decode UTF-8, reject binary, and parse rows with the csv module."""

    unsupported_boms = (
        codecs.BOM_UTF16_LE,
        codecs.BOM_UTF16_BE,
        codecs.BOM_UTF32_LE,
        codecs.BOM_UTF32_BE,
    )
    if any(data.startswith(bom) for bom in unsupported_boms):
        raise CsvEditError(f"{name!r} uses an unsupported encoding; only UTF-8 text is supported.")

    has_bom = data.startswith(codecs.BOM_UTF8)
    payload = data[len(codecs.BOM_UTF8) :] if has_bom else data
    if b"\x00" in payload:
        raise CsvEditError(f"{name!r} appears to be binary data (contains NUL bytes).")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CsvEditError(
            f"{name!r} uses an unsupported encoding or contains invalid UTF-8."
        ) from exc

    prohibited = [
        character
        for character in text
        if (ord(character) < 32 and character not in "\t\r\n")
        or ord(character) in (0x7F, 0xFFFE, 0xFFFF)
    ]
    if prohibited:
        raise CsvEditError(f"{name!r} appears to contain binary control characters.")

    without_crlf = text.replace("\r\n", "")
    if "\r" in without_crlf:
        raise CsvEditError(f"{name!r} uses unsupported bare-CR line endings.")
    has_crlf = "\r\n" in text
    if has_crlf and "\n" in without_crlf:
        raise CsvEditError(f"{name!r} mixes CRLF and LF line endings.")

    newline_text = "\r\n" if has_crlf else "\n"
    detected_delimiter = delimiter or _detect_delimiter(text)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=detected_delimiter)
    rows: list[tuple[str, ...]] = []
    for position, row in enumerate(reader):
        if len(row) > _MAX_COLUMNS:
            raise CsvEditError(
                f"{name!r} row {position} has {len(row)} columns; the limit is {_MAX_COLUMNS}."
            )
        rows.append(tuple(row))
    if len(rows) > _MAX_ROWS:
        raise CsvEditError(f"{name!r} has {len(rows)} rows; the limit is {_MAX_ROWS}.")
    final_newline = bool(text) and text.endswith(newline_text)
    return _CsvDocument(
        rows=tuple(rows),
        encoding="utf-8-sig" if has_bom else "utf-8",
        newline="CRLF" if has_crlf else "LF",
        delimiter=detected_delimiter,
        has_final_newline=final_newline,
    )


def _detect_delimiter(text: str) -> str:
    """Sniff the field separator, falling back to a first-line count."""

    sample = text[:4096] or text
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(_DELIMITERS)).delimiter
    except csv.Error:
        first_line = text.split("\n", 1)[0] if text else ""
        counts = {candidate: first_line.count(candidate) for candidate in _DELIMITERS}
        best, best_count = max(counts.items(), key=lambda item: item[1])
        return best if best_count > 0 else ","


@dataclass(frozen=True)
class _EditPlan:
    """Validated operations grouped by structural row edits and cell edits."""

    structural: dict[int, CsvEditOperation]
    cell_updates: list[tuple[int, int, str]]  # (row_index, column, new_value)
    appends: list[CsvEditOperation]


def _plan_operations(
    document: _CsvDocument,
    operations: list[CsvEditOperation],
) -> _EditPlan:
    """Validate all operations against original rows before any mutation.

    Two passes keep the conflict logic order-independent: first bucket every
    operation and detect duplicate targets, then validate ranges, expected
    values, and conflicts. A cell update conflicts with a ``replace_row`` or
    ``delete_row`` on the same row, but not with an ``insert_before_row`` on
    that row (the original row still exists after the insert).
    """

    structural: dict[int, CsvEditOperation] = {}
    cell_targets: dict[tuple[int, int], CsvEditOperation] = {}
    appends: list[CsvEditOperation] = []
    problems: list[str] = []

    for position, operation in enumerate(operations):
        label = f"operation {position} ({operation.action})"
        if operation.action == "append_row":
            appends.append(operation)
            continue
        assert operation.row_index is not None
        index = operation.row_index
        if operation.action == "update_cell":
            assert operation.column is not None
            target = (index, operation.column)
            if target in cell_targets:
                problems.append(
                    f"{label}: cell at row {index}, column {operation.column} "
                    "is updated more than once."
                )
                continue
            cell_targets[target] = operation
        else:
            if index in structural:
                problems.append(
                    f"{label}: row {index} is targeted by more than one structural operation."
                )
                continue
            structural[index] = operation

    replaced_deleted = {
        index
        for index, operation in structural.items()
        if operation.action in ("replace_row", "delete_row")
    }

    for position, operation in enumerate(operations):
        label = f"operation {position} ({operation.action})"
        if operation.action == "append_row":
            continue
        assert operation.row_index is not None
        index = operation.row_index
        if index >= len(document.rows):
            problems.append(
                f"{label}: row_index {index} is out of range; the file has "
                f"{len(document.rows)} rows."
            )
            continue
        if operation.action == "update_cell":
            assert operation.column is not None
            row = document.rows[index]
            if operation.column >= len(row):
                problems.append(
                    f"{label}: column {operation.column} is out of range for "
                    f"row {index}, which has {len(row)} columns."
                )
                continue
            if index in replaced_deleted:
                problems.append(
                    f"{label}: row {index} is also replaced or deleted in the same batch."
                )
                continue
            if row[operation.column] != operation.expected_value:
                problems.append(
                    f"{label}: expected_value does not exactly match row "
                    f"{index}, column {operation.column}."
                )
                continue
        else:
            if list(document.rows[index]) != operation.expected_row:
                problems.append(f"{label}: expected_row does not exactly match row {index}.")

    if problems:
        raise CsvEditError(
            f"no changes were made because {len(problems)} operation(s) did not "
            f"match the file: {' '.join(problems)} Re-run inspect_csv_file and retry."
        )

    cell_updates = [
        (operation.row_index, operation.column, operation.new_value)
        for operation in operations
        if operation.action == "update_cell"
        and operation.row_index is not None
        and operation.column is not None
        and operation.new_value is not None
    ]
    return _EditPlan(structural=structural, cell_updates=cell_updates, appends=appends)


def _apply_operations(document: _CsvDocument, *, plan: _EditPlan) -> _CsvDocument:
    """Apply validated cell updates first, then structural row operations."""

    rows: list[list[str]] = [list(row) for row in document.rows]
    for row_index, column, new_value in plan.cell_updates:
        rows[row_index][column] = new_value

    output: list[list[str]] = []
    for index, row in enumerate(rows):
        operation = plan.structural.get(index)
        if operation is None:
            output.append(row)
        elif operation.action == "replace_row":
            assert operation.new_row is not None
            output.append(list(operation.new_row))
        elif operation.action == "insert_before_row":
            assert operation.new_row is not None
            output.append(list(operation.new_row))
            output.append(row)
        else:  # delete_row
            continue
    output.extend(list(op.new_row) for op in plan.appends if op.new_row is not None)

    if len(output) > _MAX_ROWS:
        raise CsvEditError(f"the edited CSV has {len(output)} rows; the limit is {_MAX_ROWS}.")
    return _CsvDocument(
        rows=tuple(tuple(row) for row in output),
        encoding=document.encoding,
        newline=document.newline,
        delimiter=document.delimiter,
        has_final_newline=document.has_final_newline and bool(output),
    )


def _encode_csv(document: _CsvDocument) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(
        buffer,
        delimiter=document.delimiter,
        lineterminator=document.newline_text,
    )
    writer.writerows([list(row) for row in document.rows])
    text = buffer.getvalue()
    encoded = text.encode("utf-8")
    return codecs.BOM_UTF8 + encoded if document.encoding == "utf-8-sig" else encoded


def _publish_csv(
    document: _CsvDocument,
    *,
    session_root: Path,
    source_path: Path,
    output_name: str | None,
    max_bytes: int,
) -> Path:
    target = _reserve_output_path(
        session_root=session_root,
        source_path=source_path,
        output_name=output_name,
    )
    return _write_csv_atomically(
        document,
        target=target,
        max_bytes=max_bytes,
        size_label="edited",
    )


def _prepare_session_directory(*, session_root: Path, file_root: Path) -> Path:
    """Create and validate the session directory under the configured root."""

    try:
        root = file_root.expanduser().resolve()
        directory = session_root.expanduser().resolve()
    except OSError as exc:
        raise CsvEditError(f"could not resolve the session directory: {exc}") from exc
    if directory == root or not _is_within(directory, root):
        raise CsvEditError("the output directory is outside the configured file root.")
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CsvEditError(f"could not create the session directory: {exc}") from exc
    if not directory.is_dir():
        raise CsvEditError("the session output path is not a directory.")
    return directory


def _creation_stem(filename: str) -> str:
    """Sanitize a requested creation name and return its filename stem."""

    requested = Path(str(filename or "")).name
    if requested.lower().endswith(".csv"):
        requested = requested[: -len(".csv")]
    cleaned = _SAFE_NAME.sub("_", requested.strip().strip("."))[:150]
    if not cleaned:
        raise CsvEditError("a usable .csv filename is required.")
    return cleaned


def _reserve_created_output_path(*, directory: Path, stem: str) -> Path:
    """Exclusively reserve a free filename for a newly created CSV file."""

    for variant in range(1, _MAX_OUTPUT_VARIANTS + 1):
        name = f"{stem}.csv" if variant == 1 else f"{stem}-{variant}.csv"
        candidate = directory / name
        try:
            candidate.touch(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
        except OSError as exc:
            raise CsvEditError(f"could not create the output file: {exc}") from exc
    raise CsvEditError(
        f"too many files named {stem!r} already exist in this session; "
        "download or remove some before creating another."
    )


def _write_csv_atomically(
    document: _CsvDocument,
    *,
    target: Path,
    max_bytes: int,
    size_label: str,
) -> Path:
    """Write a reserved target through a validated temporary file."""

    try:
        handle, temp_name = tempfile.mkstemp(
            prefix=".csv_write-",
            suffix=".csv.tmp",
            dir=str(target.parent),
        )
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise CsvEditError(f"could not create a temporary file: {exc}") from exc

    temp_path = Path(temp_name)
    try:
        data = _encode_csv(document)
        if len(data) > max_bytes:
            raise CsvEditError(
                f"the {size_label} CSV file is too large ({len(data):,} bytes; "
                f"limit {max_bytes:,} bytes)."
            )
        stream = os.fdopen(handle, "wb")
        handle = -1
        with stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        validated = _read_csv_document(temp_path, delimiter=document.delimiter)
        if validated != document:
            raise CsvEditError("the saved CSV file did not pass validation.")
        os.replace(temp_path, target)
    except CsvEditError:
        if handle >= 0:
            os.close(handle)
        temp_path.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise
    except (OSError, ValueError) as exc:
        if handle >= 0:
            os.close(handle)
        temp_path.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise CsvEditError(f"could not save the CSV file: {exc}") from exc
    return target


def _reserve_output_path(
    *,
    session_root: Path,
    source_path: Path,
    output_name: str | None,
) -> Path:
    stem = _output_stem(source_path=source_path, output_name=output_name)
    try:
        directory = session_root.expanduser().resolve()
        resolved_source = source_path.resolve()
    except OSError as exc:
        raise CsvEditError(f"could not resolve the output directory: {exc}") from exc
    if not directory.is_dir() or not _is_within(resolved_source, directory):
        raise CsvEditError("the output directory is outside this chat session.")

    for variant in range(1, _MAX_OUTPUT_VARIANTS + 1):
        name = f"{stem}.csv" if variant == 1 else f"{stem}-{variant}.csv"
        candidate = directory / name
        if candidate.resolve() == resolved_source:
            continue
        try:
            candidate.touch(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
        except OSError as exc:
            raise CsvEditError(f"could not create the output file: {exc}") from exc
    raise CsvEditError(
        f"too many edited copies of {source_path.name!r} already exist in this "
        "session; download or remove some before editing again."
    )


def _output_stem(*, source_path: Path, output_name: str | None) -> str:
    if output_name:
        requested = Path(str(output_name)).name
        if requested.lower().endswith(".csv"):
            requested = requested[: -len(".csv")]
        cleaned = _SAFE_NAME.sub("_", requested.strip().strip("."))[:150]
        if cleaned:
            return cleaned
    return f"{source_path.stem}.edited"


def _build_file_artifact(*, thread_id: str, filename: str, size_bytes: int) -> dict[str, object]:
    return {
        "type": FILE_ARTIFACT_TYPE,
        "version": FILE_ARTIFACT_VERSION,
        "kind": FILE_ARTIFACT_KIND,
        "provider": FILE_ARTIFACT_PROVIDER,
        "threadId": thread_id,
        "filename": filename,
        "mimeType": CSV_MIME_TYPE,
        "sizeBytes": size_bytes,
        "url": f"/chat/{quote(thread_id, safe='')}/files/{quote(filename, safe='')}",
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


__all__ = [
    "CSV_MIME_TYPE",
    "CsvCreateInput",
    "CsvEditError",
    "CsvEditInput",
    "CsvEditOperation",
    "CsvEditResult",
    "CsvInspectInput",
    "build_csv_edit_tools",
    "create_csv_file",
    "edit_csv_file",
    "inspect_csv_file",
]
