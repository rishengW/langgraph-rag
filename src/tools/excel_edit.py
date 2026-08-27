"""Session-scoped Excel (.xlsx) inspection and editing tools.

Edits are deliberately conservative: the model must inspect a workbook first
and provide the exact current value for every cell it changes.  The uploaded
source is never overwritten; a validated edited copy is published inside the
current chat session.  Cell values, formulas, rows, columns, worksheets,
column widths, and number formats can all be edited.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal
from urllib.parse import quote

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from ._files import FileAccessError, resolve_safe_path

if TYPE_CHECKING:
    from ..config import Settings

logger = logging.getLogger(__name__)

XLSX_SUFFIXES = (".xlsx",)
XLSX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
FILE_ARTIFACT_TYPE = "file"
FILE_ARTIFACT_VERSION = 1
FILE_ARTIFACT_KIND = "download"
FILE_ARTIFACT_PROVIDER = "chat_upload"
_MAX_OPERATIONS = 50
_MAX_INSPECT_CHARS = 30_000
_MAX_INSPECT_ROWS = 200
_MAX_INSPECT_COLS = 30
_MAX_CELL_CHARS = 50_000
_MAX_FORMULA_CHARS = 4_000
_MAX_STRUCTURAL_COUNT = 1_000
_MAX_ZIP_ENTRIES = 1024
_MAX_UNCOMPRESSED_BYTES = 100_000_000
_MAX_COMPRESSION_RATIO = 200
_MAX_OUTPUT_STEM_CHARS = 150
_MAX_PORTABLE_PATH_CHARS = 240
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_CELL_PATTERN = re.compile(r"^[A-Za-z]{1,3}[1-9][0-9]{0,6}$")
_SHEET_BAD_CHARS = re.compile(r"[\\/*?:\[\]]")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# A spreadsheet cell holds a string, number, boolean, or nothing.  Strict
# types keep JSON ``1`` from silently becoming ``True`` and vice versa.
CellValue = Annotated[
    StrictStr | StrictInt | StrictFloat | StrictBool | None,
    Field(description="A string, number, boolean, or blank cell value."),
]


class ExcelEditError(Exception):
    """Raised when a workbook cannot be safely inspected or edited."""


class ExcelEditResult:
    def __init__(self, content: str, artifact: dict[str, object] | None = None) -> None:
        self.content = content
        self.artifact = artifact


class ExcelEditOperation(BaseModel):
    action: Literal[
        "set_cell",
        "set_formula",
        "clear_cell",
        "insert_rows",
        "delete_rows",
        "insert_columns",
        "delete_columns",
        "rename_sheet",
        "add_sheet",
        "delete_sheet",
        "set_column_width",
        "set_number_format",
    ] = Field(..., description="The kind of edit to apply.")
    sheet: str | None = Field(
        default=None,
        min_length=1,
        max_length=31,
        description=(
            "Worksheet name from inspect_excel_spreadsheet. Required for every "
            "action except add_sheet."
        ),
    )
    cell: str | None = Field(
        default=None,
        max_length=12,
        description=(
            "A1 cell address such as C4. Required for set_cell, set_formula, "
            "and clear_cell."
        ),
    )
    value: CellValue = Field(
        default=None,
        description="New value for set_cell: a string, number, boolean, or None.",
    )
    formula: str | None = Field(
        default=None,
        max_length=_MAX_FORMULA_CHARS,
        description="Formula for set_formula; must start with '=' such as '=SUM(A1:A10)'.",
    )
    expected_value: CellValue = Field(
        default=None,
        description=(
            "Current cell value reported by inspect_excel_spreadsheet. Required "
            "for set_cell, set_formula, and clear_cell so stale edits are rejected."
        ),
    )
    row_index: int | None = Field(
        default=None,
        ge=1,
        description="1-based row number for insert_rows / delete_rows.",
    )
    column_index: int | None = Field(
        default=None,
        ge=1,
        description=(
            "1-based column number for insert_columns / delete_columns / "
            "set_column_width / set_number_format."
        ),
    )
    count: int = Field(
        default=1,
        ge=1,
        le=_MAX_STRUCTURAL_COUNT,
        description="Number of rows or columns to insert or delete.",
    )
    new_sheet_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=31,
        description="New worksheet name for rename_sheet or add_sheet.",
    )
    number_format: str | None = Field(
        default=None,
        max_length=100,
        description="Excel number format string for set_number_format, e.g. '#,##0.00'.",
    )
    column_width: float | None = Field(
        default=None,
        ge=0,
        le=255,
        description="Column width for set_column_width (0 to 255).",
    )

    @model_validator(mode="after")
    def validate_operation(self) -> ExcelEditOperation:
        action = self.action
        if action != "add_sheet" and not (self.sheet and self.sheet.strip()):
            raise ValueError(f"action {action!r} requires a worksheet name.")
        if action in ("set_cell", "set_formula", "clear_cell"):
            if not self.cell:
                raise ValueError(f"action {action!r} requires a cell address.")
            _require_cell_address(self.cell)
        if action == "set_cell":
            if self.value is None:
                raise ValueError("set_cell requires a value (use clear_cell to blank a cell).")
            if isinstance(self.value, str):
                if self.value.startswith("="):
                    raise ValueError("set_cell values must not start with '='; use the set_formula action.")
                if len(self.value) > _MAX_CELL_CHARS or _CONTROL_CHARS.search(self.value):
                    raise ValueError("cell value is too long or contains control characters.")
        if action == "set_formula":
            if not self.formula or not self.formula.startswith("="):
                raise ValueError("set_formula requires a formula starting with '='.")
            if _CONTROL_CHARS.search(self.formula):
                raise ValueError("formula contains unsupported control characters.")
            # '|' is the DDE-command marker (e.g. =cmd|'/c calc'!A1) and is not
            # a valid Excel formula operator, so any formula containing it is
            # rejected to block DDE execution when the file is opened in Excel.
            if "|" in self.formula:
                raise ValueError("formulas containing '|' (DDE commands) are not allowed.")
        if action in ("insert_rows", "delete_rows") and not self.row_index:
            raise ValueError(f"action {action!r} requires row_index.")
        if action in (
            "insert_columns",
            "delete_columns",
            "set_column_width",
            "set_number_format",
        ) and not self.column_index:
            raise ValueError(f"action {action!r} requires column_index.")
        if action in ("rename_sheet", "add_sheet") and not (
            self.new_sheet_name and self.new_sheet_name.strip()
        ):
            raise ValueError(f"action {action!r} requires new_sheet_name.")
        if action == "set_column_width" and self.column_width is None:
            raise ValueError("set_column_width requires column_width.")
        if action == "set_number_format" and not (
            self.number_format and self.number_format.strip()
        ):
            raise ValueError("set_number_format requires number_format.")
        if self.number_format and _CONTROL_CHARS.search(self.number_format):
            raise ValueError("number_format contains unsupported control characters.")
        if self.sheet:
            _require_sheet_name(self.sheet)
        if self.new_sheet_name:
            _require_sheet_name(self.new_sheet_name)
        return self


class ExcelInspectInput(BaseModel):
    path: str = Field(..., min_length=1, description="Path to a .xlsx uploaded to this chat session.")
    max_chars: int = Field(default=_MAX_INSPECT_CHARS, ge=1, le=200_000)


class ExcelEditInput(BaseModel):
    path: str = Field(
        ...,
        min_length=1,
        description="Path to a .xlsx uploaded to this chat session. This file is never modified.",
    )
    operations: list[ExcelEditOperation] = Field(..., min_length=1, max_length=_MAX_OPERATIONS)
    output_name: str | None = Field(
        default=None,
        max_length=200,
        description="Optional name for the edited copy. Defaults to '<original>.edited.xlsx'.",
    )


def build_excel_edit_tools(
    settings: Settings, *, session_root: Path | None = None, thread_id: str = ""
) -> list[BaseTool]:
    if not (settings.file_read_enabled and settings.excel_edit_enabled):
        return []
    if session_root is None or not _THREAD_ID_PATTERN.fullmatch(thread_id):
        return []
    file_root = Path(settings.file_read_root)

    def _inspect(path: str, max_chars: int = _MAX_INSPECT_CHARS) -> str:
        return inspect_excel(
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=settings.file_read_max_bytes,
            max_chars=max_chars,
        )

    def _edit(
        path: str,
        operations: list[ExcelEditOperation],
        output_name: str | None = None,
    ) -> tuple[str, dict[str, object] | None]:
        result = edit_excel(
            path,
            operations=operations,
            session_root=session_root,
            file_root=file_root,
            max_bytes=settings.file_read_max_bytes,
            thread_id=thread_id,
            output_name=output_name,
        )
        return result.content, result.artifact

    return [
        StructuredTool.from_function(
            func=_inspect,
            name="inspect_excel_spreadsheet",
            description=(
                "List worksheets, cells, values, and formulas in an uploaded .xlsx. "
                "Call this before edit_excel_spreadsheet to obtain exact cell "
                "addresses and the expected_value for every cell you intend to change."
            ),
            args_schema=ExcelInspectInput,
        ),
        StructuredTool.from_function(
            func=_edit,
            name="edit_excel_spreadsheet",
            description=(
                "Apply structured cell, formula, row/column, worksheet, column-width, "
                "and number-format edits to an uploaded .xlsx and save a new "
                "downloadable copy. The original is never changed. Always call "
                "inspect_excel_spreadsheet first and pass its exact expected_value "
                "for set_cell, set_formula, and clear_cell operations."
            ),
            args_schema=ExcelEditInput,
            response_format="content_and_artifact",
        ),
    ]


def inspect_excel(
    path: str,
    *,
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    max_chars: int = _MAX_INSPECT_CHARS,
) -> str:
    try:
        resolved = _resolve_session_xlsx(
            path, session_root=session_root, file_root=file_root, max_bytes=max_bytes
        )
        workbook = _load_workbook(resolved)
    except ExcelEditError as exc:
        return f"Could not inspect spreadsheet: {exc}"
    try:
        return _format_workbook(resolved.name, workbook, max_chars)
    finally:
        workbook.close()


def edit_excel(
    path: str,
    *,
    operations: list[ExcelEditOperation],
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    thread_id: str = "",
    output_name: str | None = None,
) -> ExcelEditResult:
    source_name = "(unresolved)"
    workbook: Any = None
    try:
        resolved = _resolve_session_xlsx(
            path, session_root=session_root, file_root=file_root, max_bytes=max_bytes
        )
        source_name = resolved.name
        workbook = _load_workbook(resolved)
        _apply_operations(workbook, operations)
        published = _publish(
            workbook,
            session_root=session_root,
            source_path=resolved,
            output_name=output_name,
            max_bytes=max_bytes,
        )
    except ExcelEditError as exc:
        logger.info(
            "excel_edit failed: thread=%s source=%s reason=%s",
            thread_id or "(none)",
            source_name,
            type(exc).__name__,
        )
        return ExcelEditResult(f"Could not edit spreadsheet: {exc}")
    finally:
        if workbook is not None:
            workbook.close()
    summary = ", ".join(op.action for op in operations)
    size = published.stat().st_size
    return ExcelEditResult(
        f"Applied {len(operations)} edit(s) ({summary}) to {source_name}. The original "
        f"file is unchanged; the edited copy was saved as {published.name} and is "
        "available to download.",
        _build_file_artifact(thread_id=thread_id, filename=published.name, size_bytes=size),
    )


def _apply_operations(workbook: Any, operations: list[ExcelEditOperation]) -> None:
    for index, operation in enumerate(operations):
        try:
            _apply_operation(workbook, operation)
        except ExcelEditError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface any openpyxl failure as a clear error
            raise ExcelEditError(
                f"operation {index} ({operation.action}) failed: {exc}"
            ) from exc


def _apply_operation(workbook: Any, operation: ExcelEditOperation) -> None:
    action = operation.action

    if action == "add_sheet":
        new_name = _require_sheet_name(operation.new_sheet_name or "")
        if _sheet_exists(workbook, new_name):
            raise ExcelEditError(f"a worksheet named {new_name!r} already exists.")
        workbook.create_sheet(title=new_name)
        return

    if action == "delete_sheet":
        name = operation.sheet or ""
        if not _sheet_exists(workbook, name):
            raise ExcelEditError(f"worksheet {name!r} not found.")
        if len(workbook.sheetnames) <= 1:
            raise ExcelEditError("the last worksheet cannot be deleted.")
        del workbook[name]
        return

    worksheet = _require_sheet(workbook, operation.sheet or "")

    if action == "rename_sheet":
        new_name = _require_sheet_name(operation.new_sheet_name or "")
        if _sheet_exists(workbook, new_name):
            raise ExcelEditError(f"a worksheet named {new_name!r} already exists.")
        worksheet.title = new_name
        return

    if action in ("set_cell", "set_formula", "clear_cell"):
        address = _require_cell_address(operation.cell or "")
        cell = worksheet[address]
        if not _values_equal(cell.value, operation.expected_value):
            raise ExcelEditError(
                f"cell {address} on {worksheet.title!r} currently holds "
                f"{cell.value!r}, not the expected {operation.expected_value!r}; "
                "re-inspect and retry."
            )
        if action == "set_cell":
            cell.value = operation.value
        elif action == "set_formula":
            cell.value = operation.formula
        else:
            cell.value = None
        return

    if action == "insert_rows":
        worksheet.insert_rows(int(operation.row_index), amount=int(operation.count))
        return
    if action == "delete_rows":
        worksheet.delete_rows(int(operation.row_index), amount=int(operation.count))
        return
    if action == "insert_columns":
        worksheet.insert_cols(int(operation.column_index), amount=int(operation.count))
        return
    if action == "delete_columns":
        worksheet.delete_cols(int(operation.column_index), amount=int(operation.count))
        return

    if action == "set_column_width":
        letter = _column_letter(int(operation.column_index))
        worksheet.column_dimensions[letter].width = float(operation.column_width)
        return

    if action == "set_number_format":
        letter = _column_letter(int(operation.column_index))
        worksheet.column_dimensions[letter].number_format = operation.number_format
        return

    raise ExcelEditError(f"unknown action {action!r}.")


def _format_workbook(file_name: str, workbook: Any, max_chars: int) -> str:
    sheet_names = list(workbook.sheetnames)
    lines: list[str] = [f"WORKSHEETS ({len(sheet_names)}):"]
    for index, name in enumerate(sheet_names):
        lines.append(f"[{index}] {name}")
    limit = max(1, int(max_chars))
    overflow = False
    for name in sheet_names:
        worksheet = workbook[name]
        lines.append("")
        lines.append(f"Worksheet {name!r}:")
        for line in _format_sheet_cells(worksheet):
            lines.append(line)
            if sum(len(item) + 1 for item in lines) > limit:
                overflow = True
                break
        if overflow:
            break
    body = "\n".join(lines)
    if len(body) > limit:
        return (
            f"Structure of {file_name} (truncated to {limit:,} of {len(body):,} chars):\n\n"
            f"{body[:limit]}"
        )
    return f"Structure of {file_name}:\n\n{body}"


def _format_sheet_cells(worksheet: Any) -> list[str]:
    max_row = min(worksheet.max_row or 0, _MAX_INSPECT_ROWS)
    max_col = min(worksheet.max_column or 0, _MAX_INSPECT_COLS)
    if max_row < 1 or max_col < 1:
        return ["(the sheet is empty)"]
    out: list[str] = []
    for row in worksheet.iter_rows(min_row=1, max_row=max_row, min_col=1, max_col=max_col):
        for cell in row:
            value = cell.value
            if value is None:
                continue
            if isinstance(value, str) and value.startswith("="):
                out.append(f"[{cell.coordinate}] {value}  (formula)")
            else:
                out.append(f"[{cell.coordinate}] {value}")
    if not out:
        return ["(no non-empty cells in the inspected range)"]
    full_rows = worksheet.max_row or 0
    full_cols = worksheet.max_column or 0
    if full_rows > _MAX_INSPECT_ROWS or full_cols > _MAX_INSPECT_COLS:
        out.append(
            f"(showing first {max_row} of {full_rows} rows and first {max_col} "
            f"of {full_cols} columns)"
        )
    return out


def _resolve_session_xlsx(
    raw_path: str, *, session_root: Path, file_root: Path, max_bytes: int
) -> Path:
    text = (raw_path or "").strip().strip('"').strip("'")
    if not text:
        raise ExcelEditError("a non-empty file path is required.")
    try:
        session = session_root.expanduser().resolve()
    except OSError as exc:
        raise ExcelEditError(f"could not resolve the session directory: {exc}") from exc
    candidate = Path(text).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [session / candidate, file_root / candidate]
    access_errors: list[FileAccessError] = []
    found_in_scope = False
    for option in candidates:
        try:
            resolved = option.resolve()
        except OSError:
            continue
        if not _is_within(resolved, session):
            continue
        found_in_scope = True
        try:
            return resolve_safe_path(
                str(resolved), root=session, expected_suffixes=XLSX_SUFFIXES, max_bytes=max_bytes
            )
        except FileAccessError as exc:
            access_errors.append(exc)
    if not found_in_scope:
        raise ExcelEditError("access denied: the spreadsheet is not in this chat session.")
    if access_errors:
        raise ExcelEditError(str(access_errors[-1])) from access_errors[-1]
    raise ExcelEditError(f"could not resolve {text!r} inside this chat session.")


def _load_workbook(path: Path) -> Any:
    _guard_archive(path)
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise ExcelEditError(
            "openpyxl is not installed; run `python -m pip install openpyxl` to "
            "enable the Excel edit tool."
        ) from exc
    try:
        return load_workbook(path, read_only=False, data_only=False)
    except Exception as exc:  # noqa: BLE001 - any openpyxl parse failure is a user-facing error
        raise ExcelEditError(f"invalid or unreadable .xlsx file: {exc}") from exc


def _guard_archive(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > _MAX_ZIP_ENTRIES:
                raise ExcelEditError("workbook contains too many archive entries.")
            total = 0
            # Stored entries (compress_size == 0) are not compression bombs
            # (their ratio is 1) and their on-disk size is already bounded by
            # file_read_max_bytes, so only deflated entries are ratio-checked.
            # Rejecting every stored entry (as the PowerPoint guard does) would
            # risk refusing real workbooks that store small parts uncompressed.
            for info in entries:
                total += info.file_size
                if info.compress_size and info.file_size / info.compress_size > _MAX_COMPRESSION_RATIO:
                    raise ExcelEditError("workbook has an unsafe compression ratio.")
            if total > _MAX_UNCOMPRESSED_BYTES:
                raise ExcelEditError("workbook expands beyond the safety limit.")
            names = archive.namelist()
            if "[Content_Types].xml" not in names or "xl/workbook.xml" not in names:
                raise ExcelEditError("file is missing required Excel parts.")
    except ExcelEditError:
        raise
    except (OSError, EOFError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ExcelEditError(f"file is not a readable Excel archive: {exc}") from exc


def _publish(
    workbook: Any,
    *,
    session_root: Path,
    source_path: Path,
    output_name: str | None,
    max_bytes: int,
) -> Path:
    directory = session_root.expanduser().resolve()
    source = source_path.resolve()
    if not directory.is_dir() or not _is_within(source, directory):
        raise ExcelEditError("the output directory is outside this chat session.")
    requested = Path(output_name or f"{source.stem}.edited").name
    if requested.lower().endswith(".xlsx"):
        requested = requested[: -len(".xlsx")]
    portable_stem_limit = _MAX_PORTABLE_PATH_CHARS - len(str(directory)) - len("-100.xlsx") - 1
    if portable_stem_limit < 1:
        raise ExcelEditError("the session directory path is too long for a safe output file.")
    stem_limit = min(_MAX_OUTPUT_STEM_CHARS, portable_stem_limit)
    stem = _SAFE_NAME.sub("_", requested.strip().strip("."))[:stem_limit]
    if not stem:
        raise ExcelEditError("a usable output filename is required.")
    target: Path | None = None
    for variant in range(1, 101):
        name = f"{stem}.xlsx" if variant == 1 else f"{stem}-{variant}.xlsx"
        candidate = directory / name
        if candidate.resolve() == source:
            continue
        try:
            candidate.touch(exist_ok=False)
            target = candidate
            break
        except FileExistsError:
            continue
        except OSError as exc:
            raise ExcelEditError(f"could not reserve the output file: {exc}") from exc
    if target is None:
        raise ExcelEditError("too many edited copies with this name exist in the session.")
    try:
        # openpyxl refuses to load files whose name does not end in a known
        # spreadsheet suffix, so the temp must use ".xlsx" (it is still uniquely
        # named and unlinked on any failure path below).
        handle, temp_name = tempfile.mkstemp(
            prefix=".excel_edit-", suffix=".xlsx", dir=str(directory)
        )
        os.close(handle)
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise ExcelEditError(f"could not create a temporary file: {exc}") from exc
    temp = Path(temp_name)
    try:
        workbook.save(str(temp))
        if temp.stat().st_size > max_bytes:
            raise ExcelEditError("the edited spreadsheet exceeds the file size limit.")
        # _load_workbook re-validates the archive (zip-bomb guard + parse), so
        # a corrupt workbook is never published.  Close it before the atomic
        # rename so the file handle is released on Windows.
        reopened = _load_workbook(temp)
        reopened.close()
        os.replace(temp, target)
    except ExcelEditError:
        temp.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise
    except (OSError, ValueError) as exc:
        temp.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise ExcelEditError(f"could not save the edited spreadsheet: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - never leak a raw openpyxl exception
        temp.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise ExcelEditError(f"could not save the edited spreadsheet: {exc}") from exc
    return target


def _require_sheet_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned or len(cleaned) > 31:
        raise ValueError("worksheet names must be 1 to 31 characters long.")
    if _SHEET_BAD_CHARS.search(cleaned) or cleaned.startswith("'") or cleaned.endswith("'"):
        raise ValueError("worksheet name contains characters that are invalid for Excel.")
    if _CONTROL_CHARS.search(cleaned):
        raise ValueError("worksheet name contains unsupported control characters.")
    return cleaned


def _require_cell_address(cell: str) -> str:
    cleaned = (cell or "").strip().upper()
    if not _CELL_PATTERN.fullmatch(cleaned):
        raise ValueError(f"{cell!r} is not a valid A1 cell address such as C4.")
    return cleaned


def _column_letter(index: int) -> str:
    from openpyxl.utils import get_column_letter

    return get_column_letter(index)


def _sheet_exists(workbook: Any, name: str) -> bool:
    return name in workbook.sheetnames


def _require_sheet(workbook: Any, name: str) -> Any:
    if not name:
        raise ExcelEditError("a worksheet name is required.")
    if name not in workbook.sheetnames:
        available = ", ".join(workbook.sheetnames) or "(none)"
        raise ExcelEditError(f"worksheet {name!r} not found. Available: {available}.")
    return workbook[name]


def _values_equal(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None
    if actual is None:
        return False
    if isinstance(actual, bool) and isinstance(expected, bool):
        return actual == expected
    if (
        isinstance(actual, (int, float))
        and isinstance(expected, (int, float))
        and not isinstance(actual, bool)
        and not isinstance(expected, bool)
    ):
        return float(actual) == float(expected)
    return str(actual).strip() == str(expected).strip()


def _build_file_artifact(*, thread_id: str, filename: str, size_bytes: int) -> dict[str, object]:
    return {
        "type": FILE_ARTIFACT_TYPE,
        "version": FILE_ARTIFACT_VERSION,
        "kind": FILE_ARTIFACT_KIND,
        "provider": FILE_ARTIFACT_PROVIDER,
        "threadId": thread_id,
        "filename": filename,
        "mimeType": XLSX_MIME_TYPE,
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
    "ExcelEditError",
    "ExcelEditInput",
    "ExcelEditOperation",
    "ExcelEditResult",
    "ExcelInspectInput",
    "XLSX_MIME_TYPE",
    "build_excel_edit_tools",
    "edit_excel",
    "inspect_excel",
]
