"""Session-scoped Excel workbook creation using the artifact-tool runtime."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
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

from ._files import FileAccessError

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)

XLSX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
FILE_ARTIFACT_TYPE = "file"
FILE_ARTIFACT_VERSION = 1
FILE_ARTIFACT_KIND = "download"
FILE_ARTIFACT_PROVIDER = "chat_upload"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_CELL_PATTERN = re.compile(r"^[A-Za-z]{1,3}[1-9][0-9]{0,6}$")
_SHEET_BAD_CHARS = re.compile(r"[\\/*?:\[\]]")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MAX_SHEETS = 20
_MAX_ROWS = 10_000
_MAX_COLUMNS = 100
_MAX_FORMULAS = 2_000
_MAX_CELL_CHARS = 50_000
_MAX_FORMULA_CHARS = 4_000
_MAX_OUTPUT_VARIANTS = 100

SpreadsheetValue = Annotated[
    StrictStr | StrictInt | StrictFloat | StrictBool | None,
    Field(description="A string, number, boolean, or blank cell value."),
]


class SpreadsheetFormula(BaseModel):
    cell: str = Field(..., min_length=2, max_length=12, description="A1 cell address.")
    formula: str = Field(..., min_length=2, max_length=_MAX_FORMULA_CHARS)
    number_format: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def validate_formula(self) -> SpreadsheetFormula:
        if not _CELL_PATTERN.fullmatch(self.cell):
            raise ValueError("cell must be a valid A1 address such as C4.")
        if not self.formula.startswith("=") or _CONTROL_CHARS.search(self.formula):
            raise ValueError("formula must start with '=' and contain no control characters.")
        if self.number_format and _CONTROL_CHARS.search(self.number_format):
            raise ValueError("number_format contains unsupported control characters.")
        self.cell = self.cell.upper()
        return self


class ColumnFormat(BaseModel):
    column: int = Field(..., ge=0, le=_MAX_COLUMNS - 1)
    format: str = Field(..., min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_format(self) -> ColumnFormat:
        if _CONTROL_CHARS.search(self.format):
            raise ValueError("format contains unsupported control characters.")
        return self


class SpreadsheetSheet(BaseModel):
    name: str = Field(..., min_length=1, max_length=31)
    rows: list[list[SpreadsheetValue]] = Field(..., min_length=1, max_length=_MAX_ROWS)
    formulas: list[SpreadsheetFormula] = Field(default_factory=list, max_length=_MAX_FORMULAS)
    header_row: bool = True
    number_formats: list[ColumnFormat] = Field(default_factory=list, max_length=_MAX_COLUMNS)
    column_widths: list[float] = Field(default_factory=list, max_length=_MAX_COLUMNS)

    @model_validator(mode="after")
    def validate_sheet(self) -> SpreadsheetSheet:
        if not self.name.strip() or _SHEET_BAD_CHARS.search(self.name) or self.name.startswith("'") or self.name.endswith("'"):
            raise ValueError("sheet name is invalid for Excel.")
        if _CONTROL_CHARS.search(self.name):
            raise ValueError("sheet name contains unsupported control characters.")
        width = len(self.rows[0])
        if width < 1 or width > _MAX_COLUMNS:
            raise ValueError(f"each sheet must have 1-{_MAX_COLUMNS} columns.")
        if any(len(row) != width for row in self.rows):
            raise ValueError("rows must be rectangular with the same column count.")
        for row in self.rows:
            for value in row:
                if isinstance(value, str) and (
                    len(value) > _MAX_CELL_CHARS or _CONTROL_CHARS.search(value)
                ):
                    raise ValueError("cell text is too long or contains control characters.")
        for formula in self.formulas:
            formula_row, formula_column = _parse_cell(formula.cell)
            if formula_row > len(self.rows) or formula_column >= width:
                raise ValueError(f"formula cell {formula.cell} is outside the supplied rows.")
        formula_cells = [formula.cell for formula in self.formulas]
        if len(set(formula_cells)) != len(formula_cells):
            raise ValueError("formula cell addresses must be unique within a worksheet.")
        if any(item.column >= width for item in self.number_formats):
            raise ValueError("number format columns must be inside the supplied rows.")
        if len(self.column_widths) > width:
            raise ValueError("column_widths cannot include columns outside the supplied rows.")
        for width_value in self.column_widths:
            if width_value <= 0 or width_value > 255:
                raise ValueError("column widths must be between 0 and 255.")
        return self


class SpreadsheetCreateInput(BaseModel):
    filename: str = Field(..., min_length=1, max_length=200)
    sheets: list[SpreadsheetSheet] = Field(..., min_length=1, max_length=_MAX_SHEETS)

    @model_validator(mode="after")
    def validate_workbook(self) -> SpreadsheetCreateInput:
        names = [sheet.name.casefold() for sheet in self.sheets]
        if len(set(names)) != len(names):
            raise ValueError("worksheet names must be unique, ignoring case.")
        return self


@dataclass(frozen=True)
class SpreadsheetCreateResult:
    content: str
    artifact: dict[str, object] | None = None


class SpreadsheetCreateError(Exception):
    """Raised when an XLSX workbook cannot be created safely."""


def build_excel_create_tools(settings: Settings, *, session_root: Path | None = None, thread_id: str = "") -> list[BaseTool]:
    if not (settings.file_read_enabled and settings.excel_create_enabled):
        return []
    if session_root is None or not _THREAD_ID_PATTERN.fullmatch(thread_id):
        return []

    def _run_create(filename: str, sheets: list[SpreadsheetSheet]) -> tuple[str, dict[str, object] | None]:
        result = create_excel_spreadsheet(
            filename,
            sheets=sheets,
            session_root=session_root,
            file_root=Path(settings.file_read_root),
            max_bytes=settings.file_read_max_bytes,
            thread_id=thread_id,
            node_executable=settings.excel_node_executable,
            node_modules_path=settings.excel_node_modules_path,
        )
        return result.content, result.artifact

    return [
        StructuredTool.from_function(
            func=_run_create,
            name="create_excel_spreadsheet",
            description=(
                "Create a new, professionally formatted Excel .xlsx workbook in the current "
                "chat session and return it as a download. Supports multiple worksheets, "
                "typed values, formulas, header styling, number formats, and column widths. "
                "Existing files are never overwritten. Use only when the user "
                "explicitly asks to create an Excel spreadsheet."
            ),
            args_schema=SpreadsheetCreateInput,
            response_format="content_and_artifact",
        )
    ]


def create_excel_spreadsheet(
    filename: str,
    *,
    sheets: list[SpreadsheetSheet],
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    thread_id: str,
    node_executable: str = "node",
    node_modules_path: str = "node_modules",
) -> SpreadsheetCreateResult:
    output_name = "(unresolved)"
    target: Path | None = None
    work_dir: Path | None = None
    try:
        request = SpreadsheetCreateInput(filename=filename, sheets=sheets)
        if not _THREAD_ID_PATTERN.fullmatch(thread_id):
            raise SpreadsheetCreateError("a valid chat thread is required to create a spreadsheet.")
        directory = _prepare_session_directory(session_root=session_root, file_root=file_root)
        target = _reserve_output_path(directory, _creation_stem(request.filename))
        output_name = target.name
        work_dir = Path(tempfile.mkdtemp(prefix=".excel-create-", dir=directory))
        input_path = work_dir / "request.json"
        output_temp = work_dir / "output.xlsx"
        inspect_path = work_dir / "inspect.ndjson"
        render_dir = work_dir / "renders"
        input_path.write_text(json.dumps(request.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8")
        adapter_path = work_dir / "builder.mjs"
        adapter_path.write_text(Path(__file__).with_suffix(".mjs").read_text(encoding="utf-8"), encoding="utf-8")
        _link_node_modules(work_dir / "node_modules", node_modules_path)
        completed = subprocess.run(
            [
                node_executable,
                str(adapter_path),
                str(input_path),
                str(output_temp),
                str(inspect_path),
                str(render_dir),
            ],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "artifact-tool failed").strip()
            raise SpreadsheetCreateError(detail[-2000:])
        if not output_temp.is_file():
            raise SpreadsheetCreateError("artifact-tool did not produce an XLSX file.")
        size = output_temp.stat().st_size
        if size > max_bytes:
            raise SpreadsheetCreateError(f"the new spreadsheet is too large ({size:,} bytes; limit {max_bytes:,} bytes).")
        _validate_xlsx(output_temp)
        os.replace(output_temp, target)
        return SpreadsheetCreateResult(
            content=f"Created {target.name} with {len(request.sheets)} worksheet(s). The Excel workbook is available to download.",
            artifact=_build_file_artifact(thread_id=thread_id, filename=target.name, size_bytes=size),
        )
    except (OSError, ValueError, FileAccessError, SpreadsheetCreateError, subprocess.SubprocessError) as exc:
        if target is not None:
            target.unlink(missing_ok=True)
        logger.info("excel_create failed: thread=%s output=%s reason=%s", thread_id or "(none)", output_name, type(exc).__name__)
        return SpreadsheetCreateResult(content=f"Could not create Excel spreadsheet: {exc}")
    finally:
        if work_dir is not None:
            shutil.rmtree(work_dir, ignore_errors=True)


def _prepare_session_directory(*, session_root: Path, file_root: Path) -> Path:
    root = file_root.expanduser().resolve()
    directory = session_root.expanduser().resolve()
    if directory == root or not _is_within(directory, root):
        raise SpreadsheetCreateError("the output directory is outside the configured file root.")
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise SpreadsheetCreateError("the session output path is not a directory.")
    return directory


def _creation_stem(filename: str) -> str:
    requested = Path(str(filename or "")).name
    if requested.lower().endswith(".xlsx"):
        requested = requested[:-5]
    cleaned = _SAFE_NAME.sub("_", requested.strip().strip("."))[:150]
    if not cleaned:
        raise SpreadsheetCreateError("a usable .xlsx filename is required.")
    return cleaned


def _reserve_output_path(directory: Path, stem: str) -> Path:
    for variant in range(1, _MAX_OUTPUT_VARIANTS + 1):
        candidate = directory / (f"{stem}.xlsx" if variant == 1 else f"{stem}-{variant}.xlsx")
        try:
            candidate.touch(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
        except OSError as exc:
            raise SpreadsheetCreateError(f"could not reserve the output file: {exc}") from exc
    raise SpreadsheetCreateError(f"too many files named {stem!r} already exist in this session.")


def _link_node_modules(link: Path, configured: str) -> None:
    if not configured.strip():
        raise SpreadsheetCreateError(
            "EXCEL_NODE_MODULES_PATH is not configured; point it to the "
            "loader-provided node_modules directory containing @oai/artifact-tool."
        )
    modules = Path(configured).expanduser()
    if not modules.is_absolute():
        modules = Path.cwd() / modules
    modules = modules.resolve()
    if not modules.is_dir():
        raise SpreadsheetCreateError(f"artifact-tool dependency directory not found: {modules}")
    try:
        link.symlink_to(modules, target_is_directory=True)
    except OSError:
        completed = subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(link), str(modules)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise SpreadsheetCreateError(
                "could not create the task-local node_modules junction."
            ) from None


def _validate_xlsx(path: Path) -> None:
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if "[Content_Types].xml" not in names or "xl/workbook.xml" not in names:
                raise SpreadsheetCreateError("artifact-tool output is not a valid XLSX archive.")
    except zipfile.BadZipFile as exc:
        raise SpreadsheetCreateError("artifact-tool output is not a readable XLSX archive.") from exc


def _parse_cell(cell: str) -> tuple[int, int]:
    match = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", cell)
    if match is None:
        raise ValueError("invalid cell address")
    column = 0
    for char in match.group(1):
        column = column * 26 + ord(char) - 64
    row = int(match.group(2))
    return row, column - 1


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
    "ColumnFormat",
    "SpreadsheetCreateError",
    "SpreadsheetCreateInput",
    "SpreadsheetCreateResult",
    "SpreadsheetFormula",
    "SpreadsheetSheet",
    "XLSX_MIME_TYPE",
    "build_excel_create_tools",
    "create_excel_spreadsheet",
]
