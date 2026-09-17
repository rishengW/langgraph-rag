from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, model_validator

from ._files import FileAccessError, resolve_safe_path

if TYPE_CHECKING:
    from src.config import Settings

_SUFFIXES = (".pdf",)
_MAX_OPERATIONS = 20


class PdfEditError(Exception):
    """Raised when a PDF edit operation cannot be completed."""


class PdfEditOperation(BaseModel):
    """A single edit to apply to a PDF copy."""

    action: Literal[
        "add_annotation",
        "add_watermark_text",
        "merge_pdfs",
        "extract_pages",
        "fill_form_field",
        "set_metadata",
    ] = Field(..., description="The edit action to perform.")
    page: int | None = Field(
        default=None,
        ge=1,
        description="1-based page number, for actions that target one page.",
    )
    text: str | None = Field(
        default=None,
        max_length=2_000,
        description="Annotation comment or watermark text.",
    )
    field_name: str | None = Field(
        default=None,
        max_length=200,
        description="Form field name, for the fill_form_field action.",
    )
    field_value: str | None = Field(
        default=None,
        max_length=2_000,
        description="Form field value, for the fill_form_field action.",
    )
    other_path: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Path of a second PDF, relative to the file-read root, for the "
            "merge_pdfs action."
        ),
    )
    pages: list[int] | None = Field(
        default=None,
        description=(
            "1-based page numbers to keep, for the extract_pages action. "
            "Defaults to all pages."
        ),
    )
    title: str | None = Field(
        default=None,
        max_length=200,
        description="Document title, for the set_metadata action.",
    )
    author: str | None = Field(
        default=None,
        max_length=200,
        description="Document author, for the set_metadata action.",
    )

    @model_validator(mode="after")
    def _check_required_fields(self) -> PdfEditOperation:
        if self.action == "add_annotation":
            if self.page is None or not (self.text or "").strip():
                raise ValueError(
                    "add_annotation requires 'page' (1-based) and non-empty 'text'."
                )
        elif self.action == "add_watermark_text":
            if not (self.text or "").strip():
                raise ValueError("add_watermark_text requires non-empty 'text'.")
        elif self.action == "merge_pdfs":
            if not (self.other_path or "").strip():
                raise ValueError("merge_pdfs requires 'other_path'.")
        elif self.action == "extract_pages":
            if not self.pages:
                raise ValueError("extract_pages requires a non-empty 'pages' list.")
        elif self.action == "fill_form_field":
            if not (self.field_name or "").strip():
                raise ValueError("fill_form_field requires 'field_name'.")
        return self


class PdfEditInput(BaseModel):
    """Input schema for the PDF editor."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .pdf file uploaded to this chat session, exactly as "
            "listed in the upload note. This file is never modified."
        ),
    )
    operations: list[PdfEditOperation] = Field(
        ...,
        min_length=1,
        max_length=_MAX_OPERATIONS,
        description="The edits to apply, in order.",
    )
    output_name: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Optional file name for the edited copy. Defaults to "
            "'<original>.edited.pdf'. A .pdf suffix is enforced."
        ),
    )


def build_pdf_edit_tools(
    settings: Settings,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> list[BaseTool]:
    """Create the session-scoped PDF edit tool.

    Returns an empty list unless file reading and PDF editing are enabled.
    Like the other session editors, output stays inside the session upload
    directory and existing files are never overwritten.
    """

    if not (settings.file_read_enabled and settings.pdf_edit_enabled):
        return []

    file_root = Path(settings.file_read_root)
    max_bytes = settings.file_read_max_bytes

    def _run_edit(
        path: str,
        operations: list[PdfEditOperation],
        output_name: str | None = None,
    ) -> tuple[str, dict[str, object] | None]:
        result = edit_pdf_document(
            path,
            operations=operations,
            file_root=file_root,
            max_bytes=max_bytes,
            output_name=output_name,
        )
        return result, None

    tool = StructuredTool.from_function(
        func=_run_edit,
        name="edit_pdf_document",
        description=(
            "Create an edited copy of an uploaded PDF. Supported actions: "
            "add_annotation (comment on a page), add_watermark_text (diagonal "
            "watermark on every page), merge_pdfs (append another PDF), "
            "extract_pages (keep only selected pages), fill_form_field (AcroForm "
            "fields), and set_metadata (title/author). The original file is "
            "never modified; the edited copy is returned as a download. "
            "Content editing of existing text is not supported."
        ),
        args_schema=PdfEditInput,
        response_format="content_and_artifact",
    )
    return [tool]


def edit_pdf_document(
    path: str,
    *,
    operations: list[PdfEditOperation],
    file_root: Path,
    max_bytes: int,
    output_name: str | None = None,
) -> str:
    """Apply the given operations and write the result as a new PDF."""

    try:
        resolved = resolve_safe_path(
            path,
            root=file_root,
            expected_suffixes=_SUFFIXES,
            max_bytes=max_bytes,
        )
    except FileAccessError as exc:
        return f"Could not read PDF: {exc}"

    try:
        import pypdf
    except ImportError as exc:
        return (
            "pypdf is not installed. Run `python -m pip install pypdf` to "
            f"enable the PDF edit tool: {exc}"
        )

    try:
        reader = pypdf.PdfReader(str(resolved))
        writer = pypdf.PdfWriter()
        writer.append(reader)
    except Exception as exc:
        return f"Could not open PDF {resolved.name!r}: {exc}"

    try:
        for index, operation in enumerate(operations, start=1):
            _apply_operation(
                writer,
                reader,
                operation,
                index=index,
                file_root=file_root,
                max_bytes=max_bytes,
            )
    except PdfEditError as exc:
        return f"PDF edit failed: {exc}"
    except Exception as exc:
        return f"PDF edit failed with an unexpected error: {exc}"

    target = _reserve_output_path(resolved, output_name)
    try:
        handle, temp_name = tempfile.mkstemp(
            prefix=".pdf_edit-",
            suffix=".pdf.tmp",
            dir=str(target.parent),
        )
    except OSError as exc:
        target.unlink(missing_ok=True)
        return f"PDF edit failed: could not create a temporary file: {exc}"

    os.close(handle)
    temp_path = Path(temp_name)
    try:
        with open(temp_path, "wb") as stream:
            writer.write(stream)
        output_size = temp_path.stat().st_size
        if output_size > max_bytes:
            raise PdfEditError(
                f"the edited PDF is too large ({output_size:,} bytes; "
                f"limit {max_bytes:,} bytes)."
            )
        os.replace(temp_path, target)
    except PdfEditError as exc:
        temp_path.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        return f"PDF edit failed: {exc}"
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        return f"PDF edit failed while writing output: {exc}"

    return (
        f"Created edited PDF: {target.name} ({len(operations)} operation(s) "
        f"applied to {resolved.name}; {target.stat().st_size:,} bytes)."
    )


def _apply_operation(
    writer: Any,
    reader: Any,
    operation: PdfEditOperation,
    *,
    index: int,
    file_root: Path,
    max_bytes: int,
) -> None:
    """Apply one operation to the in-memory writer, or raise PdfEditError."""

    action = operation.action
    total_pages = len(writer.pages)

    if action == "add_annotation":
        page_index = _page_index(operation.page, total_pages)
        page = writer.pages[page_index]
        try:
            from pypdf.annotations import FreeText

            annotation = FreeText(
                text=operation.text or "",
                rect=(50, 50, 350, 80),
                font="Helvetica",
                font_size="10pt",
            )
            writer.add_annotation(page_number=page_index, annotation=annotation)
        except Exception as exc:
            raise PdfEditError(
                f"operation {index}: could not add annotation to page "
                f"{operation.page}: {exc}"
            ) from exc

    elif action == "add_watermark_text":
        try:
            watermark = _build_text_watermark(
                writer,
                operation.text or "",
            )
            for page_index in range(total_pages):
                page = writer.pages[page_index]
                page.merge_page(watermark)
        except PdfEditError:
            raise
        except Exception as exc:
            raise PdfEditError(
                f"operation {index}: could not add watermark: {exc}"
            ) from exc

    elif action == "merge_pdfs":
        try:
            other = resolve_safe_path(
                operation.other_path or "",
                root=file_root,
                expected_suffixes=_SUFFIXES,
                max_bytes=max_bytes,
            )
            other_reader = reader.__class__(str(other))
        except FileAccessError as exc:
            raise PdfEditError(f"operation {index}: could not read merge source: {exc}")
        except Exception as exc:
            raise PdfEditError(
                f"operation {index}: could not open merge source PDF: {exc}"
            )
        try:
            writer.append(other_reader)
        except Exception as exc:
            raise PdfEditError(f"operation {index}: could not merge PDFs: {exc}")

    elif action == "extract_pages":
        wanted = operation.pages or []
        zero_based: list[int] = []
        for number in wanted:
            if number < 1 or number > total_pages:
                raise PdfEditError(
                    f"operation {index}: page {number} is out of range "
                    f"(document has {total_pages} page(s))."
                )
            zero_based.append(number - 1)
        try:
            for page_index in sorted(set(range(total_pages)) - set(zero_based),
                                     reverse=True):
                writer.remove_page(page_index)
        except Exception as exc:
            raise PdfEditError(
                f"operation {index}: could not extract pages: {exc}"
            ) from exc

    elif action == "fill_form_field":
        try:
            writer.update_page_form_field_values(
                writer.pages[0],
                {operation.field_name or "": operation.field_value or ""},
                auto_regenerate=False,
            )
        except Exception as exc:
            raise PdfEditError(
                f"operation {index}: could not fill form field "
                f"{operation.field_name!r}: {exc}"
            ) from exc

    elif action == "set_metadata":
        metadata: dict[str, str] = {}
        if operation.title is not None:
            metadata["/Title"] = operation.title
        if operation.author is not None:
            metadata["/Author"] = operation.author
        if not metadata:
            raise PdfEditError(
                f"operation {index}: set_metadata requires 'title' and/or 'author'."
            )
        try:
            writer.add_metadata(metadata)
        except Exception as exc:
            raise PdfEditError(
                f"operation {index}: could not set metadata: {exc}"
            ) from exc


def _page_index(page: int | None, total_pages: int) -> int:
    """Convert a 1-based page number to a 0-based index, validating range."""

    if page is None:
        raise PdfEditError("a page number is required for this action.")
    if page < 1 or page > total_pages:
        raise PdfEditError(
            f"page {page} is out of range (document has {total_pages} page(s))."
        )
    return page - 1


def _build_text_watermark(writer: Any, text: str) -> Any:
    """Build a light diagonal text watermark page sized to the first page."""

    try:
        import io

        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise PdfEditError(
            "reportlab is not installed. Run `python -m pip install reportlab` "
            "to enable text watermarks."
        ) from exc

    first = writer.pages[0]
    width = float(first.mediabox.width)
    height = float(first.mediabox.height)

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(width, height))
    pdf.saveState()
    pdf.setFont("Helvetica", 48)
    pdf.setFillAlpha(0.15)
    pdf.translate(width / 2, height / 2)
    pdf.rotate(45)
    pdf.drawCentredString(0, 0, text)
    pdf.restoreState()
    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    from pypdf import PdfReader

    return PdfReader(buffer).pages[0]


def _reserve_output_path(source_path: Path, output_name: str | None) -> Path:
    """Pick a non-existent output path next to the source file."""

    stem = source_path.stem
    suffix = source_path.suffix.lower() or ".pdf"
    if output_name:
        candidate_stem = Path(output_name.strip()).stem or f"{stem}.edited"
        candidate = source_path.with_name(f"{candidate_stem}{suffix}")
        if not candidate.exists():
            return candidate
        counter = 2
        while True:
            numbered = source_path.with_name(f"{candidate_stem}-{counter}{suffix}")
            if not numbered.exists():
                return numbered
            counter += 1
    candidate = source_path.with_name(f"{stem}.edited{suffix}")
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        numbered = source_path.with_name(f"{stem}.edited-{counter}{suffix}")
        if not numbered.exists():
            return numbered
        counter += 1


__all__ = [
    "PdfEditError",
    "PdfEditInput",
    "PdfEditOperation",
    "build_pdf_edit_tools",
    "edit_pdf_document",
]
