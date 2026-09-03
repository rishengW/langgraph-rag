"""Session-scoped Markdown (.md) creation, inspection, and editing tools.

The tools accept only UTF-8 ``.md`` files in the current chat session. New and
edited files are written to a temporary file, decoded again, and atomically
published as downloads. Existing files are never overwritten.
"""

from __future__ import annotations

import codecs
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

MARKDOWN_MIME_TYPE = "text/markdown"
FILE_ARTIFACT_TYPE = "file"
FILE_ARTIFACT_VERSION = 1
FILE_ARTIFACT_KIND = "download"
FILE_ARTIFACT_PROVIDER = "chat_upload"

_SUFFIXES = (".md",)
_INSPECT_MAX_CHARS = 20_000
_MAX_OPERATIONS = 100
_MAX_LINE_CHARS = 50_000
_MAX_OUTPUT_VARIANTS = 100
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class MarkdownEditError(Exception):
    """Raised when a Markdown file cannot be inspected or edited safely."""


@dataclass(frozen=True)
class MarkdownEditResult:
    """Internal content/artifact result for a Markdown-file write."""

    content: str
    artifact: dict[str, object] | None = None


@dataclass(frozen=True)
class _MarkdownDocument:
    """Decoded text plus byte-level characteristics that edits preserve."""

    lines: tuple[str, ...]
    encoding: Literal["utf-8", "utf-8-sig"]
    newline: Literal["LF", "CRLF"]
    has_final_newline: bool

    @property
    def newline_text(self) -> str:
        return "\r\n" if self.newline == "CRLF" else "\n"


class MarkdownEditOperation(BaseModel):
    """One line-oriented edit, addressed by the inspector's zero-based index."""

    action: Literal[
        "replace_line",
        "delete_line",
        "insert_before_line",
        "append_line",
    ] = Field(..., description="The kind of line edit to apply.")
    line_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Zero-based line number from inspect_markdown_file. Required for "
            "replace_line, delete_line, and insert_before_line."
        ),
    )
    expected_text: str | None = Field(
        default=None,
        max_length=_MAX_LINE_CHARS,
        description=(
            "The exact current text of the indexed line. Required for every "
            "action except append_line."
        ),
    )
    new_text: str | None = Field(
        default=None,
        max_length=_MAX_LINE_CHARS,
        description=(
            "The new single line. Required for replace_line, "
            "insert_before_line, and append_line."
        ),
    )

    @model_validator(mode="after")
    def _check_fields(self) -> MarkdownEditOperation:
        indexed = self.action != "append_line"
        needs_text = self.action != "delete_line"
        missing: list[str] = []
        if indexed and self.line_index is None:
            missing.append("line_index")
        if indexed and self.expected_text is None:
            missing.append("expected_text")
        if needs_text and self.new_text is None:
            missing.append("new_text")
        if missing:
            raise ValueError(f"{self.action} requires: {', '.join(missing)}.")

        for name in ("expected_text", "new_text"):
            value = getattr(self, name)
            if value is not None and ("\r" in value or "\n" in value):
                raise ValueError(f"{name} must contain exactly one line.")
        return self


class MarkdownInspectInput(BaseModel):
    """Input schema for the Markdown-file inspector."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .md file uploaded to this chat session, exactly as "
            "listed in the upload note."
        ),
    )
    max_chars: int = Field(
        default=_INSPECT_MAX_CHARS,
        ge=1,
        le=200_000,
        description="Maximum characters of the numbered listing to return.",
    )


class MarkdownCreateInput(BaseModel):
    """Input schema for creating a Markdown file."""

    filename: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "Bare filename for the new Markdown file. A .md suffix is added when "
            "needed. Existing files are never overwritten."
        ),
    )
    content: str = Field(
        ...,
        description=(
            "Complete UTF-8 Markdown text to write (headings, lists, code "
            "blocks, tables, etc.). LF or CRLF line endings are preserved; "
            "mixed line endings and binary control characters are refused."
        ),
    )


class MarkdownEditInput(BaseModel):
    """Input schema for the Markdown-file editor."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .md file uploaded to this chat session. The source is "
            "never modified."
        ),
    )
    operations: list[MarkdownEditOperation] = Field(
        ...,
        min_length=1,
        max_length=_MAX_OPERATIONS,
        description=(
            "Line edits to validate against the original file and then apply "
            "as one transaction."
        ),
    )
    output_name: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Optional name for the edited copy. Defaults to "
            "'<original>.edited.md'; a .md suffix is enforced."
        ),
    )


def build_markdown_edit_tools(
    settings: Settings,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> list[BaseTool]:
    """Create MD tools only when enabled and safely scoped to a chat."""

    if not (settings.file_read_enabled and settings.markdown_edit_enabled):
        return []
    if session_root is None or not _THREAD_ID_PATTERN.fullmatch(thread_id):
        return []

    file_root = Path(settings.file_read_root)
    max_bytes = settings.file_read_max_bytes

    def _run_create(
        filename: str,
        content: str,
    ) -> tuple[str, dict[str, object] | None]:
        result = create_markdown_file(
            filename,
            content,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
            thread_id=thread_id,
        )
        return result.content, result.artifact

    def _run_inspect(path: str, max_chars: int = _INSPECT_MAX_CHARS) -> str:
        return inspect_markdown_file(
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    def _run_edit(
        path: str,
        operations: list[MarkdownEditOperation],
        output_name: str | None = None,
    ) -> tuple[str, dict[str, object] | None]:
        result = edit_markdown_file(
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
        name="create_markdown_file",
        description=(
            "Create a new UTF-8 Markdown .md file in the current chat session "
            "and return it as a download. Pass full Markdown text (headings, "
            "lists, code blocks, tables). Existing files are never "
            "overwritten; a numbered filename is chosen on collision. Use only "
            "when the user explicitly asks to create a Markdown file."
        ),
        args_schema=MarkdownCreateInput,
        response_format="content_and_artifact",
    )
    inspect_tool = StructuredTool.from_function(
        func=_run_inspect,
        name="inspect_markdown_file",
        description=(
            "List the zero-based lines and text format of a .md file uploaded "
            "to this chat session. Call this before edit_markdown_file to "
            "obtain the indexes and exact current line text required by edit "
            "operations."
        ),
        args_schema=MarkdownInspectInput,
    )
    edit_tool = StructuredTool.from_function(
        func=_run_edit,
        name="edit_markdown_file",
        description=(
            "Apply structured line edits to an uploaded .md file and create a "
            "new downloadable copy without changing the original. Supports "
            "replace, delete, insert-before, and append. Call "
            "inspect_markdown_file first and supply exact expected_text for "
            "indexed operations. Use only when the user explicitly asks to "
            "change the file."
        ),
        args_schema=MarkdownEditInput,
        response_format="content_and_artifact",
    )
    return [create_tool, inspect_tool, edit_tool]


def create_markdown_file(
    filename: str,
    content: str,
    *,
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    thread_id: str = "",
) -> MarkdownEditResult:
    """Create a session-scoped UTF-8 MD file and downloadable artifact."""

    output_name = "(unresolved)"
    try:
        if not _THREAD_ID_PATTERN.fullmatch(thread_id):
            raise MarkdownEditError("a valid chat thread is required to create a Markdown file.")
        directory = _prepare_session_directory(
            session_root=session_root,
            file_root=file_root,
        )
        stem = _creation_stem(filename)
        document = _decode_markdown(content.encode("utf-8"), name=f"{stem}.md")
        target = _reserve_created_output_path(directory=directory, stem=stem)
        output_name = target.name
        published = _write_markdown_atomically(
            document,
            target=target,
            max_bytes=max_bytes,
            size_label="new",
        )
        size_bytes = published.stat().st_size
    except (OSError, UnicodeError, MarkdownEditError) as exc:
        logger.info(
            "markdown_create failed: thread=%s output=%s reason=%s",
            thread_id or "(none)",
            output_name,
            type(exc).__name__,
        )
        return MarkdownEditResult(content=f"Could not create Markdown file: {exc}")

    logger.info(
        "markdown_create succeeded: thread=%s output=%s bytes=%d",
        thread_id,
        published.name,
        size_bytes,
    )
    return MarkdownEditResult(
        content=(
            f"Created {published.name} in this chat session. "
            "The file is available to download."
        ),
        artifact=_build_file_artifact(
            thread_id=thread_id,
            filename=published.name,
            size_bytes=size_bytes,
        ),
    )


def inspect_markdown_file(
    path: str,
    *,
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    max_chars: int = _INSPECT_MAX_CHARS,
) -> str:
    """Return format metadata and a zero-based listing of exact lines."""

    try:
        resolved = _resolve_session_markdown(
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
        )
        document = _read_markdown_document(resolved)
    except MarkdownEditError as exc:
        return f"Could not inspect Markdown file: {exc}"

    encoding = "UTF-8 with BOM" if document.encoding == "utf-8-sig" else "UTF-8"
    final_newline = "yes" if document.has_final_newline else "no"
    lines = [
        f"Markdown file {resolved.name}:",
        f"Encoding: {encoding}",
        f"Newline: {document.newline}",
        f"Final newline: {final_newline}",
        f"LINES ({len(document.lines)}):",
    ]
    lines.extend(
        f"[{index}] {json.dumps(line, ensure_ascii=False)}"
        for index, line in enumerate(document.lines)
    )
    body = "\n".join(lines)
    limit = max(1, int(max_chars))
    if len(body) > limit:
        return f"{body[:limit]}\n\n[listing truncated to {limit:,} of {len(body):,} chars]"
    return body


def edit_markdown_file(
    path: str,
    *,
    operations: list[MarkdownEditOperation],
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    thread_id: str = "",
    output_name: str | None = None,
) -> MarkdownEditResult:
    """Validate and apply a batch, then publish a copy-on-write MD artifact."""

    source_name = "(unresolved)"
    try:
        resolved = _resolve_session_markdown(
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
        )
        source_name = resolved.name
        document = _read_markdown_document(resolved)
        indexed, appends = _plan_operations(document, operations)
        edited = _apply_operations(document, indexed=indexed, appends=appends)
        published = _publish_markdown(
            edited,
            session_root=session_root,
            source_path=resolved,
            output_name=output_name,
            max_bytes=max_bytes,
        )
    except MarkdownEditError as exc:
        logger.info(
            "markdown_edit failed: thread=%s source=%s operations=%d reason=%s",
            thread_id or "(none)",
            source_name,
            len(operations),
            type(exc).__name__,
        )
        return MarkdownEditResult(content=f"Could not edit Markdown file: {exc}")

    size_bytes = published.stat().st_size
    logger.info(
        "markdown_edit succeeded: thread=%s source=%s output=%s operations=%d bytes=%d",
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
    return MarkdownEditResult(
        content=content,
        artifact=_build_file_artifact(
            thread_id=thread_id,
            filename=published.name,
            size_bytes=size_bytes,
        ),
    )


def _resolve_session_markdown(
    raw_path: str,
    *,
    session_root: Path,
    file_root: Path,
    max_bytes: int,
) -> Path:
    """Resolve a bare or upload-note path within this session directory."""

    text = (raw_path or "").strip().strip('"').strip("'")
    if not text:
        raise MarkdownEditError("a non-empty file path is required.")

    try:
        session_resolved = session_root.expanduser().resolve()
    except OSError as exc:
        raise MarkdownEditError(f"could not resolve the session directory: {exc}") from exc

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
        raise MarkdownEditError(
            f"access denied: {text!r} is outside this chat session's Markdown-file "
            "directory. Only files uploaded to this session can be edited."
        )
    if access_errors:
        raise MarkdownEditError(str(access_errors[-1])) from access_errors[-1]
    raise MarkdownEditError(f"could not resolve {text!r} inside this chat session.")


def _read_markdown_document(path: Path) -> _MarkdownDocument:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise MarkdownEditError(f"could not read {path.name!r}: {exc}") from exc
    return _decode_markdown(data, name=path.name)


def _decode_markdown(data: bytes, *, name: str) -> _MarkdownDocument:
    """Strictly decode UTF-8 and reject binary or ambiguous line formats."""

    unsupported_boms = (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE, codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)
    if any(data.startswith(bom) for bom in unsupported_boms):
        raise MarkdownEditError(
            f"{name!r} uses an unsupported encoding; only UTF-8 text is supported."
        )

    has_bom = data.startswith(codecs.BOM_UTF8)
    payload = data[len(codecs.BOM_UTF8) :] if has_bom else data
    if b"\x00" in payload:
        raise MarkdownEditError(f"{name!r} appears to be binary data (contains NUL bytes).")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise MarkdownEditError(
            f"{name!r} uses an unsupported encoding or contains invalid UTF-8."
        ) from exc

    prohibited = [
        character
        for character in text
        if (ord(character) < 32 and character not in "\t\r\n")
        or ord(character) in (0x7F, 0xFFFE, 0xFFFF)
    ]
    if prohibited:
        raise MarkdownEditError(f"{name!r} appears to contain binary control characters.")

    without_crlf = text.replace("\r\n", "")
    if "\r" in without_crlf:
        raise MarkdownEditError(f"{name!r} uses unsupported bare-CR line endings.")
    has_crlf = "\r\n" in text
    if has_crlf and "\n" in without_crlf:
        raise MarkdownEditError(f"{name!r} mixes CRLF and LF line endings.")

    newline_text = "\r\n" if has_crlf else "\n"
    final_newline = bool(text) and text.endswith(newline_text)
    split_lines = text.split(newline_text) if text else []
    if final_newline:
        split_lines.pop()
    return _MarkdownDocument(
        lines=tuple(split_lines),
        encoding="utf-8-sig" if has_bom else "utf-8",
        newline="CRLF" if has_crlf else "LF",
        has_final_newline=final_newline,
    )


def _plan_operations(
    document: _MarkdownDocument,
    operations: list[MarkdownEditOperation],
) -> tuple[dict[int, MarkdownEditOperation], list[MarkdownEditOperation]]:
    """Validate all operations against original lines before any mutation."""

    indexed: dict[int, MarkdownEditOperation] = {}
    appends: list[MarkdownEditOperation] = []
    problems: list[str] = []
    for position, operation in enumerate(operations):
        label = f"operation {position} ({operation.action})"
        if operation.action == "append_line":
            appends.append(operation)
            continue

        index = operation.line_index
        assert index is not None
        if index in indexed:
            problems.append(f"{label}: line {index} is targeted more than once.")
            continue
        indexed[index] = operation
        if index >= len(document.lines):
            problems.append(
                f"{label}: line_index {index} is out of range; the file has "
                f"{len(document.lines)} lines."
            )
            continue
        if document.lines[index] != operation.expected_text:
            problems.append(
                f"{label}: expected_text does not exactly match line {index}."
            )

    if problems:
        raise MarkdownEditError(
            f"no changes were made because {len(problems)} operation(s) did not "
            f"match the file: {' '.join(problems)} Re-run inspect_markdown_file and retry."
        )
    return indexed, appends


def _apply_operations(
    document: _MarkdownDocument,
    *,
    indexed: dict[int, MarkdownEditOperation],
    appends: list[MarkdownEditOperation],
) -> _MarkdownDocument:
    output: list[str] = []
    for index, line in enumerate(document.lines):
        operation = indexed.get(index)
        if operation is None:
            output.append(line)
        elif operation.action == "replace_line":
            output.append(operation.new_text or "")
        elif operation.action == "insert_before_line":
            output.extend((operation.new_text or "", line))
        else:  # delete_line
            continue
    output.extend(operation.new_text or "" for operation in appends)
    return _MarkdownDocument(
        lines=tuple(output),
        encoding=document.encoding,
        newline=document.newline,
        # An empty file cannot carry a line-ending style or final newline.
        # Normalize that state when a batch deletes every source line so the
        # encoded file can round-trip through the publication validator.
        has_final_newline=document.has_final_newline and bool(output),
    )


def _encode_markdown(document: _MarkdownDocument) -> bytes:
    text = document.newline_text.join(document.lines)
    if document.has_final_newline and document.lines:
        text += document.newline_text
    encoded = text.encode("utf-8")
    return codecs.BOM_UTF8 + encoded if document.encoding == "utf-8-sig" else encoded


def _publish_markdown(
    document: _MarkdownDocument,
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
    return _write_markdown_atomically(
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
        raise MarkdownEditError(f"could not resolve the session directory: {exc}") from exc
    if directory == root or not _is_within(directory, root):
        raise MarkdownEditError("the output directory is outside the configured file root.")
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MarkdownEditError(f"could not create the session directory: {exc}") from exc
    if not directory.is_dir():
        raise MarkdownEditError("the session output path is not a directory.")
    return directory


def _creation_stem(filename: str) -> str:
    """Sanitize a requested creation name and return its filename stem."""

    requested = Path(str(filename or "")).name
    if requested.lower().endswith(".md"):
        requested = requested[: -len(".md")]
    cleaned = _SAFE_NAME.sub("_", requested.strip().strip("."))[:150]
    if not cleaned:
        raise MarkdownEditError("a usable .md filename is required.")
    return cleaned


def _reserve_created_output_path(*, directory: Path, stem: str) -> Path:
    """Exclusively reserve a free filename for a newly created Markdown file."""
    suffix = _SUFFIXES[0]
    for variant in range(1, _MAX_OUTPUT_VARIANTS + 1):
        name = f"{stem}{suffix}" if variant == 1 else f"{stem}-{variant}{suffix}"
        candidate = directory / name
        try:
            candidate.touch(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
        except OSError as exc:
            raise MarkdownEditError(f"could not create the output file: {exc}") from exc
    raise MarkdownEditError(
        f"too many files named {stem!r} already exist in this session; "
        "download or remove some before creating another."
    )


def _write_markdown_atomically(
    document: _MarkdownDocument,
    *,
    target: Path,
    max_bytes: int,
    size_label: str,
) -> Path:
    """Write a reserved target through a validated temporary file."""

    try:
        handle, temp_name = tempfile.mkstemp(
            prefix=".markdown_write-",
            suffix=".md.tmp",
            dir=str(target.parent),
        )
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise MarkdownEditError(f"could not create a temporary file: {exc}") from exc

    temp_path = Path(temp_name)
    try:
        data = _encode_markdown(document)
        if len(data) > max_bytes:
            raise MarkdownEditError(
                f"the {size_label} Markdown file is too large ({len(data):,} bytes; "
                f"limit {max_bytes:,} bytes)."
            )
        stream = os.fdopen(handle, "wb")
        handle = -1
        with stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        validated = _read_markdown_document(temp_path)
        if validated != document:
            raise MarkdownEditError("the saved Markdown file did not pass validation.")
        os.replace(temp_path, target)
    except MarkdownEditError:
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
        raise MarkdownEditError(f"could not save the Markdown file: {exc}") from exc
    return target


def _reserve_output_path(
    *,
    session_root: Path,
    source_path: Path,
    output_name: str | None,
) -> Path:
    suffix = _SUFFIXES[0]
    stem = _output_stem(source_path=source_path, output_name=output_name)
    try:
        directory = session_root.expanduser().resolve()
        resolved_source = source_path.resolve()
    except OSError as exc:
        raise MarkdownEditError(f"could not resolve the output directory: {exc}") from exc
    if not directory.is_dir() or not _is_within(resolved_source, directory):
        raise MarkdownEditError("the output directory is outside this chat session.")

    for variant in range(1, _MAX_OUTPUT_VARIANTS + 1):
        name = f"{stem}{suffix}" if variant == 1 else f"{stem}-{variant}{suffix}"
        candidate = directory / name
        if candidate.resolve() == resolved_source:
            continue
        try:
            candidate.touch(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
        except OSError as exc:
            raise MarkdownEditError(f"could not create the output file: {exc}") from exc
    raise MarkdownEditError(
        f"too many edited copies of {source_path.name!r} already exist in this "
        "session; download or remove some before editing again."
    )


def _output_stem(*, source_path: Path, output_name: str | None) -> str:
    suffix = _SUFFIXES[0]
    if output_name:
        requested = Path(str(output_name)).name
        if requested.lower().endswith(suffix):
            requested = requested[: -len(suffix)]
        cleaned = _SAFE_NAME.sub("_", requested.strip().strip("."))[:150]
        if cleaned:
            return cleaned
    return f"{source_path.stem}.edited"


def _build_file_artifact(
    *, thread_id: str, filename: str, size_bytes: int
) -> dict[str, object]:
    return {
        "type": FILE_ARTIFACT_TYPE,
        "version": FILE_ARTIFACT_VERSION,
        "kind": FILE_ARTIFACT_KIND,
        "provider": FILE_ARTIFACT_PROVIDER,
        "threadId": thread_id,
        "filename": filename,
        "mimeType": MARKDOWN_MIME_TYPE,
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
    "MARKDOWN_MIME_TYPE",
    "MarkdownCreateInput",
    "MarkdownEditError",
    "MarkdownEditInput",
    "MarkdownEditOperation",
    "MarkdownEditResult",
    "MarkdownInspectInput",
    "build_markdown_edit_tools",
    "create_markdown_file",
    "edit_markdown_file",
    "inspect_markdown_file",
]
