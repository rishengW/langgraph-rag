"""Shared engine for session-scoped plain-text source-file tools.

The R, Rust, Go, SQL, and JSONL tools are the same line-oriented create,
inspect, and edit flow (the one used by the .txt/.md/.tsx editors), so the
logic lives here once and each language supplies a
:class:`SourceEditConfig`. New and edited files are written to a temporary
file, decoded again, and atomically published as downloads. Existing files
are never overwritten.
"""

from __future__ import annotations

import codecs
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
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

FILE_ARTIFACT_TYPE = "file"
FILE_ARTIFACT_VERSION = 1
FILE_ARTIFACT_KIND = "download"
FILE_ARTIFACT_PROVIDER = "chat_upload"

_INSPECT_MAX_CHARS = 20_000
_MAX_OPERATIONS = 100
_MAX_LINE_CHARS = 50_000
_MAX_OUTPUT_VARIANTS = 100
# Cap returned characters so a large file does not blow up the LLM context.
_MAX_CHARS_READER = 20_000
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class SourceEditError(Exception):
    """Raised when a source file cannot be inspected or edited safely."""


@dataclass(frozen=True)
class SourceEditResult:
    """Internal content/artifact result for a source-file write."""

    content: str
    artifact: dict[str, object] | None = None


# Returns None when the line is acceptable, else a human-readable reason.
LineValidator = Callable[[str, int], str | None]


@dataclass(frozen=True)
class SourceEditConfig:
    """Per-language identity and rules for the shared source-edit engine."""

    label: str
    suffixes: tuple[str, ...]
    creation_suffix: str
    tool_prefix: str
    mime_type: str
    flag_name: str
    temp_prefix: str
    line_validator: LineValidator | None = None

    @property
    def suffix_text(self) -> str:
        return "/".join(self.suffixes)


@dataclass(frozen=True)
class _SourceDocument:
    """Decoded text plus byte-level characteristics that edits preserve."""

    lines: tuple[str, ...]
    encoding: Literal["utf-8", "utf-8-sig"]
    newline: Literal["LF", "CRLF"]
    has_final_newline: bool

    @property
    def newline_text(self) -> str:
        return "\r\n" if self.newline == "CRLF" else "\n"


class SourceEditOperation(BaseModel):
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
            "Zero-based line number from the inspect tool. Required for "
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
    def _check_fields(self) -> SourceEditOperation:
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


class SourceInspectInput(BaseModel):
    """Input schema for the source-file inspector."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a file uploaded to this chat session, exactly as listed "
            "in the upload note."
        ),
    )
    max_chars: int = Field(
        default=_INSPECT_MAX_CHARS,
        ge=1,
        le=200_000,
        description="Maximum characters of the numbered listing to return.",
    )


def build_source_edit_tools(
    settings: Settings,
    config: SourceEditConfig,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> list[BaseTool]:
    """Create the language's tools only when enabled and scoped to a chat."""

    if not (settings.file_read_enabled and getattr(settings, config.flag_name)):
        return []
    if session_root is None or not _THREAD_ID_PATTERN.fullmatch(thread_id):
        return []

    file_root = Path(settings.file_read_root)
    max_bytes = settings.file_read_max_bytes
    create_name = f"create_{config.tool_prefix}_file"
    inspect_name = f"inspect_{config.tool_prefix}_file"
    edit_name = f"edit_{config.tool_prefix}_file"
    suffixes = config.suffix_text

    def _run_create(
        filename: str,
        content: str,
    ) -> tuple[str, dict[str, object] | None]:
        result = create_source_file(
            config,
            filename,
            content,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
            thread_id=thread_id,
        )
        return result.content, result.artifact

    def _run_inspect(path: str, max_chars: int = _INSPECT_MAX_CHARS) -> str:
        return inspect_source_file(
            config,
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    def _run_edit(
        path: str,
        operations: list[SourceEditOperation],
        output_name: str | None = None,
    ) -> tuple[str, dict[str, object] | None]:
        result = edit_source_file(
            config,
            path,
            operations=operations,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
            thread_id=thread_id,
            output_name=output_name,
        )
        return result.content, result.artifact

    create_input = _build_create_input_model(config)
    edit_input = _build_edit_input_model(config)

    create_tool = StructuredTool.from_function(
        func=_run_create,
        name=create_name,
        description=(
            f"Create a new UTF-8 {config.label} file ({suffixes}) in the "
            "current chat session and return it as a download. Pass the "
            "complete source text. Existing files are never overwritten; a "
            "numbered filename is chosen on collision. Use only when the "
            f"user explicitly asks to create a {config.label} file."
        ),
        args_schema=create_input,
        response_format="content_and_artifact",
    )
    inspect_tool = StructuredTool.from_function(
        func=_run_inspect,
        name=inspect_name,
        description=(
            f"List the zero-based lines and text format of a {suffixes} file "
            f"uploaded to this chat session. Call this before {edit_name} to "
            "obtain the indexes and exact current line text required by edit "
            "operations."
        ),
        args_schema=SourceInspectInput,
    )
    edit_tool = StructuredTool.from_function(
        func=_run_edit,
        name=edit_name,
        description=(
            f"Apply structured line edits to an uploaded {suffixes} file and "
            "create a new downloadable copy without changing the original. "
            "Supports replace, delete, insert-before, and append. Call "
            f"{inspect_name} first and supply exact expected_text for indexed "
            "operations. Use only when the user explicitly asks to change the "
            "file."
        ),
        args_schema=edit_input,
        response_format="content_and_artifact",
    )
    return [create_tool, inspect_tool, edit_tool]


def _build_create_input_model(config: SourceEditConfig) -> type[BaseModel]:
    suffix_hint = config.creation_suffix

    class _ConfiguredCreateInput(BaseModel):
        """Input schema for creating this language's file."""

        filename: str = Field(
            ...,
            min_length=1,
            max_length=200,
            description=(
                f"Bare filename for the new {config.label} file. A {suffix_hint} "
                "suffix is added when needed. Existing files are never "
                "overwritten."
            ),
        )
        content: str = Field(
            ...,
            description=(
                f"Complete UTF-8 {config.label} source to write. LF or CRLF "
                "line endings are preserved; mixed line endings and binary "
                "control characters are refused."
            ),
        )

    _ConfiguredCreateInput.__name__ = f"{config.tool_prefix.capitalize()}CreateInput"
    _ConfiguredCreateInput.__qualname__ = _ConfiguredCreateInput.__name__
    return _ConfiguredCreateInput


def _build_edit_input_model(config: SourceEditConfig) -> type[BaseModel]:
    default_output = f"<original>.edited{config.creation_suffix}"

    class _ConfiguredEditInput(BaseModel):
        """Input schema for this language's editor."""

        path: str = Field(
            ...,
            min_length=1,
            description=(
                f"Path to a {config.suffix_text} file uploaded to this chat "
                "session. The source is never modified."
            ),
        )
        operations: list[SourceEditOperation] = Field(
            ...,
            min_length=1,
            max_length=_MAX_OPERATIONS,
            description=(
                "Line edits to validate against the original file and then "
                "apply as one transaction."
            ),
        )
        output_name: str | None = Field(
            default=None,
            max_length=200,
            description=(
                "Optional name for the edited copy. Defaults to "
                f"'{default_output}'; the source's suffix is enforced."
            ),
        )

    _ConfiguredEditInput.__name__ = f"{config.tool_prefix.capitalize()}EditInput"
    _ConfiguredEditInput.__qualname__ = _ConfiguredEditInput.__name__
    return _ConfiguredEditInput


def create_source_file(
    config: SourceEditConfig,
    filename: str,
    content: str,
    *,
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    thread_id: str = "",
) -> SourceEditResult:
    """Create a session-scoped UTF-8 source file and downloadable artifact."""

    output_name = "(unresolved)"
    try:
        if not _THREAD_ID_PATTERN.fullmatch(thread_id):
            raise SourceEditError(
                f"a valid chat thread is required to create a {config.label} file."
            )
        directory = _prepare_session_directory(
            session_root=session_root,
            file_root=file_root,
        )
        stem, suffix = _creation_stem(config, filename)
        document = decode_source(content.encode("utf-8"), name=f"{stem}{suffix}")
        _validate_document_lines(config, document)
        target = _reserve_created_output_path(
            directory=directory,
            stem=stem,
            suffix=suffix,
        )
        output_name = target.name
        published = _write_source_atomically(
            config,
            document,
            target=target,
            max_bytes=max_bytes,
            size_label="new",
        )
        size_bytes = published.stat().st_size
    except (OSError, UnicodeError, SourceEditError) as exc:
        logger.info(
            "%s_create failed: thread=%s output=%s reason=%s",
            config.tool_prefix,
            thread_id or "(none)",
            output_name,
            type(exc).__name__,
        )
        return SourceEditResult(
            content=f"Could not create {config.label} file: {exc}"
        )

    logger.info(
        "%s_create succeeded: thread=%s output=%s bytes=%d",
        config.tool_prefix,
        thread_id,
        published.name,
        size_bytes,
    )
    return SourceEditResult(
        content=(
            f"Created {published.name} in this chat session. "
            "The file is available to download."
        ),
        artifact=_build_file_artifact(
            config,
            thread_id=thread_id,
            filename=published.name,
            size_bytes=size_bytes,
        ),
    )


def inspect_source_file(
    config: SourceEditConfig,
    path: str,
    *,
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    max_chars: int = _INSPECT_MAX_CHARS,
) -> str:
    """Return format metadata and a zero-based listing of exact lines."""

    try:
        resolved = _resolve_session_source(
            config,
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
        )
        document = _read_source_document(resolved)
    except SourceEditError as exc:
        return f"Could not inspect {config.label} file: {exc}"

    encoding = "UTF-8 with BOM" if document.encoding == "utf-8-sig" else "UTF-8"
    final_newline = "yes" if document.has_final_newline else "no"
    lines = [
        f"{config.label} file {resolved.name}:",
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


def edit_source_file(
    config: SourceEditConfig,
    path: str,
    *,
    operations: list[SourceEditOperation],
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    thread_id: str = "",
    output_name: str | None = None,
) -> SourceEditResult:
    """Validate and apply a batch, then publish a copy-on-write artifact."""

    source_name = "(unresolved)"
    try:
        resolved = _resolve_session_source(
            config,
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
        )
        source_name = resolved.name
        document = _read_source_document(resolved)
        indexed, appends = _plan_operations(config, document, operations)
        edited = _apply_operations(document, indexed=indexed, appends=appends)
        _validate_document_lines(config, edited)
        published = _publish_source(
            config,
            edited,
            session_root=session_root,
            source_path=resolved,
            output_name=output_name,
            max_bytes=max_bytes,
        )
    except SourceEditError as exc:
        logger.info(
            "%s_edit failed: thread=%s source=%s operations=%d reason=%s",
            config.tool_prefix,
            thread_id or "(none)",
            source_name,
            len(operations),
            type(exc).__name__,
        )
        return SourceEditResult(content=f"Could not edit {config.label} file: {exc}")

    size_bytes = published.stat().st_size
    logger.info(
        "%s_edit succeeded: thread=%s source=%s output=%s operations=%d bytes=%d",
        config.tool_prefix,
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
            config,
            thread_id=thread_id,
            filename=published.name,
            size_bytes=size_bytes,
        ),
    )


def _validate_document_lines(
    config: SourceEditConfig,
    document: _SourceDocument,
) -> None:
    """Run the language's per-line validator (e.g. JSONL) over every line."""

    if config.line_validator is None:
        return
    problems: list[str] = []
    for index, line in enumerate(document.lines):
        reason = config.line_validator(line, index)
        if reason:
            problems.append(reason)
    if problems:
        raise SourceEditError(
            f"{len(problems)} line(s) failed {config.label} validation: "
            f"{' '.join(problems[:5])}"
        )


def _resolve_session_source(
    config: SourceEditConfig,
    raw_path: str,
    *,
    session_root: Path,
    file_root: Path,
    max_bytes: int,
) -> Path:
    """Resolve a bare or upload-note path within this session directory."""

    text = (raw_path or "").strip().strip('"').strip("'")
    if not text:
        raise SourceEditError("a non-empty file path is required.")

    try:
        session_resolved = session_root.expanduser().resolve()
    except OSError as exc:
        raise SourceEditError(
            f"could not resolve the session directory: {exc}"
        ) from exc

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
                expected_suffixes=config.suffixes,
                max_bytes=max_bytes,
            )
        except FileAccessError as exc:
            access_errors.append(exc)

    if not found_in_scope:
        raise SourceEditError(
            f"access denied: {text!r} is outside this chat session's "
            f"{config.label}-file directory. Only files uploaded to this "
            "session can be edited."
        )
    if access_errors:
        raise SourceEditError(str(access_errors[-1])) from access_errors[-1]
    raise SourceEditError(f"could not resolve {text!r} inside this chat session.")


def _read_source_document(path: Path) -> _SourceDocument:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SourceEditError(f"could not read {path.name!r}: {exc}") from exc
    return decode_source(data, name=path.name)


def decode_source(data: bytes, *, name: str) -> _SourceDocument:
    """Strictly decode UTF-8 and reject binary or ambiguous line formats."""

    unsupported_boms = (
        codecs.BOM_UTF16_LE,
        codecs.BOM_UTF16_BE,
        codecs.BOM_UTF32_LE,
        codecs.BOM_UTF32_BE,
    )
    if any(data.startswith(bom) for bom in unsupported_boms):
        raise SourceEditError(
            f"{name!r} uses an unsupported encoding; only UTF-8 text is supported."
        )

    has_bom = data.startswith(codecs.BOM_UTF8)
    payload = data[len(codecs.BOM_UTF8) :] if has_bom else data
    if b"\x00" in payload:
        raise SourceEditError(f"{name!r} appears to be binary data (contains NUL bytes).")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SourceEditError(
            f"{name!r} uses an unsupported encoding or contains invalid UTF-8."
        ) from exc

    prohibited = [
        character
        for character in text
        if (ord(character) < 32 and character not in "\t\r\n")
        or ord(character) in (0x7F, 0xFFFE, 0xFFFF)
    ]
    if prohibited:
        raise SourceEditError(f"{name!r} appears to contain binary control characters.")

    without_crlf = text.replace("\r\n", "")
    if "\r" in without_crlf:
        raise SourceEditError(f"{name!r} uses unsupported bare-CR line endings.")
    has_crlf = "\r\n" in text
    if has_crlf and "\n" in without_crlf:
        raise SourceEditError(f"{name!r} mixes CRLF and LF line endings.")

    newline_text = "\r\n" if has_crlf else "\n"
    final_newline = bool(text) and text.endswith(newline_text)
    split_lines = text.split(newline_text) if text else []
    if final_newline:
        split_lines.pop()
    return _SourceDocument(
        lines=tuple(split_lines),
        encoding="utf-8-sig" if has_bom else "utf-8",
        newline="CRLF" if has_crlf else "LF",
        has_final_newline=final_newline,
    )


def encode_source(document: _SourceDocument) -> bytes:
    text = document.newline_text.join(document.lines)
    if document.has_final_newline and document.lines:
        text += document.newline_text
    encoded = text.encode("utf-8")
    return codecs.BOM_UTF8 + encoded if document.encoding == "utf-8-sig" else encoded


def _plan_operations(
    config: SourceEditConfig,
    document: _SourceDocument,
    operations: list[SourceEditOperation],
) -> tuple[dict[int, SourceEditOperation], list[SourceEditOperation]]:
    """Validate all operations against original lines before any mutation."""

    indexed: dict[int, SourceEditOperation] = {}
    appends: list[SourceEditOperation] = []
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
        raise SourceEditError(
            f"no changes were made because {len(problems)} operation(s) did not "
            f"match the file: {' '.join(problems)} Re-run "
            f"inspect_{config.tool_prefix}_file and retry."
        )
    return indexed, appends


def _apply_operations(
    document: _SourceDocument,
    *,
    indexed: dict[int, SourceEditOperation],
    appends: list[SourceEditOperation],
) -> _SourceDocument:
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
    return _SourceDocument(
        lines=tuple(output),
        encoding=document.encoding,
        newline=document.newline,
        # An empty file cannot carry a line-ending style or final newline.
        # Normalize that state when a batch deletes every source line so the
        # encoded file can round-trip through the publication validator.
        has_final_newline=document.has_final_newline and bool(output),
    )


def _publish_source(
    config: SourceEditConfig,
    document: _SourceDocument,
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
    return _write_source_atomically(
        config,
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
        raise SourceEditError(
            f"could not resolve the session directory: {exc}"
        ) from exc
    if directory == root or not _is_within(directory, root):
        raise SourceEditError(
            "the output directory is outside the configured file root."
        )
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SourceEditError(
            f"could not create the session directory: {exc}"
        ) from exc
    if not directory.is_dir():
        raise SourceEditError("the session output path is not a directory.")
    return directory


def _creation_stem(config: SourceEditConfig, filename: str) -> tuple[str, str]:
    """Sanitize a requested creation name into a stem and enforced suffix."""

    requested = Path(str(filename or "")).name
    suffix = config.creation_suffix
    matched = False
    for option in config.suffixes:
        if requested.lower().endswith(option):
            suffix = option
            matched = True
            break
    if matched:
        requested = requested[: -len(suffix)]
    cleaned = _SAFE_NAME.sub("_", requested.strip().strip("."))[:150]
    if not cleaned:
        raise SourceEditError(f"a usable {suffix} filename is required.")
    return cleaned, suffix


def _reserve_created_output_path(
    *,
    directory: Path,
    stem: str,
    suffix: str,
) -> Path:
    """Exclusively reserve a free filename for a newly created source file."""

    for variant in range(1, _MAX_OUTPUT_VARIANTS + 1):
        name = f"{stem}{suffix}" if variant == 1 else f"{stem}-{variant}{suffix}"
        candidate = directory / name
        try:
            candidate.touch(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
        except OSError as exc:
            raise SourceEditError(
                f"could not create the output file: {exc}"
            ) from exc
    raise SourceEditError(
        f"too many files named {stem!r} already exist in this session; "
        "download or remove some before creating another."
    )


def _write_source_atomically(
    config: SourceEditConfig,
    document: _SourceDocument,
    *,
    target: Path,
    max_bytes: int,
    size_label: str,
) -> Path:
    """Write a reserved target through a validated temporary file."""

    try:
        handle, temp_name = tempfile.mkstemp(
            prefix=config.temp_prefix,
            suffix=f"{config.creation_suffix}.tmp",
            dir=str(target.parent),
        )
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise SourceEditError(
            f"could not create a temporary file: {exc}"
        ) from exc

    temp_path = Path(temp_name)
    try:
        data = encode_source(document)
        if len(data) > max_bytes:
            raise SourceEditError(
                f"the {size_label} {config.label} file is too large "
                f"({len(data):,} bytes; limit {max_bytes:,} bytes)."
            )
        stream = os.fdopen(handle, "wb")
        handle = -1
        with stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        validated = _read_source_document(temp_path)
        if validated != document:
            raise SourceEditError(
                f"the saved {config.label} file did not pass validation."
            )
        os.replace(temp_path, target)
    except SourceEditError:
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
        raise SourceEditError(
            f"could not save the {config.label} file: {exc}"
        ) from exc
    return target


def _reserve_output_path(
    *,
    session_root: Path,
    source_path: Path,
    output_name: str | None,
) -> Path:
    suffix = source_path.suffix.lower()
    stem = _output_stem(source_path=source_path, output_name=output_name)
    try:
        directory = session_root.expanduser().resolve()
        resolved_source = source_path.resolve()
    except OSError as exc:
        raise SourceEditError(
            f"could not resolve the output directory: {exc}"
        ) from exc
    if not directory.is_dir() or not _is_within(resolved_source, directory):
        raise SourceEditError(
            "the output directory is outside this chat session."
        )

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
            raise SourceEditError(
                f"could not create the output file: {exc}"
            ) from exc
    raise SourceEditError(
        f"too many edited copies of {source_path.name!r} already exist in this "
        "session; download or remove some before editing again."
    )


def _output_stem(*, source_path: Path, output_name: str | None) -> str:
    suffix = source_path.suffix.lower()
    if output_name:
        requested = Path(str(output_name)).name
        if requested.lower().endswith(suffix):
            requested = requested[: -len(suffix)]
        cleaned = _SAFE_NAME.sub("_", requested.strip().strip("."))[:150]
        if cleaned:
            return cleaned
    return f"{source_path.stem}.edited"


def _build_file_artifact(
    config: SourceEditConfig,
    *,
    thread_id: str,
    filename: str,
    size_bytes: int,
) -> dict[str, object]:
    return {
        "type": FILE_ARTIFACT_TYPE,
        "version": FILE_ARTIFACT_VERSION,
        "kind": FILE_ARTIFACT_KIND,
        "provider": FILE_ARTIFACT_PROVIDER,
        "threadId": thread_id,
        "filename": filename,
        "mimeType": config.mime_type,
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
    "LineValidator",
    "SourceEditConfig",
    "SourceEditError",
    "SourceEditOperation",
    "SourceEditResult",
    "SourceFileInput",
    "SourceInspectInput",
    "build_source_edit_tools",
    "build_source_file_tool",
    "create_source_file",
    "decode_source",
    "edit_source_file",
    "encode_source",
    "inspect_source_file",
    "read_source_file",
]


class SourceFileInput(BaseModel):
    """Input schema for the source-code file reader."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to the file, relative to the configured file-read root "
            "directory. This also covers files uploaded to the current chat "
            "session (under chat_uploads/<thread-id>/)."
        ),
    )
    max_chars: int = Field(
        default=_MAX_CHARS_READER,
        ge=1,
        le=200_000,
        description="Maximum characters of file content to return.",
    )


def build_source_file_tool(
    settings: Settings,
    config: SourceEditConfig,
) -> BaseTool:
    """Create a source-file reader confined to the file-read root."""

    root = Path(settings.file_read_root)
    max_bytes = settings.file_read_max_bytes
    tool_name = f"read_{config.tool_prefix}_file"

    def _run_source_file(path: str, max_chars: int = _MAX_CHARS_READER) -> str:
        return read_source_file(
            config,
            path,
            root=root,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    return StructuredTool.from_function(
        func=_run_source_file,
        name=tool_name,
        description=(
            f"Read the contents of a {config.label} file ({config.suffix_text}) "
            "from the local document directory (including files uploaded to "
            "the current chat session). Use when the user references the file "
            "by name and wants its contents read, summarized, or searched. "
            "Returns the raw source text (possibly truncated)."
        ),
        args_schema=SourceFileInput,
    )


def read_source_file(
    config: SourceEditConfig,
    path: str,
    *,
    root: Path,
    max_bytes: int,
    max_chars: int = _MAX_CHARS_READER,
) -> str:
    try:
        resolved = resolve_safe_path(
            path,
            root=root,
            expected_suffixes=config.suffixes,
            max_bytes=max_bytes,
        )
    except FileAccessError as exc:
        return f"Could not read {config.label} file: {exc}"

    try:
        from ._file_cache import PARSED_FILE_CACHE

        # Source code and plain text share a parser key, so identical uploaded
        # content is decoded only once even when reached through either tool.
        text = PARSED_FILE_CACHE.get_or_compute(
            resolved,
            parser_key="utf8-text-v1",
            loader=lambda: resolved.read_text(encoding="utf-8", errors="replace"),
        )
    except OSError as exc:
        return f"Could not read {config.label} file {resolved.name!r}: {exc}"

    limit = max(1, int(max_chars))
    truncated = len(text) > limit
    body = text[:limit]
    header = f"Contents of {resolved.name} ({len(text):,} chars):"
    if truncated:
        header = (
            f"Contents of {resolved.name} (showing first {limit:,} of "
            f"{len(text):,} chars):"
        )
    return f"{header}\n\n{body}"
