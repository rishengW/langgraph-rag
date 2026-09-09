"""Session-scoped YAML (.yaml/.yml) creation, inspection, and editing tools.

Like the JSON editor, YAML is addressed structurally: the inspector lists
dotted paths (``$.users[0].name``) and the editor applies path-based
operations whose expectations are checked against the document before
anything is written. Edits re-serialize the document with ``yaml.safe_dump``
(so comments, custom anchors, and non-canonical formatting are normalized on
the published copy); the uploaded source is never modified. Only
single-document YAML streams are supported; a multi-document file is refused
with a clear error. Parsing uses ``yaml.safe_load`` exclusively, so
arbitrary Python tags cannot execute.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
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

YAML_MIME_TYPE = "text/yaml"
_MAX_VALUE_CHARS = 100_000
_MAX_INSPECT_ENTRIES = 1_000
_MAX_SCALAR_CHARS = 200

YAML_CONFIG = SourceEditConfig(
    label="YAML",
    suffixes=(".yaml", ".yml"),
    creation_suffix=".yaml",
    tool_prefix="yaml",
    mime_type=YAML_MIME_TYPE,
    flag_name="yaml_edit_enabled",
    temp_prefix=".yaml_write-",
)


class YamlEditError(Exception):
    """Raised when a YAML file cannot be inspected or edited safely."""


# (kind, key) steps flatten the dotted path grammar into a walk. "key" steps
# always carry a str, "index" steps always carry an int.
_Step = tuple[Literal["key"], str] | tuple[Literal["index"], int]


def _is_root_path(path: str) -> bool:
    """True for '', '$', '$.', and other spellings of the whole document."""

    return path.strip().strip("$").strip(".") == ""


class YamlEditOperation(BaseModel):
    """One path-addressed edit, validated against the current document."""

    action: Literal["set_value", "delete_key", "append_to_array"] = Field(
        ..., description="The kind of YAML edit to apply."
    )
    path: str = Field(
        default="",
        max_length=500,
        description=(
            "Dotted path to the target, e.g. $.users[0].name; '$' or '' is "
            "the whole document. Required."
        ),
    )
    value: str | None = Field(
        default=None,
        max_length=_MAX_VALUE_CHARS,
        description=(
            "JSON-encoded replacement for set_value / element for "
            "append_to_array, e.g. '\"text\"', '42', 'true', '{\"a\": 1}'."
        ),
    )
    expected_value: str | None = Field(
        default=None,
        max_length=_MAX_VALUE_CHARS,
        description=(
            "JSON-encoded current value at the path, required by set_value "
            "on an existing path and by every delete_key."
        ),
    )
    expected_missing: bool = Field(
        default=False,
        description=(
            "set_value only: declare the path does not exist yet so a new "
            "key can be created without an expected_value."
        ),
    )
    expected_length: int | None = Field(
        default=None,
        ge=0,
        description="Current array length, required by every append_to_array.",
    )

    @model_validator(mode="after")
    def _check_fields(self) -> YamlEditOperation:
        try:
            _parse_path(self.path)
        except YamlEditError as exc:
            raise ValueError(str(exc)) from exc

        if self.action == "set_value":
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
                    "(path exists) or expected_missing=true (create it)."
                )
            if _is_root_path(self.path) and self.expected_missing:
                raise ValueError("the document root always exists; pass expected_value.")
            self._require_parsed("value")
            if self.expected_value is not None:
                self._require_parsed("expected_value")
            if self.expected_length is not None:
                raise ValueError("set_value does not accept expected_length.")
        elif self.action == "delete_key":
            if self.expected_value is None:
                raise ValueError("delete_key requires: expected_value.")
            self._require_parsed("expected_value")
            if self.expected_missing:
                raise ValueError("delete_key does not accept expected_missing.")
            if self.expected_length is not None:
                raise ValueError("delete_key does not accept expected_length.")
            if not self.path.strip() or self.path.strip() == "$":
                raise ValueError("delete_key cannot target the document root.")
        else:  # append_to_array
            if self.value is None:
                raise ValueError("append_to_array requires: value.")
            self._require_parsed("value")
            if self.expected_length is None:
                raise ValueError("append_to_array requires: expected_length.")
            if self.expected_value is not None:
                raise ValueError("append_to_array does not accept expected_value.")
            if self.expected_missing:
                raise ValueError("append_to_array does not accept expected_missing.")
            if not self.path.strip():
                raise ValueError("append_to_array requires a path to an array.")
        return self

    def _require_parsed(self, name: str) -> None:
        raw = getattr(self, name)
        try:
            json.loads(raw)
        except ValueError as exc:
            raise ValueError(f"{name} is not valid JSON: {exc}") from exc


class YamlInspectInput(BaseModel):
    """Input schema for the YAML inspector."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .yaml or .yml file uploaded to this chat session, "
            "exactly as listed in the upload note."
        ),
    )
    max_chars: int = Field(
        default=_INSPECT_MAX_CHARS,
        ge=1,
        le=200_000,
        description="Maximum characters of the path listing to return.",
    )


class YamlCreateInput(BaseModel):
    """Input schema for creating a YAML file."""

    filename: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "Bare filename for the new YAML file. A .yaml suffix is added "
            "when needed. Existing files are never overwritten."
        ),
    )
    content: str = Field(
        ...,
        description=(
            "Complete UTF-8 YAML text to write. It must parse as valid "
            "(single-document) YAML; formatting is preserved as provided."
        ),
    )


class YamlEditInput(BaseModel):
    """Input schema for the YAML editor."""

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Path to a .yaml or .yml file uploaded to this chat session. "
            "The source is never modified."
        ),
    )
    operations: list[YamlEditOperation] = Field(
        ...,
        min_length=1,
        max_length=_MAX_OPERATIONS,
        description=(
            "Path-based edits to validate against the original document and "
            "then apply as one transaction."
        ),
    )
    output_name: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Optional name for the edited copy. Defaults to "
            "'<original>.edited.yaml'; a .yaml/.yml suffix is enforced."
        ),
    )


def build_yaml_edit_tools(
    settings: Settings,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> list[BaseTool]:
    """Create YAML tools only when enabled and safely scoped to a chat."""

    if not (settings.file_read_enabled and settings.yaml_edit_enabled):
        return []
    if session_root is None or not _THREAD_ID_PATTERN.fullmatch(thread_id):
        return []

    file_root = Path(settings.file_read_root)
    max_bytes = settings.file_read_max_bytes

    def _run_create(
        filename: str,
        content: str,
    ) -> tuple[str, dict[str, object] | None]:
        result = create_yaml_file(
            filename,
            content,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
            thread_id=thread_id,
        )
        return result.content, result.artifact

    def _run_inspect(path: str, max_chars: int = _INSPECT_MAX_CHARS) -> str:
        return inspect_yaml_file(
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
            max_chars=max_chars,
        )

    def _run_edit(
        path: str,
        operations: list[YamlEditOperation],
        output_name: str | None = None,
    ) -> tuple[str, dict[str, object] | None]:
        result = edit_yaml_file(
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
        name="create_yaml_file",
        description=(
            "Create a new UTF-8 .yaml (or .yml) file in the current chat "
            "session and return it as a download. The content must parse as "
            "valid single-document YAML. Existing files are never "
            "overwritten; a numbered filename is chosen on collision. Use "
            "only when the user explicitly asks to create a YAML file."
        ),
        args_schema=YamlCreateInput,
        response_format="content_and_artifact",
    )
    inspect_tool = StructuredTool.from_function(
        func=_run_inspect,
        name="inspect_yaml_file",
        description=(
            "List the documented paths ($, $.key, $.items[0], ...) and "
            "values of a .yaml or .yml file uploaded to this chat session. "
            "Call this before edit_yaml_file to obtain the exact paths and "
            "current values required by edit operations."
        ),
        args_schema=YamlInspectInput,
    )
    edit_tool = StructuredTool.from_function(
        func=_run_edit,
        name="edit_yaml_file",
        description=(
            "Apply path-based edits (set_value, delete_key, append_to_array) "
            "to an uploaded .yaml or .yml file and create a new downloadable "
            "copy without changing the original. Call inspect_yaml_file "
            "first; pass expected_value for existing paths or "
            "expected_missing=true to create a path. The copy is "
            "re-serialized, so comments and custom formatting are "
            "normalized. Use only when the user explicitly asks to change "
            "the file."
        ),
        args_schema=YamlEditInput,
        response_format="content_and_artifact",
    )
    return [create_tool, inspect_tool, edit_tool]


def create_yaml_file(
    filename: str,
    content: str,
    *,
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    thread_id: str = "",
) -> SourceEditResult:
    """Create a session-scoped UTF-8 YAML file and downloadable artifact."""

    output_name = "(unresolved)"
    try:
        if not _THREAD_ID_PATTERN.fullmatch(thread_id):
            raise YamlEditError("a valid chat thread is required to create a YAML file.")
        directory = _prepare_session_directory(
            session_root=session_root,
            file_root=file_root,
        )
        stem, suffix = _creation_stem(YAML_CONFIG, filename)
        document = decode_source(content.encode("utf-8"), name=f"{stem}{suffix}")
        _parse_document_text(document, f"{stem}{suffix}")
        target = _reserve_created_output_path(
            directory=directory,
            stem=stem,
            suffix=suffix,
        )
        output_name = target.name
        published = _write_source_atomically(
            YAML_CONFIG,
            document,
            target=target,
            max_bytes=max_bytes,
            size_label="new",
        )
        size_bytes = published.stat().st_size
    except (OSError, UnicodeError, SourceEditError, YamlEditError) as exc:
        logger.info(
            "yaml_create failed: thread=%s output=%s reason=%s",
            thread_id or "(none)",
            output_name,
            type(exc).__name__,
        )
        return SourceEditResult(content=f"Could not create YAML file: {exc}")

    logger.info(
        "yaml_create succeeded: thread=%s output=%s bytes=%d",
        thread_id,
        published.name,
        size_bytes,
    )
    return SourceEditResult(
        content=(
            f"Created {published.name} in this chat session. The file is available to download."
        ),
        artifact=_build_file_artifact(
            YAML_CONFIG,
            thread_id=thread_id,
            filename=published.name,
            size_bytes=size_bytes,
        ),
    )


def inspect_yaml_file(
    path: str,
    *,
    session_root: Path,
    file_root: Path,
    max_bytes: int,
    max_chars: int = _INSPECT_MAX_CHARS,
) -> str:
    """Return format metadata and a listing of the document's paths."""

    try:
        resolved = _resolve_session_source(
            YAML_CONFIG,
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
        )
        document = _read_source_document(resolved)
        data = _parse_document_text(document, resolved.name)
    except SourceEditError as exc:
        return f"Could not inspect YAML file: {exc}"
    except YamlEditError as exc:
        return f"Could not inspect YAML file: {exc}"

    encoding = "UTF-8 with BOM" if document.encoding == "utf-8-sig" else "UTF-8"
    final_newline = "yes" if document.has_final_newline else "no"
    lines = [
        f"YAML file {resolved.name}:",
        f"Encoding: {encoding}",
        f"Newline: {document.newline}",
        f"Final newline: {final_newline}",
        "PATHS:",
    ]
    entries: list[str] = []
    _walk_paths(data, "", entries)
    truncated_entries = len(entries) > _MAX_INSPECT_ENTRIES
    lines.extend(entries[:_MAX_INSPECT_ENTRIES])
    if truncated_entries:
        lines.append(
            f"[… {_MAX_INSPECT_ENTRIES:,} of {len(entries):,} paths shown; "
            "inspect sub-paths by editing a copy]"
        )
    body = "\n".join(lines)
    limit = max(1, int(max_chars))
    if len(body) > limit:
        return f"{body[:limit]}\n\n[listing truncated to {limit:,} of {len(body):,} chars]"
    return body


def edit_yaml_file(
    path: str,
    *,
    operations: list[YamlEditOperation],
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
            YAML_CONFIG,
            path,
            session_root=session_root,
            file_root=file_root,
            max_bytes=max_bytes,
        )
        source_name = resolved.name
        document = _read_source_document(resolved)
        original_text = document.newline_text.join(document.lines)
        data = _parse_document_text(document, resolved.name)

        problems: list[str] = []
        for position, operation in enumerate(operations):
            problem = _check_operation(data, operation)
            if problem:
                problems.append(f"operation {position} ({operation.action}): {problem}")
        if problems:
            raise YamlEditError(
                f"no changes were made because {len(problems)} operation(s) "
                f"did not match the file: {' '.join(problems)} Re-run "
                "inspect_yaml_file and retry."
            )

        working = copy.deepcopy(data)
        for operation in operations:
            working = _apply_operation(working, operation)

        indent = _detect_indent(original_text)
        edited_text = _serialize_yaml(working, indent=indent)
        edited = _document_from_text(
            edited_text,
            encoding=document.encoding,
            newline=document.newline,
            has_final_newline=document.has_final_newline,
        )
        published = _publish_edited(
            YAML_CONFIG,
            edited,
            session_root=session_root,
            source_path=resolved,
            output_name=output_name,
            max_bytes=max_bytes,
        )
    except SourceEditError as exc:
        logger.info(
            "yaml_edit failed: thread=%s source=%s operations=%d reason=%s",
            thread_id or "(none)",
            source_name,
            len(operations),
            type(exc).__name__,
        )
        return SourceEditResult(content=f"Could not edit YAML file: {exc}")
    except YamlEditError as exc:
        logger.info(
            "yaml_edit failed: thread=%s source=%s operations=%d reason=%s",
            thread_id or "(none)",
            source_name,
            len(operations),
            type(exc).__name__,
        )
        return SourceEditResult(content=f"Could not edit YAML file: {exc}")

    size_bytes = published.stat().st_size
    logger.info(
        "yaml_edit succeeded: thread=%s source=%s output=%s operations=%d bytes=%d",
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
            YAML_CONFIG,
            thread_id=thread_id,
            filename=published.name,
            size_bytes=size_bytes,
        ),
    )


def _parse_document_text(document: _SourceDocument, name: str) -> Any:
    """Parse the decoded document text, raising YamlEditError on bad YAML.

    ``yaml.safe_load`` is used exclusively so arbitrary ``!!python/...`` tags
    can never execute. Multi-document streams are refused because path-based
    editing only addresses a single root.
    """

    text = document.newline_text.join(document.lines)
    try:
        documents = list(yaml.safe_load_all(text))
    except yaml.YAMLError as exc:
        raise YamlEditError(f"{name!r} is not valid YAML: {exc}") from exc
    if len(documents) > 1:
        raise YamlEditError(
            f"{name!r} is a multi-document YAML stream; only single-document "
            "YAML files are supported."
        )
    return documents[0] if documents else None


def _serialize_yaml(data: Any, *, indent: int) -> str:
    """Re-serialize the edited document as canonical block YAML text."""

    text = yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        indent=max(1, indent),
        width=4096,
    )
    # safe_dump always ends with a trailing newline; drop it so line
    # splitting is clean and the document's preserved newline re-adds it.
    if text.endswith("\n"):
        text = text[:-1]
    return text


def _document_from_text(
    text: str,
    *,
    encoding: Literal["utf-8", "utf-8-sig"],
    newline: Literal["LF", "CRLF"],
    has_final_newline: bool,
) -> _SourceDocument:
    """Rebuild a line document from serialized YAML text."""

    return _SourceDocument(
        lines=tuple(text.split("\n")),
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


def _parse_path(path: str) -> list[tuple[str, tuple[int, ...]]]:
    """Parse ``$.users[0].name`` into (name, indexes) segments."""

    text = (path or "").strip()
    if text.startswith("$"):
        text = text[1:]
    text = text.lstrip(".")
    segments: list[tuple[str, tuple[int, ...]]] = []
    for raw in text.split(".") if text else []:
        match = re.fullmatch(r"([^\[\]]*)((?:\[\d+\])*)", raw)
        if not match:
            raise YamlEditError(f"invalid path segment {raw!r} in {path!r}.")
        name, brackets = match.groups()
        indexes = tuple(int(value) for value in re.findall(r"\[(\d+)\]", brackets))
        if not name and not indexes:
            raise YamlEditError(f"invalid empty path segment in {path!r}.")
        segments.append((name, indexes))
    return segments


def _steps(path: str) -> list[_Step]:
    steps: list[_Step] = []
    for name, indexes in _parse_path(path):
        if name:
            steps.append(("key", name))
        for index in indexes:
            steps.append(("index", index))
    return steps


def _step(node: Any, kind: str, key: str | int, where: str) -> Any:
    if isinstance(key, str):
        if not isinstance(node, dict) or key not in node:
            raise YamlEditError(f"path {where} does not exist.")
        return node[key]
    if not isinstance(node, list) or key >= len(node):
        raise YamlEditError(f"path {where} does not exist (index out of range).")
    return node[key]


def _node_at(document: Any, steps: list[_Step], where: str) -> Any:
    node = document
    for kind, key in steps:
        node = _step(node, kind, key, where)
    return node


def _check_operation(document: Any, operation: YamlEditOperation) -> str | None:
    """Return a problem description, or None when the operation may apply."""

    try:
        where = operation.path or "$"
        steps = _steps(operation.path)
        if operation.action == "set_value":
            if not steps:
                assert operation.expected_value is not None
                expected = json.loads(operation.expected_value)
                return (
                    None
                    if document == expected
                    else "expected_value does not match the current document."
                )
            parent = _node_at(document, steps[:-1], where)
            kind, key = steps[-1]
            if kind == "key":
                if not isinstance(parent, dict):
                    return f"path {where} is not inside an object."
                if key in parent:
                    if operation.expected_missing:
                        return f"path {where} already exists but expected_missing was set."
                    assert operation.expected_value is not None
                    if parent[key] != json.loads(operation.expected_value):
                        return "expected_value does not match the current value."
                elif not operation.expected_missing:
                    return f"path {where} does not exist; set expected_missing=true to create it."
            else:
                if not isinstance(parent, list):
                    return f"path {where} is not inside an array."
                assert isinstance(key, int)
                if key >= len(parent):
                    return f"array index at {where} is out of range."
                if operation.expected_value is None:
                    return "expected_value is required to update an array element."
                if parent[key] != json.loads(operation.expected_value):
                    return "expected_value does not match the current value."
        elif operation.action == "delete_key":
            parent = _node_at(document, steps[:-1], where)
            kind, key = steps[-1]
            assert operation.expected_value is not None
            if kind == "key":
                if not isinstance(parent, dict) or key not in parent:
                    return f"path {where} does not exist."
                if parent[key] != json.loads(operation.expected_value):
                    return "expected_value does not match the current value."
            else:
                assert isinstance(key, int)
                if not isinstance(parent, list) or key >= len(parent):
                    return f"path {where} does not exist (index out of range)."
                if parent[key] != json.loads(operation.expected_value):
                    return "expected_value does not match the current value."
        else:  # append_to_array
            node = _node_at(document, steps, where)
            if not isinstance(node, list):
                return f"path {where} is not an array."
            if operation.expected_length != len(node):
                return (
                    f"expected_length {operation.expected_length} does not match "
                    f"the array length {len(node)} at {where}."
                )
    except YamlEditError as exc:
        return str(exc)
    except (ValueError, TypeError) as exc:
        return f"could not decode a JSON value: {exc}"
    return None


def _apply_operation(document: Any, operation: YamlEditOperation) -> Any:
    """Apply one already-validated operation and return the (new) root."""

    problem = _check_operation(document, operation)
    if problem:
        raise YamlEditError(f"{operation.action} at {operation.path or '$'}: {problem}")

    steps = _steps(operation.path)
    value = json.loads(operation.value) if operation.value is not None else None
    if operation.action == "set_value":
        if not steps:
            return value  # root replacement swaps the whole document
        parent = _node_at(document, steps[:-1], operation.path or "$")
        kind, key = steps[-1]
        parent[key] = value
        return document
    if operation.action == "delete_key":
        parent = _node_at(document, steps[:-1], operation.path or "$")
        kind, key = steps[-1]
        del parent[key]
        return document
    # append_to_array
    node = _node_at(document, steps, operation.path)
    node.append(value)
    return document


def _walk_paths(node: Any, path: str, out: list[str]) -> None:
    here = path or "$"
    if isinstance(node, dict):
        out.append(f"{here} = object ({len(node)} keys)")
        for key, value in node.items():
            _walk_paths(value, f"{here}.{key}", out)
    elif isinstance(node, list):
        out.append(f"{here} = array ({len(node)} items)")
        for index, value in enumerate(node):
            _walk_paths(value, f"{here}[{index}]", out)
    else:
        encoded = _encode_scalar(node)
        if len(encoded) > _MAX_SCALAR_CHARS:
            encoded = f"{encoded[:_MAX_SCALAR_CHARS]}… [+{len(encoded) - _MAX_SCALAR_CHARS} chars]"
        out.append(f"{here} = {encoded}")


def _encode_scalar(node: Any) -> str:
    """Render a scalar for the path listing in a YAML-appropriate form."""

    if node is None:
        return "null"
    if isinstance(node, bool):
        return "true" if node else "false"
    if isinstance(node, (int, float)):
        return repr(node) if isinstance(node, float) else str(node)
    # Strings (including those that look like numbers/bools in YAML) are
    # quoted so the listing is unambiguous, matching the JSON inspector.
    return json.dumps(node, ensure_ascii=False)


def _detect_indent(text: str) -> int:
    """Return the original's indentation unit in spaces (default 2).

    Tabs are illegal for YAML indentation, so a tab-prefixed line is skipped
    rather than honored.
    """

    minimum: int | None = None
    for line in text.splitlines()[1:]:
        if not line.strip():
            continue
        stripped = line[: len(line) - len(line.lstrip())]
        if not stripped or stripped[0] == "\t":
            continue
        width = len(stripped)
        minimum = width if minimum is None else min(minimum, width)
    return minimum or 2


__all__ = [
    "YAML_MIME_TYPE",
    "YamlCreateInput",
    "YamlEditError",
    "YamlEditInput",
    "YamlEditOperation",
    "YamlInspectInput",
    "build_yaml_edit_tools",
    "create_yaml_file",
    "edit_yaml_file",
    "inspect_yaml_file",
]
