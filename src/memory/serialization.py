"""JSON serializer/parser pair for the long-term memory document.

The store is rewritten whole on every mutation, so these two functions are the
entire on-disk contract. They are kept pure — the parser returns warnings
instead of logging them — so the round-trip property can be tested directly:

    parse_document(serialize_document(doc)).records == doc.records

``RECORD_FIELDS`` in :mod:`.models` fixes the field order both sides use. Keep
them in sync or the round-trip breaks.
"""

from __future__ import annotations

import json
from typing import Any, cast

from .models import (
    CATEGORIES,
    DEFAULT_CATEGORY,
    RECORD_FIELDS,
    SCHEMA_VERSION,
    SCOPES,
    MemoryDocument,
    MemoryRecord,
)

_REQUIRED_STRING_FIELDS = ("id", "content", "scope")


class CorruptDocumentError(ValueError):
    """The file is not usable as a memory document and cannot be salvaged.

    Raised only for a whole-document failure (unparseable JSON, or a top-level
    value that is not an object). Individual bad records are skipped instead.
    """


def serialize_document(doc: MemoryDocument) -> str:
    """Render a document as UTF-8-ready JSON text with a trailing newline."""

    payload: dict[str, Any] = {
        "version": doc.version,
        "updated_at": doc.updated_at,
        "records": [_record_to_json(record) for record in doc.records],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def parse_document(
    raw: str,
    *,
    fallback_updated_at: str,
) -> tuple[MemoryDocument, list[str]]:
    """Parse document text, returning the document and any warnings.

    Warnings are returned rather than logged so this stays a pure function; the
    store emits them. Unknown record keys are dropped and are therefore never
    written back.
    """

    warnings: list[str] = []

    try:
        payload = json.loads(raw)
    except ValueError as exc:  # includes json.JSONDecodeError
        raise CorruptDocumentError(f"memory store is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise CorruptDocumentError(
            "memory store top-level value is not a JSON object "
            f"(got {type(payload).__name__})"
        )

    version = payload.get("version")
    if not _is_readable_version(version):
        warnings.append(
            f"memory store version {version!r} is not readable by schema "
            f"version {SCHEMA_VERSION}; ignoring all records"
        )
        return MemoryDocument.empty(fallback_updated_at), warnings

    doc_updated_at = payload.get("updated_at")
    if not isinstance(doc_updated_at, str) or not doc_updated_at.strip():
        doc_updated_at = fallback_updated_at

    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        warnings.append("memory store 'records' is not a list; ignoring all records")
        raw_records = []

    records: list[MemoryRecord] = []
    for index, item in enumerate(raw_records):
        record, record_warnings = _parse_record(
            item, index=index, doc_updated_at=doc_updated_at
        )
        warnings.extend(record_warnings)
        if record is not None:
            records.append(record)

    document = MemoryDocument(
        version=cast(int, version),
        updated_at=doc_updated_at,
        records=tuple(records),
    )
    return document, warnings


def _record_to_json(record: MemoryRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in RECORD_FIELDS:
        value = getattr(record, name)
        payload[name] = list(value) if name == "tags" else value
    return payload


def _is_readable_version(version: Any) -> bool:
    if isinstance(version, bool) or not isinstance(version, int):
        return False
    return 1 <= version <= SCHEMA_VERSION


def _parse_record(
    item: Any,
    *,
    index: int,
    doc_updated_at: str,
) -> tuple[MemoryRecord | None, list[str]]:
    if not isinstance(item, dict):
        return None, [f"memory record at position {index} is not a JSON object; skipped"]

    values: dict[str, str] = {}
    for name in _REQUIRED_STRING_FIELDS:
        value = item.get(name)
        if not isinstance(value, str) or not value.strip():
            return None, [
                f"memory record at position {index} has an unusable "
                f"{name!r} field; skipped"
            ]
        values[name] = value

    scope = values["scope"]
    if scope not in SCOPES:
        return None, [
            f"memory record at position {index} has an unusable 'scope' field; skipped"
        ]

    warnings: list[str] = []

    scope_id = item.get("scope_id")
    if scope == "global":
        if scope_id is not None:
            warnings.append(
                f"memory record at position {index} has a 'scope_id' on a global "
                "record; discarded"
            )
        scope_id = None
    elif not isinstance(scope_id, str) or not scope_id.strip():
        warnings.append(
            f"memory record at position {index} has an unusable 'scope_id' field; "
            "discarded"
        )
        scope_id = None

    category = item.get("category")
    if not isinstance(category, str) or category not in CATEGORIES:
        if category is not None:
            warnings.append(
                f"memory record at position {index} has an unusable 'category' "
                f"field; defaulted to {DEFAULT_CATEGORY!r}"
            )
        category = DEFAULT_CATEGORY

    tags, tag_warnings = _parse_tags(item.get("tags"), index=index)
    warnings.extend(tag_warnings)

    created_at, created_warning = _parse_timestamp(
        item.get("created_at"), index=index, name="created_at", fallback=doc_updated_at
    )
    warnings.extend(created_warning)
    updated_at, updated_warning = _parse_timestamp(
        item.get("updated_at"), index=index, name="updated_at", fallback=doc_updated_at
    )
    warnings.extend(updated_warning)

    last_recalled_at = item.get("last_recalled_at")
    if last_recalled_at is not None and (
        not isinstance(last_recalled_at, str) or not last_recalled_at.strip()
    ):
        warnings.append(
            f"memory record at position {index} has an unusable "
            "'last_recalled_at' field; discarded"
        )
        last_recalled_at = None

    record = MemoryRecord(
        id=values["id"],
        scope=scope,  # type: ignore[arg-type]
        scope_id=scope_id,
        category=category,  # type: ignore[arg-type]
        content=values["content"],
        tags=tags,
        created_at=created_at,
        updated_at=updated_at,
        last_recalled_at=last_recalled_at,
    )
    return record, warnings


def _parse_tags(raw: Any, *, index: int) -> tuple[tuple[str, ...], list[str]]:
    if raw is None:
        return (), []
    if not isinstance(raw, list):
        return (), [
            f"memory record at position {index} has an unusable 'tags' field; "
            "defaulted to an empty list"
        ]

    tags = [item for item in raw if isinstance(item, str) and item.strip()]
    warnings: list[str] = []
    if len(tags) != len(raw):
        warnings.append(
            f"memory record at position {index} has unusable entries in 'tags'; "
            "they were discarded"
        )
    return tuple(tags), warnings


def _parse_timestamp(
    raw: Any,
    *,
    index: int,
    name: str,
    fallback: str,
) -> tuple[str, list[str]]:
    if isinstance(raw, str) and raw.strip():
        return raw, []
    warnings = []
    if raw is not None:
        warnings.append(
            f"memory record at position {index} has an unusable {name!r} field; "
            "defaulted to the document timestamp"
        )
    return fallback, warnings


__all__ = [
    "CorruptDocumentError",
    "parse_document",
    "serialize_document",
]
