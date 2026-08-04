"""Data model for the chat agent's long-term memory store.

Pure value types plus the shared limits. This module imports nothing from the
rest of the project so it can be used by the serializer, the store, the tool
layer, and the chat layer without creating an import cycle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Literal

MemoryScope = Literal["global", "session"]
MemoryCategory = Literal["fact", "preference", "entity", "task"]

CATEGORIES: Final[tuple[str, ...]] = ("fact", "preference", "entity", "task")
SCOPES: Final[tuple[str, ...]] = ("global", "session")

DEFAULT_CATEGORY: Final[str] = "fact"
DEFAULT_STORE_PATH: Final[str] = "memory/long_term_memory.json"

#: Bumped only when the on-disk field set changes incompatibly. A document
#: carrying a higher version is refused rather than downgraded.
SCHEMA_VERSION: Final[int] = 1

MAX_TAGS: Final[int] = 10
MAX_TAG_CHARS: Final[int] = 40
MAX_QUERY_CHARS: Final[int] = 500
MAX_QUERY_TERMS: Final[int] = 50
MIN_TERM_CHARS: Final[int] = 2
MAX_FORGET_DELETES: Final[int] = 10
MAX_SCOPE_ID_CHARS: Final[int] = 200
MAX_TOOL_CALLS_PER_TURN: Final[int] = 10

#: Prefix for the provenance tag applied to automatically extracted records, so
#: they can be listed or bulk-removed without inspecting their content.
EXTRACTION_TAG_PREFIX: Final[str] = "auto:"

#: 32 lowercase hex characters, i.e. ``uuid4().hex``.
ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"\A[0-9a-f]{32}\Z")

#: Field order used by the serializer. The round-trip property depends on this
#: sequence being stable, so treat it as part of the on-disk contract.
RECORD_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "scope",
    "scope_id",
    "category",
    "content",
    "tags",
    "created_at",
    "updated_at",
    "last_recalled_at",
)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string, seconds precision."""

    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One durable unit of remembered information.

    Frozen so every mutation goes through :func:`dataclasses.replace`, which is
    what makes "leave every other field unchanged" cheap to guarantee and easy
    to assert on.
    """

    id: str
    scope: MemoryScope
    scope_id: str | None
    category: MemoryCategory
    content: str
    tags: tuple[str, ...]
    created_at: str
    updated_at: str
    last_recalled_at: str | None

    @property
    def recency(self) -> str:
        """Effective recency used for capacity eviction.

        A record that has never been recalled falls back to its creation time,
        so newly saved records are not evicted ahead of stale recalled ones.
        """

        return self.last_recalled_at or self.created_at


@dataclass(frozen=True, slots=True)
class MemoryDocument:
    """The whole on-disk document: a schema version plus the record list."""

    version: int
    updated_at: str
    records: tuple[MemoryRecord, ...]

    @classmethod
    def empty(cls, updated_at: str) -> MemoryDocument:
        return cls(version=SCHEMA_VERSION, updated_at=updated_at, records=())


__all__ = [
    "CATEGORIES",
    "DEFAULT_CATEGORY",
    "DEFAULT_STORE_PATH",
    "ID_PATTERN",
    "MAX_FORGET_DELETES",
    "MAX_QUERY_CHARS",
    "MAX_QUERY_TERMS",
    "MAX_SCOPE_ID_CHARS",
    "MAX_TAGS",
    "MAX_TAG_CHARS",
    "EXTRACTION_TAG_PREFIX",
    "MAX_TOOL_CALLS_PER_TURN",
    "MIN_TERM_CHARS",
    "RECORD_FIELDS",
    "SCHEMA_VERSION",
    "SCOPES",
    "MemoryCategory",
    "MemoryDocument",
    "MemoryRecord",
    "MemoryScope",
    "utc_now_iso",
]
