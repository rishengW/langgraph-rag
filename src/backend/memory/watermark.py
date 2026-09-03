"""Persistence seam for the extraction watermark.

The watermark records how many turns of a thread have already been extracted, so
a Round is never distilled twice. It is stored in the existing session metadata
``config`` mapping, which is already free-form JSON, so no database migration and
no schema-version bump is needed.

This module deliberately declares a Protocol instead of importing
``src.backend.sessions``: ``src/backend/memory/`` stays free of a dependency on the session layer,
and tests get a two-line fake.
"""

from __future__ import annotations

import threading
from typing import Final, Protocol

#: Key used inside ``SessionMetadata.config``.
WATERMARK_KEY: Final[str] = "extraction_watermark"

#: Sanity ceiling. A value above this is treated as corrupt rather than trusted.
MAX_WATERMARK: Final[int] = 1_000_000


def coerce_watermark(raw: object) -> int:
    """Return a usable watermark, or 0 when the stored value is unusable.

    ``bool`` is rejected explicitly: it is an ``int`` subclass, so ``True`` would
    otherwise silently read as a watermark of 1 and skip a turn.
    """

    if isinstance(raw, bool) or not isinstance(raw, int):
        return 0
    if raw < 0 or raw > MAX_WATERMARK:
        return 0
    return raw


class WatermarkStore(Protocol):
    """Read and write one thread's extraction watermark."""

    def get(self, thread_id: str) -> int:
        """Return the persisted watermark, or 0 when absent or unusable."""

    def set(self, thread_id: str, value: int) -> None:
        """Persist the watermark for one thread."""


class InMemoryWatermarkStore:
    """Thread-safe in-process watermark store, for tests."""

    def __init__(self, initial: dict[str, int] | None = None) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, int] = dict(initial or {})

    def get(self, thread_id: str) -> int:
        with self._lock:
            return coerce_watermark(self._values.get(thread_id, 0))

    def set(self, thread_id: str, value: int) -> None:
        with self._lock:
            self._values[thread_id] = coerce_watermark(value)


__all__ = [
    "MAX_WATERMARK",
    "WATERMARK_KEY",
    "InMemoryWatermarkStore",
    "WatermarkStore",
    "coerce_watermark",
]
