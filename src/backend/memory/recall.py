"""Rendering, automatic recall injection, and the per-turn tool-call budget.

This module deliberately does **not** import :mod:`.store` at module level.
``store`` imports :func:`format_records` from here, so a top-level import back
would form a cycle; :func:`build_turn_messages` imports the store lazily.

Automatic injection is read-only on purpose. It goes through ``store.in_scope``
and :func:`rank_records`, never ``store.recall``, so ``last_recalled_at`` is
touched only by an explicit ``recall_memory`` call. That is what keeps recall
output deterministic and keeps pruning recency meaningful.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .models import MAX_QUERY_CHARS, MAX_TOOL_CALLS_PER_TURN, MemoryRecord
from .relevance import derive_query_terms, rank_records

if TYPE_CHECKING:  # pragma: no cover - avoids an import cycle at runtime
    from src.config import Settings

    from .store import MemoryStore

logger = logging.getLogger(__name__)

#: Prefix identifying an injected memory note. Short, stable, and used by no
#: other message the chat agent adds to a turn.
MEMORY_NOTE_LABEL = "LONG-TERM MEMORY (recalled):"

#: Zero-width marker appended to the label so automatic extraction can identify
#: an injected recall note without inspecting its text, and therefore never
#: re-extract a recalled memory into a new record. Zero-width so the text the
#: model reads is unchanged.
MEMORY_NOTE_MARKER = "\u200b\u200b"

TRUNCATION_MARKER = " ... [truncated]"

#: Cap on tracked threads so a long-lived server cannot grow the budget map
#: without bound.
_MAX_TRACKED_THREADS = 1024


def format_record(record: MemoryRecord, *, include_category: bool = True) -> str:
    """Render one record as a single line for the model to read."""

    if include_category:
        head = f"- [{record.id}] ({record.category}, updated {record.updated_at})"
    else:
        head = f"- [{record.id}]"
    return f"{head} {record.content}"


def format_records(
    records: Sequence[MemoryRecord],
    *,
    max_chars: int,
    include_category: bool = True,
) -> str:
    """Render records as whole lines, dropping trailing ones to fit ``max_chars``.

    Entries are dropped from the end rather than cut mid-record, so the model
    never sees a partial memory. The first entry is always emitted even if it
    alone exceeds the budget; the tool layer applies the final length cap.
    """

    limit = max(1, int(max_chars))
    lines: list[str] = []
    total = 0

    for record in records:
        line = format_record(record, include_category=include_category)
        cost = len(line) + (1 if lines else 0)
        if lines and total + cost > limit:
            break
        lines.append(line)
        total += cost

    return "\n".join(lines)


def build_memory_note(
    store: MemoryStore,
    *,
    message: str,
    thread_id: str | None,
    top_k: int,
    max_chars: int,
) -> str | None:
    """Return the memory ``SystemMessage`` body for a turn, or ``None``.

    Never raises and never writes. Any failure degrades the turn to "no memory
    context" instead of breaking it.
    """

    try:
        terms = derive_query_terms((message or "")[:MAX_QUERY_CHARS])
        if not terms:
            return None

        ranked = rank_records(store.in_scope(thread_id), terms, top_k=top_k)
        if not ranked:
            return None

        overhead = len(MEMORY_NOTE_LABEL) + len(MEMORY_NOTE_MARKER) + 1
        budget = max(1, int(max_chars) - overhead)
        body = format_records(ranked, max_chars=budget)
        if not body:
            return None
        return f"{MEMORY_NOTE_LABEL}{MEMORY_NOTE_MARKER}\n{body}"
    except Exception as exc:  # noqa: BLE001 - a bad store must not break a turn
        logger.warning("skipping long-term memory recall for this turn: %s", exc)
        return None


class MemoryCallBudget:
    """Bounds how many memory tool calls one chat turn may make.

    Reset once per turn by the turn builder. Without a bound, a model that
    mis-reads a tool result can loop on ``save_memory`` for a whole turn.
    """

    def __init__(self, limit: int = MAX_TOOL_CALLS_PER_TURN) -> None:
        self._limit = max(1, int(limit))
        self._lock = threading.Lock()
        self._used: dict[str, int] = {}

    @property
    def limit(self) -> int:
        return self._limit

    def reset(self, thread_id: str | None) -> None:
        key = _budget_key(thread_id)
        with self._lock:
            if len(self._used) > _MAX_TRACKED_THREADS:
                self._used.clear()
            self._used[key] = 0

    def consume(self, thread_id: str | None) -> bool:
        """Claim one call. Returns ``False`` once the turn budget is spent."""

        key = _budget_key(thread_id)
        with self._lock:
            used = self._used.get(key, 0)
            if used >= self._limit:
                return False
            self._used[key] = used + 1
            return True

    def used(self, thread_id: str | None) -> int:
        with self._lock:
            return self._used.get(_budget_key(thread_id), 0)


#: Process-wide budget shared by the tools and both turn builders.
MEMORY_BUDGET = MemoryCallBudget()


def build_turn_messages(
    settings: Settings,
    *,
    thread_id: str | None,
    message: str,
    upload_note: str | None = None,
    budget: MemoryCallBudget | None = None,
) -> list[Any]:
    """Build one chat turn's message list, memory note first.

    Shared by the FastAPI turn builder and the CLI REPL so the two entry points
    cannot drift on selection, placement, or limits. The memory note is placed
    ahead of the upload-context note and ahead of the user message.
    """

    from langchain_core.messages import HumanMessage, SystemMessage

    (budget or MEMORY_BUDGET).reset(thread_id)

    messages: list[Any] = []

    if getattr(settings, "memory_enabled", False) and getattr(
        settings, "memory_auto_recall_enabled", False
    ):
        note = _memory_note_for(settings, thread_id=thread_id, message=message)
        if note is not None:
            messages.append(SystemMessage(content=note))

    if upload_note:
        messages.append(SystemMessage(content=upload_note))

    messages.append(HumanMessage(content=message))
    return messages


def _memory_note_for(
    settings: Settings,
    *,
    thread_id: str | None,
    message: str,
) -> str | None:
    from .store import get_memory_store

    try:
        store = get_memory_store(settings)
    except Exception as exc:  # noqa: BLE001 - e.g. a rejected store path
        logger.warning("long-term memory is unavailable this turn: %s", exc)
        return None

    return build_memory_note(
        store,
        message=message,
        thread_id=thread_id,
        top_k=settings.memory_recall_top_k,
        max_chars=settings.memory_context_max_chars,
    )


def _budget_key(thread_id: str | None) -> str:
    return (thread_id or "").strip() or "<no-thread>"


__all__ = [
    "MEMORY_BUDGET",
    "MEMORY_NOTE_LABEL",
    "MEMORY_NOTE_MARKER",
    "TRUNCATION_MARKER",
    "MemoryCallBudget",
    "build_memory_note",
    "build_turn_messages",
    "format_record",
    "format_records",
]
