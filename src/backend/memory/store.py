"""Persistence for the chat agent's long-term memory.

One JSON document holds every record, rewritten whole on each mutation. At the
configured cap (500 short records by default) that costs almost nothing and
removes every partial-update failure mode.

Reads are cached on the file's ``(mtime_ns, size)`` and re-``stat``-ed on every
call, so a cache can never serve a record set that a completed write has
superseded. Locks and store instances are keyed by resolved path, so the tool
layer, the recall injector, and session deletion all share one lock.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .models import (
    CATEGORIES,
    DEFAULT_CATEGORY,
    DEFAULT_STORE_PATH,
    ID_PATTERN,
    MAX_FORGET_DELETES,
    MAX_QUERY_CHARS,
    MAX_TAG_CHARS,
    MAX_TAGS,
    SCHEMA_VERSION,
    SCOPES,
    MemoryDocument,
    MemoryRecord,
    utc_now_iso,
)
from .recall import format_records
from .relevance import (
    derive_query_terms,
    normalize_content,
    rank_records,
    timestamp_epoch,
)
from .secrets import find_secret_in_any, find_secret_match
from .serialization import CorruptDocumentError, parse_document, serialize_document

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from src.config import Settings

logger = logging.getLogger(__name__)


class MemoryStoreError(RuntimeError):
    """Base class for memory store failures."""


class MemoryPathError(MemoryStoreError):
    """The configured store path is not usable."""


class MemoryWriteError(MemoryStoreError):
    """The store document could not be persisted."""


@dataclass(frozen=True)
class SaveOutcome:
    """Result of a save. ``ok`` false means nothing was persisted."""

    ok: bool
    message: str
    record_count: int
    record_id: str | None = None
    created: bool = False
    updated: bool = False
    pruned: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecallOutcome:
    """Result of a recall. ``message`` is the rendered record list."""

    ok: bool
    message: str
    record_count: int
    records: tuple[MemoryRecord, ...] = ()


@dataclass(frozen=True)
class ForgetOutcome:
    """Result of a forget. ``unmatched`` counts matches left in place."""

    ok: bool
    message: str
    record_count: int
    deleted: tuple[str, ...] = ()
    unmatched: int = 0


# One initial attempt plus three retries. On Windows a concurrent reader holding
# the target open surfaces as PermissionError from os.replace, and it clears in
# milliseconds, so a short bounded backoff beats failing the whole tool call.
_REPLACE_BACKOFF: tuple[float, ...] = (0.05, 0.10, 0.20)
_REPLACE_ATTEMPTS = len(_REPLACE_BACKOFF) + 1

_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()

_STORES: dict[Path, MemoryStore] = {}
_STORES_GUARD = threading.Lock()


def resolve_store_path(settings: Settings) -> Path:
    """Resolve the store path from Settings only.

    No tool parameter can influence this, which is what keeps the agent from
    steering reads or writes at an arbitrary file. A ``..`` segment is refused
    outright rather than normalized away.
    """

    raw = (getattr(settings, "memory_store_path", "") or "").strip()
    candidate = Path(raw or DEFAULT_STORE_PATH).expanduser()

    if any(part == ".." for part in candidate.parts):
        raise MemoryPathError(
            "memory_store_path must not contain a '..' path segment; "
            f"got {raw or DEFAULT_STORE_PATH!r}"
        )

    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.normpath(candidate))


def in_scope_records(
    records: tuple[MemoryRecord, ...],
    thread_id: str | None,
) -> tuple[MemoryRecord, ...]:
    """Filter records visible to ``thread_id``.

    Global records are visible everywhere, including when no thread is
    resolved. Session records are visible only on an exact thread match, so a
    session record whose ``scope_id`` is missing is visible nowhere.
    """

    resolved = (thread_id or "").strip() or None
    return tuple(
        record
        for record in records
        if record.scope == "global"
        or (resolved is not None and record.scope_id == resolved)
    )


class MemoryStore:
    """Read/write access to one memory document."""

    def __init__(
        self,
        path: Path,
        *,
        max_records: int,
        max_record_chars: int,
        default_scope: str = "global",
        clock: Callable[[], str] = utc_now_iso,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._path = Path(path)
        self._max_records = max(1, int(max_records))
        self._max_record_chars = max(1, int(max_record_chars))
        self._default_scope = default_scope if default_scope in SCOPES else "global"
        self._clock = clock
        self._sleeper = sleeper
        self._lock = _lock_for(self._path)
        self._cache: tuple[tuple[int, int], tuple[MemoryRecord, ...]] | None = None

    # ---- introspection ----------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def max_records(self) -> int:
        return self._max_records

    @property
    def max_record_chars(self) -> int:
        return self._max_record_chars

    @property
    def default_scope(self) -> str:
        return self._default_scope

    def update_limits(
        self,
        *,
        max_records: int,
        max_record_chars: int,
        default_scope: str | None = None,
    ) -> None:
        """Refresh the bounds without discarding the shared lock or cache."""

        self._max_records = max(1, int(max_records))
        self._max_record_chars = max(1, int(max_record_chars))
        if default_scope is not None and default_scope in SCOPES:
            self._default_scope = default_scope

    # ---- reads ------------------------------------------------------------

    def read(self) -> tuple[MemoryRecord, ...]:
        """Return every persisted record, or an empty tuple when none exist.

        A missing file is not an error and does not create anything. An
        unparseable file is quarantined and treated as empty so one bad
        document cannot take the chat agent down.
        """

        with self._lock:
            return self._read_locked()

    def in_scope(self, thread_id: str | None) -> tuple[MemoryRecord, ...]:
        """Return the records visible to ``thread_id``."""

        return in_scope_records(self.read(), thread_id)

    def _read_locked(self, *, force: bool = False) -> tuple[MemoryRecord, ...]:
        """Read the document, optionally bypassing the cache.

        Mutations pass ``force=True`` so a read-modify-write always starts from
        the file itself. The ``(mtime_ns, size)`` cache is an optimization for
        the read-only paths only, where a stale hit costs nothing worse than a
        slightly old recall.
        """

        try:
            stat = self._path.stat()
        except FileNotFoundError:
            self._cache = None
            return ()
        except OSError as exc:
            raise MemoryStoreError(f"could not stat the memory store: {exc}") from exc

        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._cache
        if not force and cached is not None and cached[0] == signature:
            return cached[1]

        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._cache = None
            return ()
        except OSError as exc:
            raise MemoryStoreError(f"could not read the memory store: {exc}") from exc

        try:
            document, warnings = parse_document(
                raw, fallback_updated_at=self._clock()
            )
        except CorruptDocumentError as exc:
            self._quarantine(exc)
            return ()

        for message in warnings:
            logger.warning("%s", message)

        records = document.records
        if len(records) > self._max_records:
            logger.warning(
                "memory store holds %d records, above the configured cap of %d; "
                "it will be reduced on the next write",
                len(records),
                self._max_records,
            )

        self._cache = (signature, records)
        return records

    def _quarantine(self, reason: Exception) -> None:
        """Move an unreadable document aside and continue with an empty store."""

        self._cache = None
        stamp = self._clock().replace(":", "")
        target = self._path.with_name(f"{self._path.name}.corrupt-{stamp}")
        suffix = 1
        while target.exists():
            target = self._path.with_name(
                f"{self._path.name}.corrupt-{stamp}-{suffix}"
            )
            suffix += 1

        try:
            os.replace(self._path, target)
        except OSError as exc:
            logger.warning(
                "memory store at %s is unreadable (%s) and could not be moved "
                "aside (%s); continuing with an empty store",
                self._path,
                reason,
                exc,
            )
            return

        logger.warning(
            "memory store was unreadable (%s); moved it to %s and continued "
            "with an empty store",
            reason,
            target,
        )

    def invalidate_cache(self) -> None:
        """Drop the cached record set. Mainly for tests."""

        with self._lock:
            self._cache = None

    # ---- writes -----------------------------------------------------------

    def _persist_locked(
        self,
        records: Sequence[MemoryRecord],
    ) -> tuple[MemoryRecord, ...]:
        """Write ``records`` as the complete document. Caller holds the lock."""

        ordered = tuple(records)
        document = MemoryDocument(
            version=SCHEMA_VERSION,
            updated_at=self._clock(),
            records=ordered,
        )
        self._write_document(document)
        return ordered

    def _write_document(self, document: MemoryDocument) -> None:
        """Write the whole document, then swap it in with one atomic replace.

        A reader therefore observes either the previous complete document or the
        new one, never a partial write. ``os.replace`` is atomic on POSIX and
        Windows and overwrites an existing target.
        """

        text = serialize_document(document)

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise MemoryWriteError(self._write_failure_reason(exc)) from exc

        tmp_path: Path | None = None
        try:
            handle_fd, tmp_name = tempfile.mkstemp(
                dir=self._path.parent,
                prefix=f"{self._path.name}.",
                suffix=".tmp",
            )
            tmp_path = Path(tmp_name)
            with os.fdopen(handle_fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            self._replace_with_retry(tmp_path)
            tmp_path = None
        except MemoryWriteError:
            raise
        except OSError as exc:
            raise MemoryWriteError(self._write_failure_reason(exc)) from exc
        finally:
            # A failed write must leave no debris and must not disturb the
            # document already on disk.
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(
                        "could not remove the temporary memory file %s: %s",
                        tmp_path,
                        exc,
                    )

        self._cache = None

    def _replace_with_retry(self, tmp_path: Path) -> None:
        last: OSError | None = None
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp_path, self._path)
                return
            except PermissionError as exc:
                last = exc
                if attempt < len(_REPLACE_BACKOFF):
                    self._sleeper(_REPLACE_BACKOFF[attempt])

        assert last is not None  # noqa: S101 - loop always sets it before exit
        raise MemoryWriteError(self._write_failure_reason(last)) from last

    def _write_failure_reason(self, exc: OSError) -> str:
        """Describe a write failure as one of the three documented causes."""

        if self._path.is_dir():
            detail = "the configured path is a directory"
        elif isinstance(exc, PermissionError):
            detail = "the configured path is not writable"
        else:
            detail = f"the filesystem reported an error: {exc}"
        return f"could not write the memory store at {self._path}: {detail}"

    # ---- save -------------------------------------------------------------

    def save(
        self,
        *,
        content: object,
        category: object = None,
        tags: object = None,
        scope: object = None,
        thread_id: str | None = None,
    ) -> SaveOutcome:
        """Create or update one record.

        Parameters are validated in a fixed order so a call breaking several
        rules always reports the same one: content presence, content length,
        category, scope, tag count, tag length. Credential screening runs after
        validation and before duplicate detection.
        """

        trimmed = _as_text(content).strip()
        if not trimmed:
            return self._save_error("content must not be empty")

        if len(trimmed) > self._max_record_chars:
            return self._save_error(
                f"content is {len(trimmed)} characters, above the "
                f"memory_max_record_chars limit of {self._max_record_chars}"
            )

        resolved_category = _resolve_category(category)
        if resolved_category is None:
            return self._save_error(
                "category must be one of: " + ", ".join(CATEGORIES)
            )

        resolved_scope = _resolve_scope(scope, default=self._default_scope)
        if resolved_scope is None:
            return self._save_error("scope must be one of: " + ", ".join(SCOPES))

        resolved_thread = (thread_id or "").strip() or None
        if resolved_scope == "session" and resolved_thread is None:
            return self._save_error(
                "session-scoped memory requires an active chat session; "
                "use scope 'global' for this request"
            )
        scope_id = resolved_thread if resolved_scope == "session" else None

        raw_tags = _as_tag_list(tags)
        if raw_tags is None:
            return self._save_error("tags must be a list of short strings")
        if len(raw_tags) > MAX_TAGS:
            return self._save_error(
                f"tags has {len(raw_tags)} entries, above the limit of {MAX_TAGS}"
            )

        normalized_tags, oversized = _normalize_tags(raw_tags)
        if oversized:
            return self._save_error(
                f"tags must each be at most {MAX_TAG_CHARS} characters"
            )

        if find_secret_match(trimmed) is not None:
            return self._save_error(
                "credential-like content is refused (parameter: content)"
            )
        if find_secret_in_any(normalized_tags) is not None:
            return self._save_error(
                "credential-like content is refused (parameter: tags)"
            )

        with self._lock:
            records = list(self._read_locked(force=True))
            normalized = normalize_content(trimmed)

            for index, existing in enumerate(records):
                if (
                    existing.scope == resolved_scope
                    and existing.scope_id == scope_id
                    and existing.category == resolved_category
                    and normalize_content(existing.content) == normalized
                ):
                    records[index] = replace(
                        existing,
                        tags=normalized_tags,
                        updated_at=self._clock(),
                    )
                    self._persist_locked(records)
                    return SaveOutcome(
                        ok=True,
                        message=(
                            f"Updated the existing memory {existing.id} "
                            f"({resolved_category})."
                        ),
                        record_count=len(records),
                        record_id=existing.id,
                        updated=True,
                    )

            pruned = self._prune_for_insert(records)

            timestamp = self._clock()
            new_record = MemoryRecord(
                id=uuid.uuid4().hex,
                scope=resolved_scope,  # type: ignore[arg-type]
                scope_id=scope_id,
                category=resolved_category,  # type: ignore[arg-type]
                content=trimmed,
                tags=normalized_tags,
                created_at=timestamp,
                updated_at=timestamp,
                last_recalled_at=None,
            )
            records.append(new_record)
            self._persist_locked(records)

            return SaveOutcome(
                ok=True,
                message=f"Saved memory {new_record.id} ({resolved_category}).",
                record_count=len(records),
                record_id=new_record.id,
                created=True,
                pruned=pruned,
            )

    def _prune_for_insert(self, records: list[MemoryRecord]) -> tuple[str, ...]:
        """Evict across the whole store until one slot is free.

        Eviction is store-wide, not scoped to the caller's thread: scoping it
        would make the global cap unreachable once one session filled it.
        """

        pruned: list[str] = []
        while len(records) >= self._max_records:
            victim = min(
                range(len(records)),
                key=lambda index: (
                    timestamp_epoch(records[index].recency),
                    records[index].id,
                ),
            )
            pruned.append(records.pop(victim).id)

        if pruned:
            logger.info(
                "pruned %d memory record(s) to stay within the cap of %d: %s; "
                "%d record(s) remain before the insert",
                len(pruned),
                self._max_records,
                ", ".join(pruned),
                len(records),
            )
        return tuple(pruned)

    def _save_error(self, message: str) -> SaveOutcome:
        return SaveOutcome(ok=False, message=message, record_count=self._safe_count())

    # ---- recall -----------------------------------------------------------

    def recall(
        self,
        *,
        query: object,
        thread_id: str | None = None,
        top_k: int,
        max_chars: int,
    ) -> RecallOutcome:
        """Return the records matching ``query``, stamping ``last_recalled_at``.

        The stamp affects pruning recency only: it appears in neither the
        rendered output nor the ranking key, so two identical calls with no
        intervening change return identical text.
        """

        text = _as_text(query).strip()
        if not text:
            return RecallOutcome(
                ok=False, message="query must not be empty", record_count=0
            )
        if len(text) > MAX_QUERY_CHARS:
            return RecallOutcome(
                ok=False,
                message=(
                    f"query is {len(text)} characters, above the limit of "
                    f"{MAX_QUERY_CHARS}"
                ),
                record_count=0,
            )

        terms = derive_query_terms(text)

        with self._lock:
            try:
                records = self._read_locked(force=True)
            except MemoryStoreError as exc:
                return RecallOutcome(ok=False, message=str(exc), record_count=0)

            ranked = rank_records(
                in_scope_records(records, thread_id), terms, top_k=top_k
            )
            if not ranked:
                return RecallOutcome(
                    ok=True,
                    message="No matching memory was found.",
                    record_count=len(records),
                )

            stamp = self._clock()
            recalled = {record.id for record in ranked}
            updated = tuple(
                replace(record, last_recalled_at=stamp)
                if record.id in recalled
                else record
                for record in records
            )
            self._persist_locked(updated)

            return RecallOutcome(
                ok=True,
                message=format_records(ranked, max_chars=max_chars),
                record_count=len(updated),
                records=tuple(ranked),
            )

    # ---- forget -----------------------------------------------------------

    def forget(
        self,
        *,
        memory_id: object = None,
        query: object = None,
        thread_id: str | None = None,
    ) -> ForgetOutcome:
        """Delete by identifier, or by keyword when no identifier is given.

        An identifier takes precedence and any query supplied alongside it is
        ignored, so a model that sends both cannot delete more than one record.
        """

        identifier = _as_text(memory_id).strip()
        text = _as_text(query).strip()

        if not identifier and not text:
            return self._forget_error(
                "either memory_id or query is required"
            )

        if identifier:
            return self._forget_by_id(
                identifier, thread_id=thread_id, ignored_query=bool(text)
            )
        return self._forget_by_query(text, thread_id=thread_id)

    def _forget_by_id(
        self,
        identifier: str,
        *,
        thread_id: str | None,
        ignored_query: bool,
    ) -> ForgetOutcome:
        normalized = identifier.lower()
        if not ID_PATTERN.match(normalized):
            return self._forget_error(
                "memory_id must be exactly 32 hexadecimal characters"
            )

        suffix = (
            " The query parameter was ignored because memory_id was supplied."
            if ignored_query
            else ""
        )

        with self._lock:
            records = list(self._read_locked(force=True))
            visible = {
                record.id for record in in_scope_records(tuple(records), thread_id)
            }
            if normalized not in visible:
                return ForgetOutcome(
                    ok=True,
                    message=f"No memory with id {normalized} was found.{suffix}",
                    record_count=len(records),
                )

            remaining = [record for record in records if record.id != normalized]
            self._persist_locked(remaining)
            return ForgetOutcome(
                ok=True,
                message=f"Deleted memory {normalized}.{suffix}",
                record_count=len(remaining),
                deleted=(normalized,),
            )

    def _forget_by_query(
        self,
        text: str,
        *,
        thread_id: str | None,
    ) -> ForgetOutcome:
        if len(text) > MAX_QUERY_CHARS:
            return self._forget_error(
                f"query is {len(text)} characters, above the limit of "
                f"{MAX_QUERY_CHARS}"
            )

        terms = derive_query_terms(text)

        with self._lock:
            records = list(self._read_locked(force=True))
            ranked = rank_records(
                in_scope_records(tuple(records), thread_id), terms
            )
            if not ranked:
                return ForgetOutcome(
                    ok=True,
                    message="No matching memory was found.",
                    record_count=len(records),
                )

            doomed = {record.id for record in ranked[:MAX_FORGET_DELETES]}
            unmatched = len(ranked) - len(doomed)
            remaining = [record for record in records if record.id not in doomed]
            self._persist_locked(remaining)

            plural = "memory" if len(doomed) == 1 else "memories"
            message = f"Deleted {len(doomed)} {plural}."
            if unmatched:
                message += (
                    f" {unmatched} further match(es) were left in place; "
                    "call forget_memory again to remove more."
                )

            return ForgetOutcome(
                ok=True,
                message=message,
                record_count=len(remaining),
                deleted=tuple(sorted(doomed)),
                unmatched=unmatched,
            )

    def _forget_error(self, message: str) -> ForgetOutcome:
        return ForgetOutcome(ok=False, message=message, record_count=self._safe_count())

    # ---- session cleanup --------------------------------------------------

    def purge_session(self, thread_id: str) -> int:
        """Delete the session-scoped records belonging to ``thread_id``."""

        resolved = (thread_id or "").strip()
        if not resolved:
            return 0

        with self._lock:
            records = list(self._read_locked(force=True))
            remaining = [
                record
                for record in records
                if not (record.scope == "session" and record.scope_id == resolved)
            ]
            removed = len(records) - len(remaining)
            if removed:
                self._persist_locked(remaining)
                logger.info(
                    "purged %d session-scoped memory record(s) for thread %s; "
                    "%d record(s) remain",
                    removed,
                    resolved,
                    len(remaining),
                )
            return removed

    def _safe_count(self) -> int:
        try:
            return len(self.read())
        except MemoryStoreError:
            return 0


def _as_text(value: object) -> str:
    """Coerce a tool-supplied value to a string without raising."""

    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _resolve_category(category: object) -> str | None:
    """Return the category, defaulting a blank one, or ``None`` if invalid."""

    text = _as_text(category).strip()
    if not text:
        return DEFAULT_CATEGORY
    lowered = text.casefold()
    return lowered if lowered in CATEGORIES else None


def _resolve_scope(scope: object, *, default: str) -> str | None:
    """Return the scope, defaulting a blank one, or ``None`` if invalid."""

    text = _as_text(scope).strip()
    if not text:
        return default
    lowered = text.casefold()
    return lowered if lowered in SCOPES else None


def _as_tag_list(tags: object) -> list[str] | None:
    """Normalize the tags parameter to a list, or ``None`` if unusable."""

    if tags is None:
        return []
    if isinstance(tags, str):
        return [tags]
    if isinstance(tags, (list, tuple)):
        return [_as_text(tag) for tag in tags]
    return None


def _normalize_tags(raw_tags: Sequence[str]) -> tuple[tuple[str, ...], bool]:
    """Trim, drop blanks, de-duplicate case-insensitively, preserving order.

    Returns the surviving tags and whether any tag exceeded the length limit.
    """

    kept: list[str] = []
    seen: set[str] = set()
    oversized = False

    for tag in raw_tags:
        trimmed = tag.strip()
        if not trimmed:
            continue
        if len(trimmed) > MAX_TAG_CHARS:
            oversized = True
            continue
        folded = trimmed.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        kept.append(trimmed)

    return tuple(kept), oversized


def _lock_for(path: Path) -> threading.RLock:
    """Return the process-wide lock for one resolved path."""

    with _LOCKS_GUARD:
        lock = _LOCKS.get(path)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[path] = lock
        return lock


def get_memory_store(settings: Settings) -> MemoryStore:
    """Return the cached store for the Settings-resolved path.

    One instance per path means the three tools, the recall injector, and
    session deletion all serialize against the same lock.
    """

    path = resolve_store_path(settings)
    with _STORES_GUARD:
        store = _STORES.get(path)
        if store is None:
            store = MemoryStore(
                path,
                max_records=settings.memory_max_records,
                max_record_chars=settings.memory_max_record_chars,
                default_scope=settings.memory_default_scope,
            )
            _STORES[path] = store
        else:
            store.update_limits(
                max_records=settings.memory_max_records,
                max_record_chars=settings.memory_max_record_chars,
                default_scope=settings.memory_default_scope,
            )
        return store


def reset_store_cache() -> None:
    """Forget every cached store instance. Test helper."""

    with _STORES_GUARD:
        _STORES.clear()


__all__ = [
    "ForgetOutcome",
    "MemoryPathError",
    "MemoryStore",
    "MemoryStoreError",
    "MemoryWriteError",
    "RecallOutcome",
    "SaveOutcome",
    "get_memory_store",
    "in_scope_records",
    "reset_store_cache",
    "resolve_store_path",
]
