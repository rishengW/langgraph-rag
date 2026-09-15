"""SQLite-backed persistence wrapper for LangGraph memory checkpoints."""

from __future__ import annotations

import pickle
import sqlite3
import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from langgraph.checkpoint.memory import MemorySaver

from src.errors import ResourceNotFoundError

from ..security import ResourceOwner


class SQLiteMemorySaver(MemorySaver):
    """Persist LangGraph ``MemorySaver`` checkpoint maps to SQLite.

    The installed LangGraph version in this project exposes only the in-memory
    saver. This class keeps that saver API and stores each thread's already-
    serialized checkpoint bytes in its own SQLite row after every write, so a
    checkpoint save costs O(one thread) instead of O(all threads). Legacy
    databases that kept the whole store in a single ``checkpoint_state`` row
    are migrated to per-thread rows on first open.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        enforce_ownership: bool = False,
    ) -> None:
        self._path = Path(database_path)
        self._enforce_ownership = enforce_ownership
        # The reentrant lock keeps the inherited ``super().put`` /
        # ``super().put_writes`` mutations and our per-thread SQLite snapshot
        # in the same critical section. Without it, concurrent ``/chat``
        # requests on the same session thread can raise
        # ``RuntimeError: dictionary changed size during iteration``.
        self._lock = threading.RLock()
        super().__init__()
        self._ensure_database()
        self._load_state()

    @property
    def database_path(self) -> Path:
        """Return the SQLite checkpoint database path."""

        return self._path

    @property
    def ownership_enforced(self) -> bool:
        """Return whether protected checkpoint operations require an owner."""

        return self._enforce_ownership

    def register_owner(self, thread_id: str, owner: ResourceOwner) -> None:
        """Atomically record one checkpoint thread's immutable owner."""

        if not isinstance(owner, ResourceOwner):
            raise TypeError("Checkpoint owner must be a ResourceOwner")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT principal_id, tenant_id FROM checkpoint_owners WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if row is not None:
                existing = ResourceOwner(principal_id=str(row[0]), tenant_id=row[1])
                if existing != owner:
                    raise ResourceNotFoundError("Resource not found.")
                return
            connection.execute(
                """
                INSERT INTO checkpoint_owners (thread_id, principal_id, tenant_id)
                VALUES (?, ?, ?)
                """,
                (thread_id, owner.principal_id, owner.tenant_id),
            )

    def owner_for_thread(self, thread_id: str) -> ResourceOwner | None:
        """Return internal owner metadata without authorizing a caller."""

        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT principal_id, tenant_id FROM checkpoint_owners WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        return ResourceOwner(principal_id=str(row[0]), tenant_id=row[1])

    def put(
        self,
        config: Any,
        checkpoint: Any,
        metadata: Any,
        new_versions: Any,
    ) -> Any:
        """Save one checkpoint and persist the backing store.

        Thread-safe: the inherited ``MemorySaver.put`` mutation and the
        SQLite persistence happen under ``self._lock`` so concurrent
        invocations on the same session thread do not race on the
        ``self.writes`` / ``self.storage`` dicts.
        """

        self._require_registered_config(config)
        thread_id = _thread_id_from_config(config)
        with self._lock:
            result = super().put(config, checkpoint, metadata, new_versions)
            self._persist_thread(thread_id)
        return result

    def put_writes(
        self,
        config: Any,
        writes: Any,
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save checkpoint writes and persist the backing store.

        Thread-safe: the inherited ``MemorySaver.put_writes`` mutation and
        the SQLite persistence happen under ``self._lock`` for the same
        reason as :meth:`put`.
        """

        self._require_registered_config(config)
        thread_id = _thread_id_from_config(config)
        with self._lock:
            super().put_writes(config, writes, task_id, task_path)
            self._persist_thread(thread_id)

    def _ensure_database(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_owners (
                    thread_id TEXT PRIMARY KEY,
                    principal_id TEXT NOT NULL,
                    tenant_id TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS thread_state (
                    thread_id TEXT PRIMARY KEY,
                    payload BLOB NOT NULL
                )
                """
            )
            self._migrate_legacy_state(connection)

    @staticmethod
    def _migrate_legacy_state(connection: sqlite3.Connection) -> None:
        """Split a legacy whole-store ``checkpoint_state`` row into per-thread rows."""

        legacy_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'checkpoint_state'"
        ).fetchone()
        if legacy_exists is None:
            return
        row = connection.execute("SELECT payload FROM checkpoint_state WHERE id = 1").fetchone()
        if row is not None:
            for thread_id, payload in _split_legacy_state(row[0]).items():
                connection.execute(
                    """
                    INSERT INTO thread_state (thread_id, payload)
                    VALUES (?, ?)
                    ON CONFLICT(thread_id) DO UPDATE SET payload=excluded.payload
                    """,
                    (thread_id, payload),
                )
        connection.execute("DROP TABLE checkpoint_state")

    def _load_state(self) -> None:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT thread_id, payload FROM thread_state").fetchall()
        if not rows:
            return

        storage: Any = defaultdict(lambda: defaultdict(dict))
        writes: Any = defaultdict(dict)
        blobs: dict[Any, Any] = {}
        if hasattr(self, "blobs"):
            blobs = dict(cast(Any, self).blobs)
        for thread_id, payload in rows:
            state = pickle.loads(payload)
            thread_storage = state.get("storage")
            if thread_storage:
                storage[str(thread_id)].update(
                    {
                        namespace: dict(checkpoints)
                        for namespace, checkpoints in thread_storage.items()
                    }
                )
            for key, entries in state.get("writes", {}).items():
                writes[tuple(key)] = dict(entries)
            blobs.update(state.get("blobs", {}))

        cast_self = cast(Any, self)
        cast_self.storage = storage
        cast_self.writes = writes
        if hasattr(self, "blobs"):
            cast_self.blobs = blobs

    def _thread_state_dict(self, thread_id: str) -> dict[str, Any]:
        """Snapshot one thread's backing maps. Callers must hold ``self._lock``."""

        namespaces = self.storage.get(thread_id)
        storage = (
            None
            if namespaces is None
            else {namespace: dict(checkpoints) for namespace, checkpoints in namespaces.items()}
        )
        writes = {
            key: dict(entries)
            for key, entries in self.writes.items()
            if key and key[0] == thread_id
        }
        blobs: dict[Any, Any] = {}
        if hasattr(self, "blobs"):
            blobs = {
                key: value
                for key, value in cast(Any, self).blobs.items()
                if key and key[0] == thread_id
            }
        return {"storage": storage, "writes": writes, "blobs": blobs}

    def _persist_thread(self, thread_id: str | None) -> None:
        """Persist one thread's checkpoint state. Callers must hold ``self._lock``."""

        if not thread_id:
            return
        state = self._thread_state_dict(thread_id)
        has_content = state["storage"] is not None or bool(state["writes"]) or bool(state["blobs"])
        with self._connect() as connection:
            if not has_content:
                connection.execute(
                    "DELETE FROM thread_state WHERE thread_id = ?",
                    (thread_id,),
                )
                return
            connection.execute(
                """
                INSERT INTO thread_state (thread_id, payload)
                VALUES (?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET payload=excluded.payload
                """,
                (thread_id, pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)),
            )

    def snapshot_thread(
        self,
        thread_id: str,
        *,
        owner: ResourceOwner | None = None,
    ) -> bytes:
        """Capture an authorized, isolated snapshot of one checkpoint thread."""

        self._require_owner(thread_id, owner)
        with self._lock:
            return pickle.dumps(
                self._thread_state_dict(thread_id),
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def restore_thread(
        self,
        thread_id: str,
        snapshot: bytes,
        *,
        owner: ResourceOwner | None = None,
    ) -> None:
        """Restore one authorized thread without disturbing other checkpoints."""

        self._require_owner(thread_id, owner)
        state = pickle.loads(snapshot)
        with self._lock:
            self.storage.pop(thread_id, None)
            storage = state.get("storage")
            if storage is not None:
                self.storage[thread_id].update(
                    {namespace: dict(checkpoints) for namespace, checkpoints in storage.items()}
                )

            for key in list(self.writes):
                if key and key[0] == thread_id:
                    self.writes.pop(key, None)
            for key, entries in state.get("writes", {}).items():
                self.writes[key] = dict(entries)

            if hasattr(self, "blobs"):
                blobs = cast(Any, self).blobs
                for key in list(blobs):
                    if key and key[0] == thread_id:
                        blobs.pop(key, None)
                blobs.update(state.get("blobs", {}))

            self._persist_thread(thread_id)

    def delete_thread(
        self,
        thread_id: str,
        *,
        owner: ResourceOwner | None = None,
    ) -> None:
        """Delete one authorized thread's checkpoints and owner metadata.

        Thread-safe: the inherited ``MemorySaver.delete_thread`` call (or
        the manual ``self.writes`` / ``self.storage`` pop fallback) and the
        SQLite persistence happen under ``self._lock`` so concurrent
        writes on the same thread cannot race the ``list(self.writes)``
        iteration.
        """

        self._require_owner(thread_id, owner)
        with self._lock:
            if hasattr(super(), "delete_thread"):
                super().delete_thread(thread_id)
            else:
                self.storage.pop(thread_id, None)
                for key in list(self.writes):
                    if key[0] == thread_id:
                        self.writes.pop(key, None)
            self._persist_thread(thread_id)
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM checkpoint_owners WHERE thread_id = ?",
                    (thread_id,),
                )

    def _require_registered_config(self, config: Any) -> None:
        if not self._enforce_ownership:
            return
        thread_id = _thread_id_from_config(config)
        if not isinstance(thread_id, str) or self.owner_for_thread(thread_id) is None:
            raise ResourceNotFoundError("Resource not found.")

    def _require_owner(self, thread_id: str, owner: ResourceOwner | None) -> None:
        if not self._enforce_ownership:
            return
        expected = self.owner_for_thread(thread_id)
        if expected is None or owner is None or expected != owner:
            raise ResourceNotFoundError("Resource not found.")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()


def _thread_id_from_config(config: Any) -> str | None:
    configurable = config.get("configurable") if isinstance(config, dict) else None
    thread_id = configurable.get("thread_id") if isinstance(configurable, dict) else None
    return thread_id if isinstance(thread_id, str) else None


def _split_legacy_state(payload: bytes) -> dict[str, bytes]:
    """Split a legacy whole-store checkpoint blob into per-thread payloads."""

    state = pickle.loads(payload)
    storage: dict[str, Any] = state.get("storage", {})
    writes: dict[tuple[str, str, str], Any] = state.get("writes", {})
    blobs: dict[Any, Any] = state.get("blobs") or {}

    thread_ids = set(storage)
    thread_ids.update(key[0] for key in writes if key)
    if blobs:
        thread_ids.update(key[0] for key in blobs if key)

    result: dict[str, bytes] = {}
    for thread_id in thread_ids:
        namespaces = storage.get(thread_id)
        thread_storage = (
            None
            if namespaces is None
            else {namespace: dict(checkpoints) for namespace, checkpoints in namespaces.items()}
        )
        thread_writes = {
            key: dict(entries) for key, entries in writes.items() if key and key[0] == thread_id
        }
        thread_blobs = {key: value for key, value in blobs.items() if key and key[0] == thread_id}
        result[thread_id] = pickle.dumps(
            {"storage": thread_storage, "writes": thread_writes, "blobs": thread_blobs},
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return result


__all__ = ["SQLiteMemorySaver"]
