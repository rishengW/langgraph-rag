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
    saver. This class keeps that saver API and stores its already-serialized
    checkpoint bytes in a local SQLite row after each write.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        enforce_ownership: bool = False,
    ) -> None:
        self._path = Path(database_path)
        self._enforce_ownership = enforce_ownership
        # REFACTOR: Use a reentrant lock so the inherited ``super().put`` /
        # ``super().put_writes`` mutations and our ``_persist_state`` snapshot
        # happen under the same critical section. The previous scope left
        # ``self.writes`` / ``self.storage`` mutation outside the lock, which
        # raised ``RuntimeError: dictionary changed size during iteration``
        # when concurrent ``/chat`` requests hit the same session thread.
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
        with self._lock:
            result = super().put(config, checkpoint, metadata, new_versions)
            self._persist_state()
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
        with self._lock:
            super().put_writes(config, writes, task_id, task_path)
            self._persist_state()

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
                CREATE TABLE IF NOT EXISTS checkpoint_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    payload BLOB NOT NULL
                )
                """
            )

    def _load_state(self) -> None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM checkpoint_state WHERE id = 1",
            ).fetchone()
        if row is None:
            return

        state = pickle.loads(row[0])
        storage: dict[str, Any] = state.get("storage", {})
        writes: dict[tuple[str, str, str], Any] = state.get("writes", {})
        blobs: dict[str, Any] | None = state.get("blobs")
        cast_self = cast(Any, self)
        cast_self.storage = _storage_defaultdict(storage)
        cast_self.writes = _writes_defaultdict(writes)
        if blobs is not None and hasattr(self, "blobs"):
            cast_self.blobs = dict(blobs)

    def _persist_state(self) -> None:
        # REFACTOR: Persist only backing maps exposed by the installed MemorySaver.
        # The snapshot of ``self.storage`` / ``self.writes`` / ``self.blobs``
        # is captured inside the critical section so the dict comprehensions
        # in ``_plain_storage`` / ``_plain_writes`` cannot iterate a dict
        # that another thread is mutating. Callers (``put``, ``put_writes``,
        # ``delete_thread``) already hold ``self._lock`` when they call
        # here, so the inner ``self._lock`` is defense-in-depth.
        state: dict[str, Any] = {
            "storage": _plain_storage(self.storage),
            "writes": _plain_writes(self.writes),
        }
        if hasattr(self, "blobs"):
            state["blobs"] = dict(cast(Any, self).blobs)
        payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO checkpoint_state (id, payload)
                VALUES (1, ?)
                ON CONFLICT(id) DO UPDATE SET payload=excluded.payload
                """,
                (payload,),
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
            return pickle.dumps(
                {"storage": storage, "writes": writes, "blobs": blobs},
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

            self._persist_state()

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
            self._persist_state()
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM checkpoint_owners WHERE thread_id = ?",
                    (thread_id,),
                )

    def _require_registered_config(self, config: Any) -> None:
        if not self._enforce_ownership:
            return
        configurable = config.get("configurable") if isinstance(config, dict) else None
        thread_id = configurable.get("thread_id") if isinstance(configurable, dict) else None
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


def _storage_defaultdict(value: dict[str, Any]) -> Any:
    storage: Any = defaultdict(lambda: defaultdict(dict))
    for thread_id, namespaces in value.items():
        storage[thread_id].update(
            {namespace: dict(checkpoints) for namespace, checkpoints in namespaces.items()}
        )
    return storage


def _writes_defaultdict(value: dict[tuple[str, str, str], Any]) -> Any:
    writes: Any = defaultdict(dict)
    for key, entries in value.items():
        writes[key] = dict(entries)
    return writes


def _plain_storage(value: Any) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        thread_id: {namespace: dict(checkpoints) for namespace, checkpoints in namespaces.items()}
        for thread_id, namespaces in value.items()
    }


def _plain_writes(value: Any) -> dict[tuple[str, str, str], dict[Any, Any]]:
    return {key: dict(entries) for key, entries in value.items()}


__all__ = ["SQLiteMemorySaver"]
