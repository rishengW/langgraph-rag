"""SQLite-backed persistence wrapper for LangGraph memory checkpoints."""

from __future__ import annotations

import pickle
import sqlite3
import threading
from collections import defaultdict
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver


class SQLiteMemorySaver(MemorySaver):
    """Persist LangGraph ``MemorySaver`` checkpoint maps to SQLite.

    The installed LangGraph version in this project exposes only the in-memory
    saver. This class keeps that saver API and stores its already-serialized
    checkpoint bytes in a local SQLite row after each write.
    """

    def __init__(self, database_path: str | Path) -> None:
        self._path = Path(database_path)
        self._lock = threading.Lock()
        super().__init__()
        self._ensure_database()
        self._load_state()

    @property
    def database_path(self) -> Path:
        """Return the SQLite checkpoint database path."""

        return self._path

    def put(self, config, checkpoint, metadata, new_versions):
        """Save one checkpoint and persist the backing store."""

        result = super().put(config, checkpoint, metadata, new_versions)
        self._persist_state()
        return result

    def put_writes(self, config, writes, task_id, task_path: str = "") -> None:
        """Save checkpoint writes and persist the backing store."""

        super().put_writes(config, writes, task_id, task_path)
        self._persist_state()

    def _ensure_database(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
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
        self.storage = _storage_defaultdict(storage)
        self.writes = _writes_defaultdict(writes)
        if blobs is not None and hasattr(self, "blobs"):
            self.blobs = dict(blobs)

    def _persist_state(self) -> None:
        # REFACTOR: Persist only backing maps exposed by the installed MemorySaver.
        state: dict[str, Any] = {
            "storage": _plain_storage(self.storage),
            "writes": _plain_writes(self.writes),
        }
        if hasattr(self, "blobs"):
            state["blobs"] = dict(self.blobs)
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

    def delete_thread(self, thread_id: str) -> None:
        """Delete one thread's checkpoints and persist the backing store."""

        if hasattr(super(), "delete_thread"):
            super().delete_thread(thread_id)
        else:
            self.storage.pop(thread_id, None)
            for key in list(self.writes):
                if key[0] == thread_id:
                    self.writes.pop(key, None)
        self._persist_state()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()


def _storage_defaultdict(value: dict[str, Any]):
    storage = defaultdict(lambda: defaultdict(dict))
    for thread_id, namespaces in value.items():
        storage[thread_id].update(
            {namespace: dict(checkpoints) for namespace, checkpoints in namespaces.items()}
        )
    return storage


def _writes_defaultdict(value: dict[tuple[str, str, str], Any]):
    writes = defaultdict(dict)
    for key, entries in value.items():
        writes[key] = dict(entries)
    return writes


def _plain_storage(value) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        thread_id: {
            namespace: dict(checkpoints)
            for namespace, checkpoints in namespaces.items()
        }
        for thread_id, namespaces in value.items()
    }


def _plain_writes(value) -> dict[tuple[str, str, str], dict[Any, Any]]:
    return {key: dict(entries) for key, entries in value.items()}


__all__ = ["SQLiteMemorySaver"]
