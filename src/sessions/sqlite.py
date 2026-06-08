# REFACTOR: Adds stdlib SQLite persistence for session metadata.
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .storage import SessionMetadata, SessionStorageError


class SQLiteStorage:
    """SQLite-backed storage for chat session metadata."""

    def __init__(self, database_path: str | Path) -> None:
        self._path = Path(database_path)
        self._lock = threading.Lock()
        self._ensure_database()

    @property
    def database_path(self) -> Path:
        """Return the SQLite database path."""

        return self._path

    def save(self, metadata: SessionMetadata) -> None:
        """Persist or replace one session metadata record."""

        params = _metadata_to_row(metadata)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    thread_id, source_urls, source_mode, config, created_at,
                    last_accessed_at, chroma_dir, isolated_chroma
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    source_urls=excluded.source_urls,
                    source_mode=excluded.source_mode,
                    config=excluded.config,
                    created_at=excluded.created_at,
                    last_accessed_at=excluded.last_accessed_at,
                    chroma_dir=excluded.chroma_dir,
                    isolated_chroma=excluded.isolated_chroma
                """,
                params,
            )

    def load(self, thread_id: str) -> SessionMetadata | None:
        """Return metadata for one session, if present."""

        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return _row_to_metadata(row) if row is not None else None

    def delete(self, thread_id: str) -> bool:
        """Delete one session metadata record."""

        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM sessions WHERE thread_id = ?",
                (thread_id,),
            )
            return cursor.rowcount > 0

    def list_ids(self) -> list[str]:
        """Return persisted session IDs ordered by creation time."""

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT thread_id FROM sessions ORDER BY created_at, thread_id",
            ).fetchall()
        return [str(row["thread_id"]) for row in rows]

    def list_metadata(self) -> list[SessionMetadata]:
        """Return all persisted session metadata records."""

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM sessions ORDER BY created_at, thread_id",
            ).fetchall()
        return [_row_to_metadata(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_database(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    thread_id TEXT PRIMARY KEY,
                    source_urls TEXT NOT NULL,
                    source_mode TEXT NOT NULL,
                    config TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL,
                    chroma_dir TEXT,
                    isolated_chroma INTEGER NOT NULL
                )
                """
            )


def _metadata_to_row(metadata: SessionMetadata) -> tuple[Any, ...]:
    return (
        metadata.thread_id,
        json.dumps(list(metadata.source_urls), ensure_ascii=True),
        metadata.source_mode,
        json.dumps(dict(metadata.config), ensure_ascii=True, sort_keys=True),
        metadata.created_at,
        metadata.last_accessed_at,
        metadata.chroma_dir,
        1 if metadata.isolated_chroma else 0,
    )


def _row_to_metadata(row: sqlite3.Row) -> SessionMetadata:
    return SessionMetadata(
        thread_id=str(row["thread_id"]),
        source_urls=_decode_urls(str(row["source_urls"]), str(row["thread_id"])),
        source_mode=str(row["source_mode"]),
        created_at=float(row["created_at"]),
        last_accessed_at=float(row["last_accessed_at"]),
        chroma_dir=_optional_str(row["chroma_dir"]),
        isolated_chroma=bool(row["isolated_chroma"]),
        config=_decode_config(str(row["config"]), str(row["thread_id"])),
    )


def _decode_urls(value: str, thread_id: str) -> list[str]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SessionStorageError(f"Invalid source_urls JSON for {thread_id}") from exc
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise SessionStorageError(f"Invalid source_urls value for {thread_id}")
    return decoded


def _decode_config(value: str, thread_id: str) -> dict[str, object]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SessionStorageError(f"Invalid config JSON for {thread_id}") from exc
    if not isinstance(decoded, dict):
        raise SessionStorageError(f"Invalid config value for {thread_id}")
    return decoded


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


__all__ = ["SQLiteStorage"]
