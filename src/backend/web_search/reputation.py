# REFACTOR: Adaptive per-domain reputation learned from real fetch outcomes.
# This replaces hand-written syndication and noise-host lists with a rolling
# prior: domains whose pages keep failing readability or relevance drift down,
# domains that keep grounding answers drift up. It is a ranking prior only and
# never rejects a URL on its own, so a cold or unseen domain is not penalized.
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .common import registrable_domain

logger = logging.getLogger(__name__)

DEFAULT_MIN_SAMPLES = 5
# Bounded so reputation nudges ties without overriding topical relevance.
MAX_REPUTATION_BONUS = 10
MAX_REPUTATION_PENALTY = 10
# Outcome recorded for one fetched page.
OUTCOME_GROUNDED = "grounded"
OUTCOME_REJECTED = "rejected"
OUTCOME_UNREACHABLE = "unreachable"


@dataclass(frozen=True)
class DomainStats:
    """Rolling fetch outcomes for one registrable domain."""

    domain: str
    attempts: int = 0
    grounded: int = 0
    rejected: int = 0
    unreachable: int = 0
    updated_at: float = 0.0

    @property
    def success_ratio(self) -> float:
        if self.attempts <= 0:
            return 0.0
        return self.grounded / self.attempts


class DomainReputationStore:
    """SQLite-backed rolling counts of per-domain fetch outcomes."""

    SCHEMA_VERSION = 1

    def __init__(self, database_path: str | Path) -> None:
        self._path = Path(database_path)
        self._lock = threading.Lock()
        self._ensure_database()

    @property
    def database_path(self) -> Path:
        return self._path

    def record(self, url: str, outcome: str) -> None:
        """Record one fetch outcome for the URL's registrable domain."""

        domain = registrable_domain(url)
        if not domain or outcome not in {
            OUTCOME_GROUNDED,
            OUTCOME_REJECTED,
            OUTCOME_UNREACHABLE,
        }:
            return
        column = {
            OUTCOME_GROUNDED: "grounded",
            OUTCOME_REJECTED: "rejected",
            OUTCOME_UNREACHABLE: "unreachable",
        }[outcome]
        with self._lock, self._connect() as connection:
            connection.execute(
                f"""
                INSERT INTO domain_reputation (
                    domain, attempts, grounded, rejected, unreachable, updated_at
                )
                VALUES (?, 1, ?, ?, ?, ?)
                ON CONFLICT(domain) DO UPDATE SET
                    attempts = domain_reputation.attempts + 1,
                    {column} = domain_reputation.{column} + 1,
                    updated_at = excluded.updated_at
                """,  # noqa: S608 - column name comes from a fixed literal map
                (
                    domain,
                    1 if column == "grounded" else 0,
                    1 if column == "rejected" else 0,
                    1 if column == "unreachable" else 0,
                    time.time(),
                ),
            )

    def record_many(self, outcomes: Sequence[tuple[str, str]]) -> None:
        """Record several (url, outcome) pairs."""

        for url, outcome in outcomes:
            self.record(url, outcome)

    def stats(self, url_or_domain: str) -> DomainStats | None:
        """Return rolling stats for a URL's domain, if any were recorded."""

        domain = registrable_domain(url_or_domain)
        if not domain:
            return None
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM domain_reputation WHERE domain = ?",
                (domain,),
            ).fetchone()
        return _row_to_stats(row) if row is not None else None

    def all_stats(self) -> list[DomainStats]:
        """Return every recorded domain, most active first."""

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM domain_reputation ORDER BY attempts DESC, domain",
            ).fetchall()
        return [_row_to_stats(row) for row in rows]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _ensure_database(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS domain_reputation (
                    domain TEXT PRIMARY KEY,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    grounded INTEGER NOT NULL DEFAULT 0,
                    rejected INTEGER NOT NULL DEFAULT 0,
                    unreachable INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL DEFAULT 0
                )
                """
            )


def reputation_delta(
    stats: DomainStats | None,
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> int:
    """Convert rolling outcomes into a bounded ranking delta.

    Returns ``0`` until the domain has enough observations, so a new domain
    competes on its own merits rather than on an empty history.
    """

    if stats is None or stats.attempts < max(1, int(min_samples)):
        return 0
    ratio = stats.success_ratio
    if ratio >= 0.5:
        return round(MAX_REPUTATION_BONUS * min(1.0, (ratio - 0.5) / 0.5))
    return -round(MAX_REPUTATION_PENALTY * min(1.0, (0.5 - ratio) / 0.5))


def build_reputation_store(settings: object) -> DomainReputationStore | None:
    """Return a store when reputation tracking is enabled, else ``None``."""

    if not getattr(settings, "web_search_domain_reputation_enabled", False):
        return None
    chroma_dir = getattr(settings, "chroma_dir", None)
    if chroma_dir is None:
        return None
    try:
        return DomainReputationStore(Path(chroma_dir) / "web-search" / "reputation.sqlite3")
    except sqlite3.Error as exc:
        logger.warning("Domain reputation store unavailable: %s", exc)
        return None


def _row_to_stats(row: sqlite3.Row) -> DomainStats:
    return DomainStats(
        domain=str(row["domain"]),
        attempts=int(row["attempts"]),
        grounded=int(row["grounded"]),
        rejected=int(row["rejected"]),
        unreachable=int(row["unreachable"]),
        updated_at=float(row["updated_at"]),
    )


__all__ = [
    "DEFAULT_MIN_SAMPLES",
    "MAX_REPUTATION_BONUS",
    "MAX_REPUTATION_PENALTY",
    "OUTCOME_GROUNDED",
    "OUTCOME_REJECTED",
    "OUTCOME_UNREACHABLE",
    "DomainReputationStore",
    "DomainStats",
    "build_reputation_store",
    "reputation_delta",
]
