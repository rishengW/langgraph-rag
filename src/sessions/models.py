from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings


@dataclass
class ChatSession:
    """Runtime metadata for one in-memory chat session."""

    thread_id: str
    graph: Any
    settings: Settings
    source_urls: list[str] = field(default_factory=list)
    source_mode: str = "defaults"
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float | None = None
    isolated_chroma: bool = False

    def __post_init__(self) -> None:
        if self.last_accessed_at is None:
            self.last_accessed_at = self.created_at

    def touch(self, now: float | None = None) -> None:
        self.last_accessed_at = time.time() if now is None else now


__all__ = ["ChatSession"]
