from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..security import ResourceOwner


@dataclass
class ChatSession:
    """Runtime metadata for one in-memory chat session."""

    thread_id: str
    graph: Any
    settings: Settings
    owner: ResourceOwner | None = None
    source_urls: list[str] = field(default_factory=list)
    source_mode: str = "defaults"
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float | None = None
    isolated_chroma: bool = False
    # Tool-ready relative paths already injected into the conversation as an
    # upload-context note, so later turns do not re-announce the same files.
    announced_uploads: set[str] = field(default_factory=set)
    extraction_watermark: int = 0
    # Serializes graph execution and rollback for this checkpoint thread.
    turn_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.last_accessed_at is None:
            self.last_accessed_at = self.created_at

    def touch(self, now: float | None = None) -> None:
        self.last_accessed_at = time.time() if now is None else now
