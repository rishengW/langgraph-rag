"""Bounded, content-addressed cache for parsed local files."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

T = TypeVar("T")
_CacheKey = tuple[str, str]


@dataclass(frozen=True)
class _FileStamp:
    size: int
    modified_ns: int


class ParsedFileCache:
    """Cache successful parser results and coalesce concurrent cache misses."""

    def __init__(self, max_entries: int = 128) -> None:
        self._max_entries = max(1, int(max_entries))
        self._lock = threading.Lock()
        self._values: OrderedDict[_CacheKey, object] = OrderedDict()
        self._digests: OrderedDict[Path, tuple[_FileStamp, str]] = OrderedDict()
        self._stripes = tuple(threading.Lock() for _ in range(32))

    def get_or_compute(
        self,
        path: Path,
        *,
        parser_key: str,
        loader: Callable[[], T],
    ) -> T:
        """Return a cached result or invoke ``loader`` once for this key."""

        digest = self._content_digest(path)
        key = (digest, parser_key)
        stripe = self._stripes[hash(key) % len(self._stripes)]
        with stripe:
            with self._lock:
                cached = self._values.pop(key, None)
                if cached is not None:
                    self._values[key] = cached
                    return cast(T, cached)

            value = loader()
            with self._lock:
                self._values[key] = value
                self._values.move_to_end(key)
                while len(self._values) > self._max_entries:
                    self._values.popitem(last=False)
            return value
    def _content_digest(self, path: Path) -> str:
        """Hash a file once per stable size/mtime pair."""

        for _ in range(2):
            before_stat = path.stat()
            stamp = _FileStamp(before_stat.st_size, before_stat.st_mtime_ns)
            with self._lock:
                known = self._digests.get(path)
                if known is not None and known[0] == stamp:
                    self._digests.move_to_end(path)
                    return known[1]

            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
            after_stat = path.stat()
            if (after_stat.st_size, after_stat.st_mtime_ns) == (
                stamp.size,
                stamp.modified_ns,
            ):
                with self._lock:
                    self._digests[path] = (stamp, digest)
                    self._digests.move_to_end(path)
                    while len(self._digests) > self._max_entries * 2:
                        self._digests.popitem(last=False)
                return digest

        raise OSError(f"file changed while it was being read: {path.name}")

    def clear(self) -> None:
        """Clear cached values, primarily for process lifecycle and tests."""

        with self._lock:
            self._values.clear()
            self._digests.clear()


PARSED_FILE_CACHE = ParsedFileCache()

__all__ = ["PARSED_FILE_CACHE", "ParsedFileCache"]
