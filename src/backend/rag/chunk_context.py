# Purpose: optional LLM-generated contextual prefixes for index-time chunks.
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.utils.retry import call_with_retry

logger = logging.getLogger(__name__)

CHUNK_CONTEXT_CACHE_FILENAME = "chunk_context_cache.json"
# Bump when the prompt, prefix format, or cache-key scheme changes; stale cache
# entries are keyed off this version and simply stop hitting.
PROMPT_VERSION = "v2"

# Document excerpt first, chunk last: providers with implicit prompt caching
# reuse the repeated document prefix across chunks of the same source.
CONTEXT_PROMPT = """<document>
{document_excerpt}
</document>
以下是该文档中的一个片段：
<chunk>
{chunk_text}
</chunk>
请用 1-2 句简洁的话说明这个片段在整篇文档中的位置和讨论内容，用于改进检索效果。只输出这段上下文，使用文档的语言。"""


@dataclass(frozen=True)
class ChunkContextConfig:
    """Configuration for LLM-generated contextual chunk prefixes."""

    enabled: bool = False
    document_excerpt_chars: int = 6000
    max_prefix_chars: int = 200
    max_concurrency: int = 4
    max_retries: int = 3


class ChunkContextCache:
    """JSON disk cache mapping chunk content hashes to generated prefixes.

    The cache survives Chroma rebuilds (see `_clear_chroma_store`) so an
    embedding-config-triggered rebuild does not re-pay the LLM cost for
    unchanged chunks. Entries never expire: a changed chunk hashes differently
    and regenerates on its own.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._entries: dict[str, str] | None = None
        # Debounce disk writes: a single indexing run generates hundreds of
        # prefixes, and rewriting the whole JSON per set() is O(N^2) I/O.
        self._dirty = False

    def _load_locked(self) -> dict[str, str]:
        if self._entries is not None:
            return self._entries
        entries: dict[str, str] = {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                entries = {str(key): str(value) for key, value in raw.items()}
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read chunk context cache %s: %s", self._path, exc)
        self._entries = entries
        return entries

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._load_locked().get(key)

    def set(self, key: str, prefix: str) -> None:
        # In-memory only; the caller flushes once per batch to avoid rewriting
        # the whole JSON file per entry (O(N^2) write amplification).
        with self._lock:
            entries = self._load_locked()
            entries[key] = prefix
            self._dirty = True

    def flush(self) -> None:
        """Persist pending in-memory entries to disk (idempotent)."""

        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._dirty:
            return
        assert self._entries is not None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Unique tmp name: concurrent processes sharing the cache directory
            # would otherwise overwrite each other's temp file mid-write.
            tmp_path = self._path.with_name(f"{self._path.name}.{os.getpid()}.tmp")
            tmp_path.write_text(json.dumps(self._entries, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(self._path)
            self._dirty = False
        except OSError as exc:
            logger.warning("Could not write chunk context cache %s: %s", self._path, exc)


def _cache_key(model_label: str, source: str, chunk_text: str) -> str:
    # The source is part of the key: identical chunk text in two documents must
    # not share a prefix, since the prefix situates the chunk in its own document.
    digest = hashlib.sha256()
    digest.update(PROMPT_VERSION.encode())
    digest.update(b"|")
    digest.update(model_label.encode())
    digest.update(b"|")
    digest.update(source.encode("utf-8"))
    digest.update(b"|")
    digest.update(chunk_text.encode("utf-8"))
    return digest.hexdigest()


def _model_label(chat_model: Any) -> str:
    label = getattr(chat_model, "model", None) or getattr(chat_model, "model_name", None)
    return str(label) if label else type(chat_model).__name__


def _grouped_by_source(chunks: Sequence[Document]) -> dict[str, list[Document]]:
    groups: dict[str, list[Document]] = {}
    for chunk in chunks:
        source = str(chunk.metadata.get("source", ""))
        groups.setdefault(source, []).append(chunk)
    return groups


def _head_excerpt(source_chunks: Sequence[Document], max_chars: int) -> str:
    """Join the source's first chunks into a head excerpt for the prompt."""

    parts: list[str] = []
    remaining = max(1, int(max_chars))
    for chunk in source_chunks:
        text = (chunk.page_content or "").strip()
        if not text:
            continue
        parts.append(text[:remaining])
        remaining -= len(parts[-1])
        if remaining <= 0:
            break
    return "\n".join(parts)


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    # Thinking-mode or tool-call-only responses can carry content=None; without
    # this guard str(None) would become a literal "None" prefix.
    if content is None:
        return ""
    if isinstance(content, list):
        content = "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content).strip().strip('"')


def _with_prefix(chunk: Document, prefix: str) -> Document:
    metadata = dict(chunk.metadata) if isinstance(chunk.metadata, dict) else {}
    return Document(page_content=f"{prefix}\n\n{chunk.page_content}", metadata=metadata)


# REFACTOR: Generate per-chunk prefixes concurrently; failures keep the original.
def apply_chunk_context(
    chunks: Sequence[Document],
    *,
    config: ChunkContextConfig,
    chat_model: Any,
    cache: ChunkContextCache | None = None,
) -> list[Document]:
    """Prepend an LLM-generated situating prefix to each chunk's page_content.

    The prefix flows into both the embedding and the final prompt because the
    retrieval path only reads page_content. Disabled or empty input is a no-op.
    """

    if not config.enabled or not chunks:
        return list(chunks)

    model_label = _model_label(chat_model)
    excerpts = {
        source: _head_excerpt(group, config.document_excerpt_chars)
        for source, group in _grouped_by_source(chunks).items()
    }
    counts = {"cached": 0, "generated": 0, "failed": 0, "skipped": 0}
    counts_lock = threading.Lock()

    def prefix_for(index: int, chunk: Document) -> tuple[int, Document]:
        if not (chunk.page_content or "").strip():
            with counts_lock:
                counts["skipped"] += 1
            return index, chunk

        if config.max_prefix_chars <= 0:
            # Nothing would survive truncation; skip the LLM call entirely
            # instead of paying for a prefix that is discarded.
            with counts_lock:
                counts["skipped"] += 1
            return index, chunk

        source = str(chunk.metadata.get("source", ""))
        key = _cache_key(model_label, source, chunk.page_content)
        if cache is not None:
            cached = cache.get(key)
            if cached is not None:
                with counts_lock:
                    counts["cached"] += 1
                # Truncation also applies on the cache-hit path: MAX_PREFIX_CHARS
                # may have been lowered after the entry was stored.
                cached = cached[: max(0, config.max_prefix_chars)].strip()
                if not cached:
                    with counts_lock:
                        counts["skipped"] += 1
                    return index, chunk
                return index, _with_prefix(chunk, cached)

        prompt_text = CONTEXT_PROMPT.format(
            document_excerpt=excerpts.get(source, ""),
            chunk_text=chunk.page_content,
        )
        try:
            prefix = _response_text(
                call_with_retry(
                    lambda: chat_model.invoke(prompt_text),
                    max_retries=config.max_retries,
                    log_label="chunk context prefix",
                )
            )
        except Exception as exc:
            logger.warning(
                "Chunk context prefix failed for chunk %d; keeping original: %s",
                index,
                exc,
            )
            with counts_lock:
                counts["failed"] += 1
            return index, chunk

        prefix = prefix[: max(0, config.max_prefix_chars)].strip()
        if not prefix:
            with counts_lock:
                counts["skipped"] += 1
            return index, chunk
        if cache is not None:
            cache.set(key, prefix)
        with counts_lock:
            counts["generated"] += 1
        return index, _with_prefix(chunk, prefix)

    worker_count = min(len(chunks), max(1, int(config.max_concurrency)))
    results: dict[int, Document] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(prefix_for, index, chunk) for index, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            index, document = future.result()
            results[index] = document

    if cache is not None:
        # One disk write per batch, not per generated prefix.
        cache.flush()

    logger.info(
        "CHUNK CONTEXT PREFIXES: %d generated, %d cached, %d failed, %d skipped",
        counts["generated"],
        counts["cached"],
        counts["failed"],
        counts["skipped"],
    )
    return [results[index] for index in range(len(chunks))]
