"""Automatic extraction of durable memories from checkpointed conversations."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.config.settings import Settings

from ..llm.prompts import MEMORY_EXTRACTION_PROMPT
from ..llm.provider import build_chat_model
from .models import EXTRACTION_TAG_PREFIX, MAX_TAG_CHARS, MAX_TAGS, SCOPES
from .store import MemoryStore, get_memory_store
from .transcript import (
    MAX_SLICE_MESSAGES,
    count_turns,
    message_role,
    render_slice,
    select_slice,
)
from .watermark import WatermarkStore, coerce_watermark

logger = logging.getLogger("src.backend.memory.extraction")


@dataclass(frozen=True)
class ExtractionCandidate:
    """One model-proposed memory, before store validation."""

    content: str
    category: str | None = None
    tags: tuple[str, ...] = ()
    scope: str = "global"


@dataclass(frozen=True)
class ExtractionOutcome:
    """Content-free summary of one extraction attempt."""

    trigger: str
    thread_id: str
    slice_messages: int
    candidates: int
    persisted: int
    refused: int
    record_count: int
    watermark_before: int
    watermark_after: int
    duration_ms: int
    status: str
    detail: str = ""


@dataclass(frozen=True)
class _ModelCallResult:
    text: str | None = None
    error_type: str = ""
    timed_out: bool = False


@dataclass(frozen=True)
class _PersistResult:
    persisted: int = 0
    refused: int = 0
    record_count: int = 0
    error_type: str = ""


class _CheckpointReadError(RuntimeError):
    """Internal marker used to distinguish an empty thread from a failed read."""


class MemoryExtractor:
    """Extract and persist memories without allowing failures into chat turns."""

    def __init__(
        self,
        settings: Settings,
        *,
        checkpointer: Any,
        watermarks: WatermarkStore,
        store: MemoryStore | None = None,
        model_factory: Callable[[Settings], Any] = build_chat_model,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._checkpointer = checkpointer
        self._watermarks = watermarks
        self._store = store if store is not None else get_memory_store(settings)
        self._model_factory = model_factory
        self._clock = clock

    def _read_thread_messages(self, thread_id: str) -> tuple[Any, ...]:
        if self._checkpointer is None:
            raise _CheckpointReadError("missing_checkpointer")

        config = {"configurable": {"thread_id": thread_id}}
        try:
            checkpoint_tuple = self._checkpointer.get_tuple(config)
        except Exception as exc:
            raise _CheckpointReadError(type(exc).__name__) from exc
        if checkpoint_tuple is None:
            raise _CheckpointReadError("missing_checkpoint")

        checkpoint = getattr(checkpoint_tuple, "checkpoint", None)
        if checkpoint is None and isinstance(checkpoint_tuple, Mapping):
            checkpoint = checkpoint_tuple.get("checkpoint", checkpoint_tuple)
        if not isinstance(checkpoint, Mapping):
            raise _CheckpointReadError("invalid_checkpoint")

        channel_values = checkpoint.get("channel_values")
        if not isinstance(channel_values, Mapping):
            raise _CheckpointReadError("invalid_checkpoint")
        messages = channel_values.get("messages", ())
        if messages is None:
            return ()
        if isinstance(messages, (str, bytes, bytearray)) or not isinstance(messages, Sequence):
            raise _CheckpointReadError("invalid_messages")
        return tuple(messages)

    def read_thread_messages(self, thread_id: str) -> tuple[Any, ...]:
        """Return checkpointed messages, degrading every read failure to empty."""

        try:
            return self._read_thread_messages(thread_id)
        except Exception:
            return ()

    def turn_count(self, thread_id: str) -> int:
        """Return the authoritative user-turn count from the checkpoint."""

        return count_turns(self.read_thread_messages(thread_id))

    def should_extract(self, thread_id: str) -> bool:
        """Return whether the thread has reached its next exact round boundary."""

        try:
            turns = self.turn_count(thread_id)
            watermark = min(coerce_watermark(self._watermarks.get(thread_id)), turns)
            interval = self._settings.memory_extraction_turn_interval
            return turns >= watermark + interval
        except Exception:
            return False

    def _parse_candidate_payload(self, text: str) -> tuple[tuple[ExtractionCandidate, ...], bool]:
        parsed: object
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            parsed = None

        if not isinstance(parsed, list):
            parsed = _first_array(text)
        if not isinstance(parsed, list):
            return (), False

        candidates: list[ExtractionCandidate] = []
        seen: set[str] = set()
        limit = self._settings.memory_extraction_max_candidates
        for entry in parsed[:100]:
            if not isinstance(entry, Mapping):
                continue
            content = entry.get("content")
            if not isinstance(content, str):
                continue
            content = content.strip()
            if not content:
                continue

            normalized = " ".join(content.split()).casefold()
            if normalized in seen:
                continue
            seen.add(normalized)

            raw_category = entry.get("category")
            category = (
                raw_category.strip()
                if isinstance(raw_category, str) and raw_category.strip()
                else None
            )

            raw_tags = entry.get("tags")
            tags = (
                tuple(tag.strip() for tag in raw_tags if isinstance(tag, str) and tag.strip())
                if isinstance(raw_tags, list)
                else ()
            )

            raw_scope = entry.get("scope")
            scope = raw_scope if raw_scope in SCOPES else "global"
            candidates.append(
                ExtractionCandidate(
                    content=content,
                    category=category,
                    tags=tags,
                    scope=scope,
                )
            )
            if len(candidates) >= limit:
                break
        return tuple(candidates), True

    def _parse_candidates(self, text: str) -> tuple[ExtractionCandidate, ...]:
        """Parse, validate, de-duplicate, and bound model candidates."""

        return self._parse_candidate_payload(text)[0]

    def _call_model(self, prompt: str) -> _ModelCallResult:
        """Invoke the model once in a daemon thread with a bounded wait."""

        try:
            model = self._model_factory(self._settings)
        except Exception as exc:
            return _ModelCallResult(error_type=type(exc).__name__)

        completed = threading.Event()
        result: dict[str, object] = {}

        def invoke() -> None:
            try:
                result["response"] = model.invoke(prompt)
            except Exception as exc:
                result["error_type"] = type(exc).__name__
            finally:
                completed.set()

        worker = threading.Thread(
            target=invoke,
            name="memory-extraction-model-call",
            daemon=True,
        )
        try:
            worker.start()
            finished = completed.wait(timeout=self._settings.memory_extraction_timeout_seconds)
        except Exception as exc:
            return _ModelCallResult(error_type=type(exc).__name__)
        if not finished:
            return _ModelCallResult(timed_out=True)
        error_type = result.get("error_type")
        if isinstance(error_type, str) and error_type:
            return _ModelCallResult(error_type=error_type)
        return _ModelCallResult(text=_response_text(result.get("response")))

    def _persist(
        self,
        candidates: Sequence[ExtractionCandidate],
        thread_id: str,
        trigger: str,
    ) -> _PersistResult:
        """Save candidates sequentially through the existing guarded store."""

        persisted = 0
        refused = 0
        record_count = self._safe_record_count()
        error_type = ""
        provenance = _provenance_tag(trigger)

        for candidate in candidates[: self._settings.memory_extraction_max_candidates]:
            tags = _with_provenance(candidate.tags, provenance)
            try:
                outcome = self._store.save(
                    content=candidate.content,
                    category=candidate.category,
                    tags=tags,
                    scope=candidate.scope,
                    thread_id=thread_id,
                )
                record_count = int(outcome.record_count)
                if outcome.ok:
                    persisted += 1
                else:
                    refused += 1
            except Exception as exc:
                if not error_type:
                    error_type = type(exc).__name__
                continue

        return _PersistResult(
            persisted=persisted,
            refused=refused,
            record_count=record_count,
            error_type=error_type,
        )

    def _safe_record_count(self) -> int:
        try:
            return len(self._store.read())
        except Exception:
            return 0

    def run(self, trigger: str, thread_id: str) -> ExtractionOutcome:
        """Run one extraction and return a content-free outcome; never raise."""

        started = self._safe_clock()
        state = {
            "slice_messages": 0,
            "candidates": 0,
            "persisted": 0,
            "refused": 0,
            "record_count": 0,
            "watermark_before": 0,
            "watermark_after": 0,
        }

        try:
            messages = self._read_thread_messages(thread_id)
            turns = count_turns(messages)

            try:
                stored_watermark = self._watermarks.get(thread_id)
            except Exception as exc:
                return self._outcome(
                    trigger,
                    thread_id,
                    started,
                    state,
                    status="failed",
                    detail=f"watermark_read:{type(exc).__name__}",
                )
            watermark = min(coerce_watermark(stored_watermark), turns)
            state["watermark_before"] = watermark
            state["watermark_after"] = watermark
            state["record_count"] = self._safe_record_count()

            selected = select_slice(messages, watermark=watermark, trigger=trigger)[
                -MAX_SLICE_MESSAGES:
            ]
            state["slice_messages"] = len(selected)
            rendered = render_slice(
                selected,
                max_chars=self._settings.memory_extraction_max_transcript_chars,
            )
            has_user = any(message_role(message) == "user" for message in selected)
            if not selected or not has_user or not rendered.strip():
                return self._advance_and_finish(
                    trigger,
                    thread_id,
                    started,
                    state,
                    turns,
                    status="skipped",
                    detail="empty_slice",
                )

            prompt = MEMORY_EXTRACTION_PROMPT.format(
                transcript=rendered,
                max_candidates=self._settings.memory_extraction_max_candidates,
            )
            call = self._call_model(prompt)
            if call.timed_out:
                return self._outcome(
                    trigger,
                    thread_id,
                    started,
                    state,
                    status="failed",
                    detail="model_timeout",
                )
            if call.error_type:
                return self._outcome(
                    trigger,
                    thread_id,
                    started,
                    state,
                    status="failed",
                    detail=f"model_call:{call.error_type}",
                )

            candidates, usable = self._parse_candidate_payload(call.text or "")
            state["candidates"] = len(candidates)
            if not candidates:
                return self._advance_and_finish(
                    trigger,
                    thread_id,
                    started,
                    state,
                    turns,
                    status="no_information",
                    detail="empty_response" if usable else "unusable_response",
                    warning=not usable,
                )

            persisted = self._persist(candidates, thread_id, trigger)
            state["persisted"] = persisted.persisted
            state["refused"] = persisted.refused
            state["record_count"] = persisted.record_count
            if persisted.error_type:
                return self._outcome(
                    trigger,
                    thread_id,
                    started,
                    state,
                    status="failed",
                    detail=f"store_write:{persisted.error_type}",
                )

            return self._advance_and_finish(
                trigger,
                thread_id,
                started,
                state,
                turns,
                status="extracted" if persisted.persisted else "no_information",
                detail="all_refused" if not persisted.persisted else "",
            )
        except _CheckpointReadError as exc:
            return self._outcome(
                trigger,
                thread_id,
                started,
                state,
                status="failed",
                detail=f"checkpoint_read:{exc}",
            )
        except Exception as exc:
            return self._outcome(
                trigger,
                thread_id,
                started,
                state,
                status="failed",
                detail=f"unexpected:{type(exc).__name__}",
            )

    def _advance_and_finish(
        self,
        trigger: str,
        thread_id: str,
        started: float,
        state: dict[str, int],
        turns: int,
        *,
        status: str,
        detail: str,
        warning: bool = False,
    ) -> ExtractionOutcome:
        try:
            self._watermarks.set(thread_id, turns)
        except Exception as exc:
            return self._outcome(
                trigger,
                thread_id,
                started,
                state,
                status="failed",
                detail=f"watermark_write:{type(exc).__name__}",
            )
        state["watermark_after"] = turns
        return self._outcome(
            trigger,
            thread_id,
            started,
            state,
            status=status,
            detail=detail,
            warning=warning,
        )

    def _safe_clock(self) -> float:
        try:
            return float(self._clock())
        except Exception:
            return 0.0

    def _outcome(
        self,
        trigger: str,
        thread_id: str,
        started: float,
        state: dict[str, int],
        *,
        status: str,
        detail: str,
        warning: bool = False,
    ) -> ExtractionOutcome:
        duration_ms = max(0, int((self._safe_clock() - started) * 1000))
        outcome = ExtractionOutcome(
            trigger=trigger,
            thread_id=thread_id,
            slice_messages=state["slice_messages"],
            candidates=state["candidates"],
            persisted=state["persisted"],
            refused=state["refused"],
            record_count=state["record_count"],
            watermark_before=state["watermark_before"],
            watermark_after=state["watermark_after"],
            duration_ms=duration_ms,
            status=status,
            detail=detail,
        )
        message = (
            "memory_extraction trigger=%s thread_id=%s slice_messages=%d "
            "candidates=%d persisted=%d refused=%d record_count=%d "
            "watermark_before=%d watermark_after=%d duration_ms=%d "
            "status=%s detail=%s"
        )
        log = logger.warning if status == "failed" or warning else logger.info
        log(
            message,
            outcome.trigger,
            outcome.thread_id,
            outcome.slice_messages,
            outcome.candidates,
            outcome.persisted,
            outcome.refused,
            outcome.record_count,
            outcome.watermark_before,
            outcome.watermark_after,
            outcome.duration_ms,
            outcome.status,
            outcome.detail or "none",
        )
        return outcome


def _first_array(text: object) -> list[object] | None:
    if not isinstance(text, str):
        return None

    start = -1
    depth = 0
    quoted = False
    escaped = False
    for index, character in enumerate(text):
        if start < 0:
            if character == "[":
                start = index
                depth = 1
            continue
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : index + 1])
                except ValueError:
                    return None
                return parsed if isinstance(parsed, list) else None
    return None


def _response_text(response: object) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if content is None and isinstance(response, Mapping):
        content = response.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content) if content is not None else str(response)


def _provenance_tag(trigger: str) -> str:
    room = MAX_TAG_CHARS - len(EXTRACTION_TAG_PREFIX)
    return EXTRACTION_TAG_PREFIX + str(trigger)[: max(0, room)]


def _with_provenance(tags: Sequence[str], provenance: str) -> tuple[str, ...]:
    kept = [tag for tag in tags if tag.casefold() != provenance.casefold()]
    return tuple(kept[: MAX_TAGS - 1]) + (provenance,)


__all__ = ["ExtractionCandidate", "ExtractionOutcome", "MemoryExtractor"]
