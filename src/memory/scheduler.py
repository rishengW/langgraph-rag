"""Bounded, fire-and-forget scheduling for automatic memory extraction."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("src.memory.scheduler")

ThreadFactory = Callable[..., threading.Thread]
ExtractionRunner = Callable[[str, str], Any]


class ExtractionScheduler:
    """Run extraction work on daemon threads without queueing requests."""

    def __init__(
        self,
        *,
        max_concurrency: int,
        runner: ExtractionRunner,
        thread_factory: ThreadFactory = threading.Thread,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")

        self._max_concurrency = max_concurrency
        self._runner = runner
        self._thread_factory = thread_factory
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._inflight: set[str] = set()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._shutdown = False

    def submit(self, trigger: str, thread_id: str) -> bool:
        """Start an extraction, or return ``False`` when it must be discarded."""

        with self._condition:
            if self._shutdown:
                return False

            if thread_id in self._inflight:
                logger.debug(
                    "memory_extraction discarded trigger=%s thread_id=%s cause=already_in_flight",
                    trigger,
                    thread_id,
                )
                return False

            if not self._slots.acquire(blocking=False):
                logger.warning(
                    "memory_extraction discarded trigger=%s thread_id=%s "
                    "cause=concurrency_limit limit=%d",
                    trigger,
                    thread_id,
                    self._max_concurrency,
                )
                return False

            self._inflight.add(thread_id)

        try:
            worker = self._thread_factory(
                target=self._run,
                args=(trigger, thread_id),
                name=f"memory-extraction-{thread_id}",
                daemon=True,
            )
            worker.start()
        except Exception as exc:
            self._finish(thread_id)
            logger.warning(
                "memory_extraction worker_start_failed trigger=%s thread_id=%s cause=%s",
                trigger,
                thread_id,
                type(exc).__name__,
            )
            return False

        return True

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait for all started workers to finish. Intended for tests only."""

        with self._condition:
            return self._condition.wait_for(lambda: not self._inflight, timeout=timeout)

    def shutdown(self) -> None:
        """Reject future submissions without waiting for in-flight workers."""

        with self._condition:
            self._shutdown = True

    def _run(self, trigger: str, thread_id: str) -> None:
        try:
            self._runner(trigger, thread_id)
        except Exception as exc:
            logger.warning(
                "memory_extraction worker_failed trigger=%s thread_id=%s cause=%s",
                trigger,
                thread_id,
                type(exc).__name__,
            )
        finally:
            self._finish(thread_id)

    def _finish(self, thread_id: str) -> None:
        with self._condition:
            self._inflight.discard(thread_id)
            self._slots.release()
            self._condition.notify_all()


__all__ = ["ExtractionScheduler"]
