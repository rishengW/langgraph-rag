from __future__ import annotations

import threading

from src.memory.scheduler import ExtractionScheduler


def test_scheduler_bounds_concurrency_deduplicates_and_has_no_queue(caplog):
    release = threading.Event()
    all_started = threading.Event()
    calls: list[tuple[str, str]] = []
    calls_lock = threading.Lock()

    def runner(trigger: str, thread_id: str) -> None:
        with calls_lock:
            calls.append((trigger, thread_id))
            if len(calls) == 2:
                all_started.set()
        release.wait()

    scheduler = ExtractionScheduler(max_concurrency=2, runner=runner)
    try:
        assert scheduler.submit("round_complete", "thread-a") is True
        assert scheduler.submit("session_start", "thread-b") is True
        assert all_started.wait(timeout=2)

        assert scheduler.submit("session_start", "thread-a") is False
        assert scheduler.submit("round_complete", "thread-c") is False
        assert set(calls) == {
            ("round_complete", "thread-a"),
            ("session_start", "thread-b"),
        }
        assert "limit=2" in caplog.text
    finally:
        release.set()

    assert scheduler.wait_idle(timeout=2)
    assert scheduler.submit("round_complete", "thread-c") is True
    assert scheduler.wait_idle(timeout=2)
    assert calls[-1] == ("round_complete", "thread-c")


def test_scheduler_workers_are_daemon_and_shutdown_does_not_join():
    workers: list[threading.Thread] = []

    class RecordingThread(threading.Thread):
        def join(self, timeout: float | None = None) -> None:
            raise AssertionError("ExtractionScheduler must not join workers")

    def thread_factory(**kwargs: object) -> threading.Thread:
        worker = RecordingThread(**kwargs)
        workers.append(worker)
        return worker

    scheduler = ExtractionScheduler(
        max_concurrency=1,
        runner=lambda trigger, thread_id: None,
        thread_factory=thread_factory,
    )

    assert scheduler.submit("round_complete", "thread-a") is True
    assert scheduler.wait_idle(timeout=2)
    assert len(workers) == 1
    assert workers[0].daemon is True

    scheduler.shutdown()
    assert scheduler.submit("round_complete", "thread-b") is False
    assert len(workers) == 1


def test_submit_returns_while_runner_is_blocked_and_exceptions_release_slots():
    started = threading.Event()
    release = threading.Event()

    def blocked_runner(trigger: str, thread_id: str) -> None:
        started.set()
        release.wait()

    scheduler = ExtractionScheduler(max_concurrency=1, runner=blocked_runner)
    try:
        assert scheduler.submit("round_complete", "blocked") is True
        assert started.wait(timeout=2)
        assert scheduler.wait_idle(timeout=0) is False
    finally:
        release.set()

    assert scheduler.wait_idle(timeout=2)

    failed_calls: list[str] = []

    def raising_runner(trigger: str, thread_id: str) -> None:
        failed_calls.append(thread_id)
        raise RuntimeError("test failure")

    failing_scheduler = ExtractionScheduler(max_concurrency=1, runner=raising_runner)
    assert failing_scheduler.submit("round_complete", "first") is True
    assert failing_scheduler.wait_idle(timeout=2)
    assert failing_scheduler.submit("round_complete", "second") is True
    assert failing_scheduler.wait_idle(timeout=2)
    assert failed_calls == ["first", "second"]
