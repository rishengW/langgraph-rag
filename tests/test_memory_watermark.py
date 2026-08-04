"""Tests for the extraction watermark seam."""

from __future__ import annotations

import pytest

from src.memory.watermark import (
    MAX_WATERMARK,
    WATERMARK_KEY,
    InMemoryWatermarkStore,
    coerce_watermark,
)


@pytest.mark.parametrize("raw,expected", [
    (0, 0),
    (7, 7),
    (MAX_WATERMARK, MAX_WATERMARK),
    (MAX_WATERMARK + 1, 0),
    (-1, 0),
    (True, 0),
    (False, 0),
    ("3", 0),
    (3.0, 0),
    (None, 0),
    ({}, 0),
    ([], 0),
])
def test_coerce_watermark(raw, expected):
    assert coerce_watermark(raw) == expected


def test_booleans_are_rejected_rather_than_read_as_ints():
    """bool subclasses int, so True would otherwise silently mean watermark 1."""

    assert coerce_watermark(True) == 0
    assert coerce_watermark(True) != 1


def test_coerce_watermark_never_raises():
    class Hostile:
        def __int__(self):
            raise RuntimeError("no")

        def __eq__(self, other):
            raise RuntimeError("no")

    assert coerce_watermark(Hostile()) == 0


def test_watermark_key_is_stable():
    assert WATERMARK_KEY == "extraction_watermark"


# ---- in-memory store -------------------------------------------------------


def test_unknown_thread_reads_zero():
    assert InMemoryWatermarkStore().get("absent") == 0


def test_set_then_get_round_trips():
    store = InMemoryWatermarkStore()

    store.set("thread-a", 12)

    assert store.get("thread-a") == 12


def test_stored_values_are_coerced_on_write_and_read():
    store = InMemoryWatermarkStore({"thread-a": -5})

    assert store.get("thread-a") == 0

    store.set("thread-b", MAX_WATERMARK + 10)
    assert store.get("thread-b") == 0


def test_threads_are_independent():
    store = InMemoryWatermarkStore()

    store.set("thread-a", 3)
    store.set("thread-b", 9)

    assert store.get("thread-a") == 3
    assert store.get("thread-b") == 9
