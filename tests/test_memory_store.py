"""Tests for the memory document pair and the store read path.

The headline test is the round-trip property: parsing then re-serializing a
document produced by the serializer must reproduce the record list exactly. No
``hypothesis`` in this project, so the domains are enumerated explicitly.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

import pytest

from src.config.settings import Settings
from src.backend.memory.models import SCHEMA_VERSION, MemoryDocument, MemoryRecord
from src.backend.memory.serialization import (
    CorruptDocumentError,
    parse_document,
    serialize_document,
)
from src.backend.memory.store import (
    MemoryPathError,
    MemoryStore,
    get_memory_store,
    in_scope_records,
    reset_store_cache,
    resolve_store_path,
)

TS = "2026-07-01T00:00:00+00:00"
LATER = "2026-07-02T12:30:45+00:00"


def make_record(
    rid: str = "a" * 32,
    *,
    scope: str = "global",
    scope_id: str | None = None,
    category: str = "fact",
    content: str = "Prefers metric units",
    tags: tuple[str, ...] = (),
    created_at: str = TS,
    updated_at: str = TS,
    last_recalled_at: str | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=rid,
        scope=scope,  # type: ignore[arg-type]
        scope_id=scope_id,
        category=category,  # type: ignore[arg-type]
        content=content,
        tags=tags,
        created_at=created_at,
        updated_at=updated_at,
        last_recalled_at=last_recalled_at,
    )


def make_store(tmp_path, *, max_records: int = 500, clock=None) -> MemoryStore:
    return MemoryStore(
        tmp_path / "long_term_memory.json",
        max_records=max_records,
        max_record_chars=1000,
        clock=clock or (lambda: TS),
    )


def write_document(store: MemoryStore, doc: MemoryDocument) -> None:
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(serialize_document(doc), encoding="utf-8")
    store.invalidate_cache()


# ---- the round-trip property ----------------------------------------------

ROUND_TRIP_DOCUMENTS = {
    "empty": (),
    "global_record_null_scope_id": (make_record(),),
    "session_record": (
        make_record("b" * 32, scope="session", scope_id="thread-1"),
    ),
    "never_recalled": (make_record("c" * 32, last_recalled_at=None),),
    "recalled": (make_record("d" * 32, last_recalled_at=LATER),),
    "with_tags": (make_record("e" * 32, tags=("style", "units", "zh")),),
    "non_ascii": (
        make_record(
            "f" * 32,
            content="\u7528\u6237\u559c\u6b22\u516c\u5236\u5355\u4f4d",
            tags=("\u5355\u4f4d",),
        ),
    ),
    "every_category": tuple(
        make_record(f"{i:032x}", category=category)
        for i, category in enumerate(("fact", "preference", "entity", "task"))
    ),
    "mixed_scopes": (
        make_record("1" * 32, scope="global", scope_id=None),
        make_record("2" * 32, scope="session", scope_id="thread-a"),
        make_record("3" * 32, scope="session", scope_id="thread-b"),
    ),
    "whitespace_in_content": (
        make_record("4" * 32, content="line one\nline two\ttabbed"),
    ),
    "max_tags": (
        make_record("5" * 32, tags=tuple(f"tag{i}" for i in range(10))),
    ),
}


@pytest.mark.parametrize("name", sorted(ROUND_TRIP_DOCUMENTS))
def test_round_trip_preserves_records_field_by_field(name):
    """Property 1: parse(serialize(doc)) reproduces the record list exactly."""

    doc = MemoryDocument(
        version=SCHEMA_VERSION, updated_at=TS, records=ROUND_TRIP_DOCUMENTS[name]
    )

    text = serialize_document(doc)
    parsed, warnings = parse_document(text, fallback_updated_at=TS)

    assert warnings == []
    assert parsed.version == doc.version
    assert parsed.records == doc.records
    # Re-serializing is byte-identical, so repeated read/write cycles are stable.
    assert serialize_document(parsed) == text


def test_round_trip_at_max_records():
    """The property holds for a document filled to the configured cap."""

    records = tuple(
        make_record(
            f"{i:032x}",
            content=f"memory number {i}",
            updated_at=TS,
            last_recalled_at=LATER if i % 2 else None,
            scope="session" if i % 3 == 0 else "global",
            scope_id=f"thread-{i}" if i % 3 == 0 else None,
        )
        for i in range(500)
    )
    doc = MemoryDocument(version=SCHEMA_VERSION, updated_at=TS, records=records)

    text = serialize_document(doc)
    parsed, warnings = parse_document(text, fallback_updated_at=TS)

    assert warnings == []
    assert parsed.records == records
    assert serialize_document(parsed) == text


def test_serialized_document_shape():
    doc = MemoryDocument(version=SCHEMA_VERSION, updated_at=TS, records=(make_record(),))

    payload = json.loads(serialize_document(doc))

    assert list(payload) == ["version", "updated_at", "records"]
    assert payload["version"] == 1
    assert list(payload["records"][0]) == [
        "id",
        "scope",
        "scope_id",
        "category",
        "content",
        "tags",
        "created_at",
        "updated_at",
        "last_recalled_at",
    ]


# ---- parse resilience ------------------------------------------------------


@pytest.mark.parametrize("field", ["id", "content", "scope"])
def test_missing_required_field_skips_only_that_record(field):
    good = {
        "id": "a" * 32,
        "scope": "global",
        "content": "keep me",
    }
    bad = dict(good, id="b" * 32)
    bad.pop(field)
    raw = json.dumps({"version": 1, "updated_at": TS, "records": [good, bad]})

    parsed, warnings = parse_document(raw, fallback_updated_at=TS)

    assert [record.content for record in parsed.records] == ["keep me"]
    assert len(warnings) == 1
    assert "position 1" in warnings[0]
    assert field in warnings[0]
    assert "keep me" not in warnings[0]


@pytest.mark.parametrize("value", [None, "", "   ", 42, [], {}])
def test_unusable_required_field_value_skips_the_record(value):
    raw = json.dumps(
        {
            "version": 1,
            "updated_at": TS,
            "records": [{"id": "a" * 32, "scope": "global", "content": value}],
        }
    )

    parsed, warnings = parse_document(raw, fallback_updated_at=TS)

    assert parsed.records == ()
    assert len(warnings) == 1


def test_scope_outside_the_accepted_set_skips_the_record():
    raw = json.dumps(
        {
            "version": 1,
            "updated_at": TS,
            "records": [{"id": "a" * 32, "scope": "team", "content": "x"}],
        }
    )

    parsed, warnings = parse_document(raw, fallback_updated_at=TS)

    assert parsed.records == ()
    assert "scope" in warnings[0]


def test_wrong_typed_optional_fields_keep_the_record_with_defaults():
    raw = json.dumps(
        {
            "version": 1,
            "updated_at": LATER,
            "records": [
                {
                    "id": "a" * 32,
                    "scope": "global",
                    "content": "keep me",
                    "category": 42,
                    "tags": "not-a-list",
                    "created_at": 7,
                    "updated_at": None,
                    "last_recalled_at": [],
                }
            ],
        }
    )

    parsed, warnings = parse_document(raw, fallback_updated_at=TS)

    assert len(parsed.records) == 1
    record = parsed.records[0]
    assert record.category == "fact"
    assert record.tags == ()
    assert record.created_at == LATER
    assert record.updated_at == LATER
    assert record.last_recalled_at is None
    assert warnings, "each substitution should be reported"


def test_global_record_with_a_scope_id_has_it_discarded():
    raw = json.dumps(
        {
            "version": 1,
            "updated_at": TS,
            "records": [
                {
                    "id": "a" * 32,
                    "scope": "global",
                    "content": "x",
                    "scope_id": "thread-1",
                }
            ],
        }
    )

    parsed, warnings = parse_document(raw, fallback_updated_at=TS)

    assert parsed.records[0].scope_id is None
    assert any("scope_id" in message for message in warnings)


def test_unknown_keys_are_dropped_and_not_written_back():
    raw = json.dumps(
        {
            "version": 1,
            "updated_at": TS,
            "extra_top_level": True,
            "records": [
                {
                    "id": "a" * 32,
                    "scope": "global",
                    "content": "x",
                    "surprise": {"nested": 1},
                }
            ],
        }
    )

    parsed, _ = parse_document(raw, fallback_updated_at=TS)
    rewritten = serialize_document(parsed)

    assert "surprise" not in rewritten
    assert "extra_top_level" not in rewritten
    assert len(parsed.records) == 1


@pytest.mark.parametrize("version", [2, 99, None, "1", 1.5, True, 0, -3])
def test_unreadable_version_yields_no_records(version):
    raw = json.dumps(
        {
            "version": version,
            "updated_at": TS,
            "records": [{"id": "a" * 32, "scope": "global", "content": "x"}],
        }
    )

    parsed, warnings = parse_document(raw, fallback_updated_at=TS)

    assert parsed.records == ()
    assert len(warnings) == 1
    assert str(SCHEMA_VERSION) in warnings[0]


def test_records_not_a_list_yields_no_records():
    raw = json.dumps({"version": 1, "updated_at": TS, "records": {"a": 1}})

    parsed, warnings = parse_document(raw, fallback_updated_at=TS)

    assert parsed.records == ()
    assert any("records" in message for message in warnings)


@pytest.mark.parametrize("raw", ["", "not json", "[1, 2, 3]", '"a string"', "42"])
def test_whole_document_failures_raise_corrupt(raw):
    with pytest.raises(CorruptDocumentError):
        parse_document(raw, fallback_updated_at=TS)


# ---- path resolution -------------------------------------------------------


def test_default_store_path_when_unset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(dashscope_api_key="x")

    path = resolve_store_path(settings)

    assert path == tmp_path / "memory" / "long_term_memory.json"


def test_blank_store_path_falls_back_to_the_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(dashscope_api_key="x", memory_store_path="   ")

    assert resolve_store_path(settings).name == "long_term_memory.json"


def test_relative_store_path_resolves_against_the_working_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    settings = Settings(dashscope_api_key="x", memory_store_path="custom/mem.json")

    assert resolve_store_path(settings) == tmp_path / "custom" / "mem.json"


def test_absolute_store_path_is_used_as_supplied(tmp_path):
    target = tmp_path / "abs" / "mem.json"
    settings = Settings(dashscope_api_key="x", memory_store_path=str(target))

    assert resolve_store_path(settings) == target


@pytest.mark.parametrize("raw", ["../escape.json", "memory/../../escape.json", ".."])
def test_parent_traversal_in_store_path_is_refused(raw):
    settings = Settings(dashscope_api_key="x", memory_store_path=raw)

    with pytest.raises(MemoryPathError, match=r"\.\."):
        resolve_store_path(settings)


# ---- read path -------------------------------------------------------------


def test_read_missing_file_returns_empty_and_creates_nothing(tmp_path):
    store = make_store(tmp_path)

    assert store.read() == ()
    assert not store.path.exists()
    assert not store.path.parent.joinpath("memory").exists()


def test_read_returns_persisted_records(tmp_path):
    store = make_store(tmp_path)
    records = (make_record("1" * 32), make_record("2" * 32, content="second"))
    write_document(store, MemoryDocument(SCHEMA_VERSION, TS, records))

    assert store.read() == records


def test_read_caches_an_unchanged_file(tmp_path):
    store = make_store(tmp_path)
    write_document(store, MemoryDocument(SCHEMA_VERSION, TS, (make_record(),)))

    first = store.read()
    second = store.read()

    assert first is second, "unchanged file should serve the cached tuple"


def test_mutation_read_bypasses_the_cache(tmp_path):
    """A read-modify-write must always start from the file, never the cache."""

    store = make_store(tmp_path)
    write_document(store, MemoryDocument(SCHEMA_VERSION, TS, (make_record(),)))
    store.read()  # populate the cache

    replacement = (make_record("9" * 32, content="replaced"),)
    store.path.write_text(
        serialize_document(MemoryDocument(SCHEMA_VERSION, LATER, replacement)),
        encoding="utf-8",
    )

    with store._lock:  # noqa: SLF001 - exercising the mutation read path
        assert store._read_locked(force=True) == replacement  # noqa: SLF001


def test_read_notices_an_out_of_band_rewrite(tmp_path):
    store = make_store(tmp_path)
    write_document(store, MemoryDocument(SCHEMA_VERSION, TS, (make_record(),)))
    assert len(store.read()) == 1

    bigger = tuple(
        make_record(f"{i:032x}", content=f"record {i}") for i in range(5)
    )
    store.path.write_text(
        serialize_document(MemoryDocument(SCHEMA_VERSION, LATER, bigger)),
        encoding="utf-8",
    )

    assert len(store.read()) == 5


def test_corrupt_file_is_quarantined_and_read_continues(tmp_path, caplog):
    store = make_store(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not json at all", encoding="utf-8")

    with caplog.at_level("WARNING"):
        assert store.read() == ()

    assert not store.path.exists()
    quarantined = list(tmp_path.glob("long_term_memory.json.corrupt-*"))
    assert len(quarantined) == 1
    assert ":" not in quarantined[0].name, "quarantine name must be filesystem-safe"
    assert any("unreadable" in message for message in caplog.messages)


def test_repeated_corrupt_quarantine_does_not_collide(tmp_path):
    store = make_store(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(3):
        store.path.write_text("{still not json", encoding="utf-8")
        assert store.read() == ()

    assert len(list(tmp_path.glob("long_term_memory.json.corrupt-*"))) == 3


def test_read_warns_when_the_document_exceeds_the_cap(tmp_path, caplog):
    store = make_store(tmp_path, max_records=2)
    records = tuple(make_record(f"{i:032x}") for i in range(5))
    write_document(store, MemoryDocument(SCHEMA_VERSION, TS, records))

    with caplog.at_level("WARNING"):
        loaded = store.read()

    assert len(loaded) == 5, "an over-capacity document still reads fully"
    assert store.path.read_text(encoding="utf-8"), "reading must not rewrite the file"
    assert any("above the configured cap" in message for message in caplog.messages)


# ---- scope filtering -------------------------------------------------------


def test_in_scope_includes_global_everywhere():
    records = (make_record("1" * 32),)

    assert in_scope_records(records, "thread-a") == records
    assert in_scope_records(records, None) == records


def test_in_scope_matches_session_records_on_exact_thread_only():
    session = make_record("2" * 32, scope="session", scope_id="thread-a")
    records = (session,)

    assert in_scope_records(records, "thread-a") == (session,)
    assert in_scope_records(records, "thread-b") == ()
    assert in_scope_records(records, None) == ()
    assert in_scope_records(records, "  ") == ()


def test_session_record_without_a_scope_id_is_visible_nowhere():
    orphan = make_record("3" * 32, scope="session", scope_id=None)

    assert in_scope_records((orphan,), "thread-a") == ()
    assert in_scope_records((orphan,), None) == ()


def test_store_in_scope_reads_through_to_the_file(tmp_path):
    store = make_store(tmp_path)
    glob = make_record("1" * 32)
    mine = make_record("2" * 32, scope="session", scope_id="mine")
    theirs = make_record("3" * 32, scope="session", scope_id="theirs")
    write_document(store, MemoryDocument(SCHEMA_VERSION, TS, (glob, mine, theirs)))

    assert store.in_scope("mine") == (glob, mine)


# ---- store cache -----------------------------------------------------------


def test_get_memory_store_returns_one_instance_per_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reset_store_cache()
    settings = Settings(dashscope_api_key="x")

    first = get_memory_store(settings)
    second = get_memory_store(settings)

    assert first is second, "one instance per path keeps the lock shared"
    reset_store_cache()


def test_get_memory_store_refreshes_limits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reset_store_cache()

    first = get_memory_store(Settings(dashscope_api_key="x", memory_max_records=10))
    assert first.max_records == 10

    second = get_memory_store(Settings(dashscope_api_key="x", memory_max_records=25))
    assert second is first
    assert second.max_records == 25
    reset_store_cache()


# ---- atomic write path -----------------------------------------------------


def persist(store: MemoryStore, records) -> None:
    """Drive one write through the locked mutation entry point."""

    with store._lock:  # noqa: SLF001 - exercising the mutation write path
        store._persist_locked(records)  # noqa: SLF001


def tmp_files(tmp_path) -> list:
    return list(tmp_path.glob("*.tmp"))


def test_write_creates_missing_parent_directories(tmp_path):
    store = MemoryStore(
        tmp_path / "nested" / "deeper" / "long_term_memory.json",
        max_records=500,
        max_record_chars=1000,
        clock=lambda: TS,
    )

    persist(store, (make_record(),))

    assert store.path.exists()
    assert store.read() == (make_record(),)


def test_write_then_read_round_trips_through_the_file(tmp_path):
    store = make_store(tmp_path)
    records = (
        make_record("1" * 32, content="first", tags=("a", "b")),
        make_record("2" * 32, scope="session", scope_id="thread-x", content="second"),
    )

    persist(store, records)

    assert store.read() == records


def test_write_leaves_no_temporary_files(tmp_path):
    store = make_store(tmp_path)

    persist(store, (make_record(),))
    persist(store, (make_record(), make_record("2" * 32)))

    assert tmp_files(tmp_path) == []


def test_write_invalidates_the_read_cache(tmp_path):
    store = make_store(tmp_path)
    persist(store, (make_record(),))
    assert len(store.read()) == 1

    persist(store, (make_record(), make_record("2" * 32, content="second")))

    assert len(store.read()) == 2


def test_replace_is_retried_then_fails_with_a_write_error(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    persist(store, (make_record("1" * 32, content="original"),))
    before = store.path.read_text(encoding="utf-8")

    waits: list[float] = []
    store._sleeper = waits.append  # noqa: SLF001
    attempts = {"count": 0}

    def always_locked(src, dst):
        attempts["count"] += 1
        raise PermissionError(32, "The process cannot access the file")

    monkeypatch.setattr(os, "replace", always_locked)

    with pytest.raises(Exception) as excinfo:
        persist(store, (make_record("2" * 32, content="replacement"),))

    monkeypatch.undo()

    assert type(excinfo.value).__name__ == "MemoryWriteError"
    assert attempts["count"] == 4, "one attempt plus three retries"
    assert waits == [0.05, 0.10, 0.20], "bounded backoff totalling 350 ms"
    assert tmp_files(tmp_path) == [], "the temporary file must be cleaned up"
    assert store.path.read_text(encoding="utf-8") == before, "target unchanged"
    assert "not writable" in str(excinfo.value)


def test_replace_succeeds_after_transient_failures(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    waits: list[float] = []
    store._sleeper = waits.append  # noqa: SLF001

    real_replace = os.replace
    attempts = {"count": 0}

    def flaky(src, dst):
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise PermissionError(32, "temporarily locked")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    persist(store, (make_record("3" * 32, content="eventually written"),))
    monkeypatch.undo()

    assert attempts["count"] == 3
    assert waits == [0.05, 0.10]
    assert tmp_files(tmp_path) == []
    assert store.read()[0].content == "eventually written"


def test_write_to_a_directory_path_reports_a_directory(tmp_path):
    target = tmp_path / "long_term_memory.json"
    target.mkdir()
    store = make_store(tmp_path)

    with pytest.raises(Exception) as excinfo:
        persist(store, (make_record(),))

    assert "directory" in str(excinfo.value)
    assert tmp_files(tmp_path) == []


def test_unwritable_location_reports_not_writable(tmp_path, monkeypatch):
    store = make_store(tmp_path)

    def refuse(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(tempfile, "mkstemp", refuse)

    with pytest.raises(Exception) as excinfo:
        persist(store, (make_record(),))

    monkeypatch.undo()

    assert "not writable" in str(excinfo.value)
    assert not store.path.exists()


def test_non_permission_os_error_reports_a_filesystem_error(tmp_path, monkeypatch):
    store = make_store(tmp_path)

    def boom(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", boom)

    with pytest.raises(Exception) as excinfo:
        persist(store, (make_record(),))

    monkeypatch.undo()

    assert "filesystem reported an error" in str(excinfo.value)
    assert tmp_files(tmp_path) == []


def test_temporary_file_cleanup_failure_is_logged_not_raised(
    tmp_path, monkeypatch, caplog
):
    store = make_store(tmp_path)

    def boom(src, dst):
        raise OSError(28, "No space left on device")

    def refuse_unlink(self, missing_ok=False):
        raise OSError(13, "cannot unlink")

    monkeypatch.setattr(os, "replace", boom)
    monkeypatch.setattr(Path, "unlink", refuse_unlink)

    with caplog.at_level("WARNING"), pytest.raises(Exception) as excinfo:
        persist(store, (make_record(),))

    monkeypatch.undo()

    assert "filesystem reported an error" in str(excinfo.value), (
        "the write failure, not the cleanup failure, is what the caller sees"
    )
    assert any("temporary memory file" in message for message in caplog.messages)

    # Remove the debris the faked unlink refused to clear.
    for leftover in tmp_files(tmp_path):
        leftover.unlink()


def test_concurrent_readers_never_see_a_partial_document(tmp_path):
    """Property 2: a reader observes the previous or the new document, never half."""

    store = make_store(tmp_path)
    persist(store, (make_record("1" * 32, content="v1"),))

    seen: list[int] = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            seen.append(len(store.read()))

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for size in range(1, 20):
            persist(
                store,
                tuple(
                    make_record(f"{i:032x}", content=f"record {i}")
                    for i in range(size)
                ),
            )
    finally:
        stop.set()
        thread.join(timeout=5)

    assert seen, "the reader should have observed at least one document"
    assert all(count >= 1 for count in seen), "no read returned a broken document"
