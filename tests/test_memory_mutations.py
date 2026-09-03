"""Tests for the memory store mutations: save, recall, forget, purge.

Durability and concurrency behaviour lives at the bottom.
"""

from __future__ import annotations

import threading

import pytest

from src.backend.memory.models import MAX_FORGET_DELETES, MAX_TAG_CHARS, MAX_TAGS
from src.backend.memory.store import MemoryStore

TS = "2026-07-01T00:00:00+00:00"


class StepClock:
    """Deterministic clock that advances one second per call."""

    def __init__(self, start: int = 0) -> None:
        self._n = start

    def __call__(self) -> str:
        value = f"2026-07-01T{self._n // 3600:02d}:{self._n // 60 % 60:02d}:{self._n % 60:02d}+00:00"
        self._n += 1
        return value


@pytest.fixture
def store(tmp_path):
    return MemoryStore(
        tmp_path / "long_term_memory.json",
        max_records=500,
        max_record_chars=1000,
        clock=StepClock(),
    )


def small_store(tmp_path, *, max_records=2, max_record_chars=1000, default_scope="global"):
    return MemoryStore(
        tmp_path / "long_term_memory.json",
        max_records=max_records,
        max_record_chars=max_record_chars,
        default_scope=default_scope,
        clock=StepClock(),
    )


# ---- save: happy paths -----------------------------------------------------


def test_save_persists_one_record(store):
    outcome = store.save(content="  Name is Ada  ", category="fact")

    assert outcome.ok and outcome.created
    assert outcome.record_id is not None
    assert len(outcome.record_id) == 32
    assert int(outcome.record_id, 16) >= 0, "identifier is hexadecimal"
    assert outcome.record_id in outcome.message

    (record,) = store.read()
    assert record.content == "Name is Ada", "content is stored trimmed"
    assert record.category == "fact"
    assert record.created_at == record.updated_at
    assert record.last_recalled_at is None
    assert record.scope == "global"
    assert record.scope_id is None


def test_save_defaults_the_category(store):
    for value in (None, "", "   "):
        outcome = store.save(content=f"content {value!r}", category=value)
        assert outcome.ok
    assert {record.category for record in store.read()} == {"fact"}


@pytest.mark.parametrize("category", ["fact", "preference", "entity", "task", "TASK"])
def test_save_accepts_every_category(store, category):
    outcome = store.save(content="something", category=category)

    assert outcome.ok
    assert store.read()[0].category == category.casefold()


def test_save_normalizes_tags(store):
    outcome = store.save(
        content="likes concise answers",
        tags=["  Style  ", "style", "", "   ", "Brevity"],
    )

    assert outcome.ok
    assert store.read()[0].tags == ("Style", "Brevity"), (
        "trimmed, blanks dropped, case-insensitive duplicates collapsed, order kept"
    )


def test_save_with_no_tags_stores_an_empty_tuple(store):
    assert store.save(content="x").ok
    assert store.read()[0].tags == ()


# ---- save: duplicate update ------------------------------------------------


def test_duplicate_save_updates_tags_and_timestamp_only(store):
    first = store.save(content="Prefers metric units", tags=["units"])
    original = store.read()[0]

    second = store.save(content="  prefers   METRIC units ", tags=["si"])

    assert second.ok and second.updated and not second.created
    assert second.record_id == first.record_id
    assert first.record_id in second.message

    (record,) = store.read()
    assert record.id == original.id
    assert record.content == original.content, "content is left as first written"
    assert record.created_at == original.created_at
    assert record.category == original.category
    assert record.scope == original.scope
    assert record.scope_id == original.scope_id
    assert record.last_recalled_at == original.last_recalled_at
    assert record.tags == ("si",), "tags are replaced"
    assert record.updated_at != original.updated_at


def test_duplicate_detection_is_scoped_by_category_and_scope(store):
    store.save(content="same text", category="fact")
    store.save(content="same text", category="preference")
    store.save(content="same text", scope="session", thread_id="thread-a")

    assert len(store.read()) == 3, "different category or scope is not a duplicate"


# ---- save: validation ------------------------------------------------------


@pytest.mark.parametrize("content", [None, "", "   ", "\t\n "])
def test_save_rejects_blank_content(store, content):
    outcome = store.save(content=content)

    assert not outcome.ok
    assert "content" in outcome.message
    assert store.read() == ()


def test_save_rejects_over_length_content(tmp_path):
    store = small_store(tmp_path, max_record_chars=10)

    outcome = store.save(content="x" * 11)

    assert not outcome.ok
    assert "content" in outcome.message
    assert "10" in outcome.message
    assert store.read() == ()


def test_save_length_check_applies_after_trimming(tmp_path):
    store = small_store(tmp_path, max_record_chars=5)

    assert store.save(content="   12345   ").ok


def test_save_rejects_unknown_category(store):
    outcome = store.save(content="x", category="opinion")

    assert not outcome.ok
    assert "category" in outcome.message
    assert store.read() == ()


def test_save_rejects_unknown_scope(store):
    outcome = store.save(content="x", scope="team")

    assert not outcome.ok
    assert "scope" in outcome.message
    assert "global" in outcome.message and "session" in outcome.message
    assert store.read() == ()


def test_save_rejects_too_many_tags(store):
    outcome = store.save(content="x", tags=[f"t{i}" for i in range(MAX_TAGS + 1)])

    assert not outcome.ok
    assert "tags" in outcome.message
    assert store.read() == ()


def test_save_rejects_over_length_tag(store):
    outcome = store.save(content="x", tags=["a" * (MAX_TAG_CHARS + 1)])

    assert not outcome.ok
    assert "tags" in outcome.message
    assert str(MAX_TAG_CHARS) in outcome.message
    assert store.read() == ()


def test_save_validation_order_reports_the_first_violation(store):
    """Content presence outranks every other violation."""

    outcome = store.save(
        content="   ",
        category="nonsense",
        scope="nonsense",
        tags=[f"t{i}" for i in range(MAX_TAGS + 5)],
    )

    assert not outcome.ok
    assert "content" in outcome.message
    assert "category" not in outcome.message


def test_save_validation_order_category_before_scope_and_tags(store):
    outcome = store.save(
        content="fine",
        category="nonsense",
        scope="nonsense",
        tags=[f"t{i}" for i in range(MAX_TAGS + 5)],
    )

    assert "category" in outcome.message
    assert "scope" not in outcome.message


def test_save_validation_order_scope_before_tags(store):
    outcome = store.save(
        content="fine",
        scope="nonsense",
        tags=[f"t{i}" for i in range(MAX_TAGS + 5)],
    )

    assert "scope" in outcome.message
    assert "tags" not in outcome.message


# ---- save: secret refusal --------------------------------------------------


def test_save_refuses_credential_like_content(store):
    secret = "sk-abcdefghijklmnopqrstuv"

    outcome = store.save(content=f"my key is {secret}")

    assert not outcome.ok
    assert "credential" in outcome.message
    assert "content" in outcome.message
    assert secret not in outcome.message, "the matched text must not be echoed"
    assert store.read() == ()


def test_save_refuses_credential_like_tag(store):
    outcome = store.save(content="harmless", tags=["AKIAIOSFODNN7EXAMPLE"])

    assert not outcome.ok
    assert "credential" in outcome.message
    assert "tags" in outcome.message
    assert "AKIAIOSFODNN7EXAMPLE" not in outcome.message
    assert store.read() == ()


def test_secret_screening_runs_after_limit_checks(tmp_path):
    """A call breaking both a limit and a pattern reports the limit."""

    store = small_store(tmp_path, max_record_chars=5)

    outcome = store.save(content="password=hunter2hunter2")

    assert not outcome.ok
    assert "content" in outcome.message
    assert "credential" not in outcome.message


# ---- save: scope -----------------------------------------------------------


def test_session_scope_uses_the_current_thread(store):
    assert store.save(content="local note", scope="session", thread_id="thread-a").ok

    (record,) = store.read()
    assert record.scope == "session"
    assert record.scope_id == "thread-a"


def test_session_scope_without_a_thread_is_refused(store):
    outcome = store.save(content="local note", scope="session")

    assert not outcome.ok
    assert "session" in outcome.message
    assert "global" in outcome.message
    assert store.read() == ()


def test_default_scope_setting_is_honoured(tmp_path):
    store = small_store(tmp_path, max_records=10, default_scope="session")

    assert store.save(content="note", thread_id="thread-a").ok

    assert store.read()[0].scope == "session"


def test_default_session_scope_without_a_thread_is_refused(tmp_path):
    store = small_store(tmp_path, max_records=10, default_scope="session")

    outcome = store.save(content="note")

    assert not outcome.ok
    assert store.read() == ()


# ---- save: capacity --------------------------------------------------------


def test_save_evicts_the_least_recently_used_record(tmp_path):
    store = small_store(tmp_path, max_records=2)
    store.save(content="oldest")
    store.save(content="middle")
    oldest_id = store.read()[0].id

    outcome = store.save(content="newest")

    assert outcome.pruned == (oldest_id,)
    contents = {record.content for record in store.read()}
    assert contents == {"middle", "newest"}
    assert len(store.read()) == 2


def test_eviction_prefers_never_recalled_over_recalled(tmp_path):
    store = small_store(tmp_path, max_records=2)
    store.save(content="alpha keyword")
    store.save(content="beta")
    # Recalling alpha makes it the most recently used, so beta should go.
    store.recall(query="keyword", top_k=5, max_chars=500)

    store.save(content="gamma")

    assert {record.content for record in store.read()} == {"alpha keyword", "gamma"}


def test_eviction_spans_scopes(tmp_path):
    store = small_store(tmp_path, max_records=2)
    store.save(content="session note", scope="session", thread_id="thread-a")
    store.save(content="global note")

    store.save(content="newest")

    assert len(store.read()) == 2
    assert "session note" not in {record.content for record in store.read()}


def test_eviction_ties_break_on_ascending_id(tmp_path):
    store = small_store(tmp_path, max_records=2)
    fixed = "2026-07-01T00:00:00+00:00"
    store._clock = lambda: fixed  # noqa: SLF001 - force identical recency
    store.save(content="one")
    store.save(content="two")
    ids = sorted(record.id for record in store.read())
    assert len(ids) == 2, "the store must be at capacity for the tie to matter"

    outcome = store.save(content="three")

    assert outcome.pruned == (ids[0],), "smallest identifier is evicted first"


def test_duplicate_update_at_capacity_evicts_nothing(tmp_path):
    store = small_store(tmp_path, max_records=2)
    store.save(content="alpha")
    store.save(content="beta")

    outcome = store.save(content="alpha", tags=["again"])

    assert outcome.updated
    assert outcome.pruned == ()
    assert len(store.read()) == 2
    assert {record.content for record in store.read()} == {"alpha", "beta"}


def test_count_invariant_holds_across_many_saves(tmp_path):
    store = small_store(tmp_path, max_records=5)

    for index in range(25):
        assert store.save(content=f"memory {index}").ok
        assert len(store.read()) <= 5

    assert len(store.read()) == 5


# ---- recall ----------------------------------------------------------------


def test_recall_returns_matching_records(store):
    store.save(content="Prefers metric units", category="preference")
    store.save(content="Works in Shanghai", category="entity")

    outcome = store.recall(query="metric", top_k=5, max_chars=2000)

    assert outcome.ok
    assert len(outcome.records) == 1
    record = outcome.records[0]
    assert record.id in outcome.message
    assert "preference" in outcome.message
    assert "Prefers metric units" in outcome.message
    assert record.updated_at in outcome.message


def test_recall_output_omits_other_fields(store):
    store.save(content="Prefers metric units", tags=["secret-tag-name"])

    outcome = store.recall(query="metric", top_k=5, max_chars=2000)

    assert "secret-tag-name" not in outcome.message
    assert "last_recalled_at" not in outcome.message


def test_recall_is_deterministic_across_repeated_calls(store):
    store.save(content="Prefers metric units")
    store.save(content="Also likes metric charts")

    first = store.recall(query="metric", top_k=5, max_chars=2000)
    second = store.recall(query="metric", top_k=5, max_chars=2000)

    assert first.message == second.message, (
        "the last_recalled_at stamp must not alter output or ordering"
    )


def test_recall_stamps_last_recalled_at_without_touching_other_fields(store):
    store.save(content="Prefers metric units")
    before = store.read()[0]

    store.recall(query="metric", top_k=5, max_chars=2000)
    after = store.read()[0]

    assert after.last_recalled_at is not None
    assert after.last_recalled_at != before.last_recalled_at
    assert after.id == before.id
    assert after.content == before.content
    assert after.category == before.category
    assert after.tags == before.tags
    assert after.created_at == before.created_at
    assert after.updated_at == before.updated_at
    assert len(store.read()) == 1, "recall must not change the record count"


def test_recall_respects_top_k(store):
    for index in range(6):
        store.save(content=f"metric memory {index}")

    outcome = store.recall(query="metric", top_k=2, max_chars=2000)

    assert len(outcome.records) == 2


def test_recall_truncation_drops_whole_entries(store):
    for index in range(5):
        store.save(content=f"metric memory number {index}")

    outcome = store.recall(query="metric", top_k=5, max_chars=120)

    lines = outcome.message.splitlines()
    assert 0 < len(lines) < 5
    for line in lines:
        assert line.startswith("- ["), "each surviving line is a whole record"


def test_recall_with_no_match_reports_it_and_changes_nothing(store):
    store.save(content="Prefers metric units")
    before = store.read()

    outcome = store.recall(query="bicycles", top_k=5, max_chars=2000)

    assert outcome.ok
    assert "No matching memory" in outcome.message
    assert store.read() == before


def test_recall_with_only_short_terms_reports_no_match(store):
    store.save(content="Prefers metric units")

    outcome = store.recall(query="a b c", top_k=5, max_chars=2000)

    assert outcome.ok
    assert "No matching memory" in outcome.message


@pytest.mark.parametrize("query", [None, "", "   "])
def test_recall_rejects_a_blank_query(store, query):
    outcome = store.recall(query=query, top_k=5, max_chars=2000)

    assert not outcome.ok
    assert "query" in outcome.message


def test_recall_rejects_an_over_length_query(store):
    outcome = store.recall(query="x" * 501, top_k=5, max_chars=2000)

    assert not outcome.ok
    assert "query" in outcome.message


def test_recall_only_sees_in_scope_records(store):
    store.save(content="global metric note")
    store.save(content="session metric note", scope="session", thread_id="mine")

    mine = store.recall(query="metric", thread_id="mine", top_k=5, max_chars=2000)
    theirs = store.recall(query="metric", thread_id="other", top_k=5, max_chars=2000)

    assert len(mine.records) == 2
    assert len(theirs.records) == 1


def test_recall_read_failure_is_reported_without_a_write(store, monkeypatch):
    store.save(content="Prefers metric units")
    before = store.path.read_text(encoding="utf-8")

    def boom(*args, **kwargs):
        from src.backend.memory.store import MemoryStoreError

        raise MemoryStoreError("disk on fire")

    monkeypatch.setattr(MemoryStore, "_read_locked", boom)

    outcome = store.recall(query="metric", top_k=5, max_chars=2000)

    monkeypatch.undo()

    assert not outcome.ok
    assert "disk on fire" in outcome.message
    assert store.path.read_text(encoding="utf-8") == before


# ---- forget ----------------------------------------------------------------


def test_forget_by_id_deletes_one_record(store):
    keep = store.save(content="keep me").record_id
    drop = store.save(content="drop me").record_id

    outcome = store.forget(memory_id=drop)

    assert outcome.ok
    assert outcome.deleted == (drop,)
    assert drop in outcome.message
    assert [record.id for record in store.read()] == [keep]


def test_forget_by_id_ignores_a_query_supplied_alongside(store):
    target = store.save(content="alpha match").record_id
    other = store.save(content="beta match").record_id

    outcome = store.forget(memory_id=target, query="match")

    assert outcome.deleted == (target,), "at most one record is deleted"
    assert "ignored" in outcome.message
    assert [record.id for record in store.read()] == [other]


def test_forget_by_id_accepts_uppercase(store):
    target = store.save(content="drop me").record_id

    outcome = store.forget(memory_id=target.upper())

    assert outcome.deleted == (target,)


def test_forget_by_id_is_idempotent(store):
    target = store.save(content="drop me").record_id

    first = store.forget(memory_id=target)
    count_after_first = len(store.read())
    second = store.forget(memory_id=target)

    assert first.deleted == (target,)
    assert second.deleted == ()
    assert "no memory with id" in second.message.lower()
    assert len(store.read()) == count_after_first


def test_forget_by_id_out_of_scope_reports_not_found(store):
    target = store.save(
        content="theirs", scope="session", thread_id="theirs"
    ).record_id

    outcome = store.forget(memory_id=target, thread_id="mine")

    assert "no memory with id" in outcome.message.lower()
    assert len(store.read()) == 1, "an out-of-scope record is left untouched"


@pytest.mark.parametrize("bad", ["abc", "z" * 32, "a" * 31, "a" * 33])
def test_forget_rejects_a_malformed_id(store, bad):
    store.save(content="keep me")

    outcome = store.forget(memory_id=bad)

    assert not outcome.ok
    assert "memory_id" in outcome.message
    assert len(store.read()) == 1


def test_forget_by_query_deletes_matches(store):
    store.save(content="metric units preferred")
    store.save(content="metric charts preferred")
    keep = store.save(content="unrelated entirely").record_id

    outcome = store.forget(query="metric")

    assert outcome.ok
    assert len(outcome.deleted) == 2
    assert "2 memories" in outcome.message
    assert [record.id for record in store.read()] == [keep]


def test_forget_by_query_caps_deletions_and_reports_the_remainder(store):
    for index in range(MAX_FORGET_DELETES + 4):
        store.save(content=f"metric memory {index}")

    outcome = store.forget(query="metric")

    assert len(outcome.deleted) == MAX_FORGET_DELETES
    assert outcome.unmatched == 4
    assert "4 further match" in outcome.message
    assert len(store.read()) == 4


def test_forget_by_query_with_no_match_changes_nothing(store):
    store.save(content="keep me")
    before = store.read()

    outcome = store.forget(query="bicycles")

    assert outcome.ok
    assert "No matching memory" in outcome.message
    assert store.read() == before


def test_forget_by_query_rejects_an_over_length_query(store):
    outcome = store.forget(query="x" * 501)

    assert not outcome.ok
    assert "query" in outcome.message


@pytest.mark.parametrize("memory_id,query", [(None, None), ("", ""), ("  ", "  ")])
def test_forget_requires_one_parameter(store, memory_id, query):
    store.save(content="keep me")

    outcome = store.forget(memory_id=memory_id, query=query)

    assert not outcome.ok
    assert "memory_id" in outcome.message and "query" in outcome.message
    assert len(store.read()) == 1


def test_forget_leaves_other_records_untouched(store):
    keeper_id = store.save(content="keep me", tags=["k"], category="task").record_id
    keeper = next(r for r in store.read() if r.id == keeper_id)
    drop = store.save(content="drop me").record_id

    store.forget(memory_id=drop)

    survivor = store.read()[0]
    assert survivor == keeper, "every field of the surviving record is unchanged"


# ---- session purge ---------------------------------------------------------


def test_purge_session_removes_only_that_session(store):
    store.save(content="global note")
    store.save(content="mine", scope="session", thread_id="mine")
    store.save(content="theirs", scope="session", thread_id="theirs")

    removed = store.purge_session("mine")

    assert removed == 1
    assert {record.content for record in store.read()} == {"global note", "theirs"}


def test_purge_session_with_nothing_to_remove_is_a_noop(store):
    store.save(content="global note")
    before = store.path.read_text(encoding="utf-8")

    assert store.purge_session("absent") == 0
    assert store.path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("thread_id", ["", "   ", None])
def test_purge_session_ignores_a_blank_thread(store, thread_id):
    store.save(content="global note")

    assert store.purge_session(thread_id) == 0
    assert len(store.read()) == 1


# ---- durability and concurrency -------------------------------------------


def test_records_survive_a_new_store_instance(tmp_path):
    first = MemoryStore(
        tmp_path / "long_term_memory.json",
        max_records=500,
        max_record_chars=1000,
        clock=StepClock(),
    )
    first.save(content="durable note", tags=["keep"], category="task")
    first.save(content="session note", scope="session", thread_id="thread-a")
    expected = first.read()

    second = MemoryStore(
        tmp_path / "long_term_memory.json",
        max_records=500,
        max_record_chars=1000,
        clock=StepClock(),
    )

    assert second.read() == expected, "every field survives a restart"


def test_concurrent_saves_lose_no_record(tmp_path):
    store = MemoryStore(
        tmp_path / "long_term_memory.json",
        max_records=500,
        max_record_chars=1000,
    )
    threads = 8
    per_thread = 20
    errors: list[str] = []

    def worker(index: int) -> None:
        for step in range(per_thread):
            outcome = store.save(content=f"memory {index}-{step}")
            if not outcome.ok:
                errors.append(outcome.message)

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join(timeout=60)

    assert errors == []
    assert len(store.read()) == threads * per_thread


def test_concurrent_saves_respect_the_cap(tmp_path):
    store = MemoryStore(
        tmp_path / "long_term_memory.json",
        max_records=10,
        max_record_chars=1000,
    )

    def worker(index: int) -> None:
        for step in range(15):
            store.save(content=f"memory {index}-{step}")

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join(timeout=60)

    assert len(store.read()) == 10


def test_over_capacity_document_is_reduced_on_the_next_write(tmp_path):
    big = MemoryStore(
        tmp_path / "long_term_memory.json",
        max_records=500,
        max_record_chars=1000,
        clock=StepClock(),
    )
    for index in range(6):
        big.save(content=f"memory {index}")

    capped = MemoryStore(
        tmp_path / "long_term_memory.json",
        max_records=3,
        max_record_chars=1000,
        clock=StepClock(100),
    )
    assert len(capped.read()) == 6, "reading does not delete anything"

    capped.save(content="triggers the reduction")

    assert len(capped.read()) == 3
