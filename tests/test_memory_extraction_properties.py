"""Exhaustive policy tests for automatic memory extraction."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.config.settings import Settings
from src.memory.extraction import MemoryExtractor
from src.memory.store import MemoryStore, SaveOutcome
from src.memory.watermark import InMemoryWatermarkStore


class Checkpointer:
    def __init__(self, messages=(), *, error: Exception | None = None):
        self.messages = list(messages)
        self.error = error

    def get_tuple(self, _config):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(checkpoint={"channel_values": {"messages": self.messages}})


class Model:
    def __init__(self, response="[]", *, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = 0

    def invoke(self, _prompt):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.response)


class RefusingStore:
    def read(self):
        return ()

    def save(self, **_kwargs):
        return SaveOutcome(ok=False, message="refused", record_count=0)


class RaisingStore:
    def read(self):
        return ()

    def save(self, **_kwargs):
        raise OSError("candidate content must never be logged")


def settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "dashscope_api_key": "test",
        "memory_enabled": True,
        "memory_extraction_enabled": True,
        "memory_store_path": str(tmp_path / "memory.json"),
    }
    values.update(overrides)
    return Settings(**values)


def extractor(
    tmp_path: Path,
    *,
    messages=None,
    response="[]",
    model_error=None,
    checkpointer=None,
    watermarks=None,
    store=None,
    **setting_overrides,
):
    messages = messages if messages is not None else [HumanMessage(content="question")]
    model = Model(response, error=model_error)
    resolved_store = store or MemoryStore(
        tmp_path / "memory.json", max_records=500, max_record_chars=1000
    )
    instance = MemoryExtractor(
        settings(tmp_path, **setting_overrides),
        checkpointer=checkpointer or Checkpointer(messages),
        watermarks=watermarks or InMemoryWatermarkStore(),
        store=resolved_store,
        model_factory=lambda _settings: model,
    )
    return instance, model, resolved_store


@pytest.mark.parametrize(
    ("case", "expected_status", "expected_watermark"),
    [
        ("persisted", "extracted", 1),
        ("empty_array", "no_information", 1),
        ("unusable", "no_information", 1),
        ("all_refused", "no_information", 1),
        ("empty_slice", "skipped", 0),
        ("model_error", "failed", 0),
        ("checkpoint_error", "failed", 0),
        ("store_error", "failed", 0),
    ],
)
def test_only_transient_failures_hold_watermark(
    tmp_path, case, expected_status, expected_watermark
):
    arguments = {}
    if case == "persisted":
        arguments["response"] = '[{"content":"durable fact"}]'
    elif case == "unusable":
        arguments["response"] = "not json"
    elif case == "all_refused":
        arguments.update(response='[{"content":"durable fact"}]', store=RefusingStore())
    elif case == "empty_slice":
        arguments["messages"] = []
    elif case == "model_error":
        arguments["model_error"] = RuntimeError("private model response")
    elif case == "checkpoint_error":
        arguments["checkpointer"] = Checkpointer(error=OSError("private transcript"))
    elif case == "store_error":
        arguments.update(response='[{"content":"durable fact"}]', store=RaisingStore())

    watermarks = InMemoryWatermarkStore()
    instance, _model, _store = extractor(tmp_path, watermarks=watermarks, **arguments)
    outcome = instance.run("round_complete", "thread")

    assert outcome.status == expected_status
    assert watermarks.get("thread") == expected_watermark


def test_timeout_holds_watermark_and_discards_late_response(tmp_path):
    release = threading.Event()

    class BlockingModel:
        def invoke(self, _prompt):
            release.wait()
            return '[{"content":"late private response"}]'

    instance, _model, store = extractor(tmp_path, memory_extraction_timeout_seconds=0)
    instance._model_factory = lambda _settings: BlockingModel()

    outcome = instance.run("round_complete", "thread")
    release.set()

    assert outcome.status == "failed"
    assert outcome.detail == "model_timeout"
    assert store.read() == ()


@pytest.mark.parametrize("failure", ["checkpoint", "model", "store", "watermark"])
def test_extraction_never_raises(tmp_path, failure):
    arguments = {"response": '[{"content":"durable fact"}]'}
    if failure == "checkpoint":
        arguments["checkpointer"] = Checkpointer(error=RuntimeError("boom"))
    elif failure == "model":
        arguments["model_error"] = RuntimeError("boom")
    elif failure == "store":
        arguments["store"] = RaisingStore()
    elif failure == "watermark":
        arguments["watermarks"] = RaisingWatermarks()

    instance, _model, _store = extractor(tmp_path, **arguments)
    outcome = instance.run("round_complete", "thread")
    assert outcome.status == "failed"


class RaisingWatermarks:
    def get(self, _thread_id):
        raise RuntimeError("boom")

    def set(self, _thread_id, _value):
        raise RuntimeError("boom")


def test_watermark_above_turn_count_is_clamped_and_repaired(tmp_path):
    watermarks = InMemoryWatermarkStore({"thread": 50})
    instance, model, _store = extractor(
        tmp_path,
        messages=[HumanMessage(content="q"), AIMessage(content="a")],
        watermarks=watermarks,
    )

    assert instance.should_extract("thread") is False
    assert model.calls == 0

    outcome = instance.run("session_start", "thread")
    assert model.calls == 0
    assert outcome.status == "skipped"
    assert outcome.watermark_before == 1
    assert watermarks.get("thread") == 1


def test_real_store_enforces_safeguards_and_capacity(tmp_path):
    store = MemoryStore(tmp_path / "memory.json", max_records=2, max_record_chars=30)
    store.save(content="duplicate fact")
    response = json.dumps(
        [
            {"content": "sk-abcdefghijklmnopqrstuv"},
            {"content": "duplicate fact"},
            {"content": "x" * 31},
            {"content": "new fact one"},
            {"content": "new fact two"},
        ]
    )
    instance, _model, _store = extractor(tmp_path, response=response, store=store)
    outcome = instance.run("round_complete", "thread")

    assert outcome.persisted == 3
    assert outcome.refused == 2
    assert outcome.record_count == 2
    assert len(store.read()) == 2


def test_session_scope_binds_to_extracted_thread(tmp_path):
    instance, _model, store = extractor(
        tmp_path,
        response='[{"content":"session fact","scope":"session"}]',
    )
    outcome = instance.run("session_start", "previous-thread")

    assert outcome.persisted == 1
    assert store.read()[0].scope_id == "previous-thread"


@pytest.mark.parametrize(
    ("response", "level"),
    [('[{"content":"candidate-private-0123456789"}]', logging.INFO), ("bad", logging.WARNING)],
)
def test_one_content_free_log_record_per_outcome(tmp_path, caplog, response, level):
    transcript = "transcript-private-0123456789"
    instance, _model, store = extractor(
        tmp_path,
        messages=[HumanMessage(content=transcript)],
        response=response,
    )

    with caplog.at_level(logging.INFO, logger="src.memory.extraction"):
        instance.run("round_complete", "thread")

    records = [record for record in caplog.records if record.name == "src.memory.extraction"]
    assert len(records) == 1
    assert records[0].levelno == level
    assert records[0].getMessage().startswith("memory_extraction")
    forbidden = [transcript, response]
    forbidden.extend(record.content for record in store.read())
    for value in forbidden:
        if len(value) >= 20:
            assert value[:20] not in records[0].getMessage()
