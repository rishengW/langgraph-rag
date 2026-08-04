"""Focused tests for the automatic memory extractor."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config.settings import Settings
from src.memory.extraction import ExtractionCandidate, MemoryExtractor
from src.memory.store import MemoryStore
from src.memory.watermark import InMemoryWatermarkStore


class Checkpointer:
    def __init__(self, messages):
        self.messages = list(messages)

    def get_tuple(self, _config):
        return SimpleNamespace(checkpoint={"channel_values": {"messages": self.messages}})


class Model:
    def __init__(self, response="[]"):
        self.response = response
        self.calls = 0

    def invoke(self, _prompt):
        self.calls += 1
        return AIMessage(content=self.response)


def make_extractor(tmp_path: Path, messages, response="[]", **overrides):
    settings = Settings(
        dashscope_api_key="test",
        memory_enabled=True,
        memory_extraction_enabled=True,
        memory_store_path=str(tmp_path / "memory.json"),
        **overrides,
    )
    model = Model(response)
    store = MemoryStore(tmp_path / "memory.json", max_records=500, max_record_chars=1000)
    extractor = MemoryExtractor(
        settings,
        checkpointer=Checkpointer(messages),
        watermarks=InMemoryWatermarkStore(),
        store=store,
        model_factory=lambda _settings: model,
    )
    return extractor, model, store


def test_checkpoint_read_and_round_boundaries(tmp_path):
    messages = [
        item
        for index in range(20)
        for item in (HumanMessage(content=f"question {index}"), AIMessage(content="answer"))
    ]
    extractor, model, _store = make_extractor(tmp_path, messages, '[{"content":"fact"}]')

    assert extractor.turn_count("thread") == 20
    assert extractor.should_extract("thread")
    outcome = extractor.run("round_complete", "thread")

    assert outcome.watermark_after == 20
    assert outcome.persisted == 1
    assert model.calls == 1


def test_round_boundaries_are_exact_across_successive_rounds(tmp_path):
    checkpointer = Checkpointer([])
    watermarks = InMemoryWatermarkStore()
    settings = Settings(
        dashscope_api_key="test",
        memory_enabled=True,
        memory_extraction_enabled=True,
        memory_extraction_turn_interval=10,
        memory_store_path=str(tmp_path / "memory.json"),
    )
    model = Model("[]")
    instance = MemoryExtractor(
        settings,
        checkpointer=checkpointer,
        watermarks=watermarks,
        store=MemoryStore(tmp_path / "memory.json", max_records=500, max_record_chars=1000),
        model_factory=lambda _settings: model,
    )

    triggered = []
    for turn in range(1, 31):
        checkpointer.messages.extend(
            [HumanMessage(content=f"q{turn}"), AIMessage(content=f"a{turn}")]
        )
        if instance.should_extract("thread"):
            triggered.append(turn)
            instance.run("round_complete", "thread")

    assert triggered == [10, 20, 30]
    assert watermarks.get("thread") == 30
    assert model.calls == 3


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[]", ()),
        ("not json", ()),
        ('{"content":"not an array"}', ()),
        ('prefix [{"content":"one"}] suffix', ("one",)),
    ],
)
def test_candidate_parsing_shapes(tmp_path, text, expected):
    extractor, _model, _store = make_extractor(tmp_path, [HumanMessage(content="q")])
    parsed = extractor._parse_candidates(text)
    assert tuple(item.content for item in parsed) == expected


def test_candidate_parsing_filters_duplicates_and_bounds(tmp_path):
    extractor, _model, _store = make_extractor(
        tmp_path,
        [HumanMessage(content="q")],
        memory_extraction_max_candidates=2,
    )
    entries = [
        {"content": "  Stable   Fact ", "category": " ", "tags": ["one", 2, " "]},
        {"content": "stable fact", "scope": "bad"},
        {"content": "Second", "tags": "not-a-list"},
        {"content": "Third"},
        {"content": ""},
        42,
    ]
    parsed = extractor._parse_candidates(json.dumps(entries))

    assert parsed == (
        ExtractionCandidate(content="Stable   Fact", tags=("one",)),
        ExtractionCandidate(content="Second"),
    )


def test_candidate_parser_never_evaluates_entry_101(tmp_path):
    extractor, _model, _store = make_extractor(
        tmp_path,
        [HumanMessage(content="q")],
        memory_extraction_max_candidates=5,
    )
    entries = [{"content": ""} for _ in range(100)]
    entries.append({"content": "too late"})

    assert extractor._parse_candidates(json.dumps(entries)) == ()


def test_unusable_response_advances_watermark_without_calling_store(tmp_path):
    extractor, model, store = make_extractor(tmp_path, [HumanMessage(content="q")], "oops")
    outcome = extractor.run("round_complete", "thread")

    assert outcome.status == "no_information"
    assert outcome.watermark_after == 1
    assert model.calls == 1
    assert store.read() == ()


def test_memory_note_and_system_messages_are_not_candidates(tmp_path):
    extractor, model, _store = make_extractor(
        tmp_path,
        [
            SystemMessage(content="LONG-TERM MEMORY (recalled):\n- private"),
            HumanMessage(content="q"),
            AIMessage(content="a"),
        ],
        "[]",
    )
    outcome = extractor.run("round_complete", "thread")
    assert outcome.slice_messages == 2
    assert model.calls == 1


def test_store_refusal_does_not_stop_later_candidates(tmp_path):
    response = json.dumps(
        [
            {"content": "sk-example-secret-value"},
            {"content": "durable fact"},
        ]
    )
    extractor, _model, store = make_extractor(
        tmp_path,
        [HumanMessage(content="q")],
        response,
    )
    outcome = extractor.run("round_complete", "thread")

    assert outcome.persisted == 1
    assert outcome.refused == 1
    assert len(store.read()) == 1


def test_model_failure_holds_watermark(tmp_path):
    class Broken:
        def invoke(self, _prompt):
            raise RuntimeError("model failed")

    settings = Settings(
        dashscope_api_key="test",
        memory_enabled=True,
        memory_extraction_enabled=True,
        memory_store_path=str(tmp_path / "memory.json"),
    )
    extractor = MemoryExtractor(
        settings,
        checkpointer=Checkpointer([HumanMessage(content="q")]),
        watermarks=InMemoryWatermarkStore(),
        store=MemoryStore(tmp_path / "memory.json", max_records=500, max_record_chars=1000),
        model_factory=lambda _settings: Broken(),
    )
    outcome = extractor.run("round_complete", "thread")
    assert outcome.status == "failed"
    assert outcome.watermark_after == 0
