"""Offline tests for the RAG retrieval benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.backend.rag import benchmark


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )
    return path


def _golden(path: Path) -> Path:
    return _write_jsonl(
        path,
        [
            {
                "query_id": "q1",
                "question": "question one",
                "expected_sources": ["https://example.test/a"],
            },
            {
                "query_id": "q2",
                "question": "question two",
                "expected_sources": ["https://example.test/a", "chat_uploads/t1/notes.txt"],
            },
        ],
    )


def _runs(path: Path) -> Path:
    return _write_jsonl(
        path,
        [
            {
                "query_id": "q1",
                "strategy": "default",
                "top_k": 4,
                "chunks": [
                    {"rank": 1, "source": "https://example.test/a?spm=tracking", "text_head": "x"},
                    {"rank": 2, "source": "https://example.test/b", "text_head": "y"},
                ],
            },
            {
                "query_id": "q2",
                "strategy": "hybrid",
                "top_k": 4,
                "chunks": [
                    {"rank": 1, "source": "https://example.test/a", "text_head": "x"},
                ],
            },
        ],
    )


def test_recall_ignores_query_strings_and_case(tmp_path):
    run = benchmark.RetrievalRun(
        query_id="q1",
        strategy="default",
        top_k=4,
        chunks=(
            benchmark.RetrievedChunk(rank=1, source="https://example.test/A?spm=1", text_head=""),
        ),
    )
    assert benchmark.recall_at_k(run, ["https://example.test/a/"], k=4) == 1.0


def test_recall_partial_when_one_of_two_sources_missing(tmp_path):
    run = benchmark.RetrievalRun(
        query_id="q2",
        strategy="default",
        top_k=4,
        chunks=(
            benchmark.RetrievedChunk(rank=1, source="https://example.test/a", text_head=""),
            benchmark.RetrievedChunk(rank=2, source="https://example.test/c", text_head=""),
        ),
    )
    assert benchmark.recall_at_k(
        run, ["https://example.test/a", "chat_uploads/t1/notes.txt"], k=4
    ) == pytest.approx(0.5)


def test_recall_respects_k_cutoff(tmp_path):
    run = benchmark.RetrievalRun(
        query_id="q1",
        strategy="default",
        top_k=4,
        chunks=tuple(
            benchmark.RetrievedChunk(rank=rank, source=f"https://example.test/{rank}", text_head="")
            for rank in range(1, 5)
        ),
    )
    assert benchmark.recall_at_k(run, ["https://example.test/4"], k=3) == 0.0
    assert benchmark.recall_at_k(run, ["https://example.test/4"], k=4) == 1.0


def test_reciprocal_rank_positions(tmp_path):
    run = benchmark.RetrievalRun(
        query_id="q1",
        strategy="default",
        top_k=4,
        chunks=(
            benchmark.RetrievedChunk(rank=1, source="https://example.test/x", text_head=""),
            benchmark.RetrievedChunk(rank=2, source="https://example.test/a", text_head=""),
        ),
    )
    assert benchmark.reciprocal_rank(run, ["https://example.test/a"], k=4) == pytest.approx(0.5)
    assert benchmark.reciprocal_rank(run, ["https://example.test/none"], k=4) == 0.0


def test_provenance_complete_requires_all_sources(tmp_path):
    complete = benchmark.RetrievalRun(
        query_id="q",
        strategy="default",
        top_k=4,
        chunks=(
            benchmark.RetrievedChunk(rank=1, source="https://example.test/a", text_head=""),
            benchmark.RetrievedChunk(rank=2, source="chat_uploads/t1/notes.txt", text_head=""),
        ),
    )
    incomplete = benchmark.RetrievalRun(
        query_id="q",
        strategy="default",
        top_k=4,
        chunks=(
            benchmark.RetrievedChunk(rank=1, source="https://example.test/a", text_head=""),
            benchmark.RetrievedChunk(rank=2, source="", text_head=""),
        ),
    )
    empty = benchmark.RetrievalRun(query_id="q", strategy="default", top_k=4, chunks=())
    assert benchmark.provenance_complete(complete)
    assert not benchmark.provenance_complete(incomplete)
    assert not benchmark.provenance_complete(empty)


def test_evaluate_aggregates_and_groups(tmp_path):
    golden_path = _golden(tmp_path / "golden.jsonl")
    runs_path = _runs(tmp_path / "runs.jsonl")

    golden = benchmark.load_golden_set(golden_path)
    runs = benchmark.load_runs(runs_path)
    report = benchmark.evaluate(runs, golden, k=4)

    assert report.k == 4
    assert report.query_count == 2
    # q1: both chunks within k=4, expected found at rank 1 -> recall 1, rr 1.
    # q2: one of two expected sources -> recall 0.5, rr 1.
    assert report.aggregate.recall_at_k == pytest.approx(0.75)
    assert report.aggregate.mrr == pytest.approx(1.0)
    assert report.aggregate.provenance_complete_rate == 1.0
    assert report.aggregate.empty_run_rate == 0.0
    strategies = {group.name: group for group in report.groups}
    assert strategies["default"].recall_at_k == pytest.approx(1.0)
    assert strategies["hybrid"].recall_at_k == pytest.approx(0.5)


def test_evaluate_rejects_unknown_query_ids(tmp_path):
    golden_path = _golden(tmp_path / "golden.jsonl")
    runs_path = _write_jsonl(
        tmp_path / "runs.jsonl",
        [
            {
                "query_id": "mystery",
                "strategy": "default",
                "top_k": 4,
                "chunks": [],
            }
        ],
    )

    with pytest.raises(ValueError, match="unknown query ids"):
        benchmark.evaluate(
            benchmark.load_runs(runs_path), benchmark.load_golden_set(golden_path), k=4
        )


def test_load_golden_set_rejects_duplicates_and_empty_sources(tmp_path):
    dup = _write_jsonl(
        tmp_path / "dup.jsonl",
        [
            {"query_id": "q", "question": "a", "expected_sources": ["https://x.test/1"]},
            {"query_id": "q", "question": "b", "expected_sources": ["https://x.test/2"]},
        ],
    )
    with pytest.raises(ValueError, match="Duplicate query_id"):
        benchmark.load_golden_set(dup)

    no_sources = _write_jsonl(
        tmp_path / "empty_sources.jsonl",
        [{"query_id": "q", "question": "a", "expected_sources": []}],
    )
    with pytest.raises(ValueError, match="expected_sources must not be empty"):
        benchmark.load_golden_set(no_sources)


def test_capture_runs_records_sources(tmp_path):
    from langchain_core.documents import Document

    golden_path = _golden(tmp_path / "golden.jsonl")
    golden = benchmark.load_golden_set(golden_path)

    class FakeProvider:
        def __init__(self) -> None:
            self.calls = 0

        def retrieve(self, query: str, k: int = 4):
            self.calls += 1
            return [
                Document(page_content="chunk body", metadata={"source": "https://example.test/a"})
            ]

    fake = FakeProvider()
    runs = benchmark.capture_runs(
        golden=golden,
        top_k=4,
        strategy="test",
        provider_factory=lambda: fake,
    )

    assert len(runs) == len(golden)
    assert all(run.strategy == "test" for run in runs)
    assert all(len(run.chunks) == 1 for run in runs)
    assert runs[0].chunks[0].source == "https://example.test/a"
    assert runs[0].chunks[0].text_head == "chunk body"


def test_write_and_reload_runs_roundtrip(tmp_path):
    golden_path = _golden(tmp_path / "golden.jsonl")
    golden = benchmark.load_golden_set(golden_path)
    runs = benchmark.capture_runs(
        golden=golden,
        top_k=2,
        strategy="roundtrip",
        provider_factory=lambda: type(
            "P",
            (),
            {
                "retrieve": lambda self, query, k=4: [
                    Document(page_content="body", metadata={"source": "s"})
                ]
            },
        )(),
    )
    out = tmp_path / "nested" / "runs.jsonl"
    benchmark.write_runs(runs, out)
    reloaded = benchmark.load_runs(out)
    assert reloaded == runs


def test_render_table_contains_groups_and_metrics(tmp_path):
    golden_path = _golden(tmp_path / "golden.jsonl")
    runs_path = _runs(tmp_path / "runs.jsonl")
    report = benchmark.evaluate(
        benchmark.load_runs(runs_path),
        benchmark.load_golden_set(golden_path),
        k=4,
    )
    table = benchmark.render_table(report)
    assert "recall@k" in table
    assert "default" in table and "hybrid" in table
    assert "all" in table


def test_main_evaluate_json_output(tmp_path, capsys):
    golden_path = _golden(tmp_path / "golden.jsonl")
    runs_path = _runs(tmp_path / "runs.jsonl")
    exit_code = benchmark.main(
        [
            "evaluate",
            "--runs",
            str(runs_path),
            "--golden",
            str(golden_path),
            "--k",
            "4",
            "--format",
            "json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["k"] == 4
    assert payload["aggregate"]["run_count"] == 2
