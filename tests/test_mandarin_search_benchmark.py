from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.backend.web_search.benchmark import (
    ALLOWED_CATEGORIES,
    DEFAULT_QUERY_SET,
    BenchmarkQuery,
    RelevanceJudgment,
    SearchHit,
    SearchRun,
    evaluate,
    format_report,
    load_judgments,
    load_queries,
    load_runs,
    main,
)


def _queries() -> tuple[BenchmarkQuery, ...]:
    return (
        BenchmarkQuery("q1", "count", "南京地铁有多少条线路？", "2025-12-31"),
        BenchmarkQuery("q2", "policy", "充电宝乘机规定是什么？", "2025-07-01"),
    )


def _runs() -> tuple[SearchRun, ...]:
    return (
        SearchRun(
            query_id="q1",
            provider="bing",
            strategy="exact",
            query="南京地铁 运营线路 总数",
            latency_ms=100.0,
            results=(
                SearchHit(1, "https://metro.example.cn/official"),
                SearchHit(2, "https://news.example.cn/relevant"),
                SearchHit(3, "https://spam.example.cn/unrelated"),
            ),
        ),
        SearchRun(
            query_id="q2",
            provider="bing",
            strategy="exact",
            query="充电宝 民航 规定",
            latency_ms=300.0,
            results=(
                SearchHit(1, "https://caac.example.cn/rules"),
                SearchHit(2, "https://shop.example.cn/power-bank"),
            ),
        ),
    )


def _judgments() -> tuple[RelevanceJudgment, ...]:
    return (
        RelevanceJudgment("q1", "https://metro.example.cn/official", True, True, True),
        RelevanceJudgment("q1", "https://news.example.cn/relevant", True, False, False),
        RelevanceJudgment("q1", "https://spam.example.cn/unrelated", False, False, False),
        RelevanceJudgment("q2", "https://caac.example.cn/rules", True, True, False),
        RelevanceJudgment("q2", "https://shop.example.cn/power-bank", False, False, False),
    )


@pytest.mark.skipif(
    not DEFAULT_QUERY_SET.exists(),
    reason="benchmarks/ is gitignored; the manifest only exists in local data checkouts",
)
def test_default_manifest_covers_all_mandarin_intent_categories() -> None:
    queries = load_queries(DEFAULT_QUERY_SET)

    assert len(queries) >= 15
    assert {query.category for query in queries} == ALLOWED_CATEGORIES
    assert all(any("\u3400" <= char <= "\u9fff" for char in query.question) for query in queries)
    assert len({query.query_id for query in queries}) == len(queries)


def test_evaluate_reports_quality_coverage_and_latency() -> None:
    report = evaluate(_queries(), _runs(), _judgments(), k=3)
    summary = report.aggregate

    assert report.query_count == 2
    assert summary.run_count == 2
    assert summary.precision_at_k == pytest.approx(0.5)
    assert summary.official_source_rate_at_k == pytest.approx(0.4)
    assert summary.answerable_at_k == pytest.approx(0.5)
    assert summary.irrelevant_url_rate == pytest.approx(0.4)
    assert summary.judged_result_count == 5
    assert summary.unjudged_result_count == 0
    assert summary.latency_mean_ms == 200.0
    assert summary.latency_median_ms == 200.0
    assert summary.latency_p95_ms == 300.0
    assert summary.latency_min_ms == 100.0
    assert summary.latency_max_ms == 300.0
    assert [group.name for group in report.groups] == ["bing/exact"]


def test_evaluate_requires_labels_unless_exploratory_mode_is_explicit() -> None:
    incomplete = _judgments()[:-1]

    with pytest.raises(ValueError, match="lack judgments"):
        evaluate(_queries(), _runs(), incomplete)

    report = evaluate(_queries(), _runs(), incomplete, allow_unjudged=True)

    assert report.aggregate.unjudged_result_count == 1
    assert report.aggregate.irrelevant_url_rate == pytest.approx(0.4)


def test_evaluate_rejects_an_empty_capture() -> None:
    with pytest.raises(ValueError, match="At least one captured provider run"):
        evaluate(_queries(), (), _judgments())


def test_loaders_validate_jsonl_and_normalize_equivalent_urls(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.jsonl"
    judgments_path = tmp_path / "judgments.jsonl"
    runs_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "provider": "BING",
                "strategy": "EXACT",
                "query": "南京地铁",
                "latency_ms": 10,
                "results": [{"rank": 1, "url": "https://EXAMPLE.cn:443/path/", "title": "标题"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    judgments_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "url": "https://example.cn/path",
                "relevant": True,
                "official": False,
                "answerable": True,
            }
        ),
        encoding="utf-8",
    )

    runs = load_runs(runs_path)
    judgments = load_judgments(judgments_path)
    report = evaluate((_queries()[0],), runs, judgments)

    assert runs[0].provider == "bing"
    assert runs[0].strategy == "exact"
    assert report.aggregate.precision_at_k == pytest.approx(1 / 3)


def test_table_and_json_cli_outputs_are_deterministic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    queries_path = tmp_path / "queries.json"
    runs_path = tmp_path / "runs.jsonl"
    judgments_path = tmp_path / "judgments.jsonl"
    queries_path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "category": "count",
                    "question": "南京地铁有多少条线路？",
                    "as_of_date": "2025-12-31",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    runs_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "provider": "bing",
                "strategy": "exact",
                "query": "南京地铁线路总数",
                "latency_ms": 25,
                "results": [{"rank": 1, "url": "https://example.cn/metro"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    judgments_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "url": "https://example.cn/metro",
                "relevant": True,
                "official": True,
                "answerable": True,
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--queries",
                str(queries_path),
                "--runs",
                str(runs_path),
                "--judgments",
                str(judgments_path),
                "--format",
                "json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["aggregate"]["precision_at_k"] == pytest.approx(1 / 3)
    assert "P@3" in format_report(
        evaluate(load_queries(queries_path), load_runs(runs_path), load_judgments(judgments_path))
    )
