from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit, urlunsplit

DEFAULT_QUERY_SET = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "mandarin_search" / "queries.json"
)
ALLOWED_CATEGORIES = frozenset({"comparison", "count", "current_status", "date", "policy", "price"})


@dataclass(frozen=True)
class BenchmarkQuery:
    """One stable Mandarin question in the offline benchmark."""

    query_id: str
    category: str
    question: str
    as_of_date: str


@dataclass(frozen=True)
class SearchHit:
    """A provider result in the order presented to the search pipeline."""

    rank: int
    url: str
    title: str = ""
    snippet: str = ""


@dataclass(frozen=True)
class SearchRun:
    """Captured output from one provider/query-strategy invocation."""

    query_id: str
    provider: str
    strategy: str
    query: str
    latency_ms: float
    results: tuple[SearchHit, ...]

    @property
    def group_name(self) -> str:
        return f"{self.provider}/{self.strategy}"


@dataclass(frozen=True)
class RelevanceJudgment:
    """Human judgment for one benchmark question and normalized URL."""

    query_id: str
    url: str
    relevant: bool
    official: bool
    answerable: bool


@dataclass(frozen=True)
class MetricSummary:
    """Search-quality and latency metrics for a set of captured runs."""

    name: str
    run_count: int
    precision_at_k: float
    official_source_rate_at_k: float
    answerable_at_k: float
    irrelevant_url_rate: float
    judged_result_count: int
    unjudged_result_count: int
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_min_ms: float
    latency_max_ms: float


@dataclass(frozen=True)
class BenchmarkReport:
    """Complete benchmark output, including aggregate and group summaries."""

    k: int
    query_count: int
    aggregate: MetricSummary
    groups: tuple[MetricSummary, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "query_count": self.query_count,
            "aggregate": asdict(self.aggregate),
            "groups": [asdict(group) for group in self.groups],
        }


def _object(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return cast(dict[str, object], value)


def _string(record: Mapping[str, object], field: str, *, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{field} must be a non-empty string")
    return value.strip()


def _boolean(record: Mapping[str, object], field: str, *, context: str) -> bool:
    value = record.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"{context}.{field} must be a boolean")
    return value


def _number(record: Mapping[str, object], field: str, *, context: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}.{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{context}.{field} must be a finite non-negative number")
    return number


def _optional_string(record: Mapping[str, object], field: str) -> str:
    value = record.get(field, "")
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string when provided")
    return value.strip()


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
        records.append(_object(value, context=f"{path}:{line_number}"))
    return records


def load_queries(path: Path = DEFAULT_QUERY_SET) -> tuple[BenchmarkQuery, ...]:
    """Load and validate the benchmark query manifest."""

    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")

    queries: list[BenchmarkQuery] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload):
        context = f"{path}[{index}]"
        record = _object(item, context=context)
        query = BenchmarkQuery(
            query_id=_string(record, "query_id", context=context),
            category=_string(record, "category", context=context),
            question=_string(record, "question", context=context),
            as_of_date=_string(record, "as_of_date", context=context),
        )
        if query.query_id in seen_ids:
            raise ValueError(f"Duplicate query_id: {query.query_id}")
        if query.category not in ALLOWED_CATEGORIES:
            raise ValueError(f"Unsupported category {query.category!r} for query {query.query_id}")
        seen_ids.add(query.query_id)
        queries.append(query)

    if not queries:
        raise ValueError(f"{path} contains no benchmark queries")
    return tuple(queries)


def load_runs(path: Path) -> tuple[SearchRun, ...]:
    """Load provider result records from newline-delimited JSON."""

    runs: list[SearchRun] = []
    seen_runs: set[tuple[str, str, str]] = set()
    for index, record in enumerate(_load_jsonl(path), 1):
        context = f"{path}:record {index}"
        raw_results = record.get("results")
        if not isinstance(raw_results, list):
            raise ValueError(f"{context}.results must be a JSON array")

        hits: list[SearchHit] = []
        seen_ranks: set[int] = set()
        for result_index, raw_hit in enumerate(raw_results):
            hit_context = f"{context}.results[{result_index}]"
            hit_record = _object(raw_hit, context=hit_context)
            raw_rank = hit_record.get("rank")
            if isinstance(raw_rank, bool) or not isinstance(raw_rank, int) or raw_rank < 1:
                raise ValueError(f"{hit_context}.rank must be a positive integer")
            if raw_rank in seen_ranks:
                raise ValueError(f"{context} has duplicate result rank {raw_rank}")
            url = _string(hit_record, "url", context=hit_context)
            _normalized_url(url)
            seen_ranks.add(raw_rank)
            hits.append(
                SearchHit(
                    rank=raw_rank,
                    url=url,
                    title=_optional_string(hit_record, "title"),
                    snippet=_optional_string(hit_record, "snippet"),
                )
            )

        run = SearchRun(
            query_id=_string(record, "query_id", context=context),
            provider=_string(record, "provider", context=context).lower(),
            strategy=_string(record, "strategy", context=context).lower(),
            query=_string(record, "query", context=context),
            latency_ms=_number(record, "latency_ms", context=context),
            results=tuple(sorted(hits, key=lambda hit: hit.rank)),
        )
        run_key = (run.query_id, run.provider, run.strategy)
        if run_key in seen_runs:
            raise ValueError(f"Duplicate run for query/provider/strategy: {run_key}")
        seen_runs.add(run_key)
        runs.append(run)

    if not runs:
        raise ValueError(f"{path} contains no provider runs")
    return tuple(runs)


def load_judgments(path: Path) -> tuple[RelevanceJudgment, ...]:
    """Load human URL judgments from newline-delimited JSON."""

    judgments: list[RelevanceJudgment] = []
    seen: set[tuple[str, str]] = set()
    for index, record in enumerate(_load_jsonl(path), 1):
        context = f"{path}:record {index}"
        judgment = RelevanceJudgment(
            query_id=_string(record, "query_id", context=context),
            url=_string(record, "url", context=context),
            relevant=_boolean(record, "relevant", context=context),
            official=_boolean(record, "official", context=context),
            answerable=_boolean(record, "answerable", context=context),
        )
        _normalized_url(judgment.url)
        if judgment.answerable and not judgment.relevant:
            raise ValueError(f"{context}: answerable=true requires relevant=true")
        key = (judgment.query_id, _normalized_url(judgment.url))
        if key in seen:
            raise ValueError(f"Duplicate judgment for query/URL: {key}")
        seen.add(key)
        judgments.append(judgment)
    return tuple(judgments)


def _normalized_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Expected an absolute HTTP(S) URL, got {url!r}")

    hostname = parsed.hostname.lower()
    port = parsed.port
    default_port = (parsed.scheme.lower() == "http" and port == 80) or (
        parsed.scheme.lower() == "https" and port == 443
    )
    netloc = hostname if port is None or default_port else f"{hostname}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), netloc, path, parsed.query, ""))


def evaluate(
    queries: Sequence[BenchmarkQuery],
    runs: Sequence[SearchRun],
    judgments: Sequence[RelevanceJudgment],
    *,
    k: int = 3,
    allow_unjudged: bool = False,
) -> BenchmarkReport:
    """Join captured runs to judgments and calculate deterministic metrics."""

    if k < 1:
        raise ValueError("k must be at least 1")
    if not runs:
        raise ValueError("At least one captured provider run is required")
    query_ids = {query.query_id for query in queries}
    unknown_query_ids = sorted({run.query_id for run in runs} - query_ids)
    if unknown_query_ids:
        raise ValueError(f"Runs reference unknown query IDs: {', '.join(unknown_query_ids)}")

    by_key: dict[tuple[str, str], RelevanceJudgment] = {}
    for judgment in judgments:
        if judgment.query_id not in query_ids:
            raise ValueError(f"Judgment references unknown query ID: {judgment.query_id}")
        key = (judgment.query_id, _normalized_url(judgment.url))
        if key in by_key:
            raise ValueError(f"Duplicate judgment for query/URL: {key}")
        by_key[key] = judgment

    missing = sorted(
        {
            (run.query_id, hit.url)
            for run in runs
            for hit in run.results
            if (run.query_id, _normalized_url(hit.url)) not in by_key
        }
    )
    if missing and not allow_unjudged:
        examples = ", ".join(f"{query_id}: {url}" for query_id, url in missing[:3])
        suffix = " ..." if len(missing) > 3 else ""
        raise ValueError(
            f"{len(missing)} result URL(s) lack judgments ({examples}{suffix}); "
            "label them or pass allow_unjudged=True"
        )

    grouped: dict[str, list[SearchRun]] = defaultdict(list)
    for run in runs:
        grouped[run.group_name].append(run)

    aggregate = _summarize("overall", runs, by_key, k=k)
    summaries = tuple(_summarize(name, grouped[name], by_key, k=k) for name in sorted(grouped))
    return BenchmarkReport(
        k=k,
        query_count=len({run.query_id for run in runs}),
        aggregate=aggregate,
        groups=summaries,
    )


def _summarize(
    name: str,
    runs: Sequence[SearchRun],
    judgments: Mapping[tuple[str, str], RelevanceJudgment],
    *,
    k: int,
) -> MetricSummary:
    relevant_at_k = 0
    official_at_k = 0
    returned_at_k = 0
    runs_with_answer = 0
    irrelevant = 0
    judged = 0
    unjudged = 0

    for run in runs:
        top_hits = run.results[:k]
        top_judgments = [
            judgments.get((run.query_id, _normalized_url(hit.url))) for hit in top_hits
        ]
        returned_at_k += len(top_hits)
        relevant_at_k += sum(bool(item and item.relevant) for item in top_judgments)
        official_at_k += sum(bool(item and item.official) for item in top_judgments)
        runs_with_answer += int(any(item and item.answerable for item in top_judgments))

        for hit in run.results:
            item = judgments.get((run.query_id, _normalized_url(hit.url)))
            if item is None:
                unjudged += 1
                irrelevant += 1
            else:
                judged += 1
                irrelevant += int(not item.relevant)

    run_count = len(runs)
    result_count = judged + unjudged
    latencies = [run.latency_ms for run in runs]
    return MetricSummary(
        name=name,
        run_count=run_count,
        precision_at_k=_ratio(relevant_at_k, run_count * k),
        official_source_rate_at_k=_ratio(official_at_k, returned_at_k),
        answerable_at_k=_ratio(runs_with_answer, run_count),
        irrelevant_url_rate=_ratio(irrelevant, result_count),
        judged_result_count=judged,
        unjudged_result_count=unjudged,
        latency_mean_ms=statistics.fmean(latencies),
        latency_median_ms=statistics.median(latencies),
        latency_p95_ms=_nearest_rank_percentile(latencies, 0.95),
        latency_min_ms=min(latencies),
        latency_max_ms=max(latencies),
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _nearest_rank_percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def format_report(report: BenchmarkReport) -> str:
    """Render a compact human-readable table."""

    headers = (
        "Group",
        "Runs",
        f"P@{report.k}",
        f"Official@{report.k}",
        f"Answerable@{report.k}",
        "Irrelevant",
        "Labels",
        "Missing",
        "Mean ms",
        "P95 ms",
    )
    rows = [report.aggregate, *report.groups]
    values = [
        (
            row.name,
            str(row.run_count),
            f"{row.precision_at_k:.3f}",
            f"{row.official_source_rate_at_k:.3f}",
            f"{row.answerable_at_k:.3f}",
            f"{row.irrelevant_url_rate:.3f}",
            str(row.judged_result_count),
            str(row.unjudged_result_count),
            f"{row.latency_mean_ms:.1f}",
            f"{row.latency_p95_ms:.1f}",
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]

    def render(row: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "  ".join("-" * width for width in widths)
    return "\n".join((render(headers), separator, *(render(row) for row in values)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate captured Mandarin provider results against human judgments."
    )
    parser.add_argument("--runs", type=Path, required=True, help="Provider-run JSONL file")
    parser.add_argument(
        "--judgments", type=Path, required=True, help="Relevance-judgment JSONL file"
    )
    parser.add_argument(
        "--queries", type=Path, default=DEFAULT_QUERY_SET, help="Benchmark query JSON file"
    )
    parser.add_argument("--k", type=int, default=3, help="Evaluation cutoff (default: 3)")
    parser.add_argument(
        "--allow-unjudged",
        action="store_true",
        help="Treat unjudged results as irrelevant and report them as missing labels",
    )
    parser.add_argument(
        "--format", choices=("table", "json"), default="table", help="Output format"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate(
        load_queries(args.queries),
        load_runs(args.runs),
        load_judgments(args.judgments),
        k=args.k,
        allow_unjudged=args.allow_unjudged,
    )
    if args.format == "json":
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
