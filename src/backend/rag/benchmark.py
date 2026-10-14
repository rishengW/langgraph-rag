"""Offline RAG retrieval benchmark: capture runs, then score recall/MRR.

Capture loads the persisted Chroma store (no rebuild, no network beyond the
embedding provider) and records what ``ChromaRetriever.retrieve`` returns for
each golden-set question. Evaluate is fully offline: it scores the captured
runs against the golden sources and reports recall@k, MRR, and a provenance
check (every retrieved chunk must carry a non-empty ``source``).

Layout mirrors ``src/backend/web_search/benchmark.py`` so both benchmarks
feel the same to operate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

DEFAULT_GOLDEN_SET = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "rag_quality" / "golden_set.jsonl"
)
DEFAULT_RUNS_PATH = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "rag_quality" / "runs.jsonl"
)
RETRIEVED_TEXT_HEAD_CHARS = 200


@dataclass(frozen=True)
class GoldenQuery:
    """One benchmark question with the sources expected to ground its answer."""

    query_id: str
    question: str
    expected_sources: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class RetrievedChunk:
    """One ranked chunk as captured from the retriever."""

    rank: int
    source: str
    text_head: str


@dataclass(frozen=True)
class RetrievalRun:
    """Captured retriever output for one golden question."""

    query_id: str
    strategy: str
    top_k: int
    chunks: tuple[RetrievedChunk, ...]


@dataclass(frozen=True)
class MetricSummary:
    """Retrieval-quality metrics for a set of runs."""

    name: str
    run_count: int
    recall_at_k: float
    mrr: float
    provenance_complete_rate: float
    empty_run_rate: float


@dataclass(frozen=True)
class BenchmarkReport:
    """Complete benchmark output."""

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


def _string_list(
    record: Mapping[str, object],
    field: str,
    *,
    context: str,
) -> tuple[str, ...]:
    value = record.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{context}.{field} must be a JSON array of strings")
    return tuple(str(item).strip() for item in value if str(item).strip())


def _rank(chunk: Mapping[str, object], position: int) -> int:
    value = chunk.get("rank", position + 1)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"chunk {position}.rank must be an integer")
    return value


def _positive_int(record: Mapping[str, object], field: str, *, context: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{context}.{field} must be a positive integer")
    return value


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


def load_golden_set(path: Path = DEFAULT_GOLDEN_SET) -> tuple[GoldenQuery, ...]:
    """Load and validate the golden question manifest."""

    queries: list[GoldenQuery] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(_load_jsonl(path), 1):
        context = f"{path}:record {index}"
        query = GoldenQuery(
            query_id=_string(record, "query_id", context=context),
            question=_string(record, "question", context=context),
            expected_sources=_string_list(record, "expected_sources", context=context),
            note=str(record.get("note", "") or ""),
        )
        if query.query_id in seen_ids:
            raise ValueError(f"Duplicate query_id: {query.query_id}")
        if not query.expected_sources:
            raise ValueError(f"{context}.expected_sources must not be empty")
        seen_ids.add(query.query_id)
        queries.append(query)

    if not queries:
        raise ValueError(f"{path} contains no golden queries")
    return tuple(queries)


def load_runs(path: Path) -> tuple[RetrievalRun, ...]:
    """Load captured retriever runs from newline-delimited JSON."""

    runs: list[RetrievalRun] = []
    seen: set[tuple[str, str, int]] = set()
    for index, record in enumerate(_load_jsonl(path), 1):
        context = f"{path}:record {index}"
        raw_chunks = record.get("chunks")
        if not isinstance(raw_chunks, list):
            raise ValueError(f"{context}.chunks must be a JSON array")
        chunks = tuple(
            RetrievedChunk(
                rank=_rank(chunk, position),
                source=_string(chunk, "source", context=f"{context}.chunk {position}"),
                text_head=str(chunk.get("text_head", "") or ""),
            )
            for position, chunk in enumerate(
                _object(item, context=f"{context}.chunk") for item in raw_chunks
            )
        )
        run = RetrievalRun(
            query_id=_string(record, "query_id", context=context),
            strategy=_string(record, "strategy", context=context),
            top_k=_positive_int(record, "top_k", context=context),
            chunks=chunks,
        )
        key = (run.query_id, run.strategy, run.top_k)
        if key in seen:
            raise ValueError(f"Duplicate run for {run.query_id}/{run.strategy}@{run.top_k}")
        seen.add(key)
        runs.append(run)
    if not runs:
        raise ValueError(f"{path} contains no runs")
    return tuple(runs)


def _normalized_source(source: str) -> str:
    """Compare sources loosely: strip trailing punctuation and query strings.

    Captured sources are file paths or URLs; a golden URL with tracking
    parameters must still match a chunk stored under the canonical link.
    """

    text = (source or "").strip().rstrip(".,;)")
    if "?" in text:
        text = text.split("?", 1)[0]
    return text.rstrip("/").casefold()


def recall_at_k(run: RetrievalRun, expected: Sequence[str], *, k: int) -> float:
    """Fraction of expected sources present in the run's top-k chunks."""

    if not expected:
        return 0.0
    expected_set = {_normalized_source(item) for item in expected}
    seen = {_normalized_source(chunk.source) for chunk in run.chunks[:k]}
    hits = sum(1 for item in expected_set if item in seen)
    return hits / len(expected_set)


def reciprocal_rank(run: RetrievalRun, expected: Sequence[str], *, k: int) -> float:
    """1/rank of the first expected source within the top-k chunks."""

    if not expected:
        return 0.0
    expected_set = {_normalized_source(item) for item in expected}
    for position, chunk in enumerate(run.chunks[:k], 1):
        if _normalized_source(chunk.source) in expected_set:
            return 1.0 / position
    return 0.0


def provenance_complete(run: RetrievalRun) -> bool:
    """True when every retrieved chunk names its source."""

    return bool(run.chunks) and all(chunk.source.strip() for chunk in run.chunks)


def summarize(
    name: str,
    runs: Sequence[tuple[RetrievalRun, GoldenQuery]],
    *,
    k: int,
) -> MetricSummary:
    """Aggregate recall@k, MRR, provenance completeness, and emptiness."""

    if not runs:
        return MetricSummary(name, 0, 0.0, 0.0, 0.0, 0.0)
    recalls = [recall_at_k(run, query.expected_sources, k=k) for run, query in runs]
    rrs = [reciprocal_rank(run, query.expected_sources, k=k) for run, query in runs]
    provenance = [1.0 if provenance_complete(run) else 0.0 for run, _ in runs]
    empties = [1.0 if not run.chunks else 0.0 for run, _ in runs]
    return MetricSummary(
        name=name,
        run_count=len(runs),
        recall_at_k=sum(recalls) / len(recalls),
        mrr=sum(rrs) / len(rrs),
        provenance_complete_rate=sum(provenance) / len(provenance),
        empty_run_rate=sum(empties) / len(empties),
    )


def evaluate(
    runs: Sequence[RetrievalRun],
    golden: Sequence[GoldenQuery],
    *,
    k: int,
) -> BenchmarkReport:
    """Score every captured run against its golden question."""

    golden_by_id = {query.query_id: query for query in golden}
    unknown = sorted({run.query_id for run in runs} - set(golden_by_id))
    if unknown:
        raise ValueError(f"Runs reference unknown query ids: {', '.join(unknown)}")

    paired: list[tuple[RetrievalRun, GoldenQuery]] = [
        (run, golden_by_id[run.query_id]) for run in runs
    ]
    grouped: dict[str, list[tuple[RetrievalRun, GoldenQuery]]] = defaultdict(list)
    for pair in paired:
        grouped[pair[0].strategy].append(pair)

    return BenchmarkReport(
        k=k,
        query_count=len({run.query_id for run in runs}),
        aggregate=summarize("all", paired, k=k),
        groups=tuple(
            summarize(strategy, pairs, k=k) for strategy, pairs in sorted(grouped.items())
        ),
    )


def capture_runs(
    *,
    golden: Sequence[GoldenQuery],
    top_k: int,
    strategy: str,
    provider_factory: Callable[[], Any] | None = None,
) -> tuple[RetrievalRun, ...]:
    """Run the persisted-store retriever over every golden question.

    ``provider_factory`` defaults to building a ``ChromaRetriever`` from
    ``load_settings``; tests inject a fake. Requires the embedding provider
    (DASHSCOPE_API_KEY) but never rebuilds the index.
    """

    runs: list[RetrievalRun] = []
    for query in golden:
        factory = provider_factory or _default_provider
        provider = factory()
        documents = provider.retrieve(query.question, k=top_k)
        chunks = tuple(
            RetrievedChunk(
                rank=position,
                source=str(document.metadata.get("source", "") or ""),
                text_head=document.page_content[:RETRIEVED_TEXT_HEAD_CHARS],
            )
            for position, document in enumerate(documents, 1)
        )
        runs.append(
            RetrievalRun(query_id=query.query_id, strategy=strategy, top_k=top_k, chunks=chunks)
        )
    return tuple(runs)


def _default_provider() -> Any:
    from src.backend.rag.chroma_retriever import ChromaRetriever
    from src.config import load_settings

    return ChromaRetriever(load_settings())


def write_runs(runs: Sequence[RetrievalRun], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for run in runs:
            payload = {
                "query_id": run.query_id,
                "strategy": run.strategy,
                "top_k": run.top_k,
                "chunks": [asdict(chunk) for chunk in run.chunks],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _format_summary_row(summary: MetricSummary) -> str:
    return (
        f"{summary.name:<24} {summary.run_count:>4} "
        f"{summary.recall_at_k:>8.3f} {summary.mrr:>8.3f} "
        f"{summary.provenance_complete_rate:>10.3f} {summary.empty_run_rate:>9.3f}"
    )


def render_table(report: BenchmarkReport) -> str:
    header = f"{'group':<24} {'runs':>4} {'recall@k':>8} {'mrr':>8} {'provenance':>10} {'empty':>9}"
    lines = [
        f"k={report.k}, queries={report.query_count}",
        header,
        "-" * len(header),
        _format_summary_row(report.aggregate),
    ]
    lines.extend(_format_summary_row(group) for group in report.groups)
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG retrieval quality benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture",
        help="run the retriever over the golden set and write runs.jsonl",
    )
    capture.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_SET)
    capture.add_argument("--out", type=Path, default=DEFAULT_RUNS_PATH)
    capture.add_argument("--top-k", type=int, default=4)
    capture.add_argument("--strategy", default="default")

    evaluate_parser = subparsers.add_parser("evaluate", help="score captured runs")
    evaluate_parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS_PATH)
    evaluate_parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_SET)
    evaluate_parser.add_argument("--k", type=int, default=4)
    evaluate_parser.add_argument("--format", choices=("table", "json"), default="table")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "capture":
        golden = load_golden_set(args.golden)
        runs = capture_runs(
            golden=golden,
            top_k=max(1, int(args.top_k)),
            strategy=args.strategy,
        )
        write_runs(runs, args.out)
        print(f"Wrote {len(runs)} run(s) to {args.out}")
        return 0

    golden = load_golden_set(args.golden)
    runs = load_runs(args.runs)
    report = evaluate(runs, golden, k=max(1, int(args.k)))
    if args.format == "json":
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
