from __future__ import annotations

import re
import statistics
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..config import Settings

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_MAX_VALUES = 100_000


class StatisticsInput(BaseModel):
    """Input schema for the descriptive statistics tool."""

    numbers: str = Field(
        ...,
        min_length=1,
        description=(
            "A list of numbers separated by commas or spaces, e.g. "
            "'4, 8, 15, 16, 23, 42'."
        ),
    )


def build_statistics_tool(
    _settings: Settings,
) -> BaseTool:
    """Create a descriptive-statistics tool over a list of numbers."""

    def _run_statistics(numbers: str) -> str:
        return compute_statistics(numbers)

    return StructuredTool.from_function(
        func=_run_statistics,
        name="compute_statistics",
        description=(
            "Compute descriptive statistics for a list of numbers: count, sum, "
            "mean, median, mode, variance, standard deviation, min, max, range, "
            "and quartiles. Use for 'average', 'median', 'standard deviation', "
            "spread, or summary-statistics questions over a dataset."
        ),
        args_schema=StatisticsInput,
    )


def compute_statistics(numbers: str) -> str:
    values = _parse_numbers(numbers)
    if not values:
        return "Statistics requires at least one number."
    if len(values) > _MAX_VALUES:
        return f"Too many values ({len(values):,}); limit is {_MAX_VALUES:,}."

    count = len(values)
    total = sum(values)
    mean = statistics.fmean(values)
    minimum = min(values)
    maximum = max(values)

    rows: list[tuple[str, str]] = [
        ("Count", str(count)),
        ("Sum", _fmt(total)),
        ("Mean", _fmt(mean)),
        ("Median", _fmt(statistics.median(values))),
        ("Mode", _mode(values)),
        ("Min", _fmt(minimum)),
        ("Max", _fmt(maximum)),
        ("Range", _fmt(maximum - minimum)),
    ]

    if count >= 2:
        rows.append(("Variance (sample)", _fmt(statistics.variance(values))))
        rows.append(("Std dev (sample)", _fmt(statistics.stdev(values))))
        rows.append(("Variance (pop.)", _fmt(statistics.pvariance(values))))
        rows.append(("Std dev (pop.)", _fmt(statistics.pstdev(values))))
    quartiles = _quartiles(values)
    if quartiles is not None:
        rows.append(("Quartiles (Q1, Q2, Q3)", quartiles))

    return "Descriptive statistics:\n\n" + _markdown_table(["Statistic", "Value"], rows)


def _parse_numbers(raw: str) -> list[float]:
    matches = _NUMBER_RE.findall(raw or "")
    values: list[float] = []
    for token in matches:
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _mode(values: list[float]) -> str:
    try:
        modes = statistics.multimode(values)
    except statistics.StatisticsError:
        return "n/a"
    if not modes or len(modes) == len(set(values)):
        return "no unique mode"
    return ", ".join(_fmt(m) for m in modes)


def _quartiles(values: list[float]) -> str | None:
    if len(values) < 2:
        return None
    try:
        # n=4 yields the three cut points Q1, Q2, Q3.
        cuts = statistics.quantiles(values, n=4, method="inclusive")
    except statistics.StatisticsError:
        return None
    return ", ".join(_fmt(c) for c in cuts)


def _fmt(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.6g}"


def _markdown_table(headers: list[str], rows: list[tuple[str, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


__all__ = [
    "StatisticsInput",
    "build_statistics_tool",
    "compute_statistics",
]
