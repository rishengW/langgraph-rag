from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.config import Settings

_OPERATIONS = ("now", "difference", "add", "convert_timezone", "weekday")


class DateTimeInput(BaseModel):
    """Input schema for the date/time tool."""

    operation: str = Field(
        ...,
        description=(
            "What to compute: 'now' (current date/time, optionally in a "
            "timezone), 'difference' (between two datetimes), 'add' (offset a "
            "datetime by days/hours/minutes), 'convert_timezone' (between two "
            "zones), or 'weekday' (day of week for a date)."
        ),
    )
    start: str | None = Field(
        default=None,
        description="Primary date/time, ISO format e.g. '2026-06-30' or "
        "'2026-06-30T15:00'. Required for all operations except 'now'.",
    )
    end: str | None = Field(
        default=None,
        description="Second date/time for 'difference', ISO format.",
    )
    days: int = Field(default=0, description="Days to add (can be negative) for 'add'.")
    hours: int = Field(default=0, description="Hours to add (can be negative) for 'add'.")
    minutes: int = Field(
        default=0, description="Minutes to add (can be negative) for 'add'."
    )
    timezone_name: str | None = Field(
        default=None,
        description="IANA timezone, e.g. 'Asia/Tokyo'. Source zone for "
        "'convert_timezone' / 'now', or the zone of 'start'.",
    )
    to_timezone: str | None = Field(
        default=None,
        description="Target IANA timezone for 'convert_timezone'.",
    )


def build_datetime_tool(
    _settings: Settings,
) -> BaseTool:
    """Create a stdlib-backed date/time calculation tool."""

    def _run_datetime(
        operation: str,
        start: str | None = None,
        end: str | None = None,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        timezone_name: str | None = None,
        to_timezone: str | None = None,
    ) -> str:
        return calculate_datetime(
            operation=operation,
            start=start,
            end=end,
            days=days,
            hours=hours,
            minutes=minutes,
            timezone_name=timezone_name,
            to_timezone=to_timezone,
        )

    return StructuredTool.from_function(
        func=_run_datetime,
        name="calculate_datetime",
        description=(
            "Perform exact date and time calculations: current time in a "
            "timezone, the difference between two dates/times, adding or "
            "subtracting days/hours/minutes, converting a time between "
            "timezones, or the weekday of a date. Use for any date math or "
            "timezone question instead of computing it yourself."
        ),
        args_schema=DateTimeInput,
    )


def calculate_datetime(
    *,
    operation: str,
    start: str | None = None,
    end: str | None = None,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    timezone_name: str | None = None,
    to_timezone: str | None = None,
) -> str:
    op = (operation or "").strip().lower()
    if op not in _OPERATIONS:
        return f"Unsupported operation {operation!r}. Use one of: {', '.join(_OPERATIONS)}."

    try:
        if op == "now":
            return _do_now(timezone_name)
        if op == "difference":
            return _do_difference(start, end)
        if op == "add":
            return _do_add(start, days, hours, minutes)
        if op == "convert_timezone":
            return _do_convert(start, timezone_name, to_timezone)
        return _do_weekday(start)
    except _DateError as exc:
        return f"Date calculation failed: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        return f"Date calculation failed: {exc}"


class _DateError(Exception):
    """Internal error with a user-readable message."""


def _zone(name: str | None) -> ZoneInfo | None:
    if not name or not name.strip():
        return None
    try:
        return ZoneInfo(name.strip())
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise _DateError(f"unknown timezone {name!r}") from exc


def _parse_dt(value: str | None, *, field: str = "start") -> datetime:
    text = (value or "").strip()
    if not text:
        raise _DateError(f"{field} date/time is required.")
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    # Fall back to date-only.
    try:
        parsed_date = date.fromisoformat(text)
    except ValueError as exc:
        raise _DateError(
            f"could not parse {field}={value!r}; use ISO format like "
            "'2026-06-30' or '2026-06-30T15:00'."
        ) from exc
    return datetime(parsed_date.year, parsed_date.month, parsed_date.day)


def _do_now(timezone_name: str | None) -> str:
    zone = _zone(timezone_name)
    now = datetime.now(zone) if zone else datetime.now()
    label = f" ({timezone_name.strip()})" if zone and timezone_name else ""
    return f"Current date/time{label}: {now.isoformat(timespec='seconds')} ({now.strftime('%A')})"


def _do_difference(start: str | None, end: str | None) -> str:
    start_dt = _parse_dt(start, field="start")
    end_dt = _parse_dt(end, field="end")
    # If exactly one side is timezone-aware, treat the naive one as UTC so the
    # subtraction does not raise.
    start_dt, end_dt = _align_awareness(start_dt, end_dt)
    delta = end_dt - start_dt
    total_seconds = int(delta.total_seconds())
    sign = "" if total_seconds >= 0 else "-"
    secs = abs(total_seconds)
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    return (
        f"Difference between {start_dt.isoformat()} and {end_dt.isoformat()}: "
        f"{sign}{days} day(s), {hours} hour(s), {minutes} minute(s) "
        f"(total {sign}{abs(delta.days)} days)"
    )


def _do_add(start: str | None, days: int, hours: int, minutes: int) -> str:
    start_dt = _parse_dt(start, field="start")
    result = start_dt + timedelta(days=days, hours=hours, minutes=minutes)
    return (
        f"{start_dt.isoformat()} + ({days}d {hours}h {minutes}m) = "
        f"{result.isoformat()} ({result.strftime('%A')})"
    )


def _do_convert(
    start: str | None,
    timezone_name: str | None,
    to_timezone: str | None,
) -> str:
    from_zone = _zone(timezone_name)
    to_zone = _zone(to_timezone)
    if from_zone is None or to_zone is None:
        raise _DateError(
            "convert_timezone requires both timezone_name (source) and "
            "to_timezone (target)."
        )
    assert timezone_name is not None
    assert to_timezone is not None
    start_dt = _parse_dt(start, field="start")
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=from_zone)
    converted = start_dt.astimezone(to_zone)
    return (
        f"{start_dt.isoformat()} ({timezone_name.strip()}) = "
        f"{converted.isoformat()} ({to_timezone.strip()})"
    )


def _do_weekday(start: str | None) -> str:
    start_dt = _parse_dt(start, field="start")
    return f"{start_dt.date().isoformat()} is a {start_dt.strftime('%A')}."


def _align_awareness(a: datetime, b: datetime) -> tuple[datetime, datetime]:
    a_aware = a.tzinfo is not None
    b_aware = b.tzinfo is not None
    if a_aware == b_aware:
        return a, b
    if not a_aware:
        a = a.replace(tzinfo=UTC)
    if not b_aware:
        b = b.replace(tzinfo=UTC)
    return a, b


__all__ = [
    "DateTimeInput",
    "build_datetime_tool",
    "calculate_datetime",
]
