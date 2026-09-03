from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._http import JsonRequester, request_json

if TYPE_CHECKING:
    from src.config import Settings

GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
}


class WeatherInput(BaseModel):
    """Input schema for weather lookup."""

    city: str | None = Field(
        default=None,
        description="City or place name. Use this unless coordinates are known.",
    )
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    forecast_days: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of forecast days to include, from 1 to 16.",
    )


def build_weather_tool(
    _settings: Settings,
    *,
    requester: JsonRequester | None = None,
) -> BaseTool:
    """Create an Open-Meteo weather forecast tool."""

    def _run_weather(
        city: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        forecast_days: int = 1,
    ) -> str:
        return get_weather(
            city=city,
            latitude=latitude,
            longitude=longitude,
            forecast_days=forecast_days,
            requester=requester,
        )

    return StructuredTool.from_function(
        func=_run_weather,
        name="get_weather",
        description=(
            "Get current weather and a short forecast for a city or latitude/"
            "longitude. Use for current temperature, humidity, wind, and "
            "weather forecast questions."
        ),
        args_schema=WeatherInput,
    )


def get_weather(
    *,
    city: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    forecast_days: int = 1,
    requester: JsonRequester | None = None,
) -> str:
    try:
        resolved = _resolve_location(
            city=city,
            latitude=latitude,
            longitude=longitude,
            requester=requester,
        )
    except Exception as exc:
        return f"Weather lookup failed: {exc}"

    if resolved is None:
        return "Weather lookup requires either a city or latitude and longitude."

    label, lat, lon = resolved
    try:
        payload = request_json(
            FORECAST_API_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                "forecast_days": forecast_days,
                "timezone": "auto",
            },
            requester=requester,
        )
    except Exception as exc:
        return f"Weather lookup failed for {label}: {exc}"

    return _format_weather(label, payload)


def _resolve_location(
    *,
    city: str | None,
    latitude: float | None,
    longitude: float | None,
    requester: JsonRequester | None,
) -> tuple[str, float, float] | None:
    if latitude is not None and longitude is not None:
        return f"{latitude:g},{longitude:g}", float(latitude), float(longitude)

    query = (city or "").strip()
    if not query:
        return None

    payload = request_json(
        GEOCODING_API_URL,
        params={"name": query, "count": 1, "language": "en", "format": "json"},
        requester=requester,
    )
    results = payload.get("results") or []
    if not results:
        raise ValueError(f"no matching location found for {query!r}")

    first = results[0]
    lat = float(first["latitude"])
    lon = float(first["longitude"])
    label_parts = [
        str(first.get("name") or query),
        str(first.get("admin1") or "").strip(),
        str(first.get("country") or "").strip(),
    ]
    label = ", ".join(part for part in label_parts if part)
    return label, lat, lon


def _format_weather(label: str, payload: dict[str, Any]) -> str:
    raw_current = payload.get("current")
    raw_daily = payload.get("daily")
    raw_units = payload.get("current_units")
    raw_daily_units = payload.get("daily_units")
    current: dict[str, Any] = raw_current if isinstance(raw_current, dict) else {}
    daily: dict[str, Any] = raw_daily if isinstance(raw_daily, dict) else {}
    units: dict[str, Any] = raw_units if isinstance(raw_units, dict) else {}
    daily_units: dict[str, Any] = (
        raw_daily_units if isinstance(raw_daily_units, dict) else {}
    )

    temperature = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    wind = current.get("wind_speed_10m")
    weather_code = current.get("weather_code")
    description = _weather_description(weather_code)

    # Emit a GitHub-flavored markdown table so the rendered answer is reliably
    # structured regardless of how the model decides to phrase it.
    current_rows: list[tuple[str, str]] = []
    if temperature is not None:
        unit = units.get("temperature_2m", "C")
        current_rows.append(("Current temperature", f"{temperature}{unit}"))
    if humidity is not None:
        current_rows.append(
            ("Humidity", f"{humidity}{units.get('relative_humidity_2m', '%')}")
        )
    if wind is not None:
        current_rows.append(("Wind", f"{wind}{units.get('wind_speed_10m', ' km/h')}"))
    if description:
        current_rows.append(("Conditions", description))

    sections: list[str] = [f"Weather for {label}:"]
    if current_rows:
        sections.append(_markdown_table(["Detail", "Value"], current_rows))

    forecast_table = _daily_forecast_table(daily, daily_units)
    if forecast_table:
        sections.append("Forecast:")
        sections.append(forecast_table)
    return "\n\n".join(sections)


def _markdown_table(headers: list[str], rows: Sequence[tuple[str, ...]]) -> str:
    """Render a GitHub-flavored markdown table from headers and row tuples."""

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _daily_forecast_table(
    daily: dict[str, Any],
    units: dict[str, Any],
) -> str:
    dates = daily.get("time") or []
    max_temps = daily.get("temperature_2m_max") or []
    min_temps = daily.get("temperature_2m_min") or []
    codes = daily.get("weather_code") or []
    temp_unit = units.get("temperature_2m_max", "C")

    rows: list[tuple[str, str, str]] = []
    for index, day in enumerate(dates[:5]):
        high = _list_get(max_temps, index)
        low = _list_get(min_temps, index)
        description = _weather_description(_list_get(codes, index))
        temp_range = (
            f"{low}{temp_unit}-{high}{temp_unit}"
            if high is not None and low is not None
            else ""
        )
        rows.append((str(day), temp_range, description))

    if not rows:
        return ""
    return _markdown_table(["Date", "Range", "Conditions"], rows)


def _weather_description(code: Any) -> str:
    try:
        return WEATHER_CODES.get(int(code), f"weather code {code}")
    except (TypeError, ValueError):
        return ""


def _list_get(values: Any, index: int) -> Any:
    if isinstance(values, list) and 0 <= index < len(values):
        return values[index]
    return None


__all__ = [
    "WeatherInput",
    "build_weather_tool",
    "get_weather",
]
