from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._http import JsonRequester, request_json

if TYPE_CHECKING:
    from ..config import Settings

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
    current = payload.get("current") if isinstance(payload.get("current"), dict) else {}
    daily = payload.get("daily") if isinstance(payload.get("daily"), dict) else {}
    units = payload.get("current_units") if isinstance(payload.get("current_units"), dict) else {}
    daily_units = payload.get("daily_units") if isinstance(payload.get("daily_units"), dict) else {}

    temperature = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    wind = current.get("wind_speed_10m")
    weather_code = current.get("weather_code")
    description = _weather_description(weather_code)

    lines = [f"Weather for {label}:"]
    if temperature is not None:
        unit = units.get("temperature_2m", "C")
        lines.append(f"Current temperature: {temperature}{unit}")
    if humidity is not None:
        lines.append(f"Humidity: {humidity}{units.get('relative_humidity_2m', '%')}")
    if wind is not None:
        lines.append(f"Wind: {wind}{units.get('wind_speed_10m', ' km/h')}")
    if description:
        lines.append(f"Conditions: {description}")

    forecast_lines = _daily_forecast_lines(daily, daily_units)
    if forecast_lines:
        lines.append("Forecast:")
        lines.extend(forecast_lines)
    return "\n".join(lines)


def _daily_forecast_lines(
    daily: dict[str, Any],
    units: dict[str, Any],
) -> list[str]:
    dates = daily.get("time") or []
    max_temps = daily.get("temperature_2m_max") or []
    min_temps = daily.get("temperature_2m_min") or []
    codes = daily.get("weather_code") or []
    temp_unit = units.get("temperature_2m_max", "C")

    lines: list[str] = []
    for index, day in enumerate(dates[:5]):
        high = _list_get(max_temps, index)
        low = _list_get(min_temps, index)
        description = _weather_description(_list_get(codes, index))
        parts = [str(day)]
        if high is not None and low is not None:
            parts.append(f"{low}{temp_unit}-{high}{temp_unit}")
        if description:
            parts.append(description)
        lines.append("- " + ": ".join([parts[0], ", ".join(parts[1:])]))
    return lines


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
