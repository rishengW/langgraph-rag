from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._http import JsonRequester, request_json

if TYPE_CHECKING:
    from ..config import Settings

GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
OSRM_ROUTE_API_URL = "https://router.project-osrm.org/route/v1"

# OSRM's public demo server is built with the car profile. We expose a mode
# field for forward compatibility, but map every mode onto the supported
# "driving" profile so a self-hosted OSRM with foot/bike profiles can be
# swapped in via OSRM_ROUTE_API_URL without changing the tool contract.
_PROFILE_BY_MODE = {
    "driving": "driving",
    "car": "driving",
    "walking": "foot",
    "foot": "foot",
    "cycling": "bike",
    "bike": "bike",
}


class DirectionsInput(BaseModel):
    """Input schema for the directions tool."""

    origin: str = Field(
        ...,
        min_length=1,
        description="Start location: a place/city name or 'latitude,longitude'.",
    )
    destination: str = Field(
        ...,
        min_length=1,
        description="End location: a place/city name or 'latitude,longitude'.",
    )
    mode: str = Field(
        default="driving",
        description="Travel mode: driving, walking, or cycling.",
    )


def build_directions_tool(
    _settings: Settings,
    *,
    requester: JsonRequester | None = None,
) -> BaseTool:
    """Create an OSRM-backed driving/walking/cycling directions tool."""

    def _run_directions(
        origin: str,
        destination: str,
        mode: str = "driving",
    ) -> str:
        return get_directions(
            origin=origin,
            destination=destination,
            mode=mode,
            requester=requester,
        )

    return StructuredTool.from_function(
        func=_run_directions,
        name="get_directions",
        description=(
            "Get the route, distance, and estimated travel time between two "
            "places. Use for directions, how far apart two locations are, how "
            "long a trip takes, or which roads to take. Accepts city/place "
            "names or 'latitude,longitude' for origin and destination."
        ),
        args_schema=DirectionsInput,
    )


def get_directions(
    *,
    origin: str,
    destination: str,
    mode: str = "driving",
    requester: JsonRequester | None = None,
) -> str:
    start = (origin or "").strip()
    end = (destination or "").strip()
    if not start or not end:
        return "Directions require both an origin and a destination."

    profile = _PROFILE_BY_MODE.get(mode.strip().lower(), "driving")

    try:
        start_label, start_lat, start_lon = _resolve_point(start, requester=requester)
        end_label, end_lat, end_lon = _resolve_point(end, requester=requester)
    except Exception as exc:
        return f"Directions lookup failed: {exc}"

    coords = f"{start_lon:f},{start_lat:f};{end_lon:f},{end_lat:f}"
    url = f"{OSRM_ROUTE_API_URL}/{profile}/{coords}"
    try:
        payload = request_json(
            url,
            params={"overview": "false", "alternatives": "false", "steps": "false"},
            requester=requester,
        )
    except Exception as exc:
        return f"Directions lookup failed for {start_label} to {end_label}: {exc}"

    return _format_directions(start_label, end_label, mode.strip().lower(), payload)


def _resolve_point(
    location: str,
    *,
    requester: JsonRequester | None,
) -> tuple[str, float, float]:
    """Resolve a place name or 'lat,lon' string to (label, lat, lon)."""

    coord = _parse_coordinates(location)
    if coord is not None:
        lat, lon = coord
        return f"{lat:g},{lon:g}", lat, lon

    payload = request_json(
        GEOCODING_API_URL,
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        requester=requester,
    )
    results = payload.get("results") or []
    if not results:
        raise ValueError(f"no matching location found for {location!r}")

    first = results[0]
    lat = float(first["latitude"])
    lon = float(first["longitude"])
    label_parts = [
        str(first.get("name") or location),
        str(first.get("admin1") or "").strip(),
        str(first.get("country") or "").strip(),
    ]
    label = ", ".join(part for part in label_parts if part)
    return label, lat, lon


def _parse_coordinates(value: str) -> tuple[float, float] | None:
    if "," not in value:
        return None
    lat_str, _, lon_str = value.partition(",")
    try:
        lat = float(lat_str.strip())
        lon = float(lon_str.strip())
    except ValueError:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon


def _format_directions(
    start_label: str,
    end_label: str,
    mode: str,
    payload: dict[str, Any],
) -> str:
    code = str(payload.get("code") or "")
    routes = payload.get("routes") or []
    if code and code != "Ok":
        message = payload.get("message") or code
        return f"No route found from {start_label} to {end_label}: {message}"
    if not routes:
        return f"No route found from {start_label} to {end_label}."

    route = routes[0] if isinstance(routes[0], dict) else {}
    distance_m = route.get("distance")
    duration_s = route.get("duration")

    rows: list[tuple[str, str]] = [
        ("From", start_label),
        ("To", end_label),
        ("Mode", mode or "driving"),
    ]
    if isinstance(distance_m, (int, float)):
        rows.append(("Distance", f"{distance_m / 1000:.1f} km"))
    if isinstance(duration_s, (int, float)):
        rows.append(("Estimated time", _format_duration(float(duration_s))))

    return f"Directions from {start_label} to {end_label}:\n\n" + _markdown_table(
        ["Detail", "Value"], rows
    )


def _format_duration(seconds: float) -> str:
    total_minutes = int(round(seconds / 60))
    hours, minutes = divmod(total_minutes, 60)
    if hours and minutes:
        return f"{hours} h {minutes} min"
    if hours:
        return f"{hours} h"
    return f"{minutes} min"


def _markdown_table(headers: list[str], rows: list[tuple[str, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


__all__ = [
    "DirectionsInput",
    "build_directions_tool",
    "get_directions",
]
