from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._http import JsonRequester, request_json

if TYPE_CHECKING:
    from ..config import Settings

GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
OSM_MAP_URL = "https://www.openstreetmap.org"


class MapInput(BaseModel):
    """Input schema for the map locator tool."""

    place: str = Field(
        ...,
        min_length=1,
        description="Place, city, landmark, or address to locate on a map.",
    )
    zoom: int = Field(
        default=12,
        ge=1,
        le=19,
        description="Map zoom level (1 world .. 19 building).",
    )


def build_map_tool(
    _settings: Settings,
    *,
    requester: JsonRequester | None = None,
) -> BaseTool:
    """Create an Open-Meteo geocoding + OpenStreetMap locator tool."""

    def _run_map(place: str, zoom: int = 12) -> str:
        return find_on_map(place, zoom=zoom, requester=requester)

    return StructuredTool.from_function(
        func=_run_map,
        name="find_on_map",
        description=(
            "Locate a place, city, landmark, or address on a map. Returns its "
            "coordinates, region/country, and an OpenStreetMap link centered on "
            "it. Use for 'where is X', 'show X on a map', or to get the "
            "latitude/longitude of a place."
        ),
        args_schema=MapInput,
    )


def find_on_map(
    place: str,
    *,
    zoom: int = 12,
    requester: JsonRequester | None = None,
) -> str:
    term = (place or "").strip()
    if not term:
        return "Map lookup requires a non-empty place name."

    try:
        payload = request_json(
            GEOCODING_API_URL,
            params={"name": term, "count": 1, "language": "en", "format": "json"},
            requester=requester,
        )
    except Exception as exc:
        return f"Map lookup failed for {term!r}: {exc}"

    results = payload.get("results") or []
    if not results:
        return f"No map location found for: {term}"

    first = results[0] if isinstance(results[0], dict) else {}
    return _format_location(term, first, zoom)


def _format_location(term: str, first: dict[str, Any], zoom: int) -> str:
    try:
        lat = float(first["latitude"])
        lon = float(first["longitude"])
    except (KeyError, TypeError, ValueError):
        return f"No map location found for: {term}"

    name = str(first.get("name") or term)
    label_parts = [
        name,
        str(first.get("admin1") or "").strip(),
        str(first.get("country") or "").strip(),
    ]
    label = ", ".join(part for part in label_parts if part)

    rows: list[tuple[str, str]] = [
        ("Place", label),
        ("Latitude", f"{lat:.4f}"),
        ("Longitude", f"{lon:.4f}"),
    ]
    population = first.get("population")
    if isinstance(population, (int, float)):
        rows.append(("Population", f"{int(population):,}"))
    rows.append(("Map", _map_url(lat, lon, zoom)))

    return f"Map location for {label}:\n\n" + _markdown_table(
        ["Detail", "Value"], rows
    )


def _map_url(lat: float, lon: float, zoom: int) -> str:
    return f"{OSM_MAP_URL}/?mlat={lat:.5f}&mlon={lon:.5f}#map={zoom}/{lat:.5f}/{lon:.5f}"


def _markdown_table(headers: list[str], rows: list[tuple[str, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


__all__ = [
    "MapInput",
    "build_map_tool",
    "find_on_map",
]
