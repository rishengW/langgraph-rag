from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._geocoding import GeocodedPlace, geocode_place
from ._http import JsonRequester

if TYPE_CHECKING:
    from ..config import Settings

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
            "Locate a place, city, district, landmark, campus, business park, "
            "or address on a map and get its coordinates. Returns latitude, "
            "longitude, region/country, and an OpenStreetMap link centered on "
            "it. USE THIS FIRST for any 'where is X', 'show X on a map', "
            "'find X's location', or latitude/longitude question, including "
            "Chinese place names, instead of a web search. Works for non-Latin "
            "names. If the result is reported as an APPROXIMATE MATCH or no "
            "location is found, then fall back to a web search."
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
        candidates = geocode_place(term, limit=3, requester=requester)
    except Exception as exc:
        return f"Map lookup failed for {term!r}: {exc}"

    if not candidates:
        return (
            f"No map location found for: {term}. The gazetteers cover cities, "
            "towns, and mapped points of interest; a very new or unmapped "
            "site may need a web search instead."
        )
    return _format_location(term, candidates, zoom)


def _format_location(term: str, candidates: list[GeocodedPlace], zoom: int) -> str:
    best = candidates[0]
    rows: list[tuple[str, ...]] = [
        ("Place", best.label),
        ("Latitude", f"{best.latitude:.4f}"),
        ("Longitude", f"{best.longitude:.4f}"),
    ]
    if best.kind:
        rows.append(("Type", best.kind))
    if best.population is not None:
        rows.append(("Population", f"{best.population:,}"))
    rows.append(("Map", _map_url(best.latitude, best.longitude, zoom)))
    rows.append(("Source", best.provider))

    heading = f"Map location for {best.label}:"
    table = _markdown_table(["Detail", "Value"], rows)
    sections = [heading, "", table]

    if not best.is_confident:
        # Text-similarity geocoding can return a neighbouring or unrelated
        # place with a plausible name. Say so instead of asserting it.
        sections.extend(
            [
                "",
                f"APPROXIMATE MATCH: the closest mapped entry to {term!r} is "
                f"{best.label!r}, which does not clearly match the requested "
                "name. Treat the coordinates as nearby, not exact, and verify "
                "with a web search before presenting them as the answer.",
            ]
        )

    if _is_ambiguous(candidates):
        # Text-similarity ranking has no notion of prominence: several unrelated
        # places can share a name and the famous one may not rank first. Flag
        # the tie rather than picking arbitrarily.
        sections.extend(
            [
                "",
                f"AMBIGUOUS: several mapped places match {term!r} equally well and "
                "none is a major populated place. Do not present one as the answer "
                "without disambiguating from the candidates below, the user's "
                "context, or a web search.",
            ]
        )

    alternatives = [
        f"- {candidate.label} ({candidate.latitude:.4f}, {candidate.longitude:.4f})"
        for candidate in candidates[1:]
    ]
    if alternatives:
        sections.extend(["", "Other candidates:", *alternatives])
    return "\n".join(sections)


def _is_ambiguous(candidates: list[GeocodedPlace]) -> bool:
    """Return whether the top candidates are an unresolved same-name tie.

    A populated place with a population figure is treated as prominent enough to
    win a tie, so ordinary city lookups are not flagged.
    """

    best = candidates[0]
    if best.population is not None:
        return False
    ties = [candidate for candidate in candidates[1:] if candidate.match_score >= best.match_score]
    return any(
        (candidate.country or candidate.state) != (best.country or best.state) for candidate in ties
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
