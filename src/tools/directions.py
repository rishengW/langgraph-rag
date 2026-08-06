from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._amap import (
    AMapPosition,
    AMapRoute,
    amap_route,
    build_amap_navigation_uri,
    build_route_artifact,
    convert_wgs84_to_gcj02,
    normalize_travel_mode,
    parse_latlon,
    safe_amap_error,
)
from ._geocoding import geocode_place
from ._http import JsonRequester

if TYPE_CHECKING:
    from ..config import Settings


@dataclass(frozen=True)
class ResolvedRoutePoint:
    """One route endpoint resolved to AMap GCJ-02 coordinates."""

    label: str
    position: AMapPosition
    raw_wgs84: AMapPosition | None = None


@dataclass(frozen=True)
class DirectionsLookupResult:
    """Internal content/artifact result for directions."""

    content: str
    artifact: dict[str, object] | None = None


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
    settings: Settings,
    *,
    requester: JsonRequester | None = None,
) -> BaseTool:
    """Create an AMap-backed driving/walking/cycling directions tool."""

    def _run_directions(
        origin: str,
        destination: str,
        mode: str = "driving",
    ) -> tuple[str, dict[str, object] | None]:
        result = get_directions_result(
            origin=origin,
            destination=destination,
            mode=mode,
            settings=settings,
            requester=requester,
        )
        return result.content, result.artifact

    return StructuredTool.from_function(
        func=_run_directions,
        name="get_directions",
        description=(
            "Get the route, distance, and estimated travel time between two "
            "places using AMap. USE THIS FIRST for directions, how far apart two "
            "locations are, how long a trip takes, or which roads to take, instead "
            "of a web search. Accepts city/place names, points of interest, "
            "addresses (including Chinese names), or raw 'latitude,longitude' WGS84 "
            "coordinates. Supports driving, walking, and cycling aliases only; "
            "transit directions are not supported. Distances and durations are "
            "free-flow estimates and exclude live traffic."
        ),
        args_schema=DirectionsInput,
        response_format="content_and_artifact",
    )


def get_directions(
    *,
    origin: str,
    destination: str,
    mode: str = "driving",
    requester: JsonRequester | None = None,
    api_key: str | None = None,
    timeout_seconds: int = 10,
) -> str:
    """Return a string directions result for compatibility with direct callers."""

    return get_directions_result(
        origin=origin,
        destination=destination,
        mode=mode,
        requester=requester,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    ).content


def get_directions_result(
    *,
    origin: str,
    destination: str,
    mode: str = "driving",
    settings: Settings | None = None,
    requester: JsonRequester | None = None,
    api_key: str | None = None,
    timeout_seconds: int | None = None,
) -> DirectionsLookupResult:
    start = (origin or "").strip()
    end = (destination or "").strip()
    if not start or not end:
        return DirectionsLookupResult("Directions require both an origin and a destination.")

    travel_mode = normalize_travel_mode(mode)
    if travel_mode is None:
        return DirectionsLookupResult(
            "Transit directions are not supported. Use driving, walking, or cycling."
        )

    service_key = api_key if api_key is not None else str(getattr(settings, "amap_web_service_key", ""))
    timeout = timeout_seconds
    if timeout is None:
        timeout = int(getattr(settings, "amap_api_timeout_seconds", 10) or 10)

    try:
        start_point = _resolve_point(
            start,
            api_key=service_key,
            requester=requester,
            timeout_seconds=timeout,
        )
        end_point = _resolve_point(
            end,
            api_key=service_key,
            requester=requester,
            timeout_seconds=timeout,
        )
    except Exception as exc:
        return DirectionsLookupResult(f"Directions lookup failed: {safe_amap_error(exc)}")

    fallback_url = build_amap_navigation_uri(
        start_point.position,
        end_point.position,
        origin_name=start_point.label,
        destination_name=end_point.label,
        mode=travel_mode,
    )

    try:
        route = amap_route(
            start_point.position,
            end_point.position,
            mode=travel_mode,
            api_key=service_key,
            requester=requester,
            timeout_seconds=timeout,
        )
    except Exception as exc:
        content = (
            f"Directions lookup failed for {start_point.label} to {end_point.label}: "
            f"{safe_amap_error(exc)}\n\nAMap route link: {fallback_url}"
        )
        artifact = build_route_artifact(
            start_point.position,
            end_point.position,
            route=None,
            fallback_url=fallback_url,
            mode=travel_mode,
            origin_label=start_point.label,
            destination_label=end_point.label,
        )
        return DirectionsLookupResult(content=content, artifact=artifact)

    if route is None:
        content = f"No route found from {start_point.label} to {end_point.label}.\n\nAMap route link: {fallback_url}"
    else:
        content = _format_directions(
            start_point.label,
            end_point.label,
            travel_mode,
            route,
            fallback_url,
        )
    artifact = build_route_artifact(
        start_point.position,
        end_point.position,
        route=route,
        fallback_url=fallback_url,
        mode=travel_mode,
        origin_label=start_point.label,
        destination_label=end_point.label,
    )
    return DirectionsLookupResult(content=content, artifact=artifact)


def _resolve_point(
    location: str,
    *,
    api_key: str,
    requester: JsonRequester | None,
    timeout_seconds: int,
) -> ResolvedRoutePoint:
    """Resolve a place name or raw 'lat,lon' string to GCJ-02 AMap coordinates."""

    raw_wgs84 = parse_latlon(location)
    if raw_wgs84 is not None:
        position = convert_wgs84_to_gcj02(
            raw_wgs84,
            api_key=api_key,
            requester=requester,
            timeout_seconds=timeout_seconds,
        )
        return ResolvedRoutePoint(
            label=f"{raw_wgs84.lat:g},{raw_wgs84.lng:g}",
            position=position,
            raw_wgs84=raw_wgs84,
        )

    # Shared AMap geocoder so route endpoints can be points of interest,
    # addresses, districts, or non-Latin place names rather than city names only.
    candidates = geocode_place(
        location,
        limit=1,
        requester=requester,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    if not candidates:
        raise ValueError(f"no matching location found for {location!r}")

    best = candidates[0]
    return ResolvedRoutePoint(
        label=best.label,
        position=AMapPosition(lng=best.longitude, lat=best.latitude),
    )


def _format_directions(
    start_label: str,
    end_label: str,
    mode: str,
    route: AMapRoute,
    fallback_url: str,
) -> str:
    rows: list[tuple[str, str]] = [
        ("From", start_label),
        ("To", end_label),
        ("Mode", mode or "driving"),
        ("Coordinate system", "GCJ-02 (AMap)"),
    ]
    if route.distance_m is not None:
        rows.append(("Distance", f"{route.distance_m / 1000:.1f} km"))
    if route.duration_s is not None:
        rows.append(("Estimated time", _format_duration(route.duration_s)))
    rows.append(("Map", fallback_url))

    sections = [
        f"Directions from {start_label} to {end_label}:",
        "",
        _markdown_table(["Detail", "Value"], rows),
    ]

    instructions = [step.instruction for step in route.steps if step.instruction]
    if instructions:
        sections.extend(
            [
                "",
                "Route steps:",
                *[f"{index}. {instruction}" for index, instruction in enumerate(instructions[:8], 1)],
            ]
        )
    return "\n".join(sections)


def _format_duration(seconds: float) -> str:
    total_minutes = int(round(seconds / 60))
    hours, minutes = divmod(total_minutes, 60)
    if hours and minutes:
        return f"{hours} h {minutes} min"
    if hours:
        return f"{hours} h"
    return f"{minutes} min"


def _markdown_table(headers: list[str], rows: Sequence[tuple[str, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


__all__ = [
    "DirectionsInput",
    "DirectionsLookupResult",
    "ResolvedRoutePoint",
    "build_directions_tool",
    "get_directions",
    "get_directions_result",
]
