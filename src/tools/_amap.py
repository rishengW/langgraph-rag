from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlencode

from ._http import JsonRequester, request_json

AMAP_PROVIDER = "amap"
AMAP_COORDINATE_SYSTEM = "gcj02"
AMAP_REST_API_BASE_URL = "https://restapi.amap.com"
AMAP_API_V3_BASE_URL = f"{AMAP_REST_API_BASE_URL}/v3"
AMAP_API_V4_BASE_URL = f"{AMAP_REST_API_BASE_URL}/v4"
AMAP_PLACE_TEXT_URL = f"{AMAP_API_V3_BASE_URL}/place/text"
AMAP_GEOCODE_GEO_URL = f"{AMAP_API_V3_BASE_URL}/geocode/geo"
AMAP_DISTRICT_URL = f"{AMAP_API_V3_BASE_URL}/config/district"
AMAP_COORDINATE_CONVERT_URL = f"{AMAP_API_V3_BASE_URL}/assistant/coordinate/convert"
AMAP_DIRECTION_DRIVING_URL = f"{AMAP_API_V3_BASE_URL}/direction/driving"
AMAP_DIRECTION_WALKING_URL = f"{AMAP_API_V3_BASE_URL}/direction/walking"
AMAP_DIRECTION_BICYCLING_URL = f"{AMAP_API_V4_BASE_URL}/direction/bicycling"
AMAP_URI_MARKER_URL = "https://uri.amap.com/marker"
AMAP_URI_NAVIGATION_URL = "https://uri.amap.com/navigation"
AMAP_DEFAULT_TIMEOUT_SECONDS = 10
AMAP_ARTIFACT_VERSION = 1
MAX_AMAP_ROUTE_STEPS = 40
MAX_AMAP_ROUTE_POLYLINE_POINTS = 500
MAX_AMAP_ROUTE_INSTRUCTIONS = 8
MAX_AMAP_QUERY_CHARS = 180

LookupSource = Literal["poi", "geocode", "district"]
TravelMode = Literal["driving", "walking", "cycling"]

_MODE_ALIASES: dict[str, TravelMode] = {
    "drive": "driving",
    "driving": "driving",
    "car": "driving",
    "auto": "driving",
    "automobile": "driving",
    "walk": "walking",
    "walking": "walking",
    "foot": "walking",
    "pedestrian": "walking",
    "bike": "cycling",
    "bicycle": "cycling",
    "bicycling": "cycling",
    "cycle": "cycling",
    "cycling": "cycling",
    "ride": "cycling",
    "riding": "cycling",
}
_UNSUPPORTED_TRANSIT_ALIASES = frozenset(
    {"bus", "metro", "subway", "train", "transit", "public", "public transit"}
)
_ROUTE_URL_BY_MODE: dict[TravelMode, str] = {
    "driving": AMAP_DIRECTION_DRIVING_URL,
    "walking": AMAP_DIRECTION_WALKING_URL,
    "cycling": AMAP_DIRECTION_BICYCLING_URL,
}
_URI_MODE_BY_MODE: dict[TravelMode, str] = {
    "driving": "car",
    "walking": "walk",
    "cycling": "bike",
}
_SECRET_PARAM_RE = re.compile(
    r"(?i)([?&\s](?:key|jscode|security_code|amap_web_service_key)\s*=\s*)[^&\s]+"
)
_SECRET_ENV_RE = re.compile(r"(?i)\b(AMAP_[A-Z0-9_]*(?:KEY|CODE))\s*=\s*\S+")
_SECRET_DICT_RE = re.compile(
    r"(?i)(['\"](?:key|jscode|security_code|amap_web_service_key)['\"]\s*:\s*)['\"][^'\"]+['\"]"
)


class AMapError(RuntimeError):
    """Base class for AMap helper failures."""


class AMapConfigError(AMapError):
    """Raised when an AMap request cannot be made because configuration is missing."""


class AMapAPIError(AMapError):
    """Raised for AMap semantic envelope or transport failures."""

    def __init__(
        self,
        message: str,
        *,
        info: str = "",
        infocode: str = "",
        context: str = "",
    ) -> None:
        super().__init__(safe_amap_error(message))
        self.info = safe_amap_error(info)
        self.infocode = safe_amap_error(infocode)
        self.context = safe_amap_error(context)


@dataclass(frozen=True)
class AMapPosition:
    """A GCJ-02 or WGS84 coordinate in AMap's longitude,latitude order."""

    lng: float
    lat: float

    def as_dict(self) -> dict[str, float]:
        return {"lng": self.lng, "lat": self.lat}

    def to_param(self) -> str:
        return f"{self.lng:.6f},{self.lat:.6f}"


@dataclass(frozen=True)
class AMapLookupResult:
    """One parsed AMap place/geocode/district candidate."""

    name: str
    position: AMapPosition
    source: LookupSource
    address: str = ""
    province: str = ""
    city: str = ""
    district: str = ""
    country: str = "China"
    kind: str = ""
    adcode: str = ""
    amap_id: str = ""
    level: str = ""
    provider: str = AMAP_PROVIDER


@dataclass(frozen=True)
class AMapRouteStep:
    """One bounded route step parsed from AMap directions."""

    instruction: str = ""
    road: str = ""
    distance_m: float | None = None
    duration_s: float | None = None
    polyline: list[AMapPosition] = field(default_factory=list)
    bounds: dict[str, dict[str, float]] | None = None

    def as_artifact_dict(self) -> dict[str, object]:
        data: dict[str, object] = {}
        if self.instruction:
            data["instruction"] = self.instruction
        if self.road:
            data["road"] = self.road
        if self.distance_m is not None:
            data["distanceMeters"] = self.distance_m
        if self.duration_s is not None:
            data["durationSeconds"] = self.duration_s
        if self.polyline:
            data["polyline"] = [position.as_dict() for position in self.polyline]
        if self.bounds is not None:
            data["bounds"] = self.bounds
        return data


@dataclass(frozen=True)
class AMapRoute:
    """A bounded route summary parsed from AMap directions."""

    mode: TravelMode
    distance_m: float | None = None
    duration_s: float | None = None
    steps: list[AMapRouteStep] = field(default_factory=list)
    polyline: list[AMapPosition] = field(default_factory=list)
    bounds: dict[str, dict[str, float]] | None = None


def normalize_travel_mode(mode: str | None) -> TravelMode | None:
    """Return the supported AMap mode, or None for unsupported transit modes."""

    value = " ".join((mode or "driving").strip().casefold().split())
    if value in _UNSUPPORTED_TRANSIT_ALIASES:
        return None
    return _MODE_ALIASES.get(value, "driving")


def resolve_amap_web_service_key(value: str | None = None) -> str:
    """Return a stripped AMap Web Service key without reading YAML config."""

    if value is not None:
        return value.strip()

    # Imported lazily so tests that monkeypatch os.environ do not need to reload
    # this module, and so config loading remains the only normal settings source.
    import os

    return os.getenv("AMAP_WEB_SERVICE_KEY", "").strip()


def require_amap_web_service_key(value: str | None) -> str:
    key = resolve_amap_web_service_key(value)
    if not key:
        raise AMapConfigError("AMap web service key is not configured.")
    return key


def safe_amap_error(error: object) -> str:
    """Return a client-safe, secret-redacted error string."""

    text = str(error or "AMap request failed.")
    text = _SECRET_PARAM_RE.sub(r"\1***", text)
    text = _SECRET_DICT_RE.sub(r"\1'***'", text)
    text = _SECRET_ENV_RE.sub(lambda match: f"{match.group(1)}=***", text)
    text = " ".join(text.split())
    if len(text) > 240:
        return f"{text[:237]}..."
    return text


def parse_lnglat(value: object) -> AMapPosition | None:
    """Parse AMap longitude,latitude coordinates."""

    if isinstance(value, str):
        first, sep, second = value.strip().partition(",")
        if not sep:
            return None
        raw_lng, raw_lat = first.strip(), second.strip()
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if len(value) < 2:
            return None
        raw_lng, raw_lat = value[0], value[1]
    else:
        return None

    try:
        lng = float(raw_lng)
        lat = float(raw_lat)
    except (OverflowError, TypeError, ValueError):
        return None
    if not (-180.0 <= lng <= 180.0 and -90.0 <= lat <= 90.0):
        return None
    return AMapPosition(lng=lng, lat=lat)


def parse_latlon(value: str) -> AMapPosition | None:
    """Parse public raw-coordinate input in latitude,longitude WGS84 order."""

    first, sep, second = (value or "").strip().partition(",")
    if not sep:
        return None
    try:
        lat = float(first.strip())
        lng = float(second.strip())
    except ValueError:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return None
    return AMapPosition(lng=lng, lat=lat)


def parse_polyline(value: object, *, max_points: int | None = None) -> list[AMapPosition]:
    """Parse an AMap semicolon-delimited route polyline."""

    points: list[AMapPosition] = []
    for chunk in str(value or "").split(";"):
        position = parse_lnglat(chunk)
        if position is not None:
            points.append(position)
    limit = len(points) if max_points is None else max_points
    return _sample_positions(points, limit)


def route_bounds(
    positions: Sequence[AMapPosition],
) -> dict[str, dict[str, float]] | None:
    """Return southwest/northeast bounds for route or step positions."""

    if not positions:
        return None
    lngs = [position.lng for position in positions]
    lats = [position.lat for position in positions]
    return {
        "southwest": {"lng": min(lngs), "lat": min(lats)},
        "northeast": {"lng": max(lngs), "lat": max(lats)},
    }


def validate_amap_envelope(payload: dict[str, Any], *, context: str) -> dict[str, Any]:
    """Raise on AMap's semantic error envelope while preserving safe messages."""

    if not isinstance(payload, dict):
        raise AMapAPIError(f"AMap {context} returned an invalid JSON envelope.")

    if "errcode" in payload:
        errcode = str(payload.get("errcode"))
        if errcode == "0":
            return payload
        message = _safe_text(payload.get("errmsg") or payload.get("info") or "error")
        raise AMapAPIError(
            f"AMap {context} failed: {message}",
            info=message,
            infocode=errcode,
            context=context,
        )

    status = str(payload.get("status") or "")
    if status == "1":
        return payload
    if not status and any(key in payload for key in ("pois", "geocodes", "districts", "route", "data")):
        return payload

    message = _safe_text(payload.get("info") or payload.get("message") or "error")
    infocode = _safe_text(payload.get("infocode") or payload.get("code") or "")
    raise AMapAPIError(
        f"AMap {context} failed: {message}",
        info=message,
        infocode=infocode,
        context=context,
    )


def amap_request_json(
    url: str,
    *,
    params: dict[str, object],
    api_key: str,
    requester: JsonRequester | None = None,
    timeout_seconds: int = AMAP_DEFAULT_TIMEOUT_SECONDS,
    context: str,
) -> dict[str, Any]:
    """Call an AMap JSON endpoint with key injection and envelope validation."""

    key = require_amap_web_service_key(api_key)
    request_params: dict[str, object] = {**params, "key": key, "output": "json"}
    try:
        payload = request_json(
            url,
            params=request_params,
            requester=requester,
            timeout=max(1, int(timeout_seconds)),
        )
    except TimeoutError as exc:
        raise AMapAPIError(f"AMap {context} timed out.", context=context) from exc
    except Exception as exc:
        raise AMapAPIError(
            f"AMap {context} request failed: {safe_amap_error(exc)}",
            context=context,
        ) from exc
    return validate_amap_envelope(payload, context=context)


def amap_search_pois(
    query: str,
    *,
    api_key: str,
    limit: int = 3,
    requester: JsonRequester | None = None,
    timeout_seconds: int = AMAP_DEFAULT_TIMEOUT_SECONDS,
) -> list[AMapLookupResult]:
    """Search AMap POIs as the primary place lookup stage."""

    term = _bounded_query(query)
    if not term:
        return []
    payload = amap_request_json(
        AMAP_PLACE_TEXT_URL,
        params={
            "keywords": term,
            "offset": max(1, int(limit)),
            "page": 1,
            "extensions": "base",
            "citylimit": "false",
        },
        api_key=api_key,
        requester=requester,
        timeout_seconds=timeout_seconds,
        context="POI lookup",
    )
    return parse_poi_results(payload, limit=limit)


def amap_geocode_address(
    address: str,
    *,
    api_key: str,
    limit: int = 3,
    requester: JsonRequester | None = None,
    timeout_seconds: int = AMAP_DEFAULT_TIMEOUT_SECONDS,
) -> list[AMapLookupResult]:
    """Geocode an address or named place through AMap's address endpoint."""

    term = _bounded_query(address)
    if not term:
        return []
    payload = amap_request_json(
        AMAP_GEOCODE_GEO_URL,
        params={"address": term},
        api_key=api_key,
        requester=requester,
        timeout_seconds=timeout_seconds,
        context="address geocode",
    )
    return parse_geocode_results(payload, fallback_name=term, limit=limit)


def amap_lookup_districts(
    query: str,
    *,
    api_key: str,
    limit: int = 3,
    requester: JsonRequester | None = None,
    timeout_seconds: int = AMAP_DEFAULT_TIMEOUT_SECONDS,
) -> list[AMapLookupResult]:
    """Look up administrative districts as the final geocoding fallback."""

    term = _bounded_query(query)
    if not term:
        return []
    payload = amap_request_json(
        AMAP_DISTRICT_URL,
        params={"keywords": term, "subdistrict": 0, "extensions": "base"},
        api_key=api_key,
        requester=requester,
        timeout_seconds=timeout_seconds,
        context="district lookup",
    )
    return parse_district_results(payload, limit=limit)


def convert_wgs84_to_gcj02(
    position: AMapPosition,
    *,
    api_key: str,
    requester: JsonRequester | None = None,
    timeout_seconds: int = AMAP_DEFAULT_TIMEOUT_SECONDS,
) -> AMapPosition:
    """Convert one raw WGS84 coordinate to the GCJ-02 coordinates AMap requires."""

    payload = amap_request_json(
        AMAP_COORDINATE_CONVERT_URL,
        params={"locations": position.to_param(), "coordsys": "gps"},
        api_key=api_key,
        requester=requester,
        timeout_seconds=timeout_seconds,
        context="coordinate conversion",
    )
    converted = parse_lnglat(payload.get("locations"))
    if converted is None:
        raise AMapAPIError("AMap coordinate conversion returned no usable location.")
    return converted


def amap_route(
    origin: AMapPosition,
    destination: AMapPosition,
    *,
    mode: TravelMode,
    api_key: str,
    requester: JsonRequester | None = None,
    timeout_seconds: int = AMAP_DEFAULT_TIMEOUT_SECONDS,
) -> AMapRoute | None:
    """Fetch and parse an AMap route for a supported non-transit travel mode."""

    payload = amap_request_json(
        _ROUTE_URL_BY_MODE[mode],
        params={
            "origin": origin.to_param(),
            "destination": destination.to_param(),
            "extensions": "base",
        },
        api_key=api_key,
        requester=requester,
        timeout_seconds=timeout_seconds,
        context=f"{mode} route",
    )
    return parse_route_result(payload, mode=mode)


def parse_poi_results(payload: dict[str, Any], *, limit: int = 3) -> list[AMapLookupResult]:
    results: list[AMapLookupResult] = []
    for item in _as_list(payload.get("pois"))[: max(1, int(limit))]:
        if not isinstance(item, dict):
            continue
        position = parse_lnglat(item.get("location"))
        if position is None:
            continue
        name = _safe_text(item.get("name"))
        address = _safe_text(item.get("address"))
        if not name and not address:
            continue
        results.append(
            AMapLookupResult(
                name=name or address,
                position=position,
                source="poi",
                address=address,
                province=_safe_text(item.get("pname")),
                city=_safe_text(item.get("cityname")),
                district=_safe_text(item.get("adname")),
                country=_safe_text(item.get("country")) or "China",
                kind=_safe_text(item.get("type")) or _safe_text(item.get("typecode")),
                adcode=_safe_text(item.get("adcode")),
                amap_id=_safe_text(item.get("id")),
            )
        )
    return results


def parse_geocode_results(
    payload: dict[str, Any],
    *,
    fallback_name: str,
    limit: int = 3,
) -> list[AMapLookupResult]:
    results: list[AMapLookupResult] = []
    for item in _as_list(payload.get("geocodes"))[: max(1, int(limit))]:
        if not isinstance(item, dict):
            continue
        position = parse_lnglat(item.get("location"))
        if position is None:
            continue
        formatted = _safe_text(item.get("formatted_address"))
        results.append(
            AMapLookupResult(
                name=formatted or fallback_name,
                position=position,
                source="geocode",
                address=formatted,
                province=_safe_text(item.get("province")),
                city=_safe_text(item.get("city")),
                district=_safe_text(item.get("district")),
                country=_safe_text(item.get("country")) or "China",
                kind=_safe_text(item.get("level")) or "address",
                adcode=_safe_text(item.get("adcode")),
                level=_safe_text(item.get("level")),
            )
        )
    return results


def parse_district_results(payload: dict[str, Any], *, limit: int = 3) -> list[AMapLookupResult]:
    results: list[AMapLookupResult] = []
    for item in _as_list(payload.get("districts"))[: max(1, int(limit))]:
        if not isinstance(item, dict):
            continue
        position = parse_lnglat(item.get("center"))
        if position is None:
            continue
        name = _safe_text(item.get("name"))
        if not name:
            continue
        level = _safe_text(item.get("level"))
        results.append(
            AMapLookupResult(
                name=name,
                position=position,
                source="district",
                province=name if level == "province" else "",
                city=name if level == "city" else "",
                district=name if level == "district" else "",
                country=_safe_text(item.get("country")) or "China",
                kind=level or "district",
                adcode=_safe_text(item.get("adcode")),
                level=level,
            )
        )
    return results


def parse_route_result(payload: dict[str, Any], *, mode: TravelMode) -> AMapRoute | None:
    paths = _route_paths(payload)
    if not paths:
        return None
    path = paths[0]
    if not isinstance(path, dict):
        return None

    steps = _route_steps(path)
    all_points = _path_polyline(path)
    if not all_points:
        for step in steps:
            all_points.extend(step.polyline)
    all_points = _sample_positions(all_points, MAX_AMAP_ROUTE_POLYLINE_POINTS)

    distance_m = _optional_nonnegative_float(path.get("distance"))
    duration_s = _optional_nonnegative_float(path.get("duration"))
    if duration_s is None and isinstance(path.get("cost"), dict):
        duration_s = _optional_nonnegative_float(path["cost"].get("duration"))

    return AMapRoute(
        mode=mode,
        distance_m=distance_m,
        duration_s=duration_s,
        steps=steps,
        polyline=all_points,
        bounds=route_bounds(all_points),
    )


def build_amap_marker_uri(
    position: AMapPosition,
    *,
    name: str = "",
    src: str = "langgraph-rag",
) -> str:
    params = {
        "position": position.to_param(),
        "name": name,
        "src": src,
        "coordinate": "gaode",
        "callnative": "0",
    }
    return f"{AMAP_URI_MARKER_URL}?{urlencode(params)}"


def build_amap_navigation_uri(
    origin: AMapPosition,
    destination: AMapPosition,
    *,
    origin_name: str = "",
    destination_name: str = "",
    mode: TravelMode = "driving",
    src: str = "langgraph-rag",
) -> str:
    params = {
        "from": _navigation_endpoint_value(origin, origin_name),
        "to": _navigation_endpoint_value(destination, destination_name),
        "mode": _URI_MODE_BY_MODE[mode],
        "policy": "1",
        "src": src,
        "coordinate": "gaode",
        "callnative": "0",
    }
    return f"{AMAP_URI_NAVIGATION_URL}?{urlencode(params)}"


def build_marker_artifact(
    position: AMapPosition,
    *,
    fallback_url: str,
    label: str = "",
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "type": "amap",
        "version": AMAP_ARTIFACT_VERSION,
        "kind": "marker",
        "provider": AMAP_PROVIDER,
        "coordinateSystem": AMAP_COORDINATE_SYSTEM,
        "fallbackUrl": fallback_url,
        "positions": [position.as_dict()],
    }
    if label:
        artifact["label"] = label
    return artifact


def build_route_artifact(
    origin: AMapPosition,
    destination: AMapPosition,
    *,
    route: AMapRoute | None,
    fallback_url: str,
    mode: TravelMode,
    origin_label: str = "",
    destination_label: str = "",
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "type": "amap",
        "version": AMAP_ARTIFACT_VERSION,
        "kind": "route",
        "provider": AMAP_PROVIDER,
        "coordinateSystem": AMAP_COORDINATE_SYSTEM,
        "fallbackUrl": fallback_url,
        "mode": mode,
        "positions": [origin.as_dict(), destination.as_dict()],
    }
    if origin_label:
        artifact["originLabel"] = origin_label
    if destination_label:
        artifact["destinationLabel"] = destination_label
    if route is not None:
        if route.distance_m is not None:
            artifact["distanceMeters"] = route.distance_m
        if route.duration_s is not None:
            artifact["durationSeconds"] = route.duration_s
        if route.polyline:
            artifact["polyline"] = [position.as_dict() for position in route.polyline]
        if route.bounds is not None:
            artifact["bounds"] = route.bounds
        step_dicts = [step.as_artifact_dict() for step in route.steps[:MAX_AMAP_ROUTE_INSTRUCTIONS]]
        artifact["steps"] = [step for step in step_dicts if step]
    return artifact


def _route_paths(payload: dict[str, Any]) -> list[Any]:
    route = payload.get("route")
    if isinstance(route, dict):
        paths = route.get("paths")
        if isinstance(paths, list):
            return paths
    data = payload.get("data")
    if isinstance(data, dict):
        paths = data.get("paths")
        if isinstance(paths, list):
            return paths
        route = data.get("route")
        if isinstance(route, dict) and isinstance(route.get("paths"), list):
            return route["paths"]
    return []


def _route_steps(path: dict[str, Any]) -> list[AMapRouteStep]:
    steps: list[AMapRouteStep] = []
    for item in _as_list(path.get("steps"))[:MAX_AMAP_ROUTE_STEPS]:
        if not isinstance(item, dict):
            continue
        polyline = parse_polyline(item.get("polyline"), max_points=MAX_AMAP_ROUTE_POLYLINE_POINTS)
        duration_s = _optional_nonnegative_float(item.get("duration"))
        if duration_s is None and isinstance(item.get("cost"), dict):
            duration_s = _optional_nonnegative_float(item["cost"].get("duration"))
        steps.append(
            AMapRouteStep(
                instruction=_safe_text(item.get("instruction")),
                road=_safe_text(item.get("road")),
                distance_m=_optional_nonnegative_float(item.get("distance")),
                duration_s=duration_s,
                polyline=polyline,
                bounds=route_bounds(polyline),
            )
        )
    return steps


def _path_polyline(path: dict[str, Any]) -> list[AMapPosition]:
    return parse_polyline(path.get("polyline"), max_points=MAX_AMAP_ROUTE_POLYLINE_POINTS)


def _sample_positions(
    positions: Sequence[AMapPosition],
    limit: int,
) -> list[AMapPosition]:
    values = list(positions)
    if limit <= 0:
        return []
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[0]]

    last_index = len(values) - 1
    sampled: list[AMapPosition] = []
    seen: set[int] = set()
    for index in range(limit):
        source_index = round(index * last_index / (limit - 1))
        if source_index in seen:
            continue
        sampled.append(values[source_index])
        seen.add(source_index)
    return sampled


def _bounded_query(value: str) -> str:
    return " ".join((value or "").split())[:MAX_AMAP_QUERY_CHARS]


def _safe_text(value: object) -> str:
    if value is None or isinstance(value, (list, tuple, dict, set)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return safe_amap_error(text)


def _optional_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_nonnegative_float(value: object) -> float | None:
    result = _optional_float(value)
    return result if result is not None and result >= 0 else None


def _as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _navigation_endpoint_value(position: AMapPosition, name: str) -> str:
    # AMap parses this value as exactly ``lng,lat,name``. Display labels often
    # contain region separators, so embedded commas must not become extra
    # structural delimiters or the destination can be discarded.
    label = " ".join(_safe_text(name).replace(",", " ").split())
    return f"{position.to_param()},{label}" if label else position.to_param()


__all__ = [
    "AMAP_API_V3_BASE_URL",
    "AMAP_API_V4_BASE_URL",
    "AMAP_ARTIFACT_VERSION",
    "AMAP_COORDINATE_CONVERT_URL",
    "AMAP_COORDINATE_SYSTEM",
    "AMAP_DEFAULT_TIMEOUT_SECONDS",
    "AMAP_DIRECTION_BICYCLING_URL",
    "AMAP_DIRECTION_DRIVING_URL",
    "AMAP_DIRECTION_WALKING_URL",
    "AMAP_DISTRICT_URL",
    "AMAP_GEOCODE_GEO_URL",
    "AMAP_PLACE_TEXT_URL",
    "AMAP_PROVIDER",
    "AMAP_REST_API_BASE_URL",
    "AMAP_URI_MARKER_URL",
    "AMAP_URI_NAVIGATION_URL",
    "AMapAPIError",
    "AMapConfigError",
    "AMapError",
    "AMapLookupResult",
    "AMapPosition",
    "AMapRoute",
    "AMapRouteStep",
    "LookupSource",
    "MAX_AMAP_ROUTE_POLYLINE_POINTS",
    "MAX_AMAP_ROUTE_STEPS",
    "TravelMode",
    "amap_geocode_address",
    "amap_lookup_districts",
    "amap_request_json",
    "amap_route",
    "amap_search_pois",
    "build_amap_marker_uri",
    "build_amap_navigation_uri",
    "build_marker_artifact",
    "build_route_artifact",
    "convert_wgs84_to_gcj02",
    "normalize_travel_mode",
    "parse_district_results",
    "parse_geocode_results",
    "parse_latlon",
    "parse_lnglat",
    "parse_poi_results",
    "parse_polyline",
    "parse_route_result",
    "require_amap_web_service_key",
    "resolve_amap_web_service_key",
    "route_bounds",
    "safe_amap_error",
    "validate_amap_envelope",
]
