from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import quote, urlencode, urlparse

from langchain_core.messages import ToolMessage

from .events import ArtifactEvent

AMAP_TYPE = "amap"
AMAP_VERSION = 1
AMAP_KIND_MARKER = "marker"
AMAP_KIND_ROUTE = "route"
AMAP_PROVIDER = "amap"
AMAP_COORDINATE_SYSTEM = "gcj02"
AMAP_URI_HOST = "uri.amap.com"

FILE_TYPE = "file"
FILE_VERSION = 1
FILE_KIND_DOWNLOAD = "download"
FILE_PROVIDER = "chat_upload"
DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TXT_MIME_TYPE = "text/plain"
XLSX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

MAX_TITLE_CHARS = 120
MAX_LABEL_CHARS = 160
MAX_ADDRESS_CHARS = 300
MAX_URL_CHARS = 2048
MAX_TOOL_CALL_ID_CHARS = 160
MAX_MARKERS = 20
MAX_STEPS = 50
MAX_POLYLINE_POINTS = 500
MAX_STEP_POLYLINE_POINTS = 100
MAX_FILENAME_CHARS = 200
MAX_MIME_CHARS = 120
MAX_THREAD_ID_CHARS = 64
MAX_FILE_SIZE_BYTES = 100_000_000

_THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_FILE_MIME_TYPES = {
    ".docx": DOCX_MIME_TYPE,
    ".txt": TXT_MIME_TYPE,
    ".xlsx": XLSX_MIME_TYPE,
}


_ROUTE_MODES = {
    "auto": "driving",
    "car": "driving",
    "drive": "driving",
    "driving": "driving",
    "walk": "walking",
    "walking": "walking",
    "foot": "walking",
    "bike": "cycling",
    "bicycle": "cycling",
    "bicycling": "cycling",
    "cycling": "cycling",
    "ride": "cycling",
    "riding": "cycling",
    "bus": "transit",
    "public_transport": "transit",
    "public-transit": "transit",
    "transit": "transit",
}

_AMAP_URI_MODES = {
    "driving": "car",
    "walking": "walk",
    "cycling": "ride",
    "transit": "bus",
}

_MARKER_ROLES = frozenset({"origin", "destination", "waypoint", "marker"})


def normalize_artifact(value: Any, *, tool_call_id: str = "") -> dict[str, Any] | None:
    """Normalize a graph artifact value, returning only supported safe envelopes."""

    if not isinstance(value, Mapping):
        return None

    artifact_type = _token(value.get("type"))
    if artifact_type == AMAP_TYPE:
        return normalize_amap_artifact(value, tool_call_id=tool_call_id)
    elif artifact_type == FILE_TYPE:
        return normalize_file_artifact(value, tool_call_id=tool_call_id)

    return None


def normalize_amap_artifact(value: Any, *, tool_call_id: str = "") -> dict[str, Any] | None:
    """Return a safe AMap artifact envelope or ``None`` for unsupported input.

    The input is treated as untrusted tool output. Only the primitive AMap
    envelope fields used by the graph transport are copied into a rebuilt object;
    unknown fields are intentionally discarded.
    """

    if not isinstance(value, Mapping):
        return None
    if _token(value.get("type")) != AMAP_TYPE:
        return None
    version = value.get("version")
    if _is_bool(version) or not isinstance(version, int) or version != AMAP_VERSION:
        return None
    if _token(value.get("kind")) not in {AMAP_KIND_MARKER, AMAP_KIND_ROUTE}:
        return None
    if _token(value.get("coordinateSystem")) != AMAP_COORDINATE_SYSTEM:
        return None
    if _token(value.get("provider")) != AMAP_PROVIDER:
        return None

    kind = _token(value.get("kind"))
    body = (
        _normalize_marker_envelope(value)
        if kind == AMAP_KIND_MARKER
        else _normalize_route_envelope(value)
    )
    if body is None:
        return None

    artifact_id = _stable_artifact_id(body, prefix="amap-")
    safe_tool_call_id = _bounded_string(tool_call_id, MAX_TOOL_CALL_ID_CHARS)
    return {"id": artifact_id, "tool_call_id": safe_tool_call_id, **body}


def normalize_file_artifact(value: Any, *, tool_call_id: str = "") -> dict[str, Any] | None:
    """Return a safe file artifact envelope or ``None`` for unsupported input.

    The input is treated as untrusted tool output. Only known primitive fields
    are copied into a rebuilt object; unknown fields are intentionally discarded.
    The ``url`` field is rebuilt from validated components to prevent arbitrary
    URLs from being accepted.
    """

    if not isinstance(value, Mapping):
        return None
    if _token(value.get("type")) != FILE_TYPE:
        return None
    version = value.get("version")
    if _is_bool(version) or not isinstance(version, int) or version != FILE_VERSION:
        return None
    if _token(value.get("kind")) != FILE_KIND_DOWNLOAD:
        return None
    if _token(value.get("provider")) != FILE_PROVIDER:
        return None

    # Validate threadId: non-empty, bounded, matches pattern
    raw_thread_id = value.get("threadId")
    if not isinstance(raw_thread_id, str):
        return None
    thread_id = raw_thread_id.strip()
    if thread_id != raw_thread_id or len(thread_id) > MAX_THREAD_ID_CHARS:
        return None
    if not thread_id or not _THREAD_ID_PATTERN.fullmatch(thread_id):
        return None

    # Validate filename: non-empty, bounded, bare basename only
    raw_filename = value.get("filename")
    if not isinstance(raw_filename, str):
        return None
    filename = raw_filename.strip()
    if filename != raw_filename or len(filename) > MAX_FILENAME_CHARS:
        return None
    if not filename:
        return None
    if "/" in filename or "\\" in filename or "\x00" in filename:
        return None
    if filename in (".", ".."):
        return None
    suffix = f".{filename.rpartition('.')[2].lower()}"
    mime_type = _FILE_MIME_TYPES.get(suffix)
    if mime_type is None:
        return None

    # Validate sizeBytes: non-negative int within the transport limit.
    raw_size = value.get("sizeBytes")
    if _is_bool(raw_size) or raw_size is None:
        return None
    if not isinstance(raw_size, int):
        return None
    if raw_size < 0 or raw_size > MAX_FILE_SIZE_BYTES:
        return None
    size_bytes = raw_size

    # SECURITY: rebuild URL from validated components, never trust incoming url
    quoted_thread = quote(thread_id, safe="")
    quoted_filename = quote(filename, safe="")
    url = f"/chat/{quoted_thread}/files/{quoted_filename}"

    body: dict[str, Any] = {
        "type": FILE_TYPE,
        "version": FILE_VERSION,
        "kind": FILE_KIND_DOWNLOAD,
        "provider": FILE_PROVIDER,
        "threadId": thread_id,
        "filename": filename,
        "mimeType": mime_type,
        "sizeBytes": size_bytes,
        "url": url,
    }

    artifact_id = _stable_artifact_id(body, prefix="file-")
    safe_tool_call_id = _bounded_string(tool_call_id, MAX_TOOL_CALL_ID_CHARS)
    return {"id": artifact_id, "tool_call_id": safe_tool_call_id, **body}



def extract_artifacts_from_messages(messages: Any) -> list[dict[str, Any]]:
    """Extract safe artifacts from ``ToolMessage.artifact`` values in order."""

    artifacts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for message in _iter_messages(messages):
        if not _is_tool_message(message):
            continue
        tool_call_id = _bounded_string(
            str(getattr(message, "tool_call_id", "") or ""),
            MAX_TOOL_CALL_ID_CHARS,
        )
        for candidate in _iter_artifact_candidates(getattr(message, "artifact", None)):
            artifact = normalize_artifact(candidate, tool_call_id=tool_call_id)
            if artifact is None:
                continue
            artifact_id = str(artifact["id"])
            if artifact_id in seen:
                continue
            seen.add(artifact_id)
            artifacts.append(artifact)
    return artifacts


def extract_artifacts_from_node_output(node_output: Any) -> list[dict[str, Any]]:
    """Extract safe artifacts from a LangGraph node output."""

    if isinstance(node_output, Mapping):
        artifacts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for key in ("messages", "message"):
            for artifact in extract_artifacts_from_messages(node_output.get(key)):
                artifact_id = str(artifact["id"])
                if artifact_id in seen:
                    continue
                seen.add(artifact_id)
                artifacts.append(artifact)
        return artifacts
    return extract_artifacts_from_messages(node_output)


def extract_artifacts_from_chunk(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract safe artifacts from one updates-mode graph chunk."""

    artifacts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node_output in output.values():
        for artifact in extract_artifacts_from_node_output(node_output):
            artifact_id = str(artifact["id"])
            if artifact_id in seen:
                continue
            seen.add(artifact_id)
            artifacts.append(artifact)
    return artifacts


def extract_amap_artifacts_from_messages(messages: Any) -> list[dict[str, Any]]:
    """Extract safe AMap artifacts from ``ToolMessage.artifact`` values in order."""

    return [
        artifact
        for artifact in extract_artifacts_from_messages(messages)
        if artifact.get("type") == AMAP_TYPE
    ]


def extract_amap_artifacts_from_node_output(node_output: Any) -> list[dict[str, Any]]:
    """Extract safe AMap artifacts from a LangGraph node output."""

    return [
        artifact
        for artifact in extract_artifacts_from_node_output(node_output)
        if artifact.get("type") == AMAP_TYPE
    ]


def _normalize_marker_envelope(value: Mapping[str, Any]) -> dict[str, Any] | None:
    markers = _markers_from_envelope(value)
    if not markers:
        return None

    title = _first_bounded_string(value, ("title", "name", "label"), MAX_TITLE_CHARS)
    if not title:
        title = str(markers[0].get("title", "") or "")
    elif not markers[0].get("title"):
        markers[0] = {**markers[0], "title": title}

    fallback_url = _safe_amap_url(value) or _marker_fallback_url(markers[0], title)
    positions = [marker["position"] for marker in markers if "position" in marker]

    body: dict[str, Any] = _base_envelope(AMAP_KIND_MARKER)
    if title:
        body["title"] = title
    body["positions"] = positions
    body["fallbackUrl"] = fallback_url
    body["markers"] = markers
    body["url"] = fallback_url
    return body


def _normalize_route_envelope(value: Mapping[str, Any]) -> dict[str, Any] | None:
    mode = _normalize_route_mode(value.get("mode", "driving"))
    if mode is None:
        return None

    polyline = _normalize_polyline(
        _first_present(value, ("polyline", "path", "points")),
        max_points=MAX_POLYLINE_POINTS,
    )
    markers = _route_markers_from_envelope(value)
    if len(markers) < 2 and len(polyline) >= 2:
        markers = [
            {"role": "origin", "position": polyline[0]},
            {"role": "destination", "position": polyline[-1]},
        ]
    if len(markers) < 2:
        return None

    fallback_url = _safe_amap_url(value) or _route_fallback_url(markers, mode)
    positions = _normalize_positions(value.get("positions"))
    if not positions:
        positions = [marker["position"] for marker in markers[:MAX_MARKERS] if "position" in marker]

    body: dict[str, Any] = _base_envelope(AMAP_KIND_ROUTE)
    title = _first_bounded_string(value, ("title", "name", "label"), MAX_TITLE_CHARS)
    if title:
        body["title"] = title
    body["mode"] = mode
    body["positions"] = positions
    body["fallbackUrl"] = fallback_url
    body["markers"] = markers[:MAX_MARKERS]

    distance_meters = _first_nonnegative_number(
        value,
        ("distanceMeters", "distance_meters", "distanceMeter", "distance_m", "distance"),
    )
    if distance_meters is not None:
        body["distanceMeters"] = distance_meters

    duration_seconds = _first_nonnegative_number(
        value,
        ("durationSeconds", "duration_seconds", "durationSecond", "duration_s", "duration"),
    )
    if duration_seconds is not None:
        body["durationSeconds"] = duration_seconds

    steps = _normalize_steps(value.get("steps"))
    if steps:
        body["steps"] = steps
    if polyline:
        body["polyline"] = polyline
    body["url"] = fallback_url
    return body


def _base_envelope(kind: str) -> dict[str, Any]:
    return {
        "type": AMAP_TYPE,
        "version": AMAP_VERSION,
        "kind": kind,
        "coordinateSystem": AMAP_COORDINATE_SYSTEM,
        "provider": AMAP_PROVIDER,
    }


def _markers_from_envelope(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    marker_values = value.get("markers")
    if marker_values is None:
        marker_values = value.get("marker")
    markers = _normalize_markers(marker_values)
    if markers:
        return markers

    positions = _normalize_positions(value.get("positions"))
    if positions:
        return [
            {"position": position, "role": "marker"}
            for position in positions[:MAX_MARKERS]
        ]

    marker = _normalize_marker(value, fallback_role="marker")
    return [marker] if marker is not None else []


def _route_markers_from_envelope(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    origin = _normalize_marker(
        _first_present(value, ("origin", "start", "from")),
        fallback_role="origin",
        fallback_title="Origin",
    )
    destination = _normalize_marker(
        _first_present(value, ("destination", "dest", "end", "to")),
        fallback_role="destination",
        fallback_title="Destination",
    )
    if origin is not None and destination is not None:
        waypoints = _normalize_markers(value.get("waypoints"), fallback_role="waypoint")
        return [origin, *waypoints, destination][:MAX_MARKERS]

    markers = _normalize_markers(value.get("markers"))
    if markers:
        return markers

    positions = _normalize_positions(value.get("positions"))
    if len(positions) >= 2:
        origin_title = _first_bounded_string(
            value,
            ("originLabel", "origin_label"),
            MAX_TITLE_CHARS,
        ) or "Origin"
        destination_title = _first_bounded_string(
            value,
            ("destinationLabel", "destination_label"),
            MAX_TITLE_CHARS,
        ) or "Destination"
        return [
            {"position": positions[0], "role": "origin", "title": origin_title},
            *[
                {"position": position, "role": "waypoint"}
                for position in positions[1:-1]
            ],
            {
                "position": positions[-1],
                "role": "destination",
                "title": destination_title,
            },
        ][:MAX_MARKERS]
    return []


def _normalize_markers(value: Any, *, fallback_role: str = "") -> list[dict[str, Any]]:
    if value is None:
        return []

    if isinstance(value, (Mapping, str)) or _is_coordinate_pair_sequence(value):
        candidates: Iterable[Any] = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str)):
        candidates = value
    else:
        return []

    markers: list[dict[str, Any]] = []
    for candidate in candidates:
        marker = _normalize_marker(candidate, fallback_role=fallback_role)
        if marker is None:
            continue
        markers.append(marker)
        if len(markers) >= MAX_MARKERS:
            break
    return markers


def _normalize_marker(
    value: Any,
    *,
    fallback_role: str = "",
    fallback_title: str = "",
) -> dict[str, Any] | None:
    position = _position_from_value(value)
    if position is None:
        return None

    marker: dict[str, Any] = {"position": position}
    if isinstance(value, Mapping):
        title = _first_bounded_string(value, ("title", "name", "label"), MAX_TITLE_CHARS)
        address = _first_bounded_string(value, ("address", "formattedAddress"), MAX_ADDRESS_CHARS)
        role = _token(value.get("role"))
        if role not in _MARKER_ROLES:
            role = fallback_role
    else:
        title = ""
        address = ""
        role = fallback_role

    if not title:
        title = _bounded_string(fallback_title, MAX_TITLE_CHARS)
    if title:
        marker["title"] = title
    if address:
        marker["address"] = address
    if role in _MARKER_ROLES:
        marker["role"] = role
    return marker


def _normalize_positions(value: Any) -> list[dict[str, float]]:
    positions = _normalize_polyline(value, max_points=MAX_MARKERS)
    return positions[:MAX_MARKERS]


def _normalize_steps(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        candidates: Iterable[Any] = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str)):
        candidates = value
    else:
        return []

    steps: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        step: dict[str, Any] = {}
        instruction = _first_bounded_string(
            candidate,
            ("instruction", "text", "name", "road"),
            MAX_LABEL_CHARS,
        )
        if instruction:
            step["instruction"] = instruction

        distance_meters = _first_nonnegative_number(
            candidate,
            ("distanceMeters", "distance_meters", "distanceMeter", "distance_m", "distance"),
        )
        if distance_meters is not None:
            step["distanceMeters"] = distance_meters

        duration_seconds = _first_nonnegative_number(
            candidate,
            ("durationSeconds", "duration_seconds", "durationSecond", "duration_s", "duration"),
        )
        if duration_seconds is not None:
            step["durationSeconds"] = duration_seconds

        polyline = _normalize_polyline(
            _first_present(candidate, ("polyline", "path", "points")),
            max_points=MAX_STEP_POLYLINE_POINTS,
        )
        if polyline:
            step["polyline"] = polyline

        if step:
            steps.append(step)
        if len(steps) >= MAX_STEPS:
            break
    return steps


def _normalize_polyline(value: Any, *, max_points: int) -> list[dict[str, float]]:
    if value is None:
        return []

    if isinstance(value, str):
        candidates: Iterable[Any] = [part for part in value.replace("|", ";").split(";") if part]
    elif isinstance(value, Mapping) or _is_coordinate_pair_sequence(value):
        candidates = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str)):
        candidates = value
    else:
        return []

    points: list[dict[str, float]] = []
    for candidate in candidates:
        point = _position_from_value(candidate)
        if point is None:
            continue
        points.append(point)
        if len(points) >= max_points:
            break
    return points


def _position_from_value(value: Any) -> dict[str, float] | None:
    if isinstance(value, Mapping):
        direct = _position_from_mapping(value)
        if direct is not None:
            return direct
        for key in ("position", "coordinate", "coordinates", "location", "point"):
            nested = value.get(key)
            nested_position = _position_from_value(nested)
            if nested_position is not None:
                return nested_position
        return None

    if isinstance(value, str):
        return _position_from_lng_lat_pair(value.split(","))

    if _is_coordinate_pair_sequence(value):
        return _position_from_lng_lat_pair(value)
    return None


def _position_from_mapping(value: Mapping[str, Any]) -> dict[str, float] | None:
    lat_value = _first_present(value, ("lat", "latitude"))
    lng_value = _first_present(value, ("lng", "lon", "longitude"))
    if lat_value is None or lng_value is None:
        return None
    lat = _finite_number(lat_value)
    lng = _finite_number(lng_value)
    if lat is None or lng is None:
        return None
    return _position_from_lat_lng(lat, lng)


def _position_from_lng_lat_pair(values: Sequence[Any]) -> dict[str, float] | None:
    if len(values) < 2:
        return None
    lng = _finite_number(values[0])
    lat = _finite_number(values[1])
    if lat is None or lng is None:
        return None
    return _position_from_lat_lng(lat, lng)


def _position_from_lat_lng(lat: float, lng: float) -> dict[str, float] | None:
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
        return None
    return {"lat": lat, "lng": lng}


def _safe_amap_url(value: Mapping[str, Any]) -> str:
    raw_url = _first_bounded_string(
        value,
        ("url", "fallbackUrl", "fallback_url", "amapUrl", "amap_url", "uri"),
        MAX_URL_CHARS,
    )
    if not raw_url:
        return ""

    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"}:
        return ""
    if parsed.hostname != AMAP_URI_HOST:
        return ""
    if parsed.username or parsed.password or parsed.port is not None:
        return ""

    safe = parsed._replace(scheme="https", netloc=AMAP_URI_HOST).geturl()
    if len(safe) > MAX_URL_CHARS:
        return ""
    return safe


def _marker_fallback_url(marker: Mapping[str, Any], title: str) -> str:
    position = marker["position"]
    label = _url_label(title or str(marker.get("title", "") or "Marker"))
    params = {
        "position": _format_lng_lat(position),
        "name": label,
        "coordinate": "gaode",
        "callnative": "0",
    }
    return f"https://{AMAP_URI_HOST}/marker?{urlencode(params, safe=',')}"


def _route_fallback_url(markers: list[dict[str, Any]], mode: str) -> str:
    origin = markers[0]
    destination = markers[-1]
    params = {
        "from": f"{_format_lng_lat(origin['position'])},{_url_label(str(origin.get('title', '') or 'Origin'))}",
        "to": f"{_format_lng_lat(destination['position'])},{_url_label(str(destination.get('title', '') or 'Destination'))}",
        "mode": _AMAP_URI_MODES[mode],
        "coordinate": "gaode",
        "callnative": "0",
    }
    return f"https://{AMAP_URI_HOST}/navigation?{urlencode(params, safe=',')}"


def _format_lng_lat(position: Mapping[str, Any]) -> str:
    return f"{position['lng']:.6f},{position['lat']:.6f}"


def _url_label(value: str) -> str:
    return _bounded_string(value.replace(",", " "), MAX_LABEL_CHARS)


def _normalize_route_mode(value: Any) -> str | None:
    mode = _token(value)
    if not mode:
        return "driving"
    return _ROUTE_MODES.get(mode)


def _stable_artifact_id(body: Mapping[str, Any], *, prefix: str) -> str:
    payload = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"


def _iter_artifact_candidates(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        artifact_type = _token(value.get("type"))
        if artifact_type in {AMAP_TYPE, FILE_TYPE}:
            yield value
            return
        nested = value.get("artifacts")
        if isinstance(nested, Iterable) and not isinstance(nested, (bytes, bytearray, str, Mapping)):
            yield from _iter_artifact_candidates(nested)
        return

    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str)):
        for item in value:
            yield from _iter_artifact_candidates(item)


def _iter_messages(messages: Any) -> Iterable[Any]:
    if messages is None:
        return ()
    if _is_tool_message(messages):
        return (messages,)
    if isinstance(messages, Iterable) and not isinstance(messages, (bytes, bytearray, str, Mapping)):
        return messages
    return (messages,)


def _is_tool_message(message: Any) -> bool:
    return isinstance(message, ToolMessage) or type(message).__name__ == "ToolMessage"


def _first_bounded_string(
    values: Mapping[str, Any],
    keys: tuple[str, ...],
    max_chars: int,
) -> str:
    for key in keys:
        value = values.get(key)
        text = _bounded_string(value, max_chars)
        if text:
            return text
    return ""


def _bounded_string(value: Any, max_chars: int) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.replace("\x00", " ").split())
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _first_nonnegative_number(values: Mapping[str, Any], keys: tuple[str, ...]) -> int | float | None:
    for key in keys:
        number = _finite_number(values.get(key))
        if number is None or number < 0:
            continue
        return _compact_number(number)
    return None


def _finite_number(value: Any) -> float | None:
    if _is_bool(value) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _compact_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _first_present(values: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in values:
            return values[key]
    return None


def _token(value: Any) -> str:
    return _bounded_string(value, MAX_LABEL_CHARS).strip().lower()


def _is_coordinate_pair_sequence(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        return False
    return len(value) >= 2 and _finite_number(value[0]) is not None and _finite_number(value[1]) is not None


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


__all__ = [
    "AMAP_COORDINATE_SYSTEM",
    "AMAP_KIND_MARKER",
    "AMAP_KIND_ROUTE",
    "AMAP_PROVIDER",
    "AMAP_TYPE",
    "AMAP_VERSION",
    "ArtifactEvent",
    "DOCX_MIME_TYPE",
    "FILE_KIND_DOWNLOAD",
    "FILE_PROVIDER",
    "FILE_TYPE",
    "FILE_VERSION",
    "MAX_FILENAME_CHARS",
    "MAX_FILE_SIZE_BYTES",
    "MAX_MARKERS",
    "MAX_MIME_CHARS",
    "MAX_POLYLINE_POINTS",
    "MAX_STEPS",
    "MAX_THREAD_ID_CHARS",
    "TXT_MIME_TYPE",
    "XLSX_MIME_TYPE",
    "extract_amap_artifacts_from_messages",
    "extract_amap_artifacts_from_node_output",
    "extract_artifacts_from_chunk",
    "extract_artifacts_from_messages",
    "extract_artifacts_from_node_output",
    "normalize_amap_artifact",
    "normalize_artifact",
    "normalize_file_artifact",
]
