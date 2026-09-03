from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from langchain_core.messages import ToolMessage

from src.backend.graph.artifacts import (
    ArtifactEvent,
    extract_amap_artifacts_from_messages,
    extract_amap_artifacts_from_node_output,
)
from src.backend.graph.events import ArtifactEvent as GraphArtifactEvent
from src.backend.tools._amap import (
    AMapAPIError,
    AMapPosition,
    amap_request_json,
    amap_route,
    build_amap_marker_uri,
    build_amap_navigation_uri,
    build_marker_artifact,
    convert_wgs84_to_gcj02,
    normalize_travel_mode,
    parse_latlon,
    parse_lnglat,
    parse_polyline,
    route_bounds,
    safe_amap_error,
    validate_amap_envelope,
)


def test_parse_coordinates_distinguishes_amap_lnglat_from_public_latlon():
    assert parse_lnglat("121.47,31.23") == AMapPosition(lng=121.47, lat=31.23)
    assert parse_latlon("31.23,121.47") == AMapPosition(lng=121.47, lat=31.23)
    assert parse_lnglat("31.23,121.47") is None
    assert parse_latlon("121.47,31.23") is None


def test_coordinate_conversion_uses_wgs84_endpoint_and_api_timeout():
    calls: list[tuple[str, dict[str, Any]]] = []

    def requester(url: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((url, kwargs))
        return {"status": "1", "locations": "121.474000,31.234000"}

    converted = convert_wgs84_to_gcj02(
        AMapPosition(lng=121.47, lat=31.23),
        api_key="test-key",
        requester=requester,
        timeout_seconds=7,
    )

    assert converted == AMapPosition(lng=121.474, lat=31.234)
    assert calls[0][0].endswith("/assistant/coordinate/convert")
    assert calls[0][1]["params"]["locations"] == "121.470000,31.230000"
    assert calls[0][1]["params"]["coordsys"] == "gps"
    assert calls[0][1]["params"]["key"] == "test-key"
    assert calls[0][1]["timeout"] == 7


def test_validate_amap_envelope_raises_safe_semantic_errors():
    with pytest.raises(AMapAPIError) as excinfo:
        validate_amap_envelope(
            {
                "status": "0",
                "info": "INVALID_USER_KEY key=secret-value",
                "infocode": "10001",
            },
            context="POI lookup",
        )

    message = str(excinfo.value)
    assert "POI lookup" in message
    assert "secret-value" not in message
    assert "key=***" in message


def test_numeric_zero_is_a_successful_v4_envelope():
    payload = {"errcode": 0, "errmsg": "OK", "data": {"paths": []}}

    assert validate_amap_envelope(payload, context="cycling route") is payload


def test_amap_request_json_does_not_allow_params_to_override_credentials():
    captured: dict[str, Any] = {}

    def requester(_url: str, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs["params"])
        return {"status": "1", "pois": []}

    amap_request_json(
        "https://restapi.amap.com/v3/place/text",
        params={"key": "caller-key", "output": "xml", "keywords": "Shanghai"},
        api_key="configured-key",
        requester=requester,
        context="POI lookup",
    )

    assert captured["key"] == "configured-key"
    assert captured["output"] == "json"


def test_safe_error_redacts_secret_shapes():
    text = safe_amap_error(
        "url?key=secret&jscode=hidden {'key': 'dict-secret'} AMAP_WEB_SERVICE_KEY=env-secret"
    )

    assert "secret" not in text
    assert "hidden" not in text
    assert "key=***" in text
    assert "AMAP_WEB_SERVICE_KEY=***" in text


def test_route_parser_bounds_polyline_and_steps():
    payload = {
        "status": "1",
        "route": {
            "paths": [
                {
                    "distance": "2500",
                    "duration": "900",
                    "steps": [
                        {
                            "instruction": "Head east",
                            "road": "Road A",
                            "distance": "1000",
                            "duration": "300",
                            "polyline": "121.1,31.1;121.2,31.2",
                        },
                        {
                            "instruction": "Arrive",
                            "polyline": "121.2,31.2;121.3,31.4",
                        },
                    ],
                }
            ]
        },
    }

    route = amap_route(
        AMapPosition(lng=121.1, lat=31.1),
        AMapPosition(lng=121.3, lat=31.4),
        mode="walking",
        api_key="test-key",
        requester=lambda _url, **_kwargs: payload,
    )

    assert route is not None
    assert route.distance_m == 2500.0
    assert route.duration_s == 900.0
    assert route.polyline == [
        AMapPosition(lng=121.1, lat=31.1),
        AMapPosition(lng=121.2, lat=31.2),
        AMapPosition(lng=121.2, lat=31.2),
        AMapPosition(lng=121.3, lat=31.4),
    ]
    assert route.bounds == {
        "southwest": {"lng": 121.1, "lat": 31.1},
        "northeast": {"lng": 121.3, "lat": 31.4},
    }
    assert route.steps[0].bounds == {
        "southwest": {"lng": 121.1, "lat": 31.1},
        "northeast": {"lng": 121.2, "lat": 31.2},
    }


def test_route_bounds_and_polyline_sampling():
    polyline = ";".join(f"{index},0" for index in range(10))

    points = parse_polyline(polyline, max_points=3)

    assert points == [
        AMapPosition(lng=0.0, lat=0.0),
        AMapPosition(lng=4.0, lat=0.0),
        AMapPosition(lng=9.0, lat=0.0),
    ]
    assert route_bounds(points) == {
        "southwest": {"lng": 0.0, "lat": 0.0},
        "northeast": {"lng": 9.0, "lat": 0.0},
    }
    assert parse_polyline(polyline, max_points=0) == []


def test_route_parser_rejects_nonfinite_and_negative_metrics_and_reads_v4_step_cost():
    route = amap_route(
        AMapPosition(lng=121.1, lat=31.1),
        AMapPosition(lng=121.3, lat=31.4),
        mode="cycling",
        api_key="test-key",
        requester=lambda _url, **_kwargs: {
            "errcode": 0,
            "data": {
                "paths": [
                    {
                        "distance": "nan",
                        "duration": "-1",
                        "steps": [
                            {
                                "distance": "inf",
                                "cost": {"duration": "90"},
                                "polyline": "121.1,31.1;121.3,31.4",
                            }
                        ],
                    }
                ]
            },
        },
    )

    assert route is not None
    assert route.distance_m is None
    assert route.duration_s is None
    assert route.steps[0].distance_m is None
    assert route.steps[0].duration_s == 90.0


def test_uri_builders_and_artifact_primitives():
    position = AMapPosition(lng=121.47, lat=31.23)
    marker_url = build_amap_marker_uri(position, name="上海")
    nav_url = build_amap_navigation_uri(
        position,
        AMapPosition(lng=120.16, lat=30.29),
        origin_name="上海",
        destination_name="杭州",
        mode="cycling",
    )
    artifact = build_marker_artifact(position, fallback_url=marker_url, label="上海")

    assert marker_url.startswith("https://uri.amap.com/marker?")
    assert "121.470000%2C31.230000" in marker_url
    assert nav_url.startswith("https://uri.amap.com/navigation?")
    assert "mode=bike" in nav_url
    assert artifact == {
        "type": "amap",
        "version": 1,
        "kind": "marker",
        "provider": "amap",
        "coordinateSystem": "gcj02",
        "fallbackUrl": marker_url,
        "positions": [{"lng": 121.47, "lat": 31.23}],
        "label": "上海",
    }


def test_navigation_uri_removes_commas_from_endpoint_labels():
    url = build_amap_navigation_uri(
        AMapPosition(lng=121.320081, lat=31.193964),
        AMapPosition(lng=121.497253, lat=31.238235),
        origin_name="Shanghai Hongqiao Station, Minhang, Shanghai",
        destination_name="The Bund, Huangpu, Shanghai",
    )

    query = parse_qs(urlsplit(url).query)

    assert query["from"] == [
        "121.320081,31.193964,Shanghai Hongqiao Station Minhang Shanghai"
    ]
    assert query["to"] == ["121.497253,31.238235,The Bund Huangpu Shanghai"]
    assert query["from"][0].count(",") == 2
    assert query["to"][0].count(",") == 2


def test_travel_mode_aliases_and_transit_rejection():
    assert normalize_travel_mode("car") == "driving"
    assert normalize_travel_mode("foot") == "walking"
    assert normalize_travel_mode("bicycling") == "cycling"
    assert normalize_travel_mode("transit") is None
    assert normalize_travel_mode("metro") is None


def test_exact_chat_integration_helper_names_and_artifact_event_payload():
    raw_artifact = {
        "type": "amap",
        "version": 1,
        "kind": "marker",
        "provider": "amap",
        "coordinateSystem": "gcj02",
        "fallbackUrl": "https://uri.amap.com/marker?position=121.470000,31.230000",
        "positions": [{"lng": 121.47, "lat": 31.23}],
    }
    message = ToolMessage(content="Map", tool_call_id="call-1", artifact=raw_artifact)

    artifacts = extract_amap_artifacts_from_messages([message])
    node_artifacts = extract_amap_artifacts_from_node_output({"messages": [message]})
    event = ArtifactEvent(artifacts=artifacts, node="tools")

    assert ArtifactEvent is GraphArtifactEvent
    assert artifacts == node_artifacts
    assert artifacts[0]["type"] == "amap"
    assert artifacts[0]["version"] == 1
    assert artifacts[0]["kind"] == "marker"
    assert artifacts[0]["provider"] == "amap"
    assert artifacts[0]["coordinateSystem"] == "gcj02"
    assert artifacts[0]["fallbackUrl"].startswith("https://uri.amap.com/marker")
    assert artifacts[0]["positions"] == [{"lat": 31.23, "lng": 121.47}]
    assert event.artifacts == artifacts
    assert event.node == "tools"
