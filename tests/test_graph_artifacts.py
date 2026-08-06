from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from src.graph.artifacts import (
    MAX_FILE_SIZE_BYTES,
    MAX_MARKERS,
    MAX_POLYLINE_POINTS,
    MAX_STEPS,
    extract_amap_artifacts_from_messages,
    extract_artifacts_from_messages,
    extract_artifacts_from_node_output,
    normalize_amap_artifact,
    normalize_file_artifact,
)


def test_normalize_marker_rebuilds_safe_amap_envelope_with_https_url():
    raw = {
        "type": "amap",
        "version": 1,
        "kind": "marker",
        "coordinateSystem": "gcj02",
        "provider": "amap",
        "title": "  Shanghai   Office  ",
        "url": "http://uri.amap.com/marker?position=121.4737,31.2304&name=Shanghai",
        "markers": [
            {
                "position": {"lng": 121.4737, "lat": 31.2304},
                "title": "Office",
                "address": "People's Square",
                "role": "marker",
                "secret": "discard-me",
            }
        ],
        "unexpected": {"must": "be discarded"},
    }

    artifact = normalize_amap_artifact(raw, tool_call_id="call-1")

    assert artifact is not None
    assert artifact["id"].startswith("amap-")
    assert artifact["tool_call_id"] == "call-1"
    assert artifact["type"] == "amap"
    assert artifact["version"] == 1
    assert artifact["kind"] == "marker"
    assert artifact["coordinateSystem"] == "gcj02"
    assert artifact["provider"] == "amap"
    assert artifact["title"] == "Shanghai Office"
    assert artifact["url"].startswith("https://uri.amap.com/marker?")
    assert artifact["markers"] == [
        {
            "position": {"lat": 31.2304, "lng": 121.4737},
            "title": "Office",
            "address": "People's Square",
            "role": "marker",
        }
    ]
    assert "unexpected" not in artifact
    assert "secret" not in artifact["markers"][0]


def test_normalize_marker_uses_bounded_strings_and_marker_cap():
    raw = {
        "type": "amap",
        "version": 1,
        "kind": "marker",
        "coordinateSystem": "gcj02",
        "provider": "amap",
        "title": "x" * 200,
        "markers": [
            {"lng": 120.0 + index * 0.001, "lat": 30.0, "title": f"M{index}"}
            for index in range(MAX_MARKERS + 5)
        ],
    }

    artifact = normalize_amap_artifact(raw, tool_call_id="tool-call-" + "y" * 300)

    assert artifact is not None
    assert len(artifact["title"]) == 120
    assert len(artifact["tool_call_id"]) == 160
    assert len(artifact["markers"]) == MAX_MARKERS
    assert artifact["markers"][-1]["title"] == f"M{MAX_MARKERS - 1}"
    assert artifact["url"].startswith("https://uri.amap.com/marker?")


def test_normalize_marker_applies_top_level_label_to_position_artifact():
    raw = {
        "type": "amap",
        "version": 1,
        "kind": "marker",
        "coordinateSystem": "gcj02",
        "provider": "amap",
        "label": "People's Square",
        "positions": [{"lng": 121.4737, "lat": 31.2304}],
    }

    artifact = normalize_amap_artifact(raw)

    assert artifact is not None
    assert artifact["title"] == "People's Square"
    assert artifact["markers"][0]["title"] == "People's Square"


def test_normalize_route_accepts_known_modes_steps_polyline_and_discards_unknown_url():
    raw = {
        "type": "amap",
        "version": 1,
        "kind": "route",
        "coordinateSystem": "gcj02",
        "provider": "amap",
        "mode": "car",
        "url": "https://evil.example/route",
        "origin": {"lng": 116.3975, "lat": 39.9087, "title": "Start", "extra": "no"},
        "destination": {"position": "116.4075,39.9187", "title": "End"},
        "distance": 1234.5,
        "duration": "600",
        "steps": [
            {
                "instruction": "Head north",
                "distance": 100,
                "duration": 60,
                "polyline": "116.3975,39.9087;116.4000,39.9100",
                "unsafe": "discard",
            }
        ],
        "polyline": [[116.3975, 39.9087], [116.4075, 39.9187]],
        "unknown": True,
    }

    artifact = normalize_amap_artifact(raw)

    assert artifact is not None
    assert artifact["kind"] == "route"
    assert artifact["mode"] == "driving"
    assert artifact["distanceMeters"] == 1234.5
    assert artifact["durationSeconds"] == 600
    assert artifact["markers"] == [
        {"position": {"lat": 39.9087, "lng": 116.3975}, "title": "Start", "role": "origin"},
        {"position": {"lat": 39.9187, "lng": 116.4075}, "title": "End", "role": "destination"},
    ]
    assert artifact["steps"] == [
        {
            "instruction": "Head north",
            "distanceMeters": 100,
            "durationSeconds": 60,
            "polyline": [
                {"lat": 39.9087, "lng": 116.3975},
                {"lat": 39.91, "lng": 116.4},
            ],
        }
    ]
    assert artifact["polyline"] == [
        {"lat": 39.9087, "lng": 116.3975},
        {"lat": 39.9187, "lng": 116.4075},
    ]
    assert artifact["url"].startswith("https://uri.amap.com/navigation?")
    assert "unknown" not in artifact
    assert "unsafe" not in artifact["steps"][0]


def test_normalize_route_preserves_labels_from_tool_artifact_positions():
    raw = {
        "type": "amap",
        "version": 1,
        "kind": "route",
        "coordinateSystem": "gcj02",
        "provider": "amap",
        "mode": "walking",
        "positions": [
            {"lng": 116.3975, "lat": 39.9087},
            {"lng": 116.4075, "lat": 39.9187},
        ],
        "originLabel": "Tiananmen",
        "destinationLabel": "Wangfujing",
    }

    artifact = normalize_amap_artifact(raw)

    assert artifact is not None
    assert artifact["markers"] == [
        {
            "position": {"lat": 39.9087, "lng": 116.3975},
            "role": "origin",
            "title": "Tiananmen",
        },
        {
            "position": {"lat": 39.9187, "lng": 116.4075},
            "role": "destination",
            "title": "Wangfujing",
        },
    ]


def test_normalize_route_caps_steps_and_polyline_points():
    raw = {
        "type": "amap",
        "version": 1,
        "kind": "route",
        "coordinateSystem": "gcj02",
        "provider": "amap",
        "mode": "walking",
        "origin": [116.0, 39.0],
        "destination": [116.5, 39.5],
        "steps": [
            {"instruction": f"Step {index}", "distance": index}
            for index in range(MAX_STEPS + 10)
        ],
        "polyline": [[116.0 + index * 0.0001, 39.0] for index in range(MAX_POLYLINE_POINTS + 10)],
    }

    artifact = normalize_amap_artifact(raw)

    assert artifact is not None
    assert len(artifact["steps"]) == MAX_STEPS
    assert artifact["steps"][-1]["instruction"] == f"Step {MAX_STEPS - 1}"
    assert len(artifact["polyline"]) == MAX_POLYLINE_POINTS


def test_invalid_amap_envelopes_are_rejected():
    base = {
        "type": "amap",
        "version": 1,
        "kind": "marker",
        "coordinateSystem": "gcj02",
        "provider": "amap",
        "lng": 116.3975,
        "lat": 39.9087,
    }

    invalid_cases = [
        {**base, "type": "map"},
        {**base, "version": 2},
        {**base, "version": True},
        {**base, "kind": "heatmap"},
        {**base, "coordinateSystem": "wgs84"},
        {**base, "provider": "osm"},
        {**base, "lat": float("nan")},
        {**base, "lat": 91},
        {**base, "lng": 181},
        {
            **base,
            "kind": "route",
            "mode": "teleport",
            "origin": [116.0, 39.0],
            "destination": [116.1, 39.1],
        },
        {**base, "kind": "route", "mode": "walking", "origin": [116.0, 39.0]},
    ]

    assert [normalize_amap_artifact(case) for case in invalid_cases] == [None] * len(
        invalid_cases
    )


def test_extract_artifacts_from_tool_messages_preserves_order_and_dedupes():
    marker_one = {
        "type": "amap",
        "version": 1,
        "kind": "marker",
        "coordinateSystem": "gcj02",
        "provider": "amap",
        "lng": 116.3975,
        "lat": 39.9087,
        "title": "One",
    }
    marker_two = {
        "type": "amap",
        "version": 1,
        "kind": "marker",
        "coordinateSystem": "gcj02",
        "provider": "amap",
        "lng": 121.4737,
        "lat": 31.2304,
        "title": "Two",
    }
    messages = [
        AIMessage(content="ignore me"),
        ToolMessage(content="first", tool_call_id="call-a", artifact=[marker_one, marker_two]),
        ToolMessage(content="duplicate", tool_call_id="call-b", artifact=marker_one),
    ]

    artifacts = extract_artifacts_from_messages(messages)

    assert [artifact["markers"][0]["title"] for artifact in artifacts] == ["One", "Two"]
    assert artifacts[0]["tool_call_id"] == "call-a"
    assert artifacts[1]["tool_call_id"] == "call-a"


def test_extract_artifacts_from_node_output_reads_message_fields():
    marker = {
        "type": "amap",
        "version": 1,
        "kind": "marker",
        "coordinateSystem": "gcj02",
        "provider": "amap",
        "lng": 116.3975,
        "lat": 39.9087,
    }
    output = {"messages": [ToolMessage(content="map", tool_call_id="call-1", artifact=marker)]}

    artifacts = extract_artifacts_from_node_output(output)

    assert len(artifacts) == 1
    assert artifacts[0]["tool_call_id"] == "call-1"
    assert artifacts[0]["markers"][0]["position"] == {"lat": 39.9087, "lng": 116.3975}


def test_normalize_file_artifact_rebuilds_safe_download_url():
    raw = {
        "type": "file",
        "version": 1,
        "kind": "download",
        "provider": "chat_upload",
        "threadId": "thread-a",
        "filename": "Q3 report.edited.docx",
        "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "sizeBytes": 12345,
        "url": "https://evil.example/steal",
        "secret": "discard",
    }

    artifact = normalize_file_artifact(raw, tool_call_id="call-file")

    assert artifact is not None
    assert artifact["id"].startswith("file-")
    assert artifact["tool_call_id"] == "call-file"
    assert artifact["url"] == "/chat/thread-a/files/Q3%20report.edited.docx"
    assert artifact["sizeBytes"] == 12345
    assert "secret" not in artifact


def test_invalid_file_artifacts_are_rejected_instead_of_rewritten_or_clamped():
    base = {
        "type": "file",
        "version": 1,
        "kind": "download",
        "provider": "chat_upload",
        "threadId": "thread-a",
        "filename": "report.docx",
        "sizeBytes": 123,
    }
    invalid_cases = [
        {**base, "version": True},
        {**base, "kind": "inline"},
        {**base, "provider": "external"},
        {**base, "threadId": "../thread-b"},
        {**base, "threadId": "x" * 65},
        {**base, "filename": "../report.docx"},
        {**base, "filename": "folder\\report.docx"},
        {**base, "filename": "report.txt"},
        {**base, "sizeBytes": -1},
        {**base, "sizeBytes": True},
        {**base, "sizeBytes": 1.5},
        {**base, "sizeBytes": MAX_FILE_SIZE_BYTES + 1},
    ]

    assert [normalize_file_artifact(case) for case in invalid_cases] == [None] * len(
        invalid_cases
    )


def test_generic_extractor_keeps_files_while_legacy_amap_extractor_filters_them():
    marker = {
        "type": "amap",
        "version": 1,
        "kind": "marker",
        "coordinateSystem": "gcj02",
        "provider": "amap",
        "lng": 116.3975,
        "lat": 39.9087,
    }
    file_artifact = {
        "type": "file",
        "version": 1,
        "kind": "download",
        "provider": "chat_upload",
        "threadId": "thread-a",
        "filename": "report.edited.docx",
        "sizeBytes": 123,
    }
    messages = [
        ToolMessage(
            content="created",
            tool_call_id="call-1",
            artifact=[marker, file_artifact],
        )
    ]

    assert [
        artifact["type"] for artifact in extract_artifacts_from_messages(messages)
    ] == ["amap", "file"]
    assert [
        artifact["type"] for artifact in extract_amap_artifacts_from_messages(messages)
    ] == ["amap"]
