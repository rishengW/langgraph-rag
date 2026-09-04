from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.backend.tools.json_edit import (
    JSON_MIME_TYPE,
    JsonEditOperation,
    build_json_edit_tools,
    create_json_file,
    edit_json_file,
    inspect_json_file,
)

SAMPLE = json.dumps(
    {
        "name": "app",
        "port": 8080,
        "debug": False,
        "tags": ["web", "api"],
        "owner": {"team": "core"},
    },
    indent=2,
)


@pytest.fixture
def json_scope(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    file_root = tmp_path / "files"
    thread_id = "thread-a"
    session_root = file_root / "chat_uploads" / thread_id
    session_root.mkdir(parents=True)
    source = session_root / "app.json"
    source.write_text(SAMPLE, encoding="utf-8")
    return file_root, session_root, source, thread_id


def _edit(path, *, operations, json_scope, output_name=None):
    file_root, session_root, _source, thread_id = json_scope
    return edit_json_file(
        path,
        operations=operations,
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
        thread_id=thread_id,
        output_name=output_name,
    )


def test_inspect_lists_documented_paths_and_values(json_scope):
    file_root, session_root, source, _ = json_scope

    result = inspect_json_file(
        source.name,
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
    )

    assert "$ = object (5 keys)" in result
    assert '$.name = "app"' in result
    assert "$.port = 8080" in result
    assert "$.tags = array (2 items)" in result
    assert '$.tags[1] = "api"' in result
    assert '$.owner.team = "core"' in result


def test_inspect_reports_invalid_json_without_listing(json_scope):
    file_root, session_root, source, _ = json_scope
    source.write_text('{"broken": ', encoding="utf-8")

    result = inspect_json_file(
        source.name,
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
    )

    assert result.startswith("Could not inspect JSON file")
    assert "not valid JSON" in result


def test_create_validates_json_and_rejects_invalid(json_scope):
    file_root, session_root, _source, thread_id = json_scope

    good = create_json_file(
        "config.json",
        '{"ok": true}\n',
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
        thread_id=thread_id,
    )
    bad = create_json_file(
        "bad.json",
        "not json",
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
        thread_id=thread_id,
    )

    assert good.artifact is not None
    assert good.artifact["filename"] == "config.json"
    assert good.artifact["mimeType"] == JSON_MIME_TYPE
    assert (session_root / "config.json").read_bytes() == b'{"ok": true}\n'
    assert bad.artifact is None
    assert "not valid JSON" in bad.content
    assert not (session_root / "bad.json").exists()


def test_edit_set_delete_and_append_with_expectations(json_scope):
    file_root, session_root, source, _ = json_scope

    result = _edit(
        source.name,
        json_scope=json_scope,
        operations=[
            JsonEditOperation(
                action="set_value",
                path="$.port",
                value="9090",
                expected_value="8080",
            ),
            JsonEditOperation(
                action="set_value",
                path="$.replicas",
                value="2",
                expected_missing=True,
            ),
            JsonEditOperation(
                action="append_to_array",
                path="$.tags",
                value='"worker"',
                expected_length=2,
            ),
            JsonEditOperation(
                action="delete_key",
                path="$.debug",
                expected_value="false",
            ),
        ],
    )

    assert result.artifact is not None
    assert result.artifact["filename"] == "app.edited.json"
    published = json.loads((session_root / "app.edited.json").read_text("utf-8"))
    assert published == {
        "name": "app",
        "port": 9090,
        "tags": ["web", "api", "worker"],
        "owner": {"team": "core"},
        "replicas": 2,
    }
    # The upload itself is untouched.
    assert json.loads(source.read_text("utf-8"))["port"] == 8080


def test_edit_can_replace_the_whole_document_from_root(json_scope):
    file_root, session_root, source, _ = json_scope

    result = _edit(
        source.name,
        json_scope=json_scope,
        operations=[
            JsonEditOperation(
                action="set_value",
                path="$",
                value='{"replaced": true}',
                expected_value=SAMPLE,
            )
        ],
    )

    assert result.artifact is not None
    published = json.loads((session_root / "app.edited.json").read_text("utf-8"))
    assert published == {"replaced": True}


def test_wrong_expectations_publish_nothing(json_scope):
    file_root, session_root, source, _ = json_scope
    original = source.read_bytes()
    batches = [
        # Wrong expected_value on an existing key.
        JsonEditOperation(action="set_value", path="$.port", value="1", expected_value="7"),
        # expected_missing on a key that exists.
        JsonEditOperation(action="set_value", path="$.name", value='"x"', expected_missing=True),
        # Wrong array length for append.
        JsonEditOperation(action="append_to_array", path="$.tags", value='"x"', expected_length=9),
        # Wrong expected_value for delete.
        JsonEditOperation(action="delete_key", path="$.name", expected_value='"nope"'),
        # Path that does not exist.
        JsonEditOperation(action="delete_key", path="$.ghost", expected_value="null"),
    ]

    for operation in batches:
        result = _edit(source.name, json_scope=json_scope, operations=[operation])
        assert result.artifact is None
        assert "no changes were made" in result.content

    assert source.read_bytes() == original
    assert list(session_root.glob("*.json")) == [source]


def test_edited_copy_preserves_original_indentation_and_newlines(json_scope):
    file_root, session_root, source, _ = json_scope
    indented = json.dumps({"outer": {"keep": 1, "edit": "old"}}, indent=4)
    source.write_bytes(indented.replace("\n", "\r\n").encode("utf-8") + b"\r\n")

    result = _edit(
        source.name,
        json_scope=json_scope,
        operations=[
            JsonEditOperation(
                action="set_value",
                path="$.outer.edit",
                value='"new"',
                expected_value='"old"',
            )
        ],
    )

    assert result.artifact is not None
    raw = (session_root / "app.edited.json").read_bytes()
    assert b"\r\n" in raw
    # Depth-1 keys keep the original 4-space unit; depth-2 gets 8 spaces.
    assert b'\r\n    "outer": {\r\n        "keep": 1,' in raw
    assert json.loads(raw.decode("utf-8"))["outer"]["edit"] == "new"


def test_multiline_new_text_is_not_required_but_values_must_be_json():
    with pytest.raises(ValidationError, match="value is not valid JSON"):
        JsonEditOperation(action="set_value", path="$.a", value="{broken", expected_value="1")
    with pytest.raises(ValidationError, match="exactly one of expected_value"):
        JsonEditOperation(action="set_value", path="$.a", value="1")
    with pytest.raises(ValidationError, match="cannot target the document root"):
        JsonEditOperation(action="delete_key", path="$", expected_value="1")


def test_tool_registration_requires_flags_session_root_and_thread_id(
    isolated_settings, tmp_path: Path
):
    session_root = tmp_path / "files" / "chat_uploads" / "thread-a"
    enabled = isolated_settings(
        file_read_enabled=True,
        json_edit_enabled=True,
        file_read_root=str(tmp_path / "files"),
    )
    disabled = isolated_settings(
        file_read_enabled=True,
        json_edit_enabled=False,
        file_read_root=str(tmp_path / "files"),
    )

    assert build_json_edit_tools(disabled, session_root=session_root, thread_id="thread-a") == []
    assert build_json_edit_tools(enabled, session_root=None, thread_id="thread-a") == []
    assert build_json_edit_tools(enabled, session_root=session_root, thread_id="") == []
    assert [
        tool.name
        for tool in build_json_edit_tools(
            enabled, session_root=session_root, thread_id="thread-a"
        )
    ] == ["create_json_file", "inspect_json_file", "edit_json_file"]
