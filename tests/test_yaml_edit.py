from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.backend.tools.yaml_edit import (
    YAML_MIME_TYPE,
    YamlEditOperation,
    build_yaml_edit_tools,
    create_yaml_file,
    edit_yaml_file,
    inspect_yaml_file,
)

SAMPLE_DATA = {
    "name": "app",
    "port": 8080,
    "debug": False,
    "tags": ["web", "api"],
    "owner": {"team": "core"},
}
SAMPLE = yaml.safe_dump(SAMPLE_DATA, sort_keys=False, default_flow_style=False)
SAMPLE_JSON = json.dumps(SAMPLE_DATA)


@pytest.fixture
def yaml_scope(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    file_root = tmp_path / "files"
    thread_id = "thread-a"
    session_root = file_root / "chat_uploads" / thread_id
    session_root.mkdir(parents=True)
    source = session_root / "app.yaml"
    source.write_text(SAMPLE, encoding="utf-8")
    return file_root, session_root, source, thread_id


def _edit(path, *, operations, yaml_scope, output_name=None):
    file_root, session_root, _source, thread_id = yaml_scope
    return edit_yaml_file(
        path,
        operations=operations,
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
        thread_id=thread_id,
        output_name=output_name,
    )


def test_inspect_lists_documented_paths_and_values(yaml_scope):
    file_root, session_root, source, _ = yaml_scope

    result = inspect_yaml_file(
        source.name,
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
    )

    assert "$ = object (5 keys)" in result
    assert '$.name = "app"' in result
    assert "$.port = 8080" in result
    assert "$.debug = false" in result
    assert "$.tags = array (2 items)" in result
    assert '$.tags[1] = "api"' in result
    assert '$.owner.team = "core"' in result


def test_inspect_reports_invalid_yaml_without_listing(yaml_scope):
    file_root, session_root, source, _ = yaml_scope
    source.write_text("{broken: ", encoding="utf-8")

    result = inspect_yaml_file(
        source.name,
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
    )

    assert result.startswith("Could not inspect YAML file")
    assert "not valid YAML" in result


def test_inspect_refuses_multi_document_stream(yaml_scope):
    file_root, session_root, source, _ = yaml_scope
    source.write_text("a: 1\n---\nb: 2\n", encoding="utf-8")

    result = inspect_yaml_file(
        source.name,
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
    )

    assert result.startswith("Could not inspect YAML file")
    assert "multi-document" in result


def test_create_validates_yaml_and_rejects_invalid(yaml_scope):
    file_root, session_root, _source, thread_id = yaml_scope

    good = create_yaml_file(
        "config.yaml",
        "name: ok\n",
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
        thread_id=thread_id,
    )
    bad = create_yaml_file(
        "bad.yaml",
        "{not closed",
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
        thread_id=thread_id,
    )

    assert good.artifact is not None
    assert good.artifact["filename"] == "config.yaml"
    assert good.artifact["mimeType"] == YAML_MIME_TYPE
    assert (session_root / "config.yaml").read_bytes() == b"name: ok\n"
    assert bad.artifact is None
    assert "not valid YAML" in bad.content
    assert not (session_root / "bad.yaml").exists()


def test_create_preserves_yml_suffix(yaml_scope):
    file_root, session_root, _source, thread_id = yaml_scope

    result = create_yaml_file(
        "config.yml",
        "ok: true\n",
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
        thread_id=thread_id,
    )

    assert result.artifact is not None
    assert result.artifact["filename"] == "config.yml"


def test_edit_set_delete_and_append_with_expectations(yaml_scope):
    file_root, session_root, source, _ = yaml_scope

    result = _edit(
        source.name,
        yaml_scope=yaml_scope,
        operations=[
            YamlEditOperation(
                action="set_value",
                path="$.port",
                value="9090",
                expected_value="8080",
            ),
            YamlEditOperation(
                action="set_value",
                path="$.replicas",
                value="2",
                expected_missing=True,
            ),
            YamlEditOperation(
                action="append_to_array",
                path="$.tags",
                value='"worker"',
                expected_length=2,
            ),
            YamlEditOperation(
                action="delete_key",
                path="$.debug",
                expected_value="false",
            ),
        ],
    )

    assert result.artifact is not None
    assert result.artifact["filename"] == "app.edited.yaml"
    published = yaml.safe_load((session_root / "app.edited.yaml").read_text("utf-8"))
    assert published == {
        "name": "app",
        "port": 9090,
        "tags": ["web", "api", "worker"],
        "owner": {"team": "core"},
        "replicas": 2,
    }
    # The upload itself is untouched.
    assert yaml.safe_load(source.read_text("utf-8"))["port"] == 8080


def test_edit_can_replace_the_whole_document_from_root(yaml_scope):
    file_root, session_root, source, _ = yaml_scope

    result = _edit(
        source.name,
        yaml_scope=yaml_scope,
        operations=[
            YamlEditOperation(
                action="set_value",
                path="$",
                value='{"replaced": true}',
                expected_value=SAMPLE_JSON,
            )
        ],
    )

    assert result.artifact is not None
    published = yaml.safe_load((session_root / "app.edited.yaml").read_text("utf-8"))
    assert published == {"replaced": True}


def test_wrong_expectations_publish_nothing(yaml_scope):
    file_root, session_root, source, _ = yaml_scope
    original = source.read_bytes()
    batches = [
        # Wrong expected_value on an existing key.
        YamlEditOperation(action="set_value", path="$.port", value="1", expected_value="7"),
        # expected_missing on a key that exists.
        YamlEditOperation(action="set_value", path="$.name", value='"x"', expected_missing=True),
        # Wrong array length for append.
        YamlEditOperation(action="append_to_array", path="$.tags", value='"x"', expected_length=9),
        # Wrong expected_value for delete.
        YamlEditOperation(action="delete_key", path="$.name", expected_value='"nope"'),
        # Path that does not exist.
        YamlEditOperation(action="delete_key", path="$.ghost", expected_value="null"),
    ]

    for operation in batches:
        result = _edit(source.name, yaml_scope=yaml_scope, operations=[operation])
        assert result.artifact is None
        assert "no changes were made" in result.content

    assert source.read_bytes() == original
    assert list(session_root.glob("*.yaml")) == [source]


def test_edited_copy_preserves_indentation_and_newlines(yaml_scope):
    file_root, session_root, source, _ = yaml_scope
    indented = "outer:\n    keep: 1\n    edit: old\n"
    source.write_bytes(indented.replace("\n", "\r\n").encode("utf-8"))

    result = _edit(
        source.name,
        yaml_scope=yaml_scope,
        operations=[
            YamlEditOperation(
                action="set_value",
                path="$.outer.edit",
                value='"new"',
                expected_value='"old"',
            )
        ],
    )

    assert result.artifact is not None
    raw = (session_root / "app.edited.yaml").read_bytes()
    assert b"\r\n" in raw
    # The original 4-space indentation unit is preserved on the edited copy.
    assert b"\r\n    keep: 1" in raw
    assert b"\r\n    edit: new" in raw
    assert yaml.safe_load(raw.decode("utf-8"))["outer"]["edit"] == "new"


def test_edit_refuses_multi_document_stream(yaml_scope):
    file_root, session_root, source, _ = yaml_scope
    source.write_text("a: 1\n---\nb: 2\n", encoding="utf-8")

    result = _edit(
        source.name,
        yaml_scope=yaml_scope,
        operations=[
            YamlEditOperation(
                action="set_value",
                path="$.a",
                value="2",
                expected_value="1",
            )
        ],
    )

    assert result.artifact is None
    assert "multi-document" in result.content


def test_values_must_be_json_and_guards_are_enforced():
    with pytest.raises(ValidationError, match="value is not valid JSON"):
        YamlEditOperation(action="set_value", path="$.a", value="{broken", expected_value="1")
    with pytest.raises(ValidationError, match="exactly one of expected_value"):
        YamlEditOperation(action="set_value", path="$.a", value="1")
    with pytest.raises(ValidationError, match="cannot target the document root"):
        YamlEditOperation(action="delete_key", path="$", expected_value="1")


def test_tool_registration_requires_flags_session_root_and_thread_id(
    isolated_settings, tmp_path: Path
):
    session_root = tmp_path / "files" / "chat_uploads" / "thread-a"
    enabled = isolated_settings(
        file_read_enabled=True,
        yaml_edit_enabled=True,
        file_read_root=str(tmp_path / "files"),
    )
    disabled = isolated_settings(
        file_read_enabled=True,
        yaml_edit_enabled=False,
        file_read_root=str(tmp_path / "files"),
    )

    assert build_yaml_edit_tools(disabled, session_root=session_root, thread_id="thread-a") == []
    assert build_yaml_edit_tools(enabled, session_root=None, thread_id="thread-a") == []
    assert build_yaml_edit_tools(enabled, session_root=session_root, thread_id="") == []
    assert [
        tool.name
        for tool in build_yaml_edit_tools(
            enabled, session_root=session_root, thread_id="thread-a"
        )
    ] == ["create_yaml_file", "inspect_yaml_file", "edit_yaml_file"]
