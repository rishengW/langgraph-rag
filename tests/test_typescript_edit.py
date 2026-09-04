from __future__ import annotations

import codecs
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.backend.tools.typescript_edit import (
    TYPESCRIPT_MIME_TYPE,
    TypeScriptEditOperation,
    build_typescript_edit_tools,
    create_typescript_file,
    edit_typescript_file,
    inspect_typescript_file,
)


@pytest.fixture
def typescript_scope(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    file_root = tmp_path / "files"
    thread_id = "thread-a"
    session_root = file_root / "chat_uploads" / thread_id
    session_root.mkdir(parents=True)
    source = session_root / "app.ts"
    source.write_bytes(b"const alpha = 1;\r\nconst beta = 2;\r\nconst gamma = 3;\r\n")
    return file_root, session_root, source, thread_id


def _edit(
    path: str,
    *,
    operations: list[TypeScriptEditOperation],
    file_root: Path,
    session_root: Path,
    thread_id: str,
    max_bytes: int = 1_000_000,
    output_name: str | None = None,
):
    return edit_typescript_file(
        path,
        operations=operations,
        session_root=session_root,
        file_root=file_root,
        max_bytes=max_bytes,
        thread_id=thread_id,
        output_name=output_name,
    )


def _create(
    filename: str,
    content: str,
    *,
    file_root: Path,
    session_root: Path,
    thread_id: str,
    max_bytes: int = 1_000_000,
):
    return create_typescript_file(
        filename,
        content,
        session_root=session_root,
        file_root=file_root,
        max_bytes=max_bytes,
        thread_id=thread_id,
    )


def test_inspect_lists_exact_lines_and_format_metadata(typescript_scope):
    file_root, session_root, source, _ = typescript_scope

    result = inspect_typescript_file(
        source.name,
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
    )

    assert "Encoding: UTF-8" in result
    assert "Newline: CRLF" in result
    assert "Final newline: yes" in result
    assert "LINES (3)" in result
    assert '[0] "const alpha = 1;"' in result
    assert '[2] "const gamma = 3;"' in result


def test_create_writes_downloadable_typescript_and_is_collision_safe(
    typescript_scope,
):
    file_root, session_root, _source, thread_id = typescript_scope

    first = _create(
        "../meeting notes.ts",
        "export const alpha = 1;\r\n",
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )
    second = _create(
        "../meeting notes.ts",
        "export const beta = 2;",
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert first.artifact is not None
    assert second.artifact is not None
    assert first.artifact["filename"] == "meeting_notes.ts"
    assert second.artifact["filename"] == "meeting_notes-2.ts"
    assert (
        session_root / "meeting_notes.ts"
    ).read_bytes() == b"export const alpha = 1;\r\n"
    assert (
        session_root / "meeting_notes-2.ts"
    ).read_bytes() == b"export const beta = 2;"
    assert not (file_root / "meeting_notes.ts").exists()


def test_create_honors_a_requested_tsx_suffix(typescript_scope):
    file_root, session_root, _source, thread_id = typescript_scope

    result = _create(
        "widget.tsx",
        "export const Widget = () => null;\n",
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )
    default = _create(
        "widget",
        "export const Plain = () => null;\n",
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert result.artifact is not None
    assert result.artifact["filename"] == "widget.tsx"
    assert (session_root / "widget.tsx").exists()
    assert default.artifact is not None
    assert default.artifact["filename"] == "widget.ts"


def test_create_rejects_invalid_content_size_and_scope(typescript_scope):
    file_root, session_root, _source, thread_id = typescript_scope

    for content, message in (
        ("const alpha = 1;\rbeta", "bare-CR"),
        ("a\r\nb\n", "mixes CRLF"),
    ):
        result = _create(
            "invalid.ts",
            content,
            file_root=file_root,
            session_root=session_root,
            thread_id=thread_id,
        )
        assert result.artifact is None
        assert message in result.content

    oversized = _create(
        "large.ts",
        "x" * 100,
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
        max_bytes=10,
    )
    assert oversized.artifact is None
    assert "new TypeScript file is too large" in oversized.content

    outside = _create(
        "outside.ts",
        "const secret = true;",
        file_root=file_root,
        session_root=file_root.parent / "other-session",
        thread_id=thread_id,
    )
    assert outside.artifact is None
    assert "outside the configured file root" in outside.content


def test_edit_applies_all_operations_and_preserves_original_format(
    typescript_scope,
):
    file_root, session_root, source, thread_id = typescript_scope
    original = source.read_bytes()

    result = _edit(
        "chat_uploads/thread-a/app.ts",
        operations=[
            TypeScriptEditOperation(
                action="insert_before_line",
                line_index=0,
                expected_text="const alpha = 1;",
                new_text="// start",
            ),
            TypeScriptEditOperation(
                action="replace_line",
                line_index=1,
                expected_text="const beta = 2;",
                new_text="const beta = 20;",
            ),
            TypeScriptEditOperation(
                action="delete_line",
                line_index=2,
                expected_text="const gamma = 3;",
            ),
            TypeScriptEditOperation(
                action="append_line",
                new_text="export const total = alpha + beta;",
            ),
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert source.read_bytes() == original
    assert (session_root / "app.edited.ts").read_bytes() == (
        b"// start\r\nconst alpha = 1;\r\nconst beta = 20;\r\n"
        b"export const total = alpha + beta;\r\n"
    )
    assert result.artifact is not None
    assert result.artifact["filename"] == "app.edited.ts"
    assert result.artifact["mimeType"] == TYPESCRIPT_MIME_TYPE
    assert result.artifact["url"] == "/chat/thread-a/files/app.edited.ts"


def test_editing_a_tsx_source_publishes_a_tsx_copy(typescript_scope):
    file_root, session_root, _source, thread_id = typescript_scope
    source = session_root / "widget.tsx"
    source.write_bytes(b"export const Widget = () => <div />;\n")

    result = _edit(
        source.name,
        operations=[
            TypeScriptEditOperation(
                action="replace_line",
                line_index=0,
                expected_text="export const Widget = () => <div />;",
                new_text="export const Widget = () => <span />;",
            )
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert result.artifact is not None
    assert result.artifact["filename"] == "widget.edited.tsx"
    assert (session_root / "widget.edited.tsx").read_bytes() == (
        b"export const Widget = () => <span />;\n"
    )


def test_utf8_bom_lf_and_missing_final_newline_are_preserved(typescript_scope):
    file_root, session_root, source, thread_id = typescript_scope
    source.write_bytes(codecs.BOM_UTF8 + b"const cafe = 1;\nconst naive = 2;")

    result = _edit(
        source.name,
        operations=[
            TypeScriptEditOperation(
                action="replace_line",
                line_index=1,
                expected_text="const naive = 2;",
                new_text="const naive = 3;",
            )
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert result.artifact is not None
    assert (session_root / "app.edited.ts").read_bytes() == (
        codecs.BOM_UTF8 + b"const cafe = 1;\nconst naive = 3;"
    )


def test_deleting_every_line_publishes_an_empty_file(typescript_scope):
    file_root, session_root, source, thread_id = typescript_scope
    source.write_bytes(b"const only = true;\n")

    result = _edit(
        source.name,
        operations=[
            TypeScriptEditOperation(
                action="delete_line",
                line_index=0,
                expected_text="const only = true;",
            )
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert result.artifact is not None
    assert (session_root / "app.edited.ts").read_bytes() == b""


def test_mismatch_duplicate_and_out_of_range_batches_publish_nothing(
    typescript_scope,
):
    file_root, session_root, source, thread_id = typescript_scope
    batches = [
        [
            TypeScriptEditOperation(
                action="replace_line",
                line_index=0,
                expected_text="const wrong = 0;",
                new_text="const changed = 1;",
            )
        ],
        [
            TypeScriptEditOperation(
                action="replace_line",
                line_index=0,
                expected_text="const alpha = 1;",
                new_text="const first = 1;",
            ),
            TypeScriptEditOperation(
                action="delete_line",
                line_index=0,
                expected_text="const alpha = 1;",
            ),
        ],
        [
            TypeScriptEditOperation(
                action="delete_line",
                line_index=99,
                expected_text="const missing = 1;",
            )
        ],
    ]

    results = [
        _edit(
            source.name,
            operations=batch,
            file_root=file_root,
            session_root=session_root,
            thread_id=thread_id,
        )
        for batch in batches
    ]

    assert all(result.artifact is None for result in results)
    assert all("no changes were made" in result.content for result in results)
    assert list(session_root.glob("*.ts")) == [source]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"const x = '\xff';", "invalid UTF-8"),
        (codecs.BOM_UTF16_LE + b"a\x00", "unsupported encoding"),
        (b"const x = 1;\x00", "binary data"),
        (b"const x = 1;\rconst y = 2;", "bare-CR"),
        (b"const x = 1;\r\nconst y = 2;\n", "mixes CRLF and LF"),
        (b"const x = 1;\x07", "binary control characters"),
    ],
)
def test_invalid_or_binary_text_is_rejected_without_output(
    typescript_scope,
    payload: bytes,
    message: str,
):
    file_root, session_root, source, thread_id = typescript_scope
    source.write_bytes(payload)

    result = _edit(
        source.name,
        operations=[TypeScriptEditOperation(action="append_line", new_text="const n = 1;")],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert result.artifact is None
    assert message in result.content
    assert list(session_root.glob("*.ts")) == [source]


def test_path_traversal_and_cross_session_access_are_denied(typescript_scope):
    file_root, session_root, _, thread_id = typescript_scope
    other_root = file_root / "chat_uploads" / "thread-b"
    other_root.mkdir()
    other_source = other_root / "private.ts"
    other_source.write_text("const secret = true;", encoding="utf-8")

    for path in ("../thread-b/private.ts", str(other_source)):
        result = _edit(
            path,
            operations=[
                TypeScriptEditOperation(action="append_line", new_text="const no = 1;")
            ],
            file_root=file_root,
            session_root=session_root,
            thread_id=thread_id,
        )
        assert result.artifact is None
        assert "outside this chat session" in result.content

    assert not (other_root / "private.edited.ts").exists()


def test_output_collisions_are_numbered_and_output_name_cannot_escape(
    typescript_scope,
):
    file_root, session_root, source, thread_id = typescript_scope
    operation = TypeScriptEditOperation(action="append_line", new_text="const n = 1;")

    first = _edit(
        source.name,
        operations=[operation],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
        output_name="../../custom.ts",
    )
    second = _edit(
        source.name,
        operations=[operation],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
        output_name="../../custom.ts",
    )

    assert first.artifact is not None
    assert second.artifact is not None
    assert first.artifact["filename"] == "custom.ts"
    assert second.artifact["filename"] == "custom-2.ts"
    assert not (file_root / "custom.ts").exists()


def test_generated_file_must_fit_configured_size_limit(typescript_scope):
    file_root, session_root, source, thread_id = typescript_scope

    result = _edit(
        source.name,
        operations=[
            TypeScriptEditOperation(
                action="append_line",
                new_text="const padding = '" + "x" * 100 + "';",
            )
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
        max_bytes=source.stat().st_size + 1,
    )

    assert result.artifact is None
    assert "edited TypeScript file is too large" in result.content
    assert list(session_root.glob("*.ts")) == [source]


def test_multiline_operation_fields_are_rejected():
    with pytest.raises(ValidationError, match="exactly one line"):
        TypeScriptEditOperation(action="append_line", new_text="const one = 1;\nconst two = 2;")


def test_tool_registration_requires_flags_session_root_and_thread_id(
    isolated_settings,
    tmp_path: Path,
):
    session_root = tmp_path / "files" / "chat_uploads" / "thread-a"
    enabled = isolated_settings(
        file_read_enabled=True,
        typescript_edit_enabled=True,
        file_read_root=str(tmp_path / "files"),
    )
    disabled = isolated_settings(
        file_read_enabled=True,
        typescript_edit_enabled=False,
        file_read_root=str(tmp_path / "files"),
    )

    assert build_typescript_edit_tools(
        disabled, session_root=session_root, thread_id="thread-a"
    ) == []
    assert build_typescript_edit_tools(enabled, session_root=None, thread_id="thread-a") == []
    assert build_typescript_edit_tools(enabled, session_root=session_root, thread_id="") == []
    assert (
        build_typescript_edit_tools(
            enabled, session_root=session_root, thread_id="../other"
        )
        == []
    )
    assert [
        tool.name
        for tool in build_typescript_edit_tools(
            enabled,
            session_root=session_root,
            thread_id="thread-a",
        )
    ] == [
        "create_typescript_file",
        "inspect_typescript_file",
        "edit_typescript_file",
    ]


def test_create_tool_call_returns_download_artifact(isolated_settings, tmp_path: Path):
    file_root = tmp_path / "files"
    session_root = file_root / "chat_uploads" / "thread-a"
    settings = isolated_settings(
        file_read_enabled=True,
        typescript_edit_enabled=True,
        file_read_root=str(file_root),
        file_read_max_bytes=1_000_000,
    )
    create_tool = build_typescript_edit_tools(
        settings,
        session_root=session_root,
        thread_id="thread-a",
    )[0]

    message = create_tool.invoke(
        {
            "type": "tool_call",
            "name": "create_typescript_file",
            "args": {"filename": "summary.tsx", "content": "export const one = 1;\n"},
            "id": "call-create-typescript",
        }
    )

    assert message.name == "create_typescript_file"
    assert message.tool_call_id == "call-create-typescript"
    assert message.artifact["filename"] == "summary.tsx"
    assert message.artifact["mimeType"] == TYPESCRIPT_MIME_TYPE
    assert (session_root / "summary.tsx").read_bytes() == b"export const one = 1;\n"
