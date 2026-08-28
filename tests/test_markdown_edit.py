from __future__ import annotations

import codecs
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.tools.markdown_edit import (
    MARKDOWN_MIME_TYPE,
    MarkdownEditOperation,
    build_markdown_edit_tools,
    create_markdown_file,
    edit_markdown_file,
    inspect_markdown_file,
)


@pytest.fixture
def md_scope(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    file_root = tmp_path / "files"
    thread_id = "thread-a"
    session_root = file_root / "chat_uploads" / thread_id
    session_root.mkdir(parents=True)
    source = session_root / "notes.md"
    source.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")
    return file_root, session_root, source, thread_id


def _edit(
    path: str,
    *,
    operations: list[MarkdownEditOperation],
    file_root: Path,
    session_root: Path,
    thread_id: str,
    max_bytes: int = 1_000_000,
    output_name: str | None = None,
):
    return edit_markdown_file(
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
    return create_markdown_file(
        filename,
        content,
        session_root=session_root,
        file_root=file_root,
        max_bytes=max_bytes,
        thread_id=thread_id,
    )


def test_inspect_lists_exact_lines_and_format_metadata(md_scope):
    file_root, session_root, source, _ = md_scope

    result = inspect_markdown_file(
        source.name,
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
    )

    assert "Encoding: UTF-8" in result
    assert "Newline: CRLF" in result
    assert "Final newline: yes" in result
    assert "LINES (3)" in result
    assert '[0] "alpha"' in result
    assert '[2] "gamma"' in result


def test_create_writes_downloadable_utf8_text_and_is_collision_safe(md_scope):
    file_root, session_root, _source, thread_id = md_scope

    first = _create(
        "../meeting notes.md",
        "alpha\r\nbeta\r\n",
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )
    second = _create(
        "../meeting notes.md",
        "second",
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert first.artifact is not None
    assert second.artifact is not None
    assert first.artifact["filename"] == "meeting_notes.md"
    assert second.artifact["filename"] == "meeting_notes-2.md"
    assert (session_root / "meeting_notes.md").read_bytes() == b"alpha\r\nbeta\r\n"
    assert (session_root / "meeting_notes-2.md").read_bytes() == b"second"
    assert not (file_root / "meeting_notes.md").exists()


def test_create_rejects_invalid_content_size_and_scope(md_scope):
    file_root, session_root, _source, thread_id = md_scope

    for content, message in (("alpha\rbeta", "bare-CR"), ("a\r\nb\n", "mixes CRLF")):
        result = _create(
            "invalid.md",
            content,
            file_root=file_root,
            session_root=session_root,
            thread_id=thread_id,
        )
        assert result.artifact is None
        assert message in result.content

    oversized = _create(
        "large.md",
        "x" * 100,
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
        max_bytes=10,
    )
    assert oversized.artifact is None
    assert "new Markdown file is too large" in oversized.content

    outside = _create(
        "outside.md",
        "secret",
        file_root=file_root,
        session_root=file_root.parent / "other-session",
        thread_id=thread_id,
    )
    assert outside.artifact is None
    assert "outside the configured file root" in outside.content


def test_edit_applies_all_operations_and_preserves_original_format(md_scope):
    file_root, session_root, source, thread_id = md_scope
    original = source.read_bytes()

    result = _edit(
        "chat_uploads/thread-a/notes.md",
        operations=[
            MarkdownEditOperation(
                action="insert_before_line",
                line_index=0,
                expected_text="alpha",
                new_text="start",
            ),
            MarkdownEditOperation(
                action="replace_line",
                line_index=1,
                expected_text="beta",
                new_text="BETA",
            ),
            MarkdownEditOperation(
                action="delete_line",
                line_index=2,
                expected_text="gamma",
            ),
            MarkdownEditOperation(action="append_line", new_text="end"),
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert source.read_bytes() == original
    assert (session_root / "notes.edited.md").read_bytes() == (
        b"start\r\nalpha\r\nBETA\r\nend\r\n"
    )
    assert result.artifact is not None
    assert result.artifact["filename"] == "notes.edited.md"
    assert result.artifact["mimeType"] == MARKDOWN_MIME_TYPE
    assert result.artifact["url"] == "/chat/thread-a/files/notes.edited.md"


def test_utf8_bom_lf_and_missing_final_newline_are_preserved(md_scope):
    file_root, session_root, source, thread_id = md_scope
    source.write_bytes(codecs.BOM_UTF8 + b"cafe\nnaive")

    result = _edit(
        source.name,
        operations=[
            MarkdownEditOperation(
                action="replace_line",
                line_index=1,
                expected_text="naive",
                new_text="naive updated",
            )
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert result.artifact is not None
    assert (session_root / "notes.edited.md").read_bytes() == (
        codecs.BOM_UTF8 + b"cafe\nnaive updated"
    )


def test_deleting_every_line_publishes_an_empty_file(md_scope):
    file_root, session_root, source, thread_id = md_scope
    source.write_bytes(b"only line\n")

    result = _edit(
        source.name,
        operations=[
            MarkdownEditOperation(
                action="delete_line",
                line_index=0,
                expected_text="only line",
            )
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert result.artifact is not None
    assert (session_root / "notes.edited.md").read_bytes() == b""


def test_mismatch_duplicate_and_out_of_range_batches_publish_nothing(md_scope):
    file_root, session_root, source, thread_id = md_scope
    batches = [
        [
            MarkdownEditOperation(
                action="replace_line",
                line_index=0,
                expected_text="wrong",
                new_text="changed",
            )
        ],
        [
            MarkdownEditOperation(
                action="replace_line",
                line_index=0,
                expected_text="alpha",
                new_text="first",
            ),
            MarkdownEditOperation(
                action="delete_line",
                line_index=0,
                expected_text="alpha",
            ),
        ],
        [
            MarkdownEditOperation(
                action="delete_line",
                line_index=99,
                expected_text="missing",
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
    assert list(session_root.glob("*.md")) == [source]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"invalid-utf8-\xff", "invalid UTF-8"),
        (codecs.BOM_UTF16_LE + b"a\x00", "unsupported encoding"),
        (b"alpha\x00beta", "binary data"),
        (b"alpha\rbeta", "bare-CR"),
        (b"alpha\r\nbeta\n", "mixes CRLF and LF"),
        (b"alpha\x07beta", "binary control characters"),
    ],
)
def test_invalid_or_binary_markdown_is_rejected_without_output(
    md_scope,
    payload: bytes,
    message: str,
):
    file_root, session_root, source, thread_id = md_scope
    source.write_bytes(payload)

    result = _edit(
        source.name,
        operations=[MarkdownEditOperation(action="append_line", new_text="new")],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert result.artifact is None
    assert message in result.content
    assert list(session_root.glob("*.md")) == [source]


def test_path_traversal_and_cross_session_access_are_denied(md_scope):
    file_root, session_root, _, thread_id = md_scope
    other_root = file_root / "chat_uploads" / "thread-b"
    other_root.mkdir()
    other_source = other_root / "private.md"
    other_source.write_text("secret", encoding="utf-8")

    for path in ("../thread-b/private.md", str(other_source)):
        result = _edit(
            path,
            operations=[MarkdownEditOperation(action="append_line", new_text="no")],
            file_root=file_root,
            session_root=session_root,
            thread_id=thread_id,
        )
        assert result.artifact is None
        assert "outside this chat session" in result.content

    assert not (other_root / "private.edited.md").exists()


def test_output_collisions_are_numbered_and_output_name_cannot_escape(md_scope):
    file_root, session_root, source, thread_id = md_scope
    operation = MarkdownEditOperation(action="append_line", new_text="new")

    first = _edit(
        source.name,
        operations=[operation],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
        output_name="../../custom.md",
    )
    second = _edit(
        source.name,
        operations=[operation],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
        output_name="../../custom.md",
    )

    assert first.artifact is not None
    assert second.artifact is not None
    assert first.artifact["filename"] == "custom.md"
    assert second.artifact["filename"] == "custom-2.md"
    assert not (file_root / "custom.md").exists()


def test_generated_file_must_fit_configured_size_limit(md_scope):
    file_root, session_root, source, thread_id = md_scope

    result = _edit(
        source.name,
        operations=[MarkdownEditOperation(action="append_line", new_text="x" * 100)],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
        max_bytes=source.stat().st_size + 1,
    )

    assert result.artifact is None
    assert "edited Markdown file is too large" in result.content
    assert list(session_root.glob("*.md")) == [source]


def test_multiline_operation_fields_are_rejected():
    with pytest.raises(ValidationError, match="exactly one line"):
        MarkdownEditOperation(action="append_line", new_text="one\ntwo")


def test_tool_registration_requires_flags_session_root_and_thread_id(
    isolated_settings,
    tmp_path: Path,
):
    session_root = tmp_path / "files" / "chat_uploads" / "thread-a"
    enabled = isolated_settings(
        file_read_enabled=True,
        markdown_edit_enabled=True,
        file_read_root=str(tmp_path / "files"),
    )
    disabled = isolated_settings(
        file_read_enabled=True,
        markdown_edit_enabled=False,
        file_read_root=str(tmp_path / "files"),
    )

    assert build_markdown_edit_tools(disabled, session_root=session_root, thread_id="thread-a") == []
    assert build_markdown_edit_tools(enabled, session_root=None, thread_id="thread-a") == []
    assert build_markdown_edit_tools(enabled, session_root=session_root, thread_id="") == []
    assert build_markdown_edit_tools(enabled, session_root=session_root, thread_id="../other") == []
    assert [
        tool.name
        for tool in build_markdown_edit_tools(
            enabled,
            session_root=session_root,
            thread_id="thread-a",
        )
    ] == ["create_markdown_file", "inspect_markdown_file", "edit_markdown_file"]


def test_create_tool_call_returns_download_artifact(isolated_settings, tmp_path: Path):
    file_root = tmp_path / "files"
    session_root = file_root / "chat_uploads" / "thread-a"
    settings = isolated_settings(
        file_read_enabled=True,
        markdown_edit_enabled=True,
        file_read_root=str(file_root),
        file_read_max_bytes=1_000_000,
    )
    create_tool = build_markdown_edit_tools(
        settings,
        session_root=session_root,
        thread_id="thread-a",
    )[0]

    message = create_tool.invoke(
        {
            "type": "tool_call",
            "name": "create_markdown_file",
            "args": {"filename": "summary", "content": "one\ntwo\n"},
            "id": "call-create-md",
        }
    )

    assert message.name == "create_markdown_file"
    assert message.tool_call_id == "call-create-md"
    assert message.artifact["filename"] == "summary.md"
    assert message.artifact["mimeType"] == MARKDOWN_MIME_TYPE
    assert (session_root / "summary.md").read_bytes() == b"one\ntwo\n"
