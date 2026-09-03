from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.backend.tools.word_edit import (
    DOCX_MIME_TYPE,
    WordContentBlock,
    WordCreateInput,
    WordEditOperation,
    build_word_edit_tools,
    create_word_document,
    edit_word_document,
    inspect_word_document,
)

docx = pytest.importorskip("docx")


def _write_document(path: Path) -> None:
    document = docx.Document()
    document.add_paragraph("Quarterly report")
    body = document.add_paragraph()
    body.add_run("Revenue ").bold = True
    body.add_run("rose 12 percent.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Region"
    table.cell(0, 1).text = "Revenue"
    table.cell(1, 0).text = "East"
    table.cell(1, 1).text = "$120"
    document.save(path)


@pytest.fixture
def document_scope(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    file_root = tmp_path / "files"
    thread_id = "thread-a"
    session_root = file_root / "chat_uploads" / thread_id
    session_root.mkdir(parents=True)
    source = session_root / "report.docx"
    _write_document(source)
    return file_root, session_root, source, thread_id


def _edit(
    path: str,
    *,
    operations: list[WordEditOperation],
    file_root: Path,
    session_root: Path,
    thread_id: str,
    max_bytes: int = 5_000_000,
    output_name: str | None = None,
):
    return edit_word_document(
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
    *,
    blocks: list[WordContentBlock],
    file_root: Path,
    session_root: Path,
    thread_id: str,
    max_bytes: int = 5_000_000,
    title: str | None = None,
    subtitle: str | None = None,
    author: str | None = None,
):
    return create_word_document(
        filename,
        blocks=blocks,
        session_root=session_root,
        file_root=file_root,
        max_bytes=max_bytes,
        thread_id=thread_id,
        title=title,
        subtitle=subtitle,
        author=author,
    )


def test_create_writes_styled_downloadable_docx_with_real_lists_and_table_geometry(
    document_scope,
):
    file_root, session_root, _source, thread_id = document_scope

    result = _create(
        "../project brief.docx",
        blocks=[
            WordContentBlock(kind="heading", text="Overview", level=1),
            WordContentBlock(
                kind="paragraph",
                text="This document was created from structured content.",
            ),
            WordContentBlock(kind="bullet_list", items=["Fast", "Safe"]),
            WordContentBlock(kind="numbered_list", items=["Draft", "Review"]),
            WordContentBlock(
                kind="table",
                rows=[["Owner", "Status"], ["Platform", "Ready"]],
            ),
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
        title="Project Brief",
        subtitle="Creation tool smoke test",
        author="Test Author",
    )

    assert result.artifact is not None
    assert result.artifact["filename"] == "project_brief.docx"
    assert result.artifact["mimeType"] == DOCX_MIME_TYPE
    assert result.artifact["url"] == "/chat/thread-a/files/project_brief.docx"
    created_path = session_root / "project_brief.docx"

    created = docx.Document(created_path)
    assert created.core_properties.title == "Project Brief"
    assert created.core_properties.author == "Test Author"
    assert created.sections[0].page_width.inches == pytest.approx(8.5, abs=0.01)
    assert created.sections[0].page_height.inches == pytest.approx(11.0, abs=0.01)
    assert [(p.style.name, p.text) for p in created.paragraphs[:4]] == [
        ("Title", "Project Brief"),
        ("Subtitle", "Creation tool smoke test"),
        ("Heading 1", "Overview"),
        ("Normal", "This document was created from structured content."),
    ]
    assert created.tables[0].cell(1, 0).text == "Platform"

    with zipfile.ZipFile(created_path) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
        numbering_xml = archive.read("word/numbering.xml").decode("utf-8")
    assert document_xml.count("<w:numPr>") == 4
    assert '<w:numFmt w:val="bullet"' in numbering_xml
    assert '<w:numFmt w:val="decimal"' in numbering_xml
    assert '<w:tblW w:w="9360" w:type="dxa"' in document_xml
    assert '<w:tblInd w:w="120" w:type="dxa"' in document_xml
    assert '<w:tblLayout w:type="fixed"' in document_xml
    assert '<w:tblHeader w:val="true"' in document_xml


def test_create_is_collision_safe_size_limited_and_session_confined(document_scope):
    file_root, session_root, _source, thread_id = document_scope
    blocks = [WordContentBlock(kind="paragraph", text="Hello")]

    first = _create(
        "brief",
        blocks=blocks,
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )
    second = _create(
        "brief",
        blocks=blocks,
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )
    oversized = _create(
        "too-large",
        blocks=blocks,
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
        max_bytes=100,
    )
    outside = _create(
        "outside",
        blocks=blocks,
        file_root=file_root,
        session_root=file_root.parent / "other-session",
        thread_id=thread_id,
    )

    assert first.artifact is not None
    assert second.artifact is not None
    assert first.artifact["filename"] == "brief.docx"
    assert second.artifact["filename"] == "brief-2.docx"
    assert oversized.artifact is None
    assert "new document is too large" in oversized.content
    assert not (session_root / "too-large.docx").exists()
    assert outside.artifact is None
    assert "outside the configured file root" in outside.content


def test_create_block_schema_rejects_invalid_shapes_and_control_characters():
    with pytest.raises(ValidationError, match="same number of columns"):
        WordContentBlock(kind="table", rows=[["A", "B"], ["one"]])
    with pytest.raises(ValidationError, match="non-empty items"):
        WordContentBlock(kind="bullet_list", items=[])

    with pytest.raises(ValidationError, match="unsupported control characters"):
        WordCreateInput(
            filename="bad.docx",
            blocks=[WordContentBlock(kind="paragraph", text="bad\x00text")],
        )


def test_inspect_lists_numbered_paragraphs_and_table_cells(document_scope):
    file_root, session_root, source, _ = document_scope

    result = inspect_word_document(
        source.name,
        session_root=session_root,
        file_root=file_root,
        max_bytes=5_000_000,
    )

    assert "[0] Quarterly report" in result
    assert "[1] Revenue rose 12 percent." in result
    assert "[table 0 row 0 col 0] Region" in result
    assert "[table 0 row 1 col 1] $120" in result


def test_edit_applies_supported_operations_and_preserves_original(document_scope):
    file_root, session_root, source, thread_id = document_scope
    original_bytes = source.read_bytes()

    result = _edit(
        source.name,
        operations=[
            WordEditOperation(
                action="replace_paragraph",
                paragraph_index=0,
                expected_text="Quarterly report",
                new_text="Annual report",
            ),
            WordEditOperation(
                action="delete_paragraph",
                paragraph_index=1,
                expected_text="Revenue rose 12 percent.",
            ),
            WordEditOperation(action="append_paragraph", new_text="Approved for release."),
            WordEditOperation(
                action="replace_table_cell",
                table_index=0,
                row_index=1,
                column_index=1,
                expected_text="$120",
                new_text="$150",
            ),
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert result.artifact is not None
    assert result.artifact == {
        "type": "file",
        "version": 1,
        "kind": "download",
        "provider": "chat_upload",
        "threadId": thread_id,
        "filename": "report.edited.docx",
        "mimeType": DOCX_MIME_TYPE,
        "sizeBytes": result.artifact["sizeBytes"],
        "url": f"/chat/{thread_id}/files/report.edited.docx",
    }
    assert source.read_bytes() == original_bytes

    edited = docx.Document(session_root / "report.edited.docx")
    assert [paragraph.text for paragraph in edited.paragraphs] == [
        "Annual report",
        "Approved for release.",
    ]
    assert edited.tables[0].cell(1, 1).text == "$150"


def test_exact_text_mismatch_and_duplicate_target_publish_nothing(document_scope):
    file_root, session_root, source, thread_id = document_scope

    mismatch = _edit(
        source.name,
        operations=[
            WordEditOperation(
                action="replace_paragraph",
                paragraph_index=0,
                expected_text="Wrong text",
                new_text="Changed",
            )
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )
    duplicate = _edit(
        source.name,
        operations=[
            WordEditOperation(
                action="replace_paragraph",
                paragraph_index=0,
                expected_text="Quarterly report",
                new_text="First",
            ),
            WordEditOperation(
                action="delete_paragraph",
                paragraph_index=0,
                expected_text="Quarterly report",
            ),
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert mismatch.artifact is None
    assert "no changes were made" in mismatch.content
    assert duplicate.artifact is None
    assert "targeted more than once" in duplicate.content
    assert list(session_root.glob("*.docx")) == [source]


def test_corrupt_and_suspicious_archives_are_rejected(tmp_path: Path):
    file_root = tmp_path / "files"
    session_root = file_root / "chat_uploads" / "thread-a"
    session_root.mkdir(parents=True)
    corrupt = session_root / "corrupt.docx"
    corrupt.write_bytes(b"not a zip archive")
    bomb = session_root / "bomb.docx"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", "x" * 1_000_000)

    corrupt_result = inspect_word_document(
        corrupt.name,
        session_root=session_root,
        file_root=file_root,
        max_bytes=5_000_000,
    )
    bomb_result = inspect_word_document(
        bomb.name,
        session_root=session_root,
        file_root=file_root,
        max_bytes=5_000_000,
    )

    assert "not a readable .docx archive" in corrupt_result
    assert "suspiciously compressed part" in bomb_result


def test_path_traversal_and_cross_session_access_are_denied(document_scope):
    file_root, session_root, _, thread_id = document_scope
    other_root = file_root / "chat_uploads" / "thread-b"
    other_root.mkdir()
    other_source = other_root / "private.docx"
    _write_document(other_source)

    for path in ("../thread-b/private.docx", str(other_source)):
        result = _edit(
            path,
            operations=[WordEditOperation(action="append_paragraph", new_text="No")],
            file_root=file_root,
            session_root=session_root,
            thread_id=thread_id,
        )
        assert result.artifact is None
        assert "outside this chat session" in result.content

    assert not (other_root / "private.edited.docx").exists()


def test_output_collisions_are_numbered_and_output_name_cannot_escape(document_scope):
    file_root, session_root, source, thread_id = document_scope
    operation = WordEditOperation(action="append_paragraph", new_text="New note")

    first = _edit(
        source.name,
        operations=[operation],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
        output_name="../../custom.docx",
    )
    second = _edit(
        source.name,
        operations=[operation],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
        output_name="../../custom.docx",
    )

    assert first.artifact is not None
    assert second.artifact is not None
    assert first.artifact["filename"] == "custom.docx"
    assert second.artifact["filename"] == "custom-2.docx"
    docx.Document(session_root / "custom.docx")
    docx.Document(session_root / "custom-2.docx")
    assert not (file_root / "custom.docx").exists()


def test_generated_document_must_fit_configured_size_limit(document_scope):
    file_root, session_root, source, thread_id = document_scope
    source_size = source.stat().st_size
    new_text = "".join(f"value-{index:04d};" for index in range(800))

    result = _edit(
        source.name,
        operations=[WordEditOperation(action="append_paragraph", new_text=new_text)],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
        max_bytes=source_size + 8,
    )

    assert result.artifact is None
    assert "edited document is too large" in result.content
    assert list(session_root.glob("*.docx")) == [source]


def test_tool_registration_requires_flags_session_root_and_thread_id(
    isolated_settings,
    tmp_path: Path,
):
    session_root = tmp_path / "files" / "chat_uploads" / "thread-a"
    enabled = isolated_settings(
        file_read_enabled=True,
        word_edit_enabled=True,
        file_read_root=str(tmp_path / "files"),
    )
    disabled = isolated_settings(
        file_read_enabled=True,
        word_edit_enabled=False,
        file_read_root=str(tmp_path / "files"),
    )

    assert build_word_edit_tools(disabled, session_root=session_root, thread_id="thread-a") == []
    assert build_word_edit_tools(enabled, session_root=None, thread_id="thread-a") == []
    assert build_word_edit_tools(enabled, session_root=session_root, thread_id="") == []
    assert build_word_edit_tools(enabled, session_root=session_root, thread_id="../other") == []
    assert [
        tool.name
        for tool in build_word_edit_tools(
            enabled,
            session_root=session_root,
            thread_id="thread-a",
        )
    ] == ["create_word_document", "inspect_word_document", "edit_word_document"]
