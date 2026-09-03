from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.backend.tools.csv_edit import (
    CSV_MIME_TYPE,
    CsvEditOperation,
    build_csv_edit_tools,
    create_csv_file,
    edit_csv_file,
    inspect_csv_file,
)


@pytest.fixture
def csv_scope(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    file_root = tmp_path / "files"
    thread_id = "thread-a"
    session_root = file_root / "chat_uploads" / thread_id
    session_root.mkdir(parents=True)
    source = session_root / "prices.csv"
    source.write_bytes(b'name,price,qty\nApple,1.25,3\n"Banana, yellow",0.99,6\nCherry,3.00,1\n')
    return file_root, session_root, source, thread_id


def _create(
    filename: str,
    *,
    headers: list[str],
    rows: list[list[str]],
    file_root: Path,
    session_root: Path,
    thread_id: str,
    max_bytes: int = 1_000_000,
):
    return create_csv_file(
        filename,
        headers=headers,
        rows=rows,
        session_root=session_root,
        file_root=file_root,
        max_bytes=max_bytes,
        thread_id=thread_id,
    )


def _edit(
    path: str,
    *,
    operations: list[CsvEditOperation],
    file_root: Path,
    session_root: Path,
    thread_id: str,
    max_bytes: int = 1_000_000,
    output_name: str | None = None,
):
    return edit_csv_file(
        path,
        operations=operations,
        session_root=session_root,
        file_root=file_root,
        max_bytes=max_bytes,
        thread_id=thread_id,
        output_name=output_name,
    )


def test_inspect_lists_columns_and_numbered_rows(csv_scope):
    file_root, session_root, source, _ = csv_scope

    result = inspect_csv_file(
        source.name,
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
    )

    assert "Encoding: UTF-8" in result
    assert "Newline: LF" in result
    assert "Delimiter: ','" in result
    assert "Final newline: yes" in result
    assert "COLUMNS (3)" in result
    assert '[0] "name"' in result
    assert "ROWS (4)" in result
    # Quoted field with an embedded comma survives as one cell.
    assert '["Banana, yellow", "0.99", "6"]' in result


def test_create_writes_downloadable_csv_and_is_collision_safe(csv_scope):
    file_root, session_root, _source, thread_id = csv_scope

    first = _create(
        "report.csv",
        headers=["name", "value"],
        rows=[["alpha", "1"], ["beta, green", "2"]],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )
    second = _create(
        "report.csv",
        headers=["name", "value"],
        rows=[["gamma", "3"]],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert first.artifact is not None
    assert first.artifact["mimeType"] == CSV_MIME_TYPE
    assert first.artifact["filename"] == "report.csv"
    assert (session_root / "report.csv").exists()
    # Collision produces a numbered sibling rather than overwriting.
    assert second.artifact["filename"] == "report-2.csv"
    assert (session_root / "report-2.csv").exists()
    # Quoting round-trips: the embedded comma is preserved as a single field.
    body = (session_root / "report.csv").read_text(encoding="utf-8")
    assert '"beta, green",2' in body
    assert "3 records including the header" in first.content


def test_edit_replaces_row_and_leaves_source_unchanged(csv_scope):
    file_root, session_root, source, thread_id = csv_scope
    original = source.read_bytes()

    result = _edit(
        source.name,
        operations=[
            CsvEditOperation(
                action="replace_row",
                row_index=2,
                expected_row=["Banana, yellow", "0.99", "6"],
                new_row=["Banana, ripe", "1.09", "8"],
            )
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert result.artifact is not None
    edited = session_root / result.artifact["filename"]
    assert edited.exists()
    assert source.read_bytes() == original
    body = edited.read_text(encoding="utf-8")
    # The new cell contains a comma, so the writer correctly quotes it.
    assert '"Banana, ripe",1.09,8' in body
    assert "Banana, yellow" not in body


def test_edit_rejects_stale_expected_row_and_aborts(csv_scope):
    file_root, session_root, source, thread_id = csv_scope
    original = source.read_bytes()

    result = _edit(
        source.name,
        operations=[
            CsvEditOperation(
                action="replace_row",
                row_index=2,
                expected_row=["Banana, yellow", "0.99", "7"],  # stale qty
                new_row=["Banana, ripe", "1.09", "8"],
            )
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert result.artifact is None
    assert "no changes were made" in result.content
    assert source.read_bytes() == original


def test_edit_supports_insert_delete_append_and_cell_update(csv_scope):
    file_root, session_root, source, thread_id = csv_scope

    result = _edit(
        source.name,
        operations=[
            CsvEditOperation(
                action="insert_before_row",
                row_index=1,
                expected_row=["Apple", "1.25", "3"],
                new_row=["Apricot", "0.75", "2"],
            ),
            CsvEditOperation(
                action="delete_row",
                row_index=3,
                expected_row=["Cherry", "3.00", "1"],
            ),
            CsvEditOperation(
                action="append_row",
                new_row=["Date", "2.50", "4"],
            ),
            CsvEditOperation(
                action="update_cell",
                row_index=1,
                column=2,
                expected_value="3",
                new_value="5",
            ),
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    edited = session_root / result.artifact["filename"]
    body = edited.read_text(encoding="utf-8")
    lines = [line for line in body.split("\n") if line]
    # Header unchanged, Apple qty updated to 5, Apricot inserted before Apple,
    # Cherry deleted, Date appended.
    assert lines[0] == "name,price,qty"
    assert lines[1] == "Apricot,0.75,2"
    assert lines[2] == "Apple,1.25,5"
    assert lines[-1] == "Date,2.50,4"
    assert not any("Cherry" in line for line in lines)


def test_edit_rejects_cell_update_on_a_row_targeted_by_delete(csv_scope):
    file_root, session_root, source, thread_id = csv_scope

    result = _edit(
        source.name,
        operations=[
            CsvEditOperation(
                action="delete_row",
                row_index=1,
                expected_row=["Apple", "1.25", "3"],
            ),
            CsvEditOperation(
                action="update_cell",
                row_index=1,
                column=2,
                expected_value="3",
                new_value="9",
            ),
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )

    assert result.artifact is None
    assert "no changes were made" in result.content


def test_edit_rejects_out_of_range_row_and_column(csv_scope):
    file_root, session_root, source, thread_id = csv_scope

    row_out_of_range = _edit(
        source.name,
        operations=[
            CsvEditOperation(
                action="replace_row",
                row_index=99,
                expected_row=["x"],
                new_row=["y"],
            )
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )
    assert "out of range" in row_out_of_range.content

    column_out_of_range = _edit(
        source.name,
        operations=[
            CsvEditOperation(
                action="update_cell",
                row_index=1,
                column=99,
                expected_value="3",
                new_value="5",
            )
        ],
        file_root=file_root,
        session_root=session_root,
        thread_id=thread_id,
    )
    assert "out of range" in column_out_of_range.content


def test_edit_preserves_delimiter_and_newline(tmp_path: Path):
    file_root = tmp_path / "files"
    thread_id = "thread-b"
    session_root = file_root / "chat_uploads" / thread_id
    session_root.mkdir(parents=True)
    source = session_root / "sems.csv"
    source.write_bytes(b"a;b;c\r\n1;2;3\r\n4;5;6\r\n")

    result = edit_csv_file(
        source.name,
        operations=[
            CsvEditOperation(
                action="update_cell",
                row_index=1,
                column=1,
                expected_value="2",
                new_value="20",
            )
        ],
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
        thread_id=thread_id,
    )

    body = (session_root / result.artifact["filename"]).read_bytes()
    assert b"\r\n" in body
    assert body == b"a;b;c\r\n1;20;3\r\n4;5;6\r\n"


def test_edit_preserves_utf8_bom(tmp_path: Path):
    file_root = tmp_path / "files"
    thread_id = "thread-c"
    session_root = file_root / "chat_uploads" / thread_id
    session_root.mkdir(parents=True)
    source = session_root / "bom.csv"
    source.write_bytes(b"\xef\xbb\xbfh1,h2\nv1,v2\n")

    result = edit_csv_file(
        source.name,
        operations=[
            CsvEditOperation(
                action="update_cell",
                row_index=1,
                column=0,
                expected_value="v1",
                new_value="V1",
            )
        ],
        session_root=session_root,
        file_root=file_root,
        max_bytes=1_000_000,
        thread_id=thread_id,
    )

    body = (session_root / result.artifact["filename"]).read_bytes()
    assert body.startswith(b"\xef\xbb\xbf")
    assert b"h1,h2\nV1,v2\n" in body


def test_operation_validator_requires_expected_row_for_structural_actions():
    with pytest.raises(ValidationError):
        CsvEditOperation(action="replace_row", row_index=1, new_row=["a"])


def test_operation_validator_requires_new_row_for_append():
    with pytest.raises(ValidationError):
        CsvEditOperation(action="append_row")


def test_build_tools_require_enabled_flag_and_session_scope(tmp_path: Path):
    from src.config import Settings

    file_root = tmp_path / "files"
    session_root = file_root / "chat_uploads" / "thread-a"

    disabled = Settings(
        dashscope_api_key="x",
        chroma_dir=tmp_path / "chroma",
        source_urls=["https://example.com/a"],
        file_read_enabled=True,
        csv_edit_enabled=False,
    )
    assert build_csv_edit_tools(disabled, session_root=session_root, thread_id="thread-a") == []

    enabled = Settings(
        dashscope_api_key="x",
        chroma_dir=tmp_path / "chroma",
        source_urls=["https://example.com/a"],
        file_read_enabled=True,
        csv_edit_enabled=True,
    )
    tools = build_csv_edit_tools(enabled, session_root=session_root, thread_id="thread-a")
    assert [t.name for t in tools] == ["create_csv_file", "inspect_csv_file", "edit_csv_file"]

    # A missing or invalid thread_id disables the tools even when the flag is on.
    assert build_csv_edit_tools(enabled, session_root=session_root, thread_id="") == []
    assert build_csv_edit_tools(enabled, session_root=session_root, thread_id="bad thread!") == []
