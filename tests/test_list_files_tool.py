"""Tests for the list_files directory-listing tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.backend.mcp.providers import DocumentToolProvider
from src.backend.tools import build_list_files_tool
from src.config import Settings


def _settings(root: Path) -> Settings:
    return Settings(
        dashscope_api_key="test-key",
        file_read_enabled=True,
        file_read_root=str(root),
        file_read_max_bytes=5_000_000,
    )


def test_lists_root_when_path_empty(tmp_path):
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "sub").mkdir()

    tool = build_list_files_tool(_settings(tmp_path))
    result = tool.invoke({"path": ""})

    assert "file-read root directory" in result
    assert "[file] note.txt" in result
    assert "[dir]  sub/" in result


def test_lists_subdirectory_with_sizes(tmp_path):
    (tmp_path / "data.bin").write_bytes(b"x" * 2048)
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "inner.md").write_text("# hi", encoding="utf-8")

    tool = build_list_files_tool(_settings(tmp_path))
    result = tool.invoke({"path": "nested"})

    assert "Listing of nested:" in result
    assert "[file] inner.md" in result


def test_directories_sort_before_files(tmp_path):
    (tmp_path / "zfile.txt").write_text("x", encoding="utf-8")
    (tmp_path / "adir").mkdir()

    tool = build_list_files_tool(_settings(tmp_path))
    result = tool.invoke({"path": "."})

    assert result.index("[dir]  adir/") < result.index("[file] zfile.txt")


def test_hidden_entries_skipped_and_counted(tmp_path):
    (tmp_path / ".secret").write_text("x", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("x", encoding="utf-8")

    tool = build_list_files_tool(_settings(tmp_path))
    result = tool.invoke({"path": ""})

    assert ".secret" not in result
    assert "1 hidden entries skipped" in result


def test_max_entries_truncates(tmp_path):
    for index in range(5):
        (tmp_path / f"f{index}.txt").write_text("x", encoding="utf-8")

    tool = build_list_files_tool(_settings(tmp_path))
    result = tool.invoke({"path": "", "max_entries": 2})

    assert result.count("[file]") == 2
    assert "showing first 2 of 5 entries" in result


def test_empty_directory(tmp_path):
    tool = build_list_files_tool(_settings(tmp_path))
    result = tool.invoke({"path": ""})

    assert "(no entries)" in result


@pytest.mark.parametrize("path", ["../outside"])
def test_escape_refused(tmp_path, path):
    tool = build_list_files_tool(_settings(tmp_path))
    result = tool.invoke({"path": path})

    assert "Could not list directory" in result
    assert "outside the allowed directory" in result


def test_missing_directory_refused(tmp_path):
    tool = build_list_files_tool(_settings(tmp_path))
    result = tool.invoke({"path": "nope"})

    assert "directory not found" in result


def test_file_target_refused(tmp_path):
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")

    tool = build_list_files_tool(_settings(tmp_path))
    result = tool.invoke({"path": "note.txt"})

    assert "not a directory" in result


def test_protected_name_refused(tmp_path):
    (tmp_path / ".env").write_text("KEY=1", encoding="utf-8")

    tool = build_list_files_tool(_settings(tmp_path))
    result = tool.invoke({"path": ".env"})

    assert "protected path" in result


def test_tool_name():
    tool = build_list_files_tool(_settings(Path(".")))

    assert tool.name == "list_files"


def test_provider_registers_and_gates(tmp_path):
    provider = DocumentToolProvider(_settings(tmp_path))
    names = {tool.name for tool in provider.tools()}

    assert "list_files" in names
