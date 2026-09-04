from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from src.backend.tools.zip_file import build_zip_tools


@pytest.fixture
def zip_scope(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "files"
    root.mkdir()
    archive_path = root / "bundle.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("docs/", "")
        archive.writestr("readme.txt", "hello from the archive\n")
        archive.writestr("data/app.json", '{"ok": true}\n')
        archive.writestr("blob.bin", b"\x00\x01\x02binary")
        archive.writestr("huge.txt", "x" * 5000)
    archive_path.write_bytes(buffer.getvalue())
    return root, archive_path


def _tools(root: Path, max_bytes: int = 1_000_000):
    from src.config import Settings

    settings = Settings(
        dashscope_api_key="test-key",
        file_read_enabled=True,
        file_read_root=str(root),
        file_read_max_bytes=max_bytes,
    )
    return build_zip_tools(settings)


def test_build_zip_tools_exposes_inspect_and_read():
    root = Path("unused")
    assert [tool.name for tool in _tools(root)] == [
        "inspect_zip_file",
        "read_zip_entry",
    ]


def test_inspect_lists_entries_sizes_and_directories(zip_scope):
    root, archive_path = zip_scope
    inspect_tool, _read = _tools(root)

    result = inspect_tool.invoke({"path": "bundle.zip"})

    assert f"ZIP archive {archive_path.name}: 5 entries (4 files, 1 directories)" in result
    assert "- docs/ (directory)" in result
    assert "- readme.txt (" in result
    assert "- data/app.json (" in result
    assert "compressed" in result


def test_read_zip_entry_returns_text_without_extraction(zip_scope):
    root, archive_path = zip_scope
    _inspect, read = _tools(root)

    result = read.invoke({"path": "bundle.zip", "entry": "readme.txt"})

    assert f"Contents of readme.txt in {archive_path.name}" in result
    assert "hello from the archive" in result
    # Nothing was ever extracted to disk.
    assert [child.name for child in root.iterdir()] == ["bundle.zip"]


def test_read_zip_entry_truncates_to_max_chars(zip_scope):
    root, _archive_path = zip_scope
    _inspect, read = _tools(root)

    result = read.invoke({"path": "bundle.zip", "entry": "readme.txt", "max_chars": 10})

    assert "showing first 10" in result


def test_read_zip_entry_describes_binary_members(zip_scope):
    root, archive_path = zip_scope
    _inspect, read = _tools(root)

    result = read.invoke({"path": "bundle.zip", "entry": "blob.bin"})

    assert "appears to be binary" in result
    assert "\x00" not in result


def test_read_zip_entry_rejects_missing_and_directory_entries(zip_scope):
    root, archive_path = zip_scope
    _inspect, read = _tools(root)

    missing = read.invoke({"path": "bundle.zip", "entry": "nope.txt"})
    assert "not in" in missing
    assert archive_path.name in missing
    assert "inspect_zip_file" in missing

    directory = read.invoke({"path": "bundle.zip", "entry": "docs/"})
    assert "is a directory" in directory


def test_read_zip_entry_refuses_entries_over_the_size_limit(zip_scope):
    root, _archive_path = zip_scope
    # The limit sits below the entry's uncompressed size (5,000) but above
    # the archive's own on-disk size so resolve_safe_path lets it through.
    _inspect, read = _tools(root, max_bytes=1_000)

    result = read.invoke({"path": "bundle.zip", "entry": "huge.txt"})

    assert "decompresses to" in result
    assert "limit 1,000 bytes" in result


def test_inspect_and_read_reject_invalid_archives(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    (root / "broken.zip").write_bytes(b"definitely not a zip")
    inspect_tool, read = _tools(root)

    inspect_result = inspect_tool.invoke({"path": "broken.zip"})
    read_result = read.invoke({"path": "broken.zip", "entry": "any.txt"})

    assert "not a valid zip archive" in inspect_result
    assert "not a valid zip archive" in read_result


def test_zip_tools_block_traversal_wrong_suffix_and_protected_files(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    outside = tmp_path / "outside.zip"
    with zipfile.ZipFile(outside, "w") as archive:
        archive.writestr("secret.txt", "top secret")
    inspect_tool, read = _tools(root)

    traversal = inspect_tool.invoke({"path": "../outside.zip"})
    assert "access denied" in traversal.lower()
    assert "top secret" not in traversal

    (root / "archive.rar").write_bytes(b"placeholder")
    wrong_suffix = inspect_tool.invoke({"path": "archive.rar"})
    assert "unsupported file type" in wrong_suffix

    # Entry names that look like path traversal are inert: entries are only
    # ever read from the archive in memory, never extracted to disk.
    malicious = root / "traversal.zip"
    with zipfile.ZipFile(malicious, "w") as archive:
        archive.writestr("../../../etc/escape.txt", "escaped?")
    result = read.invoke(
        {"path": "traversal.zip", "entry": "../../../etc/escape.txt"}
    )
    assert "escape.txt" in result
    assert not (tmp_path / "etc").exists()
    assert not (root / "etc").exists()


def test_zip_tools_are_absent_when_file_reading_is_disabled():
    from src.config import Settings

    settings = Settings(
        dashscope_api_key="test-key",
        file_read_enabled=False,
        file_read_root=".",
    )

    assert build_zip_tools(settings) == []
