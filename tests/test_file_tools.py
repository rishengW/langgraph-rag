from __future__ import annotations

import zipfile
from pathlib import Path

import openpyxl

from src.config import Settings
from src.tools import (
    build_excel_tool,
    build_pdf_tool,
    build_text_file_tool,
    build_word_tool,
)

_DOCX_DOCUMENT_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/'
    'wordprocessingml/2006/main"><w:body>'
    "<w:p><w:r><w:t>Hello world</w:t></w:r></w:p>"
    "<w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p>"
    "</w:body></w:document>"
)


def _settings(root: Path) -> Settings:
    return Settings(
        dashscope_api_key="test-key",
        file_read_enabled=True,
        file_read_root=str(root),
        file_read_max_bytes=5_000_000,
    )


def _write_docx(path: Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", _DOCX_DOCUMENT_XML)


def _write_xlsx(path: Path) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Data"
    sheet.append(["Name", "Score"])
    sheet.append(["Alice", 90])
    sheet.append(["Bob", 85])
    workbook.save(path)


def test_text_tool_reads_file(tmp_path):
    target = tmp_path / "note.txt"
    target.write_text("line one\nline two", encoding="utf-8")

    tool = build_text_file_tool(_settings(tmp_path))
    result = tool.invoke({"path": "note.txt"})

    assert "Contents of note.txt" in result
    assert "line one" in result
    assert "line two" in result


def test_text_tool_truncates_to_max_chars(tmp_path):
    target = tmp_path / "big.txt"
    target.write_text("x" * 500, encoding="utf-8")

    tool = build_text_file_tool(_settings(tmp_path))
    result = tool.invoke({"path": "big.txt", "max_chars": 100})

    assert "showing first 100" in result


def test_text_tool_blocks_path_traversal(tmp_path):
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("top secret", encoding="utf-8")
    root = tmp_path / "docs"
    root.mkdir()

    tool = build_text_file_tool(_settings(root))
    result = tool.invoke({"path": "../secret.txt"})

    assert "access denied" in result.lower()
    assert "top secret" not in result


def test_text_tool_refuses_protected_file(tmp_path):
    (tmp_path / ".env").write_text("DASHSCOPE_API_KEY=leak", encoding="utf-8")

    tool = build_text_file_tool(_settings(tmp_path))
    # .env is both a protected name and an unsupported suffix; either way the
    # secret value must never appear in the output.
    result = tool.invoke({"path": ".env"})

    assert "leak" not in result
    assert "could not read" in result.lower()


def test_text_tool_missing_file(tmp_path):
    tool = build_text_file_tool(_settings(tmp_path))
    result = tool.invoke({"path": "nope.txt"})

    assert "not found" in result.lower()


def test_word_tool_extracts_paragraphs(tmp_path):
    target = tmp_path / "doc.docx"
    _write_docx(target)

    tool = build_word_tool(_settings(tmp_path))
    result = tool.invoke({"path": "doc.docx"})

    assert "Hello world" in result
    assert "Second paragraph" in result
    assert "2 paragraphs" in result


def test_word_tool_rejects_legacy_doc(tmp_path):
    tool = build_word_tool(_settings(tmp_path))
    result = tool.invoke({"path": "old.doc"})

    assert ".doc" in result
    assert "not supported" in result.lower()


def test_excel_tool_reads_active_sheet(tmp_path):
    target = tmp_path / "book.xlsx"
    _write_xlsx(target)

    tool = build_excel_tool(_settings(tmp_path))
    result = tool.invoke({"path": "book.xlsx"})

    assert "reading 'Data'" in result
    assert "Alice" in result
    assert "Bob" in result
    assert "| C1 | C2 |" in result


def test_excel_tool_unknown_sheet_lists_available(tmp_path):
    target = tmp_path / "book.xlsx"
    _write_xlsx(target)

    tool = build_excel_tool(_settings(tmp_path))
    result = tool.invoke({"path": "book.xlsx", "sheet": "Missing"})

    assert "not found" in result.lower()
    assert "Data" in result


def test_excel_tool_wrong_suffix_rejected(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("not a spreadsheet", encoding="utf-8")

    tool = build_excel_tool(_settings(tmp_path))
    result = tool.invoke({"path": "data.txt"})

    assert "unsupported file type" in result.lower()


def test_file_tool_builders_expose_expected_names(tmp_path):
    settings = _settings(tmp_path)

    assert build_text_file_tool(settings).name == "read_text_file"
    assert build_word_tool(settings).name == "read_word_document"
    assert build_excel_tool(settings).name == "read_excel_spreadsheet"
    assert build_pdf_tool(settings).name == "read_pdf"


def test_pdf_tool_reads_text(tmp_path):
    pdf_bytes = _MINIMAL_PDF
    target = tmp_path / "doc.pdf"
    target.write_bytes(pdf_bytes)

    tool = build_pdf_tool(_settings(tmp_path))
    result = tool.invoke({"path": "doc.pdf"})

    assert "Hello PDF" in result


def test_pdf_tool_rejects_wrong_suffix(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_text("not a pdf", encoding="utf-8")

    tool = build_pdf_tool(_settings(tmp_path))
    result = tool.invoke({"path": "notes.txt"})

    assert "unsupported file type" in result.lower()


def test_pdf_tool_blocks_path_traversal(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    tool = build_pdf_tool(_settings(root))
    result = tool.invoke({"path": "../escape.pdf"})

    assert "access denied" in result.lower()


# A minimal, valid single-page PDF with the text "Hello PDF" in its content
# stream. Handcrafted so the test needs no PDF-authoring dependency.
_MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
    b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n"
    b"4 0 obj\n<< /Length 44 >>\nstream\n"
    b"BT /F1 24 Tf 72 700 Td (Hello PDF) Tj ET\n"
    b"endstream\nendobj\n"
    b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    b"xref\n0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000241 00000 n \n"
    b"0000000334 00000 n \n"
    b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
    b"startxref\n405\n"
    b"%%EOF\n"
)
