from __future__ import annotations

import pytest

from src.web_search.content_fetcher import fetch_pages
from src.web_search.pdf_loader import PdfPageLoader, extract_pdf_text, is_pdf_url

pytest.importorskip("pypdf")


def _build_pdf(objects: list[bytes]) -> bytes:
    """Assemble a valid PDF with a correct xref table from numbered objects."""

    header = b"%PDF-1.4\n"
    body = b""
    offsets: list[int] = []
    for number, payload in enumerate(objects, start=1):
        offsets.append(len(header) + len(body))
        body += b"%d 0 obj\n" % number + payload + b"\nendobj\n"

    xref_offset = len(header) + len(body)
    xref = b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        xref += b"%010d 00000 n \n" % offset
    trailer = b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_offset,
    )
    return header + body + xref + trailer


def _pdf_bytes(text: str) -> bytes:
    """Build a minimal single-page PDF whose content stream holds ``text``."""

    stream = f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET\n".encode("latin-1")
    return _build_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )


def _blank_pdf_bytes() -> bytes:
    return _build_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
        ]
    )


class _Response:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


class _Session:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.calls: list[str] = []

    def get(self, url: str, **_kwargs: object) -> _Response:
        self.calls.append(url)
        return _Response(self.content)


def test_is_pdf_url_matches_only_pdf_paths():
    assert is_pdf_url("https://www.gov.cn/zhengce/2026/notice.pdf")
    assert is_pdf_url("https://example.com/report.PDF?download=1")
    assert not is_pdf_url("https://example.com/pdf-guide")
    assert not is_pdf_url("https://example.com/news/2026/launch")


def test_extract_pdf_text_reads_the_text_layer():
    text, pages = extract_pdf_text(_pdf_bytes("Nanjing metro operates 14 lines in 2026"))

    assert pages == 1
    assert "Nanjing metro" in text


def test_pdf_page_loader_returns_a_document_with_source_metadata():
    session = _Session(_pdf_bytes("Blackwell architecture technical brief"))
    loader = PdfPageLoader(
        "https://resources.example.com/brief.pdf",
        page_timeout=5,
        session=session,
    )

    documents = loader.load()

    assert session.calls == ["https://resources.example.com/brief.pdf"]
    assert len(documents) == 1
    assert "Blackwell architecture" in documents[0].page_content
    assert documents[0].metadata["source"] == "https://resources.example.com/brief.pdf"
    assert documents[0].metadata["fetch_method"] == "pdf"


def test_pdf_page_loader_raises_when_no_text_layer_exists():
    loader = PdfPageLoader(
        "https://example.com/scan.pdf",
        page_timeout=5,
        session=_Session(_blank_pdf_bytes()),
    )

    with pytest.raises(ValueError, match="no extractable text"):
        loader.load()


def test_fetch_pages_keeps_pdf_text_without_html_extraction():
    long_text = "Nanjing metro operated 14 lines as of 2026. " * 12
    session = _Session(_pdf_bytes(long_text))

    def loader_factory(url: str, page_timeout: int) -> PdfPageLoader:
        return PdfPageLoader(url, page_timeout, session=session)

    pages = fetch_pages(
        ["https://www.example.cn/notice/metro.pdf"],
        timeout=5,
        cache_ttl_seconds=0,
        relevance_query="Nanjing metro lines 2026",
        loader_factory=loader_factory,
    )

    assert len(pages) == 1
    assert pages[0].error is None
    assert "Nanjing metro" in pages[0].text
