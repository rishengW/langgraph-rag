# REFACTOR: PDF fetch support so official notices and whitepapers stay eligible.
from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from io import BytesIO
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

PDF_URL_RE = re.compile(r"\.pdf$", re.I)
# Bound extraction so a large report cannot dominate fetch time or memory.
MAX_PDF_PAGES = 30
MAX_PDF_BYTES = 20_000_000
PDF_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


class PdfLoaderUnavailableError(RuntimeError):
    """Raised when pypdf is not installed for remote PDF extraction."""


def is_pdf_url(url: str) -> bool:
    """Return whether a URL points at a PDF document by path extension."""

    path = unquote(urlparse((url or "").strip()).path)
    return bool(PDF_URL_RE.search(path))


class PdfPageLoader:
    """Minimal loader-compatible wrapper that extracts remote PDF text."""

    def __init__(
        self,
        url: str,
        page_timeout: int,
        *,
        headers: Mapping[str, str] | None = None,
        verify_ssl: bool = True,
        session: Any | None = None,
        max_pages: int = MAX_PDF_PAGES,
    ) -> None:
        self.url = url
        self.timeout = max(1, int(page_timeout))
        self.headers = {"User-Agent": PDF_USER_AGENT, **dict(headers or {})}
        self.verify_ssl = verify_ssl
        self.max_pages = max(1, int(max_pages))
        self._session = session or requests

    def load(self) -> list[Document]:
        """Download one PDF and return its extracted text as a Document."""

        response = self._session.get(
            self.url,
            headers=self.headers,
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        response.raise_for_status()
        content = response.content or b""
        if len(content) > MAX_PDF_BYTES:
            raise ValueError(f"PDF exceeds the {MAX_PDF_BYTES} byte fetch limit")

        text, page_count = extract_pdf_text(content, max_pages=self.max_pages)
        if not text:
            raise ValueError(f"PDF has {page_count} page(s) but no extractable text layer")
        return [
            Document(
                page_content=text,
                metadata={
                    "source": self.url,
                    "url": self.url,
                    "title": _pdf_title(self.url),
                    "fetch_method": "pdf",
                    "pdf_pages": page_count,
                },
            )
        ]


def extract_pdf_text(content: bytes, *, max_pages: int = MAX_PDF_PAGES) -> tuple[str, int]:
    """Extract text from PDF bytes, returning the text and total page count."""

    try:
        import pypdf
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise PdfLoaderUnavailableError(
            "Remote PDF extraction requires pypdf. Run "
            "`python -m pip install pypdf` to enable PDF web sources."
        ) from exc

    reader = pypdf.PdfReader(BytesIO(content))
    total_pages = len(reader.pages)
    limit = min(max(1, int(max_pages)), total_pages)
    texts: list[str] = []
    for index in range(limit):
        extracted = (reader.pages[index].extract_text() or "").strip()
        if extracted:
            texts.append(extracted)
    return "\n\n".join(texts).strip(), total_pages


def pdf_loader_factory(url: str, page_timeout: int) -> PdfPageLoader:
    """Build a loader-compatible remote PDF reader for one URL."""

    return PdfPageLoader(url, page_timeout)


def _pdf_title(url: str) -> str:
    leaf = unquote(urlparse(url or "").path).rstrip("/").rsplit("/", 1)[-1]
    return leaf or url


__all__ = [
    "MAX_PDF_BYTES",
    "MAX_PDF_PAGES",
    "PdfLoaderUnavailableError",
    "PdfPageLoader",
    "extract_pdf_text",
    "is_pdf_url",
    "pdf_loader_factory",
]
