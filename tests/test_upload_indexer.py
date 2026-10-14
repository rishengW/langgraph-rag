"""Tests for upload parsing, chunking, and per-session indexing."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.backend.rag.chroma_retriever import ChromaRetriever, _delete_chunks_by_source
from src.backend.rag.upload_indexer import (
    INDEXABLE_SUFFIXES,
    UploadIndexingError,
    chunk_upload_documents,
    parse_and_chunk_upload,
    parse_upload_document,
)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_text_upload_sets_source_metadata(tmp_path):
    file = _write(tmp_path / "notes.txt", "alpha paragraph\n\nbeta paragraph")

    documents = parse_upload_document(file, source="chat_uploads/t1/notes.txt")

    assert [doc.page_content for doc in documents] == [
        "alpha paragraph",
        "beta paragraph",
    ]
    assert all(doc.metadata["source"] == "chat_uploads/t1/notes.txt" for doc in documents)
    assert all(doc.metadata["kind"] == "upload" for doc in documents)


def test_parse_upload_rejects_unindexable_suffix(tmp_path):
    file = _write(tmp_path / "archive.zip", "binary-ish")

    with pytest.raises(UploadIndexingError, match="unsupported file type"):
        parse_upload_document(file, source="chat_uploads/t1/archive.zip")


def test_parse_upload_rejects_empty_text(tmp_path):
    file = _write(tmp_path / "empty.txt", "   \n \n")

    with pytest.raises(UploadIndexingError, match="no extractable text"):
        parse_upload_document(file, source="chat_uploads/t1/empty.txt")


def test_parse_docx_upload_extracts_paragraphs(tmp_path):
    import zipfile

    body = (
        "<w:body>"
        + "".join(f"<w:p><w:r><w:t>paragraph {index}</w:t></w:r></w:p>" for index in range(1, 4))
        + "</w:body>"
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"{body}</w:document>"
    )
    file = tmp_path / "doc.docx"
    with zipfile.ZipFile(file, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    documents = parse_upload_document(file, source="chat_uploads/t1/doc.docx")

    assert [doc.page_content for doc in documents] == [
        "paragraph 1",
        "paragraph 2",
        "paragraph 3",
    ]


def test_parse_docx_upload_rejects_corrupt_archive(tmp_path):
    file = tmp_path / "broken.docx"
    file.write_bytes(b"not a zip at all")

    with pytest.raises(UploadIndexingError, match="could not read the Word document"):
        parse_upload_document(file, source="chat_uploads/t1/broken.docx")


def test_parse_pdf_upload_extracts_pages(tmp_path):
    pypdf = pytest.importorskip("pypdf")

    writer = pypdf.PdfWriter()
    for _index in range(1, 3):
        writer.add_blank_page(width=200, height=200)
    # pypdf cannot add text without a reportlab-like producer; a blank-page
    # PDF yields no extractable text, which must raise the no-text error.
    file = tmp_path / "blank.pdf"
    with open(file, "wb") as handle:
        writer.write(handle)

    with pytest.raises(UploadIndexingError, match="no extractable text"):
        parse_upload_document(file, source="chat_uploads/t1/blank.pdf")


def test_parse_csv_upload_reads_paragraph_units(tmp_path):
    file = _write(tmp_path / "data.csv", "name,role\nada,pioneer\n\nalan,visionary")

    documents = parse_upload_document(file, source="chat_uploads/t1/data.csv")

    joined = "\n\n".join(doc.page_content for doc in documents)
    assert "name,role" in joined
    assert "ada,pioneer" in joined


def test_parse_upload_truncates_huge_text(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.backend.rag.upload_indexer.MAX_UPLOAD_DOC_CHARS",
        100,
    )
    file = _write(
        tmp_path / "big.txt", "\n\n".join(f"unit {index}" + "x" * 80 for index in range(10))
    )

    documents = parse_upload_document(file, source="chat_uploads/t1/big.txt")

    assert sum(len(doc.page_content) for doc in documents) <= 100


def test_indexable_suffixes_exclude_archives():
    assert ".zip" not in INDEXABLE_SUFFIXES
    assert ".txt" in INDEXABLE_SUFFIXES
    assert ".docx" in INDEXABLE_SUFFIXES
    assert ".pdf" in INDEXABLE_SUFFIXES


def test_chunk_upload_documents_quality_filters_and_splits(tmp_path, mock_settings):
    documents = [
        Document(
            page_content=(
                "Substantial content about retrieval pipelines, embedding "
                "indexes, chunk storage, and grounded question answering."
            )
        ),
        Document(page_content="hi"),
    ]

    result = chunk_upload_documents(
        documents,
        settings=mock_settings,
        source="chat_uploads/t1/notes.txt",
    )

    assert result.source == "chat_uploads/t1/notes.txt"
    assert result.chunks
    assert all("source" in doc.metadata for doc in result.chunks)
    assert all(doc.metadata["source"] == "chat_uploads/t1/notes.txt" for doc in result.chunks)
    # The "hi" unit is below the configured quality thresholds.
    assert all(len(doc.page_content) > 2 for doc in result.chunks)


def test_chunk_upload_documents_rejects_all_filtered(tmp_path, mock_settings):
    documents = [Document(page_content="a a a a a a a a a a a a a a a a a a")]

    with pytest.raises(UploadIndexingError, match="filtered out"):
        chunk_upload_documents(
            documents,
            settings=mock_settings,
            source="chat_uploads/t1/thin.txt",
        )


def test_parse_and_chunk_upload_enforces_chunk_cap(tmp_path, mock_settings, monkeypatch):
    from dataclasses import replace as _replace

    settings = _replace(mock_settings, upload_indexing_max_chunks=2)
    file = _write(
        tmp_path / "long.txt",
        "\n\n".join(
            "substantial retrieval embedding content paragraph number "
            f"{index} with enough unique terms to pass quality filtering"
            for index in range(20)
        ),
    )

    with pytest.raises(UploadIndexingError, match="would index .* chunks"):
        parse_and_chunk_upload(
            file,
            settings=settings,
            source="chat_uploads/t1/long.txt",
        )


def test_index_documents_adds_and_replaces_by_source(monkeypatch, isolated_settings):
    """index_documents must delete-then-add per source and refresh state."""

    class FakeCollection:
        def __init__(self) -> None:
            self.records: dict[str, dict[str, str]] = {}

        def get(self, where=None):
            if where and "source" in where:
                ids = [
                    id_ for id_, meta in self.records.items() if meta["source"] == where["source"]
                ]
                return {"ids": ids}
            return {"ids": list(self.records)}

        def delete(self, ids):
            for id_ in ids:
                self.records.pop(id_, None)

    class FakeVectorstore:
        def __init__(self) -> None:
            self.collection = FakeCollection()
            self._next_id = 0

        def add_documents(self, documents):
            for document in documents:
                self._next_id += 1
                self.collection.records[str(self._next_id)] = {
                    "source": document.metadata.get("source", "")
                }

        @property
        def _collection(self):
            return self.collection

    provider = ChromaRetriever.__new__(ChromaRetriever)
    provider.settings = isolated_settings()
    provider._vectorstore = FakeVectorstore()
    provider._embeddings = None
    provider._chroma_cls = None
    provider._corpus_nodes = None
    provider._bm25_retriever = None
    provider._parent_docstore = None
    provider._retriever = object()
    provider._prepare_enhanced_retrieval = lambda *args, **kwargs: True

    upload = "chat_uploads/t1/notes.txt"
    first = [Document(page_content="one", metadata={"source": upload})]
    second = [Document(page_content="replaced", metadata={"source": upload})]

    added = provider.index_documents(first)
    assert added == 1
    assert list(provider._vectorstore.collection.records.values()) == [{"source": upload}]

    # Re-indexing the same source replaces, not accumulates.
    provider.index_documents(second)
    records = list(provider._vectorstore.collection.records.values())
    assert records == [{"source": upload}]


def test_delete_chunks_by_source_handles_missing_capabilities():
    class Bare:
        pass

    assert _delete_chunks_by_source(Bare(), "x") == 0
    assert _delete_chunks_by_source(None, "x") == 0
