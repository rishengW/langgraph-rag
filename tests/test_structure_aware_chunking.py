"""Structure-aware chunking tests: HTML sections and markdown headings."""

from __future__ import annotations

from langchain_core.documents import Document

from src.backend.rag.document_loader import (
    _html_section_documents,
    split_markdown_sections,
)

_HTML = """<html><head><title>PAI Guide</title></head><body>
<nav>menu home about contact</nav>
<h1>Introduction</h1>
<p>PAI is a machine learning platform with fine-tuning and deployment.</p>
<h2>Fine-tuning</h2>
<p>The fine-tuning experience supports supervised and preference tuning jobs.</p>
<h2>Deployment</h2>
<p>Models deploy through EAS endpoints with autoscaling.</p>
</body></html>"""


def test_html_section_documents_splits_on_headings():
    documents = _html_section_documents(
        _HTML,
        {"source": "https://example.test/guide", "title": "PAI Guide"},
    )

    sections = [(doc.metadata.get("section"), doc.page_content) for doc in documents]
    labels = [label for label, _ in sections]
    assert "Introduction" in labels
    assert "Fine-tuning" in labels
    assert "Deployment" in labels
    for document in documents:
        assert document.metadata["source"] == "https://example.test/guide"
        assert document.metadata["title"] == "PAI Guide"
    fine_tuning_docs = [doc for doc in documents if doc.metadata.get("section") == "Fine-tuning"]
    fine_tuning_text = "\n".join(doc.page_content for doc in fine_tuning_docs)
    assert "supervised and preference tuning" in fine_tuning_text
    assert "autoscaling" not in fine_tuning_text


def test_html_section_documents_failure_returns_empty():
    # A splitter crash must degrade to "keep whole page", not raise.
    assert _html_section_documents("", {"source": "x"}) == []


def test_split_markdown_sections_splits_on_headings():
    markdown = (
        "# Overview\n\nRelevant content about retrieval pipelines.\n\n"
        "## Details\n\nChunk storage details follow here.\n\n"
        "### Subsection\n\nNested content lives here too.\n"
    )
    documents = split_markdown_sections(
        [Document(page_content=markdown, metadata={"source": "notes.md"})]
    )

    assert len(documents) == 3
    labels = [doc.metadata.get("section") for doc in documents]
    assert "Overview" in labels
    assert "Details" in labels
    assert "Subsection" in labels
    assert all(doc.metadata["source"] == "notes.md" for doc in documents)
    details = next(doc for doc in documents if doc.metadata.get("section") == "Details")
    assert "Chunk storage" in details.page_content


def test_split_markdown_sections_passes_plain_text_through():
    documents = split_markdown_sections(
        [
            Document(page_content="no headings here", metadata={"source": "a.txt"}),
            Document(page_content="# heading\ndata", metadata={"source": "b.md"}),
        ]
    )

    assert documents[0].page_content == "no headings here"
    assert "heading" in documents[1].page_content


def test_structure_aware_flag_gates_loader(tmp_path):
    from src.backend.rag.document_loader import default_loader_factory

    loader = default_loader_factory("https://example.test/x", 5)
    assert loader.structure_aware is False

    structured = default_loader_factory("https://example.test/x", 5, structure_aware=True)
    assert structured.structure_aware is True
