from __future__ import annotations

from langchain_core.documents import Document

from src import web_search
from src.web_search import content_fetcher
from src.web_search.content_fetcher import FetchedPage, extract_text, fetch_pages
from src.web_search.prompt_builder import build_web_search_prompt


# REFACTOR: Focused coverage for lightweight web-search fetch/prompt primitives.
def test_fetch_pages_delegates_to_shared_document_loader(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_load_source_documents(urls, **kwargs):
        calls.append({"urls": list(urls), **kwargs})
        return [
            Document(
                page_content="""
                <html><head><title>Ignored</title></head>
                <body><nav>Menu</nav><main>Alpha page content.</main></body></html>
                """,
                metadata={"source": "https://example.com/a", "title": "Alpha"},
            )
        ]

    monkeypatch.setattr(
        content_fetcher,
        "load_source_documents",
        fake_load_source_documents,
    )

    pages = fetch_pages(
        ["https://example.com/a", "https://example.com/missing"],
        timeout=3.8,
        max_tokens_per_page=10,
        cache_ttl_seconds=30,
        max_concurrent_loads=2,
    )

    assert calls[0]["urls"] == ["https://example.com/a", "https://example.com/missing"]
    assert calls[0]["page_load_timeout"] == 3
    assert calls[0]["max_concurrent_loads"] == 2
    assert calls[0]["page_load_cache_ttl_seconds"] == 30
    assert pages[0].url == "https://example.com/a"
    assert pages[0].title == "Alpha"
    assert pages[0].text == "Alpha page content."
    assert pages[0].error is None
    assert pages[1].url == "https://example.com/missing"
    assert pages[1].text == ""
    assert pages[1].error == "No document loaded for URL."


def test_fetch_pages_returns_error_pages_when_loader_fails(monkeypatch):
    def fake_load_source_documents(_urls, **_kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(
        content_fetcher,
        "load_source_documents",
        fake_load_source_documents,
    )

    pages = fetch_pages(["https://example.com/a", "https://example.com/b"])

    assert [page.url for page in pages] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert [page.error for page in pages] == ["network unavailable", "network unavailable"]


def test_extract_text_prefers_main_content_and_removes_boilerplate():
    text = extract_text(
        """
        <html>
          <body>
            <header>Site header</header>
            <main><h1>Title</h1><p>Useful answer text.</p></main>
            <footer>Legal links</footer>
          </body>
        </html>
        """
    )

    assert text == "Title Useful answer text."


def test_build_web_search_prompt_formats_sources_and_skips_empty_pages():
    prompt = build_web_search_prompt(
        "What launched?",
        [
            FetchedPage(
                url="https://example.com/a",
                title="Launch",
                text="The product launched with citation-worthy details.",
                fetch_time_ms=12.0,
            ),
            FetchedPage(
                url="https://example.com/empty",
                title="Empty",
                text="",
                fetch_time_ms=12.0,
                error="empty",
            ),
        ],
        max_total_tokens=80,
    )

    assert "Question: What launched?" in prompt
    assert "--- Source: https://example.com/a (Title: Launch) ---" in prompt
    assert "citation-worthy details" in prompt
    assert "https://example.com/empty" not in prompt
    assert prompt.endswith("Answer:")


def test_build_web_search_prompt_respects_total_token_budget():
    prompt = build_web_search_prompt(
        "Summarize",
        [
            FetchedPage(
                url="https://example.com/a",
                title="",
                text=" ".join(["alpha"] * 200),
                fetch_time_ms=1.0,
            ),
            FetchedPage(
                url="https://example.com/b",
                title="",
                text=" ".join(["beta"] * 200),
                fetch_time_ms=1.0,
            ),
        ],
        max_total_tokens=70,
    )

    assert "https://example.com/a" in prompt
    # The fixed instructional preamble is excluded from the budget, so assert
    # the budget governs the *source* content rather than the total length:
    # only the first source fits the 70-token source budget, the second is
    # trimmed out.
    assert "https://example.com/b" not in prompt
    assert prompt.count("--- Source:") == 1


def test_web_search_package_exports_lightweight_primitives():
    assert web_search.FetchedPage is FetchedPage
    assert web_search.fetch_pages is fetch_pages
    assert web_search.build_web_search_prompt is build_web_search_prompt
