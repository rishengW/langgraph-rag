from __future__ import annotations

import logging

from langchain_core.documents import Document

from src import web_search
from src.web_search import content_fetcher
from src.web_search.content_fetcher import (
    FetchedPage,
    estimate_tokens,
    extract_text,
    fetch_pages,
    is_readable_text,
)
from src.web_search.fetch_policy import domain_matches, resolve_fetch_policy
from src.web_search.prompt_builder import build_web_search_prompt


# REFACTOR: Focused coverage for lightweight web-search fetch/prompt primitives.
def test_fetch_pages_delegates_to_shared_document_loader(monkeypatch, caplog):
    calls: list[dict[str, object]] = []
    caplog.set_level(logging.INFO, logger="src.web_search.content_fetcher")

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
        min_readable_chars=1,
        min_readable_tokens=1,
    )

    assert calls[0]["urls"] == ["https://example.com/a", "https://example.com/missing"]
    assert calls[0]["page_load_timeout"] == 3
    assert calls[0]["max_concurrent_loads"] == 2
    assert calls[0]["page_load_cache_ttl_seconds"] == 30
    assert pages[0].url == "https://example.com/a"
    assert pages[0].title == "Alpha"
    assert pages[0].text == "Alpha page content."
    assert pages[0].error is None
    assert pages[0].extracted_chars == len("Alpha page content.")
    assert pages[0].extracted_tokens == estimate_tokens("Alpha page content.")
    assert pages[1].url == "https://example.com/missing"
    assert pages[1].text == ""
    assert pages[1].error == "No document loaded for URL."
    assert (
        "Fetched web page content: url=https://example.com/a "
        "extracted_chars=19 extracted_tokens=5 prompt_chars=19 "
        "prompt_tokens=5 error=None"
    ) in caplog.text
    assert (
        "Fetched web page content: url=https://example.com/missing "
        "extracted_chars=0 extracted_tokens=0 prompt_chars=0 "
        "prompt_tokens=0 error=No document loaded for URL."
    ) in caplog.text


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


def test_fetch_pages_marks_tiny_shell_text_unreadable_and_logs_size(
    monkeypatch,
    caplog,
):
    def fake_load_source_documents(_urls, **_kwargs):
        return [
            Document(
                page_content=(
                    "<html><head><title>Shell</title></head>"
                    "<body><main>DeepSeek loading menu</main></body></html>"
                ),
                metadata={"source": "https://example.com/shell", "title": "Shell"},
            )
        ]

    monkeypatch.setattr(
        content_fetcher,
        "load_source_documents",
        fake_load_source_documents,
    )
    caplog.set_level(logging.INFO, logger="src.web_search.content_fetcher")

    pages = fetch_pages(
        ["https://example.com/shell", "https://example.com/missing"],
        min_readable_chars=200,
        min_readable_tokens=50,
    )

    assert pages[0].title == "Shell"
    assert pages[0].text == ""
    assert pages[0].extracted_chars == len("DeepSeek loading menu")
    assert pages[0].extracted_tokens == estimate_tokens("DeepSeek loading menu")
    assert "below readability threshold" in (pages[0].error or "")
    assert pages[1].error == "No document loaded for URL."
    assert "url=https://example.com/shell extracted_chars=21" in caplog.text
    assert "extracted_tokens=6 prompt_chars=0 prompt_tokens=0" in caplog.text
    assert "url=https://example.com/missing extracted_chars=0" in caplog.text
    assert "error=No document loaded for URL." in caplog.text


def test_fetch_pages_retries_policy_domain_with_js_loader(monkeypatch):
    calls: list[str] = []

    class FakeLoader:
        def __init__(self, url: str, text: str, method: str) -> None:
            self.url = url
            self.text = text
            self.method = method

        def load(self):
            calls.append(self.method)
            return [
                Document(
                    page_content=f"<main>{self.text}</main>",
                    metadata={"source": self.url, "title": self.method},
                )
            ]

    def fake_load_source_documents(urls, **kwargs):
        documents = []
        for url in urls:
            documents.extend(kwargs["loader_factory"](url, 5).load())
        return documents

    def http_loader(url, _timeout):
        return FakeLoader(url, "loading", "http")

    def js_loader(url, _timeout):
        return FakeLoader(url, " ".join(["rendered detail"] * 40), "js")

    monkeypatch.setattr(
        content_fetcher,
        "load_source_documents",
        fake_load_source_documents,
    )

    pages = fetch_pages(
        ["https://baike.baidu.com/item/DeepSeek/65368136"],
        min_readable_chars=100,
        min_readable_tokens=20,
        loader_factory=http_loader,
        js_fallback_enabled=True,
        js_loader_factory=js_loader,
    )

    assert calls == ["http", "js"]
    assert pages[0].text.startswith("rendered detail")
    assert pages[0].fetch_method == "js_fallback"
    assert pages[0].error is None


def test_fetch_pages_force_js_domain_skips_http_loader(monkeypatch):
    calls: list[str] = []

    class FakeLoader:
        def __init__(self, url: str) -> None:
            self.url = url

        def load(self):
            calls.append("js")
            return [
                Document(
                    page_content="<main>" + " ".join(["forced render"] * 40) + "</main>",
                    metadata={"title": "JS"},
                )
            ]

    def fake_load_source_documents(urls, **kwargs):
        documents = []
        for url in urls:
            documents.extend(kwargs["loader_factory"](url, 5).load())
        return documents

    def http_loader(_url, _timeout):  # pragma: no cover - must not run
        raise AssertionError("HTTP loader should be skipped for force-JS domains")

    monkeypatch.setattr(
        content_fetcher,
        "load_source_documents",
        fake_load_source_documents,
    )

    pages = fetch_pages(
        ["https://force-js.test/page"],
        min_readable_chars=100,
        min_readable_tokens=20,
        loader_factory=http_loader,
        js_fallback_enabled=True,
        js_force_domains=["force-js.test"],
        js_loader_factory=lambda url, _timeout: FakeLoader(url),
    )

    assert calls == ["js"]
    assert pages[0].text.startswith("forced render")
    assert pages[0].fetch_method == "js"
    assert pages[0].url == "https://force-js.test/page"


def test_fetch_pages_rejects_js_loader_error_text(monkeypatch):
    class FakeLoader:
        def __init__(self, url: str) -> None:
            self.url = url

        def load(self):
            return [
                Document(
                    page_content="<main>Error: " + "browser failed " * 40 + "</main>",
                    metadata={"source": self.url, "title": "JS"},
                )
            ]

    def fake_load_source_documents(urls, **kwargs):
        documents = []
        for url in urls:
            documents.extend(kwargs["loader_factory"](url, 5).load())
        return documents

    monkeypatch.setattr(
        content_fetcher,
        "load_source_documents",
        fake_load_source_documents,
    )

    pages = fetch_pages(
        ["https://force-js.test/page"],
        min_readable_chars=100,
        min_readable_tokens=20,
        js_fallback_enabled=True,
        js_force_domains=["force-js.test"],
        js_loader_factory=lambda url, _timeout: FakeLoader(url),
    )

    assert pages[0].text == ""
    assert pages[0].fetch_method == "js"
    assert "JS loader returned error text" in (pages[0].error or "")


def test_fetch_pages_keeps_http_error_when_js_fallback_fails(monkeypatch):
    class FakeLoader:
        def __init__(self, url: str, text: str) -> None:
            self.url = url
            self.text = text

        def load(self):
            return [
                Document(
                    page_content=f"<main>{self.text}</main>",
                    metadata={"source": self.url, "title": "HTTP"},
                )
            ]

    def fake_load_source_documents(urls, **kwargs):
        loader_factory = kwargs["loader_factory"]
        if loader_factory is js_loader:
            raise RuntimeError("browser missing")
        documents = []
        for url in urls:
            documents.extend(loader_factory(url, 5).load())
        return documents

    def http_loader(url, _timeout):
        return FakeLoader(url, "loading")

    def js_loader(_url, _timeout):  # pragma: no cover - fake loader identity only
        raise AssertionError("fake_load_source_documents raises before this runs")

    monkeypatch.setattr(
        content_fetcher,
        "load_source_documents",
        fake_load_source_documents,
    )

    pages = fetch_pages(
        ["https://zhuanlan.zhihu.com/p/123"],
        min_readable_chars=100,
        min_readable_tokens=20,
        loader_factory=http_loader,
        js_fallback_enabled=True,
        js_loader_factory=js_loader,
    )

    assert pages[0].text == ""
    assert pages[0].fetch_method == "http+js_failed"
    assert "below readability threshold" in (pages[0].error or "")
    assert "browser missing" in (pages[0].error or "")


def test_resolve_fetch_policy_matches_domains_and_subdomains():
    policy = resolve_fetch_policy(
        "https://sub.example.com/page",
        js_fallback_enabled=True,
        js_fallback_domains=["example.com"],
        js_force_domains=[],
    )

    assert domain_matches("sub.example.com", ["example.com"]) is True
    assert domain_matches("badexample.com", ["example.com"]) is False
    assert policy.retry_js_on_low_text is True
    assert policy.force_js is False
    assert policy.request_headers["Accept-Language"].startswith("zh-CN")


def test_is_readable_text_uses_configurable_thresholds():
    assert is_readable_text("DeepSeek loading menu") is False
    assert is_readable_text("x" * 200) is True
    assert is_readable_text("x" * 197) is True
    assert (
        is_readable_text(
            "short enough only because thresholds are lowered",
            min_chars=1,
            min_tokens=1,
        )
        is True
    )
    assert is_readable_text("anything non-empty", min_chars=0, min_tokens=0) is True
    assert is_readable_text("", min_chars=0, min_tokens=0) is False


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
    assert web_search.is_readable_text is is_readable_text
    assert web_search.build_web_search_prompt is build_web_search_prompt
