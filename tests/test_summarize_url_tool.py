from __future__ import annotations

from src.config import Settings
from src.backend.tools import build_summarize_url_tool
from src.backend.tools.summarize_tool import summarize_url


class _FakePage:
    def __init__(self, url: str, title: str, text: str, error: str | None = None):
        self.url = url
        self.title = title
        self.text = text
        self.error = error


def _settings() -> Settings:
    return Settings(dashscope_api_key="test-key", summarize_url_enabled=True)


def test_summarize_url_rejects_non_http():
    result = summarize_url("ftp://example.com/file", settings=_settings())
    assert "unsupported url" in result.lower()


def test_summarize_url_fetches_and_summarizes(monkeypatch):
    page = _FakePage(
        "https://example.com/article",
        "Example Article",
        "The quick brown fox jumps over the lazy dog. " * 20,
    )
    monkeypatch.setattr(
        "src.backend.web_search.content_fetcher.fetch_pages",
        lambda *args, **kwargs: [page],
    )
    monkeypatch.setattr(
        "src.backend.web_search.content_fetcher.is_readable_text",
        lambda *args, **kwargs: True,
    )

    captured = {}

    def fake_invoke(model, messages, max_retries=3):
        captured["prompt"] = messages[0].content

        class _Result:
            content = "This article is about a fox and a dog."

        return _Result()

    monkeypatch.setattr("src.backend.llm.provider.build_chat_model", lambda s: object())
    monkeypatch.setattr("src.utils.retry.invoke_with_retry", fake_invoke)

    result = summarize_url(
        "https://example.com/article", focus="what is it about", settings=_settings()
    )

    assert "Summary of Example Article" in result
    assert "fox and a dog" in result
    # The page content and focus must be passed into the prompt.
    assert "what is it about" in captured["prompt"]
    assert "quick brown fox" in captured["prompt"]


def test_summarize_url_reports_unreadable(monkeypatch):
    page = _FakePage("https://example.com/empty", "", "", error="empty page")
    monkeypatch.setattr(
        "src.backend.web_search.content_fetcher.fetch_pages",
        lambda *args, **kwargs: [page],
    )
    monkeypatch.setattr(
        "src.backend.web_search.content_fetcher.is_readable_text",
        lambda *args, **kwargs: False,
    )

    result = summarize_url("https://example.com/empty", settings=_settings())
    assert "could not read content" in result.lower()


def test_summarize_url_builder_name():
    assert build_summarize_url_tool(_settings()).name == "summarize_url"
