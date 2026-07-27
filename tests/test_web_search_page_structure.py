from __future__ import annotations

from src.web_search.content_fetcher import extract_text, fetch_pages
from src.web_search.page_structure import (
    assess_page_structure,
    structure_rejection_reason,
    word_count,
)

_ARTICLE_HTML = """
<html><head><title>Nanjing metro adds two lines</title></head><body>
<nav><a href="/">Home</a><a href="/news">News</a></nav>
<article>
<h1>Nanjing metro adds two lines in 2026</h1>
<p>The Nanjing metro network reached fourteen operating lines in 2026 after two
new routes opened in the western districts. Officials said the extension adds
roughly forty kilometres of track and eighteen stations to the network.</p>
<p>Ridership grew steadily through the year, with the operator reporting an
average of three million daily journeys across the network. Further extensions
are planned but have not yet been approved by the transport ministry.</p>
<p>The operator published the full timetable alongside a revised fare table, and
said interchange stations would receive additional platform staff during the
morning peak while passengers adapt to the new routes.</p>
</article>
<footer><a href="/privacy">Privacy</a></footer>
</body></html>
"""

_LISTING_HTML = """
<html><head><title>Metro news archive</title></head><body>
<main>
<h1>Metro tag archive</h1>
<ul>
<li><a href="/a1">Nanjing metro line 6 opens</a></li>
<li><a href="/a2">Nanjing metro fare change</a></li>
<li><a href="/a3">Metro ridership report</a></li>
<li><a href="/a4">Suzhou metro extension</a></li>
<li><a href="/a5">Hangzhou metro plan</a></li>
<li><a href="/a6">Wuxi metro update</a></li>
<li><a href="/a7">Metro safety notice</a></li>
<li><a href="/a8">Metro timetable revision</a></li>
<li><a href="/a9">Metro app release</a></li>
<li><a href="/a10">Metro staff hiring</a></li>
</ul>
</main>
</body></html>
"""

_GATEWAY_HTML = """
<html><head><title>Sign in</title></head><body>
<div><p>Please enable JavaScript to continue.</p></div>
</body></html>
"""


def test_word_count_counts_cjk_characters_and_latin_words():
    assert word_count("Nanjing metro 2026") == 3
    assert word_count("\u5357\u4eac\u5730\u94c1") == 4


def test_article_page_is_classified_as_content():
    structure = assess_page_structure(_ARTICLE_HTML, extract_text(_ARTICLE_HTML))

    assert structure.measured
    assert structure.shape == "article"
    assert structure.is_content_page
    assert structure.link_density < 0.5
    assert structure_rejection_reason(structure) is None


def test_link_dense_listing_page_is_rejected_without_url_hints():
    structure = assess_page_structure(_LISTING_HTML, extract_text(_LISTING_HTML))

    assert structure.shape == "listing"
    assert not structure.is_content_page
    assert structure.link_density >= 0.5
    assert structure_rejection_reason(structure) == "listing_page"


def test_login_or_javascript_shell_is_classified_as_gateway():
    structure = assess_page_structure(_GATEWAY_HTML, extract_text(_GATEWAY_HTML))

    assert structure.shape == "gateway"
    assert structure_rejection_reason(structure) == "gateway_page"


def test_short_body_is_thin_rather_than_listing():
    html = "<html><body><p>Fourteen lines are operating in 2026.</p></body></html>"

    structure = assess_page_structure(html, extract_text(html), min_content_words=60)

    assert structure.shape == "thin"
    assert structure_rejection_reason(structure) == "thin_page"


def test_unmeasurable_page_abstains_instead_of_rejecting():
    structure = assess_page_structure("", "")

    assert not structure.measured
    assert structure.is_content_page
    assert structure_rejection_reason(structure) is None


def test_long_article_with_many_citations_is_not_a_listing():
    body = "".join(
        f"<p>Paragraph {index} explains the 2026 network expansion in detail with "
        f'measured ridership figures and <a href="/ref{index}">a source</a>.</p>'
        for index in range(12)
    )
    html = f"<html><body><article>{body}</article></body></html>"

    structure = assess_page_structure(html, extract_text(html))

    assert structure.shape == "article"


class _StubLoader:
    def __init__(self, html: str) -> None:
        self.html = html

    def load(self):
        from langchain_core.documents import Document

        return [Document(page_content=self.html, metadata={"source": self.url})]


def _loader_factory(html: str):
    def factory(url: str, _page_timeout: int):
        loader = _StubLoader(html)
        loader.url = url
        return loader

    return factory


def test_fetch_pages_attaches_structure_to_each_page():
    pages = fetch_pages(
        ["https://example.com/news/metro"],
        timeout=5,
        cache_ttl_seconds=0,
        relevance_query="Nanjing metro lines 2026",
        loader_factory=_loader_factory(_ARTICLE_HTML),
    )

    assert len(pages) == 1
    assert pages[0].structure.measured
    assert pages[0].structure.shape == "article"
    assert pages[0].structure.content_words > 60


def test_js_retry_triggers_on_unreadable_text_outside_the_domain_list():
    """A JS-heavy host that nobody listed still gets one browser render."""

    rendered_calls: list[str] = []

    def js_factory(url: str, _page_timeout: int):
        rendered_calls.append(url)
        return _loader_factory(_ARTICLE_HTML)(url, _page_timeout)

    pages = fetch_pages(
        ["https://unlisted.example.com/app/metro"],
        timeout=5,
        cache_ttl_seconds=0,
        relevance_query="Nanjing metro lines 2026",
        loader_factory=_loader_factory("<html><body><div></div></body></html>"),
        js_fallback_enabled=True,
        js_fallback_domains=[],
        js_loader_factory=js_factory,
    )

    assert rendered_calls == ["https://unlisted.example.com/app/metro"]
    assert pages[0].fetch_method == "js_fallback"
    assert "Nanjing metro" in pages[0].text


def test_js_retry_respects_its_budget_and_prefers_configured_domains():
    rendered_calls: list[str] = []

    def js_factory(url: str, _page_timeout: int):
        rendered_calls.append(url)
        return _loader_factory(_ARTICLE_HTML)(url, _page_timeout)

    pages = fetch_pages(
        [
            "https://unlisted.example.com/a",
            "https://listed.example.com/b",
            "https://other.example.com/c",
        ],
        timeout=5,
        cache_ttl_seconds=0,
        relevance_query="Nanjing metro lines 2026",
        loader_factory=_loader_factory("<html><body><div></div></body></html>"),
        js_fallback_enabled=True,
        js_fallback_domains=["listed.example.com"],
        js_loader_factory=js_factory,
        js_retry_budget=1,
    )

    assert rendered_calls == ["https://listed.example.com/b"]
    assert [page.fetch_method for page in pages] == [
        "http",
        "js_fallback",
        "http",
    ]


def test_js_retry_is_skipped_when_the_fallback_is_disabled():
    rendered_calls: list[str] = []

    def js_factory(url: str, _page_timeout: int):
        rendered_calls.append(url)
        return _loader_factory(_ARTICLE_HTML)(url, _page_timeout)

    fetch_pages(
        ["https://listed.example.com/b"],
        timeout=5,
        cache_ttl_seconds=0,
        relevance_query="Nanjing metro lines 2026",
        loader_factory=_loader_factory("<html><body><div></div></body></html>"),
        js_fallback_enabled=False,
        js_fallback_domains=["listed.example.com"],
        js_loader_factory=js_factory,
    )

    assert rendered_calls == []
