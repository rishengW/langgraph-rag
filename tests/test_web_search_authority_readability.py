from __future__ import annotations

from langchain_core.documents import Document

from src.backend.web_search import content_fetcher
from src.backend.web_search.common import (
    SearchResult,
    host_authority_class,
    host_quality_score,
    registrable_domain,
    result_quality_score,
)
from src.backend.web_search.content_fetcher import fetch_pages, is_readable_page


def test_host_authority_is_conservative_and_suffix_safe():
    assert host_authority_class("https://jtj.nanjing.gov.cn/gzdt/2026") == "government"
    assert host_authority_class("news.pku.edu.cn") == "education"
    assert host_authority_class("https://api-docs.deepseek.com/zh-cn/") == "recognized_owner"
    assert host_authority_class("https://gov.cn.example.test/article") == "standard"
    assert host_authority_class("https://example.edu/article") == "standard"


def test_host_quality_uses_bonuses_and_soft_syndication_penalties():
    assert host_quality_score("https://www.gov.cn/zhengce/content/2026") > 0
    assert host_quality_score("https://www.nju.edu.cn/info/1001/1.htm") > 0
    assert host_quality_score("https://www.deepseek.com/news/launch") > 0
    assert host_quality_score("https://example.com/article") == 0
    assert host_quality_score("https://news.sohu.com/a/123") < 0
    assert host_quality_score("https://baijiahao.baidu.com/s?id=123") < 0


def test_result_quality_prefers_primary_sources_without_blocking_syndicated_hosts():
    query = "2026年南京公共交通年度报告"
    relevance_text = "2026年南京公共交通年度报告发布"
    official = SearchResult(
        url="https://jtj.nanjing.gov.cn/report/2026.html",
        title=relevance_text,
    )
    neutral = SearchResult(
        url="https://example.com/report/2026.html",
        title=relevance_text,
    )
    syndicated = SearchResult(
        url="https://news.sohu.com/report/2026.html",
        title=relevance_text,
    )

    official_score = result_quality_score(official, query=query)
    neutral_score = result_quality_score(neutral, query=query)
    syndicated_score = result_quality_score(syndicated, query=query)

    assert official_score > neutral_score > syndicated_score
    assert syndicated_score > 0


def test_registrable_domain_groups_subdomains_without_merging_all_gov_sites():
    assert registrable_domain("https://jtj.nanjing.gov.cn/a") == "nanjing.gov.cn"
    assert registrable_domain("https://www.nanjing.gov.cn/b") == "nanjing.gov.cn"
    assert registrable_domain("https://news.sohu.com/a") == "sohu.com"
    assert registrable_domain("https://docs.example.co.uk/a") == "example.co.uk"
    assert registrable_domain("http://127.0.0.1:8000/a") == "127.0.0.1"


def test_adaptive_readability_accepts_concise_official_answer_with_evidence():
    query = "南京地铁线路数量 2026 几条线"
    text = "截至2026年，南京地铁共有14条运营线路。该数据由南京市交通部门发布并定期更新。"

    assert len(text) < 200
    assert is_readable_page(
        text,
        url="https://jtj.nanjing.gov.cn/gzdt/2026/notice.html",
        title="2026年南京地铁运营线路总数",
        query=query,
    )


def test_adaptive_readability_keeps_year_and_quantity_requirements_strict():
    query = "南京地铁线路数量 2026 几条线"

    assert not is_readable_page(
        "南京地铁共有14条运营线路。相关数据由交通部门发布并负责更新。",
        url="https://jtj.nanjing.gov.cn/notice.html",
        title="南京地铁运营线路总数",
        query=query,
    )
    assert not is_readable_page(
        "2026年南京地铁建设和运营工作有序推进。相关信息由交通部门发布。",
        url="https://jtj.nanjing.gov.cn/notice.html",
        title="2026年南京地铁工作进展",
        query=query,
    )


def test_adaptive_readability_rejects_short_syndicated_and_off_topic_pages():
    query = "南京地铁线路数量 2026 几条线"
    relevant = "截至2026年，南京地铁共有14条运营线路。本文汇总城市交通数据并持续更新。"

    assert not is_readable_page(
        relevant,
        url="https://news.sohu.com/a/123",
        title="2026年南京地铁运营线路总数",
        query=query,
    )
    assert not is_readable_page(
        "2026年南京市文旅活动共有14项。本文介绍景点、美食和酒店信息。",
        url="https://www.nanjing.gov.cn/wl/notice.html",
        title="南京文旅活动",
        query=query,
    )


def test_fetch_pages_retains_short_official_page_when_query_is_supplied(monkeypatch):
    text = "截至2026年，南京地铁共有14条运营线路。该数据由南京市交通部门发布并定期更新。"

    def fake_load_source_documents(_urls, **_kwargs):
        return [
            Document(
                page_content=f"<main>{text}</main>",
                metadata={
                    "source": "https://jtj.nanjing.gov.cn/notice.html",
                    "title": "2026年南京地铁运营线路总数",
                },
            )
        ]

    monkeypatch.setattr(
        content_fetcher,
        "load_source_documents",
        fake_load_source_documents,
    )

    page = fetch_pages(
        ["https://jtj.nanjing.gov.cn/notice.html"],
        relevance_query="南京地铁线路数量 2026 几条线",
    )[0]

    assert page.text == text
    assert page.error is None


def test_fetch_pages_retries_unlisted_authoritative_host_with_js(monkeypatch):
    url = "https://agency.example.gov.cn/policy/notice.html"
    calls: list[str] = []

    class FakeLoader:
        def __init__(self, method: str, text: str) -> None:
            self.method = method
            self.text = text

        def load(self):
            calls.append(self.method)
            return [
                Document(
                    page_content=f"<main>{self.text}</main>",
                    metadata={"source": url, "title": "新规定生效日期"},
                )
            ]

    def fake_load_source_documents(_urls, **kwargs):
        return kwargs["loader_factory"](url, 5).load()

    monkeypatch.setattr(
        content_fetcher,
        "load_source_documents",
        fake_load_source_documents,
    )

    page = fetch_pages(
        [url],
        min_readable_chars=100,
        min_readable_tokens=20,
        relevance_query="新规定什么时候生效",
        loader_factory=lambda _url, _timeout: FakeLoader("http", "loading"),
        js_fallback_enabled=True,
        js_fallback_domains=[],
        js_loader_factory=lambda _url, _timeout: FakeLoader(
            "js",
            "新规定自2026年8月1日起正式生效。" * 10,
        ),
    )[0]

    assert calls == ["http", "js"]
    assert "2026年8月1日" in page.text
    assert page.fetch_method == "js_fallback"
