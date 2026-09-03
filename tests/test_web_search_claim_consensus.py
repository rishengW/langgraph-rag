from __future__ import annotations

from types import SimpleNamespace

from src.backend.web_search.claim_consensus import assess_status_consensus


def _page(url: str, text: str):
    return SimpleNamespace(url=url, title="Status update", text=text)


def test_status_consensus_accepts_responsible_official_source():
    result = assess_status_consensus(
        [
            _page("https://jtj.example.gov.cn/news/1", "该线路现已正式开通运营。"),
            _page("https://news.example.com/old", "此前预计年底开通。"),
        ],
        "这条线路现在开通了吗？",
    )

    assert result.state == "supported"
    assert result.categorical_allowed is True


def test_status_consensus_requires_two_independent_secondary_domains():
    supported = assess_status_consensus(
        [
            _page("https://news-a.example/a", "产品现已正式上线。"),
            _page("https://news-b.test/b", "产品已经发布并正式上线。"),
        ],
        "这个产品现在上线了吗？",
    )
    insufficient = assess_status_consensus(
        [_page("https://news-a.example/a", "产品现已正式上线。")],
        "这个产品现在上线了吗？",
    )

    assert supported.state == "supported"
    assert supported.categorical_allowed is True
    assert insufficient.state == "insufficient"
    assert insufficient.categorical_allowed is False
    assert "unverified" in insufficient.prompt_instruction


def test_status_consensus_surfaces_conflicting_secondary_claims():
    result = assess_status_consensus(
        [
            _page("https://news-a.example/a", "该线路已经正式开通。"),
            _page("https://news-b.test/b", "该线路尚未开通，预计年底运营。"),
        ],
        "该线路目前开通了吗？",
    )

    assert result.state == "conflicting"
    assert result.categorical_allowed is False
    assert "conflict" in result.prompt_instruction
