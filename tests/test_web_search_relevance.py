from __future__ import annotations

from src.backend.web_search.common import (
    DEFAULT_MIN_USABLE_URL_SCORE,
    OWNER_LOOKALIKE_PENALTY,
    SearchResult,
    has_listing_path,
    is_noise_url,
    is_owner_domain_lookalike,
    is_page_text_relevant,
    page_relevance_score,
    prefetch_rejection_reason,
    result_constraint_score,
    result_quality_score,
)


def test_owner_name_lookalike_domain_is_penalized_but_still_eligible():
    query = "DeepSeek V3.2 \u548c V3.1 \u6709\u4ec0\u4e48\u533a\u522b"
    lookalike = SearchResult(
        url="https://deepseek-seek.com.cn/news/v3-2",
        title="DeepSeek V3.2 \u548c V3.1 \u5bf9\u6bd4",
        snippet="DeepSeek V3.2 \u548c V3.1 \u529f\u80fd\u5dee\u5f02",
    )
    neutral = SearchResult(
        url="https://example.com/news/v3-2",
        title=lookalike.title,
        snippet=lookalike.snippet,
    )

    assert prefetch_rejection_reason(lookalike, query) is None
    assert is_owner_domain_lookalike(lookalike.url, query)
    assert not is_owner_domain_lookalike(neutral.url, query)
    assert result_constraint_score(lookalike, query) == (
        result_constraint_score(neutral, query) - OWNER_LOOKALIKE_PENALTY
    )
    assert result_quality_score(lookalike, query=query) >= DEFAULT_MIN_USABLE_URL_SCORE


def test_first_party_host_outside_the_allowlist_stays_usable():
    """``deepseek.net`` and ``xiaomiev.com`` are real sources, not lookalikes."""

    cases = [
        ("DeepSeek V3 open source weights", "https://deepseek-ai.github.io/DeepSeek-V3/"),
        ("xiaomi SU7 range", "https://www.xiaomiev.com/su7/spec"),
    ]
    for query, url in cases:
        result = SearchResult(
            url=url,
            title=query,
            snippet=f"{query} official details",
        )

        assert prefetch_rejection_reason(result, query) is None
        assert result_quality_score(result, query=query) >= DEFAULT_MIN_USABLE_URL_SCORE


def test_pdf_and_document_paths_remain_usable_sources():
    query = "NVIDIA Blackwell architecture whitepaper"
    whitepaper = SearchResult(
        url="https://resources.nvidia.com/blackwell/architecture-brief.pdf",
        title="NVIDIA Blackwell Architecture Technical Brief",
        snippet="NVIDIA Blackwell GPU architecture whitepaper.",
    )
    download_path = SearchResult(
        url="https://www.njmetro.com.cn/download/xianlutu.html",
        title="\u5357\u4eac\u5730\u94c1\u7ebf\u8def\u56fe",
        snippet="\u5357\u4eac\u5730\u94c1\u6700\u65b0\u7ebf\u8def\u56fe",
    )

    assert not is_noise_url(whitepaper.url)
    assert not is_noise_url(download_path.url)
    assert result_quality_score(whitepaper, query=query) >= DEFAULT_MIN_USABLE_URL_SCORE


def test_listing_paths_are_demoted_instead_of_rejected():
    query = "LangGraph release notes"
    listing = SearchResult(
        url="https://changelog.langchain.com/category/langgraph",
        title="LangGraph changelog",
        snippet="Latest LangGraph release notes.",
    )
    article = SearchResult(
        url="https://changelog.langchain.com/announcements/langgraph",
        title=listing.title,
        snippet=listing.snippet,
    )

    assert not is_noise_url(listing.url)
    assert has_listing_path("/category/langgraph")
    assert result_quality_score(listing, query=query) < result_quality_score(
        article,
        query=query,
    )


def test_auth_and_search_paths_are_still_hard_rejected():
    assert is_noise_url("https://example.com/login/next")
    assert is_noise_url("https://example.com/account/settings")
    assert is_noise_url("https://example.com/search?q=rag")
    assert is_noise_url("https://example.com/assets/logo.png")


def test_prefetch_gate_enforces_site_identifier_and_quoted_title_constraints():
    policy_query = '"\u5357\u4eac\u5e02\u4f4f\u623f\u79df\u8d41\u7ba1\u7406\u529e\u6cd5" \u751f\u6548\u65e5\u671f site:gov.cn'
    wrong_host = SearchResult(
        url="https://example.com/policy",
        title="\u5357\u4eac\u5e02\u4f4f\u623f\u79df\u8d41\u7ba1\u7406\u529e\u6cd5",
    )
    wrong_title = SearchResult(
        url="https://www.nanjing.gov.cn/policy/other",
        title="\u5357\u4eac\u5e02\u4f4f\u623f\u4fdd\u969c\u7ba1\u7406\u529e\u6cd5",
    )
    wrong_model = SearchResult(
        url="https://www.xiaomi.com/auto",
        title="\u5c0f\u7c73 SU7 \u5b98\u65b9\u4ef7\u683c",
        snippet="\u5c0f\u7c73 SU7 \u5b98\u65b9\u6307\u5bfc\u4ef7",
    )

    assert prefetch_rejection_reason(wrong_host, policy_query) == "site_constraint_mismatch"
    assert prefetch_rejection_reason(wrong_title, policy_query) == "missing_quoted_title"
    assert (
        prefetch_rejection_reason(
            wrong_model,
            "\u5c0f\u7c73 SU7 Ultra \u5b98\u65b9\u552e\u4ef7",
        )
        == "missing_identifier"
    )


def test_official_owner_result_outranks_matching_secondary_source():
    query = "DeepSeek V3.2 \u5b98\u65b9 \u529f\u80fd"
    official = SearchResult(
        url="https://api-docs.deepseek.com/news/v3-2",
        title="DeepSeek V3.2 \u5b98\u65b9\u529f\u80fd",
        snippet="DeepSeek V3.2 \u6a21\u578b\u80fd\u529b\u4ecb\u7ecd",
    )
    secondary = SearchResult(
        url="https://example.com/news/deepseek-v3-2",
        title=official.title,
        snippet=official.snippet,
    )

    assert result_quality_score(official, query=query) > result_quality_score(
        secondary,
        query=query,
    )


def test_noise_gate_rejects_unresolved_redirects_and_doorway_scripts():
    assert is_noise_url("https://www.baidu.com/link?url=unresolved-token")
    assert is_noise_url("https://example.com/tools/repack.php?id=123")
    assert is_noise_url("https://example.com/redirect.php?target=article")
    assert not is_noise_url("https://example.com/news/2026/nanjing-metro")


def test_result_score_prefers_requested_year_and_penalizes_conflicting_year():
    query = "南京地铁线路数量 2025 2026 几条线"
    matching = SearchResult(
        url="https://example.com/news/nanjing-metro",
        title="南京地铁线路数量",
        snippet="截至2026年，南京地铁共运营14条线路。",
    )
    missing = SearchResult(
        url="https://example.com/news/nanjing-metro",
        title="南京地铁线路数量",
        snippet="南京地铁目前共运营14条线路。",
    )
    conflicting = SearchResult(
        url="https://example.com/news/nanjing-metro",
        title="南京地铁线路数量",
        snippet="截至2024年，南京地铁共运营13条线路。",
    )

    assert result_quality_score(matching, query=query) > result_quality_score(missing, query=query)
    assert result_quality_score(missing, query=query) > result_quality_score(
        conflicting, query=query
    )


def test_result_score_rewards_a_concrete_count_answer():
    query = "南京地铁有几条线路"
    with_answer = SearchResult(
        url="https://example.com/news/nanjing-metro",
        title="南京地铁运营线路",
        snippet="南京地铁目前共运营14条线路。",
    )
    topic_only = SearchResult(
        url="https://example.com/news/nanjing-metro",
        title="南京地铁运营线路",
        snippet="南京地铁线路建设与运营情况介绍。",
    )

    assert result_quality_score(with_answer, query=query) > result_quality_score(
        topic_only, query=query
    )


def test_non_quantity_chinese_word_does_not_trigger_count_requirement():
    assert is_page_text_relevant("欧几里得几何基础介绍。", "欧几里得几何")


def test_year_specific_count_page_requires_year_and_quantity_evidence():
    query = "南京地铁线路数量 2025 2026 几条线"

    assert is_page_text_relevant(
        "截至2026年，南京地铁共运营14条线路。",
        query,
        title="南京地铁线路数量",
    )
    assert not is_page_text_relevant(
        "南京地铁共运营14条线路。",
        query,
        title="南京地铁线路数量",
    )
    assert (
        page_relevance_score(
            "南京地铁共运营14条线路。",
            query,
            title="南京地铁线路数量",
        )
        < 0
    )
    assert not is_page_text_relevant(
        "2026年南京地铁线路建设与运营情况介绍。",
        query,
        title="南京地铁线路数量",
    )
    assert not is_page_text_relevant(
        "截至2024年，南京地铁共运营13条线路。",
        query,
        title="南京地铁线路数量",
    )
    assert not is_page_text_relevant(
        "2025年南京计划开通运营3条地铁线路，分别为5号线等。",
        query,
        title="南京今年将开通3条地铁新线",
    )
    assert not is_page_text_relevant(
        "2025年目前南京在建地铁有11条，预计今明两年多条线路将开通。",
        query,
        title="南京地铁建设进展",
    )
    assert not is_page_text_relevant(
        "2026年南京地铁线路交通指南。" + "景点介绍。" * 80 + "本文共推荐10个景点。",
        query,
        title="来南䫅必玩景点推荐",
    )


def test_metro_count_rejects_nearby_city_total_as_the_line_answer():
    query = "南京地铁线路数量 2025 2026 几条线"
    text = (
        "2025城市轨道交通运营数据发布，南京变化较大。2026年1月22日发布。"
        "截至2025年12月31日，31个省共有54个城市开通运营城市轨道交通。"
        "南京轨交运营里程增加约41.1公里，城市排名升至第七位。"
    )

    assert not is_page_text_relevant(
        text,
        query,
        title="2025城市轨道交通运营数据：南京变化最大",
    )


def test_page_score_prioritizes_h1_and_lead_over_a_buried_match():
    query = "南京地铁线路数量 2026 几条线"
    off_topic_lead = "本页介绍旅游、美食、酒店和城市活动。" * 100
    buried_match = off_topic_lead + " 2026年南京地铁共运营14条线路。"
    focused_page = "2026年南京地铁共运营14条线路。" + off_topic_lead

    assert page_relevance_score(
        focused_page,
        query,
        title="南京地铁线路数量",
    ) > page_relevance_score(
        buried_match,
        query,
        title="城市生活资讯",
    )
    assert not is_page_text_relevant(
        buried_match,
        query,
        title="城市生活资讯",
    )


def test_explicit_h1_contributes_to_primary_page_relevance():
    query = "南京地铁线路数量 2026 几条线"
    generic_lead = "城市交通年度信息。" * 100

    without_h1 = page_relevance_score(generic_lead, query, title="年度信息")
    with_h1 = page_relevance_score(
        generic_lead,
        query,
        title="年度信息",
        h1="2026年南京地铁共运营14条线路",
    )

    assert with_h1 > without_h1
