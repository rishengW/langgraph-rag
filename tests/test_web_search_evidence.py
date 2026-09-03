from __future__ import annotations

from src.backend.web_search.common import is_page_text_relevant, text_relevance_delta
from src.backend.web_search.evidence import (
    answer_evidence_delta,
    detect_query_intents,
    has_required_answer_evidence,
)


def test_detects_mandarin_answer_intents():
    assert detect_query_intents("南京地铁票价多少钱") == {"price"}
    assert detect_query_intents("新政策什么时候生效") == {"date", "policy"}
    assert detect_query_intents("比较这两个方案目前的区别") == {
        "comparison",
        "status",
    }


def test_date_question_requires_concrete_date_evidence():
    query = "新规定什么时候生效"

    assert has_required_answer_evidence("该规定自2026年8月1日起施行。", query)
    assert not has_required_answer_evidence("该规定将在近期施行。", query)
    assert answer_evidence_delta("2026年8月1日起施行", query) > 0
    assert answer_evidence_delta("近期施行", query) < 0


def test_price_question_requires_currency_amount():
    query = "南京地铁票价多少钱"

    assert has_required_answer_evidence("南京地铁起步票价为2元。", query)
    assert has_required_answer_evidence("南京地铁的 fare is CNY 2.", query)
    assert not has_required_answer_evidence("某景区起步票价为2元。", query)
    assert not has_required_answer_evidence("票价按里程计算。", query)


def test_price_evidence_must_be_near_all_product_identifiers():
    query = "小米 SU7 Ultra 官方售价是多少？"
    unrelated_price = (
        "小米 SU7 Ultra 是一款高性能车型。" + "产品配置介绍。" * 80 + "小米手环售价为499元。"
    )

    assert not has_required_answer_evidence(unrelated_price, query)
    assert has_required_answer_evidence("小米 SU7 Ultra 官方售价为52.99万元。", query)


def test_quoted_policy_title_must_exist_on_date_evidence_page():
    query = "《南京市住房租赁管理办法》什么时候生效？"

    assert has_required_answer_evidence(
        "《南京市住房租赁管理办法》自2022年5月1日起施行。",
        query,
    )
    assert not has_required_answer_evidence(
        "《南京市发展保障性租赁住房实施办法》自2022年2月10日起施行。",
        query,
    )


def test_policy_and_comparison_evidence_are_soft_ranking_signals():
    policy_query = "比较两个住房政策的区别"

    assert has_required_answer_evidence("政策介绍", policy_query)
    assert answer_evidence_delta(
        "该通知自2026年施行，两项政策相比有三点区别。",
        policy_query,
    ) > answer_evidence_delta("住房市场概览", policy_query)


def test_page_relevance_requires_date_evidence_for_mandarin_date_question():
    query = "新住房政策什么时候生效？"

    assert is_page_text_relevant(
        "南京市发布新住房政策，该政策自2026年8月1日起正式生效。",
        query,
        title="南京新住房政策通知",
    )
    assert not is_page_text_relevant(
        "南京市发布新住房政策，具体生效日期请关注后续通知。",
        query,
        title="南京新住房政策解读",
    )


def test_price_evidence_improves_result_text_relevance():
    query = "小米 SU7 Ultra 2026 款价格是多少？"
    with_price = "小米 SU7 Ultra 2026 款官方售价为52.99万元。"
    without_price = "小米 SU7 Ultra 2026 款车型配置与外观介绍。"

    assert text_relevance_delta(with_price, query) > text_relevance_delta(
        without_price,
        query,
    )
    assert is_page_text_relevant(with_price, query, title="小米 SU7 Ultra 2026 款")
    assert not is_page_text_relevant(
        without_price,
        query,
        title="小米 SU7 Ultra 2026 款",
    )
