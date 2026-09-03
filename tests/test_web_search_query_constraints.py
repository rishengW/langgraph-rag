from __future__ import annotations

from src.backend.web_search.query_constraints import (
    COMPARISON_INTENT,
    PRICE_INTENT,
    extract_query_constraints,
    validate_query_candidate,
)
from src.backend.web_search.query_prep import plan_search_queries, prepare_search_query


def test_extracts_language_entities_identifiers_years_and_intent():
    constraints = extract_query_constraints("小米 SU7 Ultra 2026 款价格是多少？")

    assert constraints.language == "mixed"
    assert {"su7", "ultra"} <= set(constraints.identifiers)
    assert "小米" in constraints.entities
    assert constraints.years == {"2026"}
    assert PRICE_INTENT in constraints.intents


def test_rewrite_validation_rejects_translation_and_invented_years():
    comparison = "DeepSeek V3.2 和 V3.1 有什么区别？"

    assert not validate_query_candidate(
        comparison,
        "What are the differences between DeepSeek V3.2 and V3.1?",
        allow_partial=True,
    )
    assert validate_query_candidate(
        comparison,
        "DeepSeek V3.2 有什么特点？",
        allow_partial=True,
    )
    assert not validate_query_candidate(
        "小米 SU7 Ultra 官方售价是多少？",
        "小米 SU7 Ultra 2026 官方售价",
        allow_partial=True,
    )


def test_policy_query_uses_exact_title_and_generic_government_site_filter():
    queries = plan_search_queries("《南京市住房租赁管理办法》什么时候生效？")

    assert len(queries) == 2
    assert '"南京市住房租赁管理办法"' in queries[1]
    assert "site:gov.cn" in queries[1]
    assert "生效日期" in queries[1]
    assert COMPARISON_INTENT not in extract_query_constraints(queries[1]).intents
    assert prepare_search_query(queries[1]) == queries[1]
