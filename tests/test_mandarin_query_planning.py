from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from src.backend.graph.nodes import decompose as decompose_module
from src.backend.web_search.query_prep import plan_search_queries, prepare_search_query


def test_prepare_mandarin_query_removes_wrappers_and_adds_intent_hint():
    result = prepare_search_query("请帮我查一下 DeepSeek V4 是什么时候发布的？")

    assert result == "DeepSeek V4 是什么时候发布的 发布时间"


def test_plan_mandarin_query_has_exact_and_official_source_variants():
    result = plan_search_queries("请问小米 SU7 Ultra 2026 款价格是多少？")

    assert result == [
        "小米 SU7 Ultra 2026 款价格是多少",
        "小米 SU7 Ultra 2026 款价格是多少 官方",
    ]
    assert all("小米 SU7 Ultra 2026" in query for query in result)


def test_plan_mandarin_query_splits_exactly_two_requested_years():
    result = plan_search_queries("南京地铁线路数量 2025、2026 年分别有几条线？")

    assert len(result) == 2
    assert "2025" in result[0] and "2026" not in result[0]
    assert "2026" in result[1] and "2025" not in result[1]
    assert all("南京地铁" in query for query in result)
    assert all("运营线路总数" in query for query in result)
    assert all(query.endswith("官方 数据") for query in result)


def test_plan_search_queries_leaves_non_mandarin_input_unchanged():
    question = "what is DeepSeek V4 pricing in 2026?"

    assert plan_search_queries(question) == [question]


def test_plan_search_queries_is_bounded_to_two_variants():
    result = plan_search_queries("2024、2025、2026 年中国人口数量分别是多少？")

    assert len(result) <= 2
    assert all(year in result[0] for year in ("2024", "2025", "2026"))


@pytest.mark.parametrize(
    "question",
    [
        "比较阿里云与腾讯云的区别",
        "谁创立了阿里巴巴，以及什么时候成立？",
        "请介绍甲方并总结乙方路线图",
        "何时发布？在哪里宣布？",
        "Alpha 是什么；Beta 有什么特点？",
    ],
)
def test_decompose_detects_mandarin_compound_questions(question, isolated_settings, monkeypatch):
    expected = [question, f"{question} 补充"]
    fake_result = type("R", (), {"sub_questions": expected})()
    calls = []
    monkeypatch.setattr(
        decompose_module,
        "new_structured_chat_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        decompose_module,
        "invoke_with_retry",
        lambda *args, **kwargs: calls.append((args, kwargs)) or fake_result,
    )

    result = decompose_module.decompose_factory(isolated_settings())(
        {"messages": [HumanMessage(content=question)]}
    )

    assert result["sub_questions"] == expected
    assert len(calls) == 1


@pytest.mark.parametrize(
    "question",
    [
        "北京和上海之间的高铁时刻表",
        "研究与开发政策",
        "请介绍阿里巴巴",
    ],
)
def test_decompose_keeps_atomic_mandarin_connectors_on_fast_path(
    question, isolated_settings, monkeypatch
):
    monkeypatch.setattr(
        decompose_module,
        "new_structured_chat_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("atomic Mandarin query must not call the LLM")
        ),
    )

    result = decompose_module.decompose_factory(isolated_settings())(
        {"messages": [HumanMessage(content=question)]}
    )

    assert result["sub_questions"] == [question]
