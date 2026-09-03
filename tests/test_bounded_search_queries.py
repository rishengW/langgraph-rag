from __future__ import annotations

from threading import Lock

from langchain_core.messages import AIMessage, HumanMessage

from src.backend.graph.builder import GraphNodeOverrides, GraphProviders, build_lightweight_graph
from src.backend.graph.nodes.search_queries import WEB_SEARCH_MAX_QUERIES, search_queries_factory
from src.backend.web_search.protocol import RankedSearchResult
from src.backend.web_search.tool import build_web_search_tool


def _search_call(query: str = "original") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "live_web_search",
                "args": {"query": query},
                "id": "call_live_web_search",
            }
        ],
    )


def test_search_queries_node_dedupes_and_clamps_batch(isolated_settings):
    calls: list[str] = []
    calls_lock = Lock()

    def discover(query, _settings, _provider):
        with calls_lock:
            calls.append(query)
        slug = query.lower().replace(" ", "-")
        return [f"https://results.test/{slug}"]

    tool = build_web_search_tool(isolated_settings(), discovery=discover)
    node = search_queries_factory([tool])
    queries = ["Q1", "Q2", "Q1", "Q3", "Q4", "Q5", "Q6", "Q7"]

    result = node({"messages": [_search_call()], "search_queries": queries})

    assert set(calls) == {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}
    assert len(calls) == WEB_SEARCH_MAX_QUERIES
    assert len(result["web_search_results"]) == WEB_SEARCH_MAX_QUERIES
    assert len(result["web_search_result_metadata"]) == WEB_SEARCH_MAX_QUERIES
    assert result["messages"][0].tool_call_id == "call_live_web_search"


def test_search_queries_preserves_tool_result_metadata(isolated_settings):
    def discover(_query, _settings, _provider):
        return [
            RankedSearchResult(
                url="https://results.test/relevant",
                title="Nanjing metro lines",
                snippet="Nanjing operates 14 metro lines in 2026",
                provider="baidu",
                provider_rank=2,
                relevance_score=41,
                quality_score=106,
            )
        ]

    tool = build_web_search_tool(isolated_settings(), discovery=discover)
    node = search_queries_factory([tool])

    result = node({"messages": [_search_call("南京地铁线路数量 2026")]})

    assert len(result["search_queries"]) == 2
    assert all("南京地铁线路数量 2026" in query for query in result["search_queries"])
    assert result["web_search_results"] == [
        ["https://results.test/relevant"],
        ["https://results.test/relevant"],
    ]
    assert result["web_search_result_metadata"] == [
        [
            {
                "url": "https://results.test/relevant",
                "title": "Nanjing metro lines",
                "snippet": "Nanjing operates 14 metro lines in 2026",
                "provider": "baidu",
                "provider_rank": 2,
                "relevance_score": 41,
                "quality_score": 106,
            }
        ],
        [
            {
                "url": "https://results.test/relevant",
                "title": "Nanjing metro lines",
                "snippet": "Nanjing operates 14 metro lines in 2026",
                "provider": "baidu",
                "provider_rank": 2,
                "relevance_score": 41,
                "quality_score": 106,
            }
        ],
    ]


def test_search_queries_plans_mandarin_variants_within_global_limit(
    isolated_settings,
):
    calls: list[str] = []

    def discover(query, _settings, _provider):
        calls.append(query)
        return [f"https://results.test/{len(calls)}"]

    tool = build_web_search_tool(isolated_settings(), discovery=discover)
    node = search_queries_factory([tool])
    raw_queries = [
        "南京地铁线路数量 2026 几条线",
        "小米 SU7 Ultra 2026 款价格是多少",
        "新住房政策什么时候生效",
        "比较两个方案目前的区别",
    ]

    result = node({"messages": [_search_call()], "search_queries": raw_queries})

    expected = [
        "南京地铁线路数量 2026 几条线 运营线路总数",
        "南京地铁线路数量 2026 几条线 运营线路总数 官方 数据",
        "小米 SU7 Ultra 2026 款价格是多少",
        "小米 SU7 Ultra 2026 款价格是多少 官方",
        "新住房政策什么时候生效 发布时间",
        "新住房政策什么时候生效 发布时间 官方 原文 site:gov.cn",
    ]
    assert len(calls) == WEB_SEARCH_MAX_QUERIES
    assert result["search_queries"] == expected
    assert set(calls) == set(expected)


def test_lightweight_graph_searches_each_decomposed_query(isolated_settings):
    calls: list[str] = []
    calls_lock = Lock()
    sub_questions = ["Alpha policy", "Beta policy", "Gamma policy"]

    def discover(query, _settings, _provider):
        with calls_lock:
            calls.append(query)
        slug = query.lower().replace(" ", "-")
        return [
            "https://results.test/shared",
            f"https://results.test/{slug}",
        ]

    tool = build_web_search_tool(isolated_settings(), discovery=discover)

    def agent(_state):
        return {"messages": [_search_call("unused original agent query")]}

    def decompose(_state):
        return {"sub_questions": list(sub_questions)}

    def web_answer(state):
        assert state["source_urls"][0] == "https://results.test/shared"
        for query in sub_questions:
            slug = query.lower().replace(" ", "-")
            assert f"https://results.test/{slug}" in state["source_urls"]
        return {
            "messages": [AIMessage(content="bounded fan-out answer")],
            "web_answer_no_readable_content": False,
        }

    graph = build_lightweight_graph(
        settings=isolated_settings(web_search_top_k=6),
        providers=GraphProviders(
            tools=[tool],
            nodes=GraphNodeOverrides(
                agent=agent,
                decompose=decompose,
                web_answer=web_answer,
            ),
        ),
    )

    state = graph.invoke({"messages": [HumanMessage(content="Compare all three policies")]})

    assert set(calls) == {"Compare all three policies", *sub_questions}
    assert "unused original agent query" not in calls
    assert state["messages"][-1].content == "bounded fan-out answer"


def test_lightweight_graph_clamps_expanded_search_retry(isolated_settings):
    calls: list[str] = []
    calls_lock = Lock()
    expanded = [f"Expanded {index}" for index in range(1, 8)]
    answer_calls = 0

    def discover(query, _settings, _provider):
        with calls_lock:
            calls.append(query)
        slug = query.lower().replace(" ", "-")
        return [f"https://results.test/{slug}"]

    tool = build_web_search_tool(isolated_settings(), discovery=discover)

    def web_answer(_state):
        nonlocal answer_calls
        answer_calls += 1
        if answer_calls == 1:
            return {
                "messages": [AIMessage(content="No readable content")],
                "web_answer_no_readable_content": True,
                "web_answer_attempts": 1,
            }
        return {
            "messages": [AIMessage(content="expanded answer")],
            "web_answer_no_readable_content": False,
            "web_answer_attempts": 2,
        }

    graph = build_lightweight_graph(
        settings=isolated_settings(web_search_top_k=20),
        providers=GraphProviders(
            tools=[tool],
            nodes=GraphNodeOverrides(
                agent=lambda _state: {"messages": [_search_call()]},
                decompose=lambda _state: {"sub_questions": ["Initial query"]},
                expand=lambda _state: {
                    "expanded_queries": list(expanded),
                    "expansion_attempted": True,
                },
                web_answer=web_answer,
            ),
        ),
    )

    state = graph.invoke({"messages": [HumanMessage(content="A difficult comparison")]})

    assert set(calls) == {
        "A difficult comparison",
        "Initial query",
        *expanded[: WEB_SEARCH_MAX_QUERIES - 1],
    }
    assert expanded[WEB_SEARCH_MAX_QUERIES - 1] not in calls
    assert expanded[-1] not in calls
    assert answer_calls == 2
    assert sum(message.type == "tool" for message in state["messages"]) == 1
    assert state["messages"][-1].content == "expanded answer"


def test_search_queries_keeps_original_mandarin_and_rejects_english_translation(
    isolated_settings,
):
    calls: list[str] = []

    def discover(query, _settings, _provider):
        calls.append(query)
        return ["https://results.test/relevant"]

    node = search_queries_factory([build_web_search_tool(isolated_settings(), discovery=discover)])
    original = "DeepSeek V3.2 和 V3.1 有什么区别？"

    result = node(
        {
            "messages": [HumanMessage(content=original), _search_call()],
            "current_question": original,
            "search_queries": [
                "What are the key features of DeepSeek V3.2?",
                "What are the key features of DeepSeek V3.1?",
            ],
        }
    )

    assert result["search_queries"] == [
        "DeepSeek V3.2 和 V3.1 有什么区别 对比",
        "DeepSeek V3.2 和 V3.1 有什么区别 对比 官方",
    ]
    assert set(calls) == set(result["search_queries"])
