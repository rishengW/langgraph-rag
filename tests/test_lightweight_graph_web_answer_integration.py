from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.backend.graph import nodes as graph_nodes
from src.backend.graph.builder import (
    GraphNodeOverrides,
    GraphProviders,
    build_lightweight_graph,
    build_memory_saver,
)
from src.backend.graph.nodes import web_answer as web_answer_module
from src.backend.web_search.content_fetcher import is_readable_page, is_readable_text


def _readable_text() -> str:
    return " ".join(["source detail"] * 30)


def _install_lightweight_web_modules(monkeypatch, fetch_pages, build_prompt) -> None:
    content_fetcher = ModuleType("src.backend.web_search.content_fetcher")
    content_fetcher.FetchedPage = SimpleNamespace
    content_fetcher.fetch_pages = fetch_pages
    content_fetcher.is_readable_page = is_readable_page
    content_fetcher.is_readable_text = is_readable_text

    prompt_builder = ModuleType("src.backend.web_search.prompt_builder")
    prompt_builder.build_web_search_prompt = build_prompt

    monkeypatch.setitem(sys.modules, "src.backend.web_search.content_fetcher", content_fetcher)
    monkeypatch.setitem(sys.modules, "src.backend.web_search.prompt_builder", prompt_builder)


def test_web_answer_fetches_state_urls_and_invokes_llm(monkeypatch, isolated_settings):
    settings = isolated_settings(
        source_urls=["https://fallback.test"],
        page_load_timeout=7,
        page_load_cache_ttl_seconds=11,
        web_search_max_page_tokens=321,
    )
    calls: dict[str, object] = {}
    pages = [SimpleNamespace(url="https://a.test/page", title="A", text=_readable_text())]

    def fake_fetch_pages(urls, **kwargs):
        calls["fetch"] = (list(urls), kwargs)
        return pages

    def fake_build_prompt(question, resolved_pages):
        calls["prompt"] = (question, resolved_pages)
        return "assembled prompt"

    def fake_invoke_with_retry(model, payload, max_retries):
        calls["llm"] = (model, payload, max_retries)
        return AIMessage(content="answer from direct web context")

    _install_lightweight_web_modules(monkeypatch, fake_fetch_pages, fake_build_prompt)
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _settings: "fake-model")
    monkeypatch.setattr(web_answer_module, "invoke_with_retry", fake_invoke_with_retry)

    node = web_answer_module.web_answer_factory(settings)
    result = node(
        {
            "messages": [HumanMessage(content="What changed?")],
            "source_urls": ["https://a.test/page", "https://a.test/page"],
        }
    )

    assert result["messages"][0].content == "answer from direct web context"
    assert result["source_urls"] == ["https://a.test/page"]
    assert calls["fetch"] == (
        ["https://a.test/page"],
        {
            "timeout": 7,
            "max_tokens_per_page": 321,
            "cache_ttl_seconds": 11,
            "max_concurrent_loads": 8,
            "min_readable_chars": 200,
            "min_readable_tokens": 50,
            "relevance_query": "What changed?",
            "js_fallback_enabled": False,
            "js_fallback_domains": [
                "baike.baidu.com",
                "zhuanlan.zhihu.com",
                "apps.microsoft.com",
                "deepseek.net",
            ],
            "js_force_domains": [],
            "js_retry_budget": 2,
            "max_link_density": 0.5,
            "min_content_words": 60,
        },
    )
    assert calls["prompt"] == ("What changed?", pages)
    assert calls["llm"][1][0].content == "assembled prompt"
    assert calls["llm"][2] == 1


def test_web_answer_removes_near_duplicate_syndicated_pages(
    monkeypatch,
    isolated_settings,
):
    settings = isolated_settings(source_urls=[])
    common_body = " ".join(["南京住房政策于2026年8月1日生效，申请条件和办理流程已经公布。"] * 20)
    distinct_body = " ".join(
        ["南京市政府公布配套问答，说明政策自2026年8月1日生效，并解释适用对象。"] * 20
    )
    pages = [
        SimpleNamespace(
            url="https://www.nanjing.gov.cn/policy/original",
            title="南京住房政策生效日期",
            text=common_body,
        ),
        SimpleNamespace(
            url="https://news.example.test/syndicated-copy",
            title="南京住房政策生效日期转载",
            text=common_body,
        ),
        SimpleNamespace(
            url="https://www.nanjing.gov.cn/policy/faq",
            title="南京住房政策配套问答",
            text=distinct_body,
        ),
    ]
    captured: dict[str, object] = {}

    def fake_fetch_pages(_urls, **_kwargs):
        return pages

    def fake_build_prompt(_question, resolved_pages):
        captured["pages"] = resolved_pages
        return "prompt"

    _install_lightweight_web_modules(monkeypatch, fake_fetch_pages, fake_build_prompt)
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _settings: "model")
    monkeypatch.setattr(
        web_answer_module,
        "invoke_with_retry",
        lambda *_args, **_kwargs: AIMessage(content="grounded answer"),
    )

    result = web_answer_module.web_answer_factory(settings)(
        {
            "messages": [HumanMessage(content="南京住房政策什么时候生效？")],
            "source_urls": [page.url for page in pages],
        }
    )

    assert result["source_urls"] == [
        "https://www.nanjing.gov.cn/policy/original",
        "https://www.nanjing.gov.cn/policy/faq",
    ]
    assert [page.url for page in captured["pages"]] == result["source_urls"]


def test_web_answer_requires_evidence_for_each_requested_year(
    monkeypatch,
    isolated_settings,
):
    settings = isolated_settings(source_urls=[])
    pages = [
        SimpleNamespace(
            url="https://jtj.nanjing.gov.cn/2025-count",
            title="2025年南京地铁运营线路总数",
            text="截至2025年，南京地铁共有14条运营线路。" * 15,
        ),
        SimpleNamespace(
            url="https://example.test/2026-progress",
            title="2026年南京地铁建设进展",
            text="2026年南京地铁新线建设按计划推进，相关工程进展顺利。" * 15,
        ),
    ]

    def fake_build_prompt(_question, _pages):  # pragma: no cover - must not run
        raise AssertionError("incomplete year coverage must not reach the prompt")

    _install_lightweight_web_modules(
        monkeypatch,
        lambda _urls, **_kwargs: pages,
        fake_build_prompt,
    )
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _settings: "model")
    monkeypatch.setattr(
        web_answer_module,
        "invoke_with_retry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("incomplete year coverage must not reach the model")
        ),
    )

    result = web_answer_module.web_answer_factory(settings)(
        {
            "messages": [HumanMessage(content="南京地铁线路数量 2025 2026 分别有几条线？")],
            "source_urls": [page.url for page in pages],
        }
    )

    assert result["source_urls"] == []
    assert result["web_answer_no_readable_content"] is True


def test_web_answer_passes_status_conflict_constraint_to_prompt(
    monkeypatch,
    isolated_settings,
):
    settings = isolated_settings(
        source_urls=[],
        web_search_min_page_chars=1,
        web_search_min_page_tokens=1,
    )
    pages = [
        SimpleNamespace(
            url="https://news-a.example/status",
            title="线路状态",
            text="某线路已经正式开通运营。" * 20,
        ),
        SimpleNamespace(
            url="https://news-b.test/status",
            title="线路进展",
            text="某线路尚未开通，预计年底投入运营。" * 20,
        ),
    ]
    captured: dict[str, object] = {}

    def fake_build_prompt(question, resolved_pages, *, grounding_note=""):
        captured["prompt"] = (question, resolved_pages, grounding_note)
        return "status conflict prompt"

    _install_lightweight_web_modules(
        monkeypatch,
        lambda _urls, **_kwargs: pages,
        fake_build_prompt,
    )
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _settings: "model")
    monkeypatch.setattr(
        web_answer_module,
        "invoke_with_retry",
        lambda *_args, **_kwargs: AIMessage(content="The sources conflict."),
    )

    result = web_answer_module.web_answer_factory(settings)(
        {
            "messages": [HumanMessage(content="某线路现在开通了吗？")],
            "source_urls": [page.url for page in pages],
        }
    )

    question, resolved_pages, grounding_note = captured["prompt"]
    assert question == "某线路现在开通了吗？"
    assert resolved_pages == pages
    assert "conflict" in grounding_note
    assert result["source_urls"] == [page.url for page in pages]


def test_web_answer_extracts_urls_from_web_search_tool_messages(
    monkeypatch,
    isolated_settings,
):
    # Tool-message URLs are a fallback used only when the session has no
    # curated source URLs. Curated settings.source_urls intentionally take
    # precedence over the agent's in-graph live_web_search results, so this
    # test exercises the no-curated-URLs case.
    settings = isolated_settings(source_urls=[])
    calls: dict[str, object] = {}

    def fake_fetch_pages(urls, **_kwargs):
        calls["urls"] = list(urls)
        # Return a readable page so the flow proceeds to prompt assembly;
        # this test verifies tool-URL extraction, not the no-content guard.
        return [SimpleNamespace(url=urls[0], title="T", text=_readable_text())]

    def fake_build_prompt(question, _pages):
        calls["question"] = question
        return "prompt with pages"

    _install_lightweight_web_modules(monkeypatch, fake_fetch_pages, fake_build_prompt)
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _settings: "fake-model")
    monkeypatch.setattr(
        web_answer_module,
        "invoke_with_retry",
        lambda *_args, **_kwargs: AIMessage(content="parsed url answer"),
    )

    node = web_answer_module.web_answer_factory(settings)
    node(
        {
            "messages": [
                HumanMessage(content="Question?"),
                ToolMessage(
                    content=(
                        "Live web search results for: Question?\n"
                        "1. https://one.test/a\n"
                        "2. https://two.test/b."
                    ),
                    tool_call_id="call_live_web_search",
                ),
            ],
            "current_question": "Question?",
        }
    )

    assert calls["urls"] == ["https://one.test/a", "https://two.test/b"]
    assert calls["question"] == "Question?"


def test_web_answer_prefers_curated_settings_urls_over_tool_messages(
    monkeypatch,
    isolated_settings,
):
    # Regression guard: the session's curated, provider-ranked source URLs
    # must win over the agent's in-graph live_web_search tool output, which is
    # a narrower re-search that can otherwise send the answer node to chase
    # low-quality/unreachable pages.
    settings = isolated_settings(source_urls=["https://curated.test/a", "https://curated.test/b"])
    calls: dict[str, object] = {}

    def fake_fetch_pages(urls, **_kwargs):
        calls["urls"] = list(urls)
        return [SimpleNamespace(url=urls[0], title="C", text=_readable_text())]

    def fake_build_prompt(question, _pages):
        calls["question"] = question
        return "prompt"

    _install_lightweight_web_modules(monkeypatch, fake_fetch_pages, fake_build_prompt)
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _settings: "fake-model")
    monkeypatch.setattr(
        web_answer_module,
        "invoke_with_retry",
        lambda *_args, **_kwargs: AIMessage(content="curated answer"),
    )

    node = web_answer_module.web_answer_factory(settings)
    node(
        {
            "messages": [
                HumanMessage(content="Question?"),
                ToolMessage(
                    content=("Live web search results for: Question?\n1. https://junk.test/x"),
                    tool_call_id="call_live_web_search",
                ),
            ],
        }
    )

    assert calls["urls"] == ["https://curated.test/a", "https://curated.test/b"]


def test_web_answer_returns_grounded_refusal_when_no_readable_pages(
    monkeypatch,
    isolated_settings,
):
    # Regression guard: when no fetched page yields readable text, the node
    # must NOT call the LLM (which would answer from stale training data) and
    # must return a grounded refusal listing the attempted URLs.
    settings = isolated_settings(source_urls=["https://dead.test/a"])
    llm_called = {"value": False}

    def fake_fetch_pages(urls, **_kwargs):
        return [SimpleNamespace(url=u, title="", text="") for u in urls]

    def fake_build_prompt(question, _pages):  # pragma: no cover - must not run
        raise AssertionError("prompt must not be built when no readable pages")

    def fake_invoke(*_args, **_kwargs):  # pragma: no cover - must not run
        llm_called["value"] = True
        return AIMessage(content="should not happen")

    _install_lightweight_web_modules(monkeypatch, fake_fetch_pages, fake_build_prompt)
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _settings: "fake-model")
    monkeypatch.setattr(web_answer_module, "invoke_with_retry", fake_invoke)

    node = web_answer_module.web_answer_factory(settings)
    result = node({"messages": [HumanMessage(content="What is new?")]})

    assert llm_called["value"] is False
    content = result["messages"][0].content
    assert "couldn't retrieve readable content" in content
    assert "https://dead.test/a" in content
    assert result["source_urls"] == []


def test_web_answer_rejects_trivial_js_shell_text(
    monkeypatch,
    isolated_settings,
):
    # Regression guard for JS-rendered or anti-bot page shells: a few tokens of
    # title/nav/loading text must not be treated as usable source content.
    settings = isolated_settings(source_urls=["https://shell.test/a"])

    def fake_fetch_pages(urls, **_kwargs):
        return [
            SimpleNamespace(
                url=urls[0],
                title="Shell",
                text="DeepSeek loading menu",
            )
        ]

    def fake_build_prompt(question, _pages):  # pragma: no cover - must not run
        raise AssertionError("prompt must not be built for trivial page text")

    def fake_invoke(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("LLM must not be called for trivial page text")

    _install_lightweight_web_modules(monkeypatch, fake_fetch_pages, fake_build_prompt)
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _settings: "fake-model")
    monkeypatch.setattr(web_answer_module, "invoke_with_retry", fake_invoke)

    node = web_answer_module.web_answer_factory(settings)
    result = node({"messages": [HumanMessage(content="What is new?")]})

    content = result["messages"][0].content
    assert "couldn't retrieve readable content" in content
    assert "https://shell.test/a" in content


def test_web_answer_filters_readable_pages_by_chinese_query_relevance(
    monkeypatch,
    isolated_settings,
):
    settings = isolated_settings(
        source_urls=["https://irrelevant.test/a", "https://relevant.test/b"],
        web_search_min_page_chars=1,
        web_search_min_page_tokens=1,
    )
    pages = [
        SimpleNamespace(
            url="https://irrelevant.test/a",
            title="Slack product updates",
            text="Slack channels, integrations, and workflow tools for teams. " * 20,
        ),
        SimpleNamespace(
            url="https://relevant.test/b",
            title="南京地铁线路",
            text="截至2026年，目前共运营14条线路。",
        ),
    ]
    calls: dict[str, object] = {}

    def fake_fetch_pages(_urls, **_kwargs):
        return pages

    def fake_build_prompt(question, resolved_pages):
        calls["prompt"] = (question, resolved_pages)
        return "prompt containing only relevant sources"

    _install_lightweight_web_modules(monkeypatch, fake_fetch_pages, fake_build_prompt)
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _settings: "fake-model")
    monkeypatch.setattr(
        web_answer_module,
        "invoke_with_retry",
        lambda *_args, **_kwargs: AIMessage(content="14条线路"),
    )

    node = web_answer_module.web_answer_factory(settings)
    result = node({"messages": [HumanMessage(content="南京地铁线路数量 2026 几条线")]})

    assert calls["prompt"] == (
        "南京地铁线路数量 2026 几条线",
        [pages[1]],
    )
    assert result["messages"][0].content == "14条线路"

    assert result["source_urls"] == ["https://relevant.test/b"]


def test_web_answer_retry_uses_preserved_question_instead_of_latest_refusal(
    monkeypatch,
    isolated_settings,
):
    question = "DeepSeek V4 July 2026 release"
    page = SimpleNamespace(
        url="https://relevant.test/deepseek-v4",
        title="DeepSeek V4 July 2026 release",
        text="DeepSeek V4 official release details for July 2026. " * 20,
    )
    captured: dict[str, object] = {}

    def fake_fetch_pages(_urls, **_kwargs):
        return [page]

    def fake_build_prompt(resolved_question, resolved_pages):
        captured["question"] = resolved_question
        captured["pages"] = resolved_pages
        return "grounded prompt"

    _install_lightweight_web_modules(monkeypatch, fake_fetch_pages, fake_build_prompt)
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _settings: "model")
    monkeypatch.setattr(
        web_answer_module,
        "invoke_with_retry",
        lambda *_args, **_kwargs: AIMessage(content="grounded answer"),
    )

    result = web_answer_module.web_answer_factory(
        isolated_settings(
            source_urls=[page.url],
            web_search_min_page_chars=1,
            web_search_min_page_tokens=1,
        ),
        graph_nodes.chat_question_resolver,
    )(
        {
            "current_question": question,
            "messages": [
                HumanMessage(content="What is the latest model?"),
                AIMessage(content="I couldn't retrieve readable content."),
            ],
        }
    )

    assert captured == {"question": question, "pages": [page]}
    assert result["messages"][-1].content == "grounded answer"


@tool
def live_web_search(query: str) -> str:
    """Return fake live web search results for lightweight graph tests."""

    return f"Live web search results for: {query}\n1. https://result.test/page"


def test_build_lightweight_graph_routes_tool_calls_to_web_answer(isolated_settings):
    seen: list[str] = []

    def agent(_state):
        seen.append("agent")
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "live_web_search",
                            "args": {"query": "latest PAI"},
                            "id": "call_live_web_search",
                        }
                    ],
                )
            ]
        }

    def web_answer(state):
        seen.append("web_answer")
        assert state["messages"][-1].content.endswith("https://result.test/page")
        return {"messages": [AIMessage(content="final lightweight answer")]}

    graph = build_lightweight_graph(
        settings=isolated_settings(),
        providers=GraphProviders(
            tools=[live_web_search],
            nodes=GraphNodeOverrides(agent=agent, web_answer=web_answer),
        ),
    )

    state = graph.invoke({"messages": [HumanMessage(content="What is new?")]})

    assert seen == ["agent", "web_answer"]
    assert state["messages"][-1].content == "final lightweight answer"


def test_build_lightweight_graph_routes_non_web_search_tools_back_to_agent(
    isolated_settings,
):
    # Regression guard: when the agent calls a non-web-search tool (weather,
    # stock, currency, wikipedia, ...), the lightweight graph must route the
    # ToolMessage back to the agent so it can synthesize a final answer from
    # the structured tool output. Routing it to ``web_answer`` would discard
    # the tool result and re-prompt the LLM against the session's curated URLs,
    # which is exactly what produces "the sources don't contain weather data"
    # answers when the agent had already fetched real weather data.
    @tool
    def get_weather(city: str) -> str:
        """Return mock weather output for tests."""

        return f"Weather for {city}: 25.2C, overcast, humidity 65%"

    invocations: list[str] = []

    def agent(state):
        # First turn: ask for the weather tool. Second turn: synthesize from
        # the ToolMessage. The router must take us back to ``agent`` after
        # the tool node, otherwise the second invocation never happens.
        last = state["messages"][-1]
        if getattr(last, "type", "") == "tool":
            invocations.append("synthesize")
            return {
                "messages": [AIMessage(content=f"final synthesized answer from {last.content}")]
            }
        invocations.append("call_tool")
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_weather",
                            "args": {"city": "Shanghai"},
                            "id": "call_get_weather",
                        }
                    ],
                )
            ]
        }

    def web_answer(_state):  # pragma: no cover - must not run
        raise AssertionError(
            "web_answer must not run for non-web-search tools; the agent must "
            "synthesize the answer from the structured tool output"
        )

    graph = build_lightweight_graph(
        settings=isolated_settings(),
        providers=GraphProviders(
            tools=[live_web_search, get_weather],
            nodes=GraphNodeOverrides(agent=agent, web_answer=web_answer),
        ),
    )

    state = graph.invoke({"messages": [HumanMessage(content="What is the weather in Shanghai?")]})

    assert invocations == ["call_tool", "synthesize"]
    final = state["messages"][-1].content
    assert final.startswith("final synthesized answer from ")
    assert "25.2C" in final


def test_build_lightweight_graph_routes_direct_agent_output_to_end(
    isolated_settings,
):
    # When the agent answers directly (no tool call), the graph must terminate
    # with the agent's reply. Routing direct answers through web_answer would
    # discard them and re-prompt against the fetched (often irrelevant) pages,
    # which is exactly the failure mode the system prompt steers away from for
    # trivial questions like math, definitions, or chitchat.
    seen: list[str] = []

    def agent(_state):
        seen.append("agent")
        return {"messages": [AIMessage(content="direct final answer")]}

    def web_answer(_state):  # pragma: no cover - must not run
        seen.append("web_answer")
        raise AssertionError(
            "web_answer must not run when the agent answers directly without a tool call"
        )

    graph = build_lightweight_graph(
        settings=isolated_settings(),
        providers=GraphProviders(
            tools=[live_web_search],
            nodes=GraphNodeOverrides(agent=agent, web_answer=web_answer),
        ),
    )

    state = graph.invoke({"messages": [HumanMessage(content="What is 1+1?")]})

    assert seen == ["agent"]
    assert state["messages"][-1].content == "direct final answer"


def test_build_lightweight_graph_chat_mode_uses_chat_state_and_checkpointer(
    isolated_settings,
):
    checkpointer = build_memory_saver()
    seen: list[str] = []

    def agent(state):
        seen.append(state["current_question"])
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "live_web_search",
                            "args": {"query": state["current_question"]},
                            "id": "call_live_web_search_chat",
                        }
                    ],
                )
            ]
        }

    def web_answer(state):
        assert state["current_question"] == "standalone chat question"
        return {"messages": [AIMessage(content="chat lightweight answer")]}

    graph = build_lightweight_graph(
        settings=isolated_settings(),
        checkpointer=checkpointer,
        providers=GraphProviders(
            tools=[live_web_search],
            nodes=GraphNodeOverrides(agent=agent, web_answer=web_answer),
        ),
    )
    config = {"configurable": {"thread_id": "lightweight-chat-thread"}}

    state = graph.invoke(
        {
            "messages": [HumanMessage(content="raw follow-up")],
            "current_question": "standalone chat question",
            "current_question_index": 0,
        },
        config=config,
    )
    snapshot = graph.get_state(config)

    assert seen == ["standalone chat question"]
    assert state["messages"][-1].content == "chat lightweight answer"
    assert snapshot.values["messages"][-1].content == "chat lightweight answer"


def test_web_answer_is_exported_from_graph_nodes():
    assert graph_nodes.web_answer_factory is web_answer_module.web_answer_factory


def test_web_answer_no_readable_content_sets_fallback_state(monkeypatch, isolated_settings):
    # Regression guard: when no fetched page yields readable text, the node
    # must signal the post-web-answer edge to retry the agent by setting
    # ``web_answer_no_readable_content`` to True and bumping the attempt
    # counter, in addition to returning the grounded refusal message.
    settings = isolated_settings(source_urls=["https://dead.test/a"])

    def fake_fetch_pages(urls, **_kwargs):
        return [SimpleNamespace(url=u, title="", text="") for u in urls]

    _install_lightweight_web_modules(monkeypatch, fake_fetch_pages, lambda *_: "unused")
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _settings: "fake-model")
    monkeypatch.setattr(
        web_answer_module,
        "invoke_with_retry",
        lambda *_a, **_k: AIMessage(content="unused"),
    )

    node = web_answer_module.web_answer_factory(settings)
    result = node({"messages": [HumanMessage(content="When was UniMelb founded?")]})

    assert result["web_answer_no_readable_content"] is True
    assert result["web_answer_attempts"] == 1
    assert result["source_urls"] == []
    assert "couldn't retrieve readable content" in result["messages"][-1].content


def test_web_answer_success_clears_fallback_state(monkeypatch, isolated_settings):
    # Regression guard: on the success branch the node must explicitly clear
    # ``web_answer_no_readable_content`` so a later failure in the same
    # session starts from a clean state, and stamp the attempt counter for
    # parity with the failure branch.
    settings = isolated_settings(source_urls=["https://good.test/a"])

    def fake_fetch_pages(urls, **_kwargs):
        return [SimpleNamespace(url=urls[0], title="T", text=_readable_text())]

    def fake_build_prompt(_question, _pages):
        return "prompt"

    def fake_invoke(*_a, **_k):
        return AIMessage(content="grounded answer")

    _install_lightweight_web_modules(monkeypatch, fake_fetch_pages, fake_build_prompt)
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _settings: "fake-model")
    monkeypatch.setattr(web_answer_module, "invoke_with_retry", fake_invoke)

    node = web_answer_module.web_answer_factory(settings)
    result = node(
        {
            "messages": [HumanMessage(content="Q?")],
            # Pre-seed stale state from a prior failure in the same session.
            "web_answer_no_readable_content": True,
            "web_answer_attempts": 1,
        }
    )

    assert result["web_answer_no_readable_content"] is False
    assert result["web_answer_attempts"] == 2
    assert result["source_urls"] == ["https://good.test/a"]
    assert result["messages"][-1].content == "grounded answer"


def test_web_answer_increments_attempts_from_existing_counter(monkeypatch, isolated_settings):
    # Regression guard: the attempt counter must be derived from the current
    # state, not always reset to 1, so a second invocation in the same
    # session reports the correct cumulative count.
    settings = isolated_settings(source_urls=["https://good.test/a"])

    def fake_fetch_pages(urls, **_kwargs):
        return [SimpleNamespace(url=urls[0], title="T", text=_readable_text())]

    _install_lightweight_web_modules(monkeypatch, fake_fetch_pages, lambda *_: "prompt")
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _settings: "fake-model")
    monkeypatch.setattr(
        web_answer_module,
        "invoke_with_retry",
        lambda *_a, **_k: AIMessage(content="ok"),
    )

    node = web_answer_module.web_answer_factory(settings)
    result = node(
        {
            "messages": [HumanMessage(content="Q?")],
            "web_answer_attempts": 3,
        }
    )

    assert result["web_answer_attempts"] == 4


def test_route_after_web_answer_routes_to_expand_on_first_failure():
    # Unit test for the post-web-answer conditional edge: when the previous
    # ``web_answer`` run produced no readable content AND expansion has
    # not yet been attempted, the edge must route to ``"expand"`` (the
    # conditional-expansion path) rather than falling straight back to
    # the agent. This is the new third outcome in
    # ``route_after_web_answer`` introduced by the 2026-06-17 expansion
    # implementation.
    from src.backend.graph.edges import route_after_web_answer

    state = {
        "web_answer_no_readable_content": True,
        "web_answer_attempts": 1,
        "expansion_attempted": False,
    }
    assert route_after_web_answer(state) == "expand"


def test_route_after_web_answer_terminates_after_expansion_without_grounding():
    # A failed expanded search must preserve the grounded refusal instead of
    # invoking a model-knowledge fallback.
    from src.backend.graph.edges import route_after_web_answer

    state = {
        "web_answer_no_readable_content": True,
        "web_answer_attempts": 1,
        "expansion_attempted": True,
    }
    assert route_after_web_answer(state) == "__end__"


def test_route_after_web_answer_terminates_above_attempt_ceiling():
    # Stale or externally supplied state beyond the normal two attempts must
    # terminate without starting another fallback.
    from src.backend.graph.edges import (
        WEB_ANSWER_FALLBACK_MAX_ATTEMPTS,
        route_after_web_answer,
    )

    state = {
        "web_answer_no_readable_content": True,
        "web_answer_attempts": WEB_ANSWER_FALLBACK_MAX_ATTEMPTS + 1,
        "expansion_attempted": True,
    }
    assert route_after_web_answer(state) == "__end__"


def test_route_after_web_answer_terminates_on_success():
    # Unit test: when ``web_answer`` succeeded, the edge must terminate so
    # the agent's synthesized answer is not overridden by a redundant
    # re-prompt through the agent node.
    from src.backend.graph.edges import route_after_web_answer

    state = {
        "web_answer_no_readable_content": False,
        "web_answer_attempts": 1,
    }
    assert route_after_web_answer(state) == "__end__"


def test_build_lightweight_graph_uses_fallback_when_no_readable_content(
    monkeypatch, isolated_settings
):
    # After one expanded search fails, the graph must run exactly one
    # tool-free fallback and terminate rather than silently keeping the
    # intermediate web-answer refusal.
    import src.backend.web_search.content_fetcher as content_fetcher_module
    import src.backend.web_search.prompt_builder as prompt_builder_module
    from src.backend.tools.live_web_search import build_web_search_tool

    settings = isolated_settings()

    # Build a real tool so the lightweight graph's ToolNode sees a real
    # ``live_web_search`` tool call from the agent. The discovery function
    # is monkeypatched to return a deterministic URL regardless of provider.
    def fake_discover(query, _settings, _provider):
        return ["https://junk.test/page"]

    def fake_fetch_pages(urls, **_kwargs):
        return [SimpleNamespace(url=urls[0], title="Junk", text="")]

    def fake_build_prompt(_question, _pages):  # pragma: no cover - must not run
        raise AssertionError("prompt must not be built when no readable pages")

    tool = build_web_search_tool(settings, discovery=fake_discover)
    monkeypatch.setattr(content_fetcher_module, "fetch_pages", fake_fetch_pages)
    monkeypatch.setattr(prompt_builder_module, "build_web_search_prompt", fake_build_prompt)
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _s: "fake-model")
    monkeypatch.setattr(
        web_answer_module,
        "invoke_with_retry",
        lambda *_a, **_k: AIMessage(content="must not run"),
    )

    seen: list[str] = []
    call_count = {"agent": 0}

    def agent(state):
        call_count["agent"] += 1
        seen.append(f"agent-run-{call_count['agent']}")
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "live_web_search",
                            "args": {"query": "University of Melbourne founded"},
                            "id": "call_live_web_search",
                        }
                    ],
                )
            ]
        }

    # REFACTOR: Inject deterministic decompose/expand/merge nodes so the
    # test does not need a live LLM for the conditional-expansion path.
    # The expand node also sets ``expansion_attempted`` to True so the
    # second web-answer failure terminates with the refusal.
    def fake_decompose(state):
        return {"sub_questions": ["University of Melbourne founded"]}

    def fake_expand(state):
        return {
            "expanded_queries": ["University of Melbourne founded"],
            "expansion_attempted": True,
        }

    def fake_merge(state):
        return {"source_urls": ["https://junk.test/page"]}

    graph = build_lightweight_graph(
        settings=settings,
        providers=GraphProviders(
            tools=[tool],
            nodes=GraphNodeOverrides(
                agent=agent,
                decompose=fake_decompose,
                expand=fake_expand,
                merge=fake_merge,
                fallback_answer=lambda _state: {
                    "messages": [AIMessage(content="unverified fallback: 1853")]
                },
            ),
        ),
    )

    state = graph.invoke(
        {"messages": [HumanMessage(content="When was the University of Melbourne founded?")]}
    )

    assert seen == ["agent-run-1"]
    assert call_count == {"agent": 1}
    final = state["messages"][-1].content
    assert final == "unverified fallback: 1853"


def test_build_lightweight_graph_grounded_refusal_cannot_start_a_third_search(
    monkeypatch, isolated_settings
):
    # End-to-end regression guard: after two failed web-answer attempts, the
    # refusal must terminate without re-entering the agent.
    import src.backend.web_search.content_fetcher as content_fetcher_module
    import src.backend.web_search.prompt_builder as prompt_builder_module
    from src.backend.tools.live_web_search import build_web_search_tool

    settings = isolated_settings()

    def fake_discover(query, _settings, _provider):
        return ["https://junk.test/page"]

    def fake_fetch_pages(urls, **_kwargs):
        return [SimpleNamespace(url=urls[0], title="Junk", text="")]

    def fake_build_prompt(_question, _pages):  # pragma: no cover - must not run
        raise AssertionError("prompt must not be built when no readable pages")

    tool = build_web_search_tool(settings, discovery=fake_discover)
    monkeypatch.setattr(content_fetcher_module, "fetch_pages", fake_fetch_pages)
    monkeypatch.setattr(prompt_builder_module, "build_web_search_prompt", fake_build_prompt)
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _s: "fake-model")
    monkeypatch.setattr(
        web_answer_module,
        "invoke_with_retry",
        lambda *_a, **_k: AIMessage(content="must not run"),
    )

    call_count = {"agent": 0, "web_answer": 0}

    def agent(state):
        call_count["agent"] += 1
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "live_web_search",
                            "args": {"query": "Q"},
                            "id": "call_live_web_search",
                        }
                    ],
                )
            ]
        }

    def web_answer(state):
        call_count["web_answer"] += 1
        # Mirror the real node's behavior: no readable content, bump the
        # attempt counter, set the fallback flag. The graph loop bound
        # must stop the iteration once the bound is reached.
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I couldn't retrieve readable content from the web "
                        f"sources for this question (call #{call_count['web_answer']})."
                    )
                )
            ],
            "web_answer_no_readable_content": True,
            "web_answer_attempts": call_count["web_answer"],
        }

    # REFACTOR: Inject deterministic decompose/expand/merge nodes so the
    # test does not need a live LLM for the conditional-expansion path.
    # The expand node sets ``expansion_attempted`` to True so the
    # second web-answer failure terminates.
    def fake_decompose(state):
        return {"sub_questions": ["Q"]}

    def fake_expand(state):
        return {
            "expanded_queries": ["Q"],
            "expansion_attempted": True,
        }

    def fake_merge(state):
        return {"source_urls": ["https://junk.test/page"]}

    graph = build_lightweight_graph(
        settings=settings,
        providers=GraphProviders(
            tools=[tool],
            nodes=GraphNodeOverrides(
                agent=agent,
                web_answer=web_answer,
                decompose=fake_decompose,
                expand=fake_expand,
                merge=fake_merge,
                fallback_answer=lambda _state: {
                    "messages": [AIMessage(content="bounded fallback")]
                },
            ),
        ),
    )

    state = graph.invoke({"messages": [HumanMessage(content="Q?")]})

    assert call_count == {"agent": 1, "web_answer": 2}
    final = state["messages"][-1].content
    assert final == "bounded fallback"
    assert "call #3" not in final


def test_web_answer_drops_structurally_noisy_pages(monkeypatch, isolated_settings):
    """A link-dense listing page is filtered on measurement, not URL shape."""

    from src.backend.web_search.page_structure import PageStructure

    settings = isolated_settings(source_urls=[], web_search_structure_filter_enabled=True)
    article = SimpleNamespace(
        url="https://example.test/news/source-detail",
        title="source detail report",
        text=_readable_text(),
        structure=PageStructure(shape="article", content_words=120, measured=True),
    )
    listing = SimpleNamespace(
        url="https://example.test/news/index",
        title="source detail archive",
        text=_readable_text(),
        structure=PageStructure(
            shape="listing",
            link_density=0.71,
            anchor_count=40,
            content_words=140,
            measured=True,
        ),
    )
    captured: dict[str, object] = {}

    def fake_build_prompt(question, resolved_pages, **_kwargs):
        captured["pages"] = list(resolved_pages)
        return "assembled prompt"

    _install_lightweight_web_modules(
        monkeypatch,
        lambda _urls, **_kwargs: [article, listing],
        fake_build_prompt,
    )
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _settings: "fake-model")
    monkeypatch.setattr(
        web_answer_module,
        "invoke_with_retry",
        lambda *_args, **_kwargs: AIMessage(content="grounded answer"),
    )

    node = web_answer_module.web_answer_factory(settings)
    result = node(
        {
            "messages": [HumanMessage(content="source detail")],
            "source_urls": [article.url, listing.url],
        }
    )

    assert result["source_urls"] == [article.url]
    assert captured["pages"] == [article]


def test_web_answer_keeps_listing_pages_when_structure_filtering_is_off(
    monkeypatch,
    isolated_settings,
):
    from src.backend.web_search.page_structure import PageStructure

    settings = isolated_settings(source_urls=[], web_search_structure_filter_enabled=False)
    listing = SimpleNamespace(
        url="https://example.test/news/index",
        title="source detail archive",
        text=_readable_text(),
        structure=PageStructure(shape="listing", link_density=0.71, measured=True),
    )

    _install_lightweight_web_modules(
        monkeypatch,
        lambda _urls, **_kwargs: [listing],
        lambda _question, _pages, **_kwargs: "assembled prompt",
    )
    monkeypatch.setattr(web_answer_module, "new_chat_model", lambda _settings: "fake-model")
    monkeypatch.setattr(
        web_answer_module,
        "invoke_with_retry",
        lambda *_args, **_kwargs: AIMessage(content="grounded answer"),
    )

    node = web_answer_module.web_answer_factory(settings)
    result = node(
        {
            "messages": [HumanMessage(content="source detail")],
            "source_urls": [listing.url],
        }
    )

    assert result["source_urls"] == [listing.url]
