from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.graph import nodes as graph_nodes
from src.graph.builder import (
    GraphNodeOverrides,
    GraphProviders,
    build_memory_saver,
    build_lightweight_graph,
)
from src.graph.nodes import web_answer as web_answer_module


def _install_lightweight_web_modules(monkeypatch, fetch_pages, build_prompt) -> None:
    content_fetcher = ModuleType("src.web_search.content_fetcher")
    content_fetcher.FetchedPage = SimpleNamespace
    content_fetcher.fetch_pages = fetch_pages

    prompt_builder = ModuleType("src.web_search.prompt_builder")
    prompt_builder.build_web_search_prompt = build_prompt

    monkeypatch.setitem(sys.modules, "src.web_search.content_fetcher", content_fetcher)
    monkeypatch.setitem(sys.modules, "src.web_search.prompt_builder", prompt_builder)


def test_web_answer_fetches_state_urls_and_invokes_llm(monkeypatch, isolated_settings):
    settings = isolated_settings(
        source_urls=["https://fallback.test"],
        page_load_timeout=7,
        page_load_cache_ttl_seconds=11,
        web_search_max_page_tokens=321,
    )
    calls: dict[str, object] = {}
    pages = [SimpleNamespace(url="https://a.test/page", title="A", text="Alpha")]

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
    assert calls["fetch"] == (
        ["https://a.test/page"],
        {
            "timeout": 7,
            "max_tokens_per_page": 321,
            "cache_ttl_seconds": 11,
            "max_concurrent_loads": 4,
        },
    )
    assert calls["prompt"] == ("What changed?", pages)
    assert calls["llm"][1][0].content == "assembled prompt"
    assert calls["llm"][2] == 1


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
        return [SimpleNamespace(url=urls[0], title="T", text="tool url text")]

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
    settings = isolated_settings(
        source_urls=["https://curated.test/a", "https://curated.test/b"]
    )
    calls: dict[str, object] = {}

    def fake_fetch_pages(urls, **_kwargs):
        calls["urls"] = list(urls)
        return [SimpleNamespace(url=urls[0], title="C", text="curated text")]

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
                    content=(
                        "Live web search results for: Question?\n"
                        "1. https://junk.test/x"
                    ),
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


def test_build_lightweight_graph_routes_direct_agent_output_to_web_answer(
    isolated_settings,
):
    seen: list[str] = []

    def agent(_state):
        seen.append("agent")
        return {"messages": [AIMessage(content="no tool needed")]}

    def web_answer(state):
        seen.append("web_answer")
        assert state["messages"][-1].content == "no tool needed"
        return {"messages": [AIMessage(content="direct final answer")]}

    graph = build_lightweight_graph(
        settings=isolated_settings(),
        providers=GraphProviders(
            tools=[live_web_search],
            nodes=GraphNodeOverrides(agent=agent, web_answer=web_answer),
        ),
    )

    state = graph.invoke({"messages": [HumanMessage(content="What is new?")]})

    assert seen == ["agent", "web_answer"]
    assert state["messages"][-1].content == "direct final answer"


def test_build_lightweight_graph_chat_mode_uses_chat_state_and_checkpointer(
    isolated_settings,
):
    checkpointer = build_memory_saver()
    seen: list[str] = []

    def agent(state):
        seen.append(state["current_question"])
        return {"messages": [AIMessage(content="chat agent output")]}

    def web_answer(state):
        assert state["current_question"] == "standalone chat question"
        return {"messages": [AIMessage(content="chat lightweight answer")]}

    graph = build_lightweight_graph(
        settings=isolated_settings(),
        mode="chat",
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
