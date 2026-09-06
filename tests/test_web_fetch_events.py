"""WebFetchEvent: per-URL fetch announcements reach the streaming executor.

The ``web_answer`` node announces every source URL it is about to read through
LangGraph's custom stream; the executor surfaces those payloads as typed
``WebFetchEvent``s on the token-streaming path (chat SSE) only.
"""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.backend.graph.builder import (
    GraphNodeOverrides,
    GraphProviders,
    build_lightweight_graph,
)
from src.backend.graph.events import WebFetchEvent
from src.backend.graph.executor import GraphExecutor
from src.backend.graph.nodes import web_answer as web_answer_module


@tool
def live_web_search(query: str) -> str:
    """Search the web for the query."""

    return "stub discovery results"


def test_web_fetch_events_flow_through_streaming_executor(monkeypatch, isolated_settings):
    from src.backend.web_search.content_fetcher import is_readable_text

    pages = [
        SimpleNamespace(
            url="https://a.test/one",
            title="A",
            text=" ".join(["atomic question source detail"] * 30),
        )
    ]

    def fake_fetch_pages(urls, **kwargs):
        return pages

    content_fetcher = ModuleType("src.backend.web_search.content_fetcher")
    content_fetcher.FetchedPage = SimpleNamespace
    content_fetcher.fetch_pages = fake_fetch_pages
    content_fetcher.is_readable_page = lambda *args, **kwargs: True
    content_fetcher.is_readable_text = is_readable_text
    monkeypatch.setitem(
        sys.modules, "src.backend.web_search.content_fetcher", content_fetcher
    )
    monkeypatch.setattr(
        "src.backend.web_search.common.is_page_text_relevant",
        lambda *args, **kwargs: True,
    )

    monkeypatch.setattr(
        web_answer_module, "new_chat_model", lambda _settings: "fake-model"
    )
    monkeypatch.setattr(
        web_answer_module,
        "invoke_with_retry",
        lambda model, payload, max_retries=1: AIMessage(content="grounded answer"),
    )

    def agent(_state):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "live_web_search",
                            "args": {"query": "atomic question"},
                            "id": "call-1",
                        }
                    ],
                )
            ]
        }

    graph = build_lightweight_graph(
        settings=isolated_settings(),
        providers=GraphProviders(
            tools=[live_web_search],
            nodes=GraphNodeOverrides(
                agent=agent,
                decompose=lambda _state: {"sub_questions": ["atomic question"]},
                merge=lambda _state: {
                    "source_urls": ["https://a.test/one", "https://b.test/two"]
                },
                fallback_answer=lambda _state: {
                    "messages": [AIMessage(content="grounded refusal")]
                },
            ),
        ),
    )
    executor = GraphExecutor(graph=graph)

    async def collect():
        return [
            event
            async for event in executor.astream(
                {"messages": [HumanMessage(content="atomic question")]},
                stream_tokens=True,
            )
        ]

    events = asyncio.run(collect())

    fetches = [event for event in events if isinstance(event, WebFetchEvent)]
    assert [event.url for event in fetches] == [
        "https://a.test/one",
        "https://b.test/two",
    ]
    assert fetches[0].total == 2
    assert fetches[0].node == "web_answer"
    answers = [
        event
        for event in events
        if getattr(event, "type", "") == "done" and event.answer == "grounded answer"
    ]
    assert answers, "expected a done event with the grounded answer"


def test_web_fetch_announcements_swallow_failures_outside_stream(monkeypatch, isolated_settings):
    """Direct node calls (no stream context) must not raise on announcement."""

    from src.backend.graph.nodes import web_answer as web_answer_module

    monkeypatch.setattr(
        web_answer_module, "new_chat_model", lambda _settings: "fake-model"
    )
    monkeypatch.setattr(
        web_answer_module,
        "invoke_with_retry",
        lambda model, payload, max_retries=1: AIMessage(content="answer"),
    )
    content_fetcher = ModuleType("src.backend.web_search.content_fetcher")
    content_fetcher.FetchedPage = SimpleNamespace
    content_fetcher.fetch_pages = lambda urls, **kwargs: [
        SimpleNamespace(
            url="https://a.test/one",
            title="A",
            text=" ".join(["atomic question source detail"] * 30),
        )
    ]
    content_fetcher.is_readable_page = lambda *args, **kwargs: True
    import src.backend.web_search.content_fetcher as real_fetcher

    content_fetcher.is_readable_text = real_fetcher.is_readable_text
    monkeypatch.setitem(
        sys.modules, "src.backend.web_search.content_fetcher", content_fetcher
    )
    monkeypatch.setattr(
        "src.backend.web_search.common.is_page_text_relevant",
        lambda *args, **kwargs: True,
    )

    settings = isolated_settings(source_urls=["https://a.test/one"])
    node = web_answer_module.web_answer_factory(settings)
    result = node(
        {
            "messages": [HumanMessage(content="atomic question")],
            "source_urls": ["https://a.test/one"],
        }
    )
    assert result["messages"][0].content == "answer"
