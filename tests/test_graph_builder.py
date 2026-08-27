from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.graph.builder import (
    GraphNodeOverrides,
    GraphProviders,
    _resolve_lightweight_tools,
    _resolve_tools,
    build_graph,
)


@tool
def retrieve_source_documents(query: str) -> str:
    """Return fake source context for graph builder tests."""

    return f"retrieved context for {query}"


def test_qa_builder_accepts_injected_nodes_and_tools():
    def agent(_state):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "retrieve_source_documents",
                            "args": {"query": "PAI"},
                            "id": "call_fake_retrieve",
                        }
                    ],
                )
            ]
        }

    def grade_documents(state):
        assert state["messages"][-1].content == "retrieved context for PAI"
        return "generate"

    def generate(state):
        assert state["messages"][-1].content == "retrieved context for PAI"
        return {"messages": [AIMessage(content="final answer")]}

    providers = GraphProviders(
        tools=[retrieve_source_documents],
        nodes=GraphNodeOverrides(
            agent=agent,
            grade_documents=grade_documents,
            rewrite=lambda _state: {"messages": [AIMessage(content="unused")]},
            generate=generate,
        ),
    )

    graph = build_graph(mode="qa", providers=providers)
    state = graph.invoke({"messages": [HumanMessage(content="What is PAI?")]})

    assert state["messages"][-1].content == "final answer"


def test_chat_builder_adds_condense_and_allows_checkpointer_injection():
    seen: list[str] = []

    def condense(state):
        seen.append("condense")
        assert state["messages"][-1].content == "raw follow-up"
        return {
            "current_question": "standalone follow-up",
            "current_question_index": 0,
            "rewrite_count": 0,
        }

    def agent(state):
        seen.append("agent")
        assert state["current_question"] == "standalone follow-up"
        return {"messages": [AIMessage(content="direct answer")]}

    providers = GraphProviders(
        checkpointer=None,
        nodes=GraphNodeOverrides(
            condense=condense,
            agent=agent,
            retrieve=lambda _state: {"messages": [AIMessage(content="unused")]},
            grade_documents=lambda _state: "generate",
            rewrite=lambda _state: {"messages": [AIMessage(content="unused")]},
            generate=lambda _state: {"messages": [AIMessage(content="unused")]},
        ),
    )

    graph = build_graph(mode="chat", providers=providers)
    state = graph.invoke({"messages": [HumanMessage(content="raw follow-up")]})

    assert seen == ["condense", "agent"]
    assert state["messages"][-1].content == "direct answer"


def test_builder_requires_settings_for_default_nodes():
    with pytest.raises(ValueError, match="settings are required"):
        build_graph(mode="qa")


def test_resolve_tools_adds_web_search_when_enabled(monkeypatch, isolated_settings):
    import src.core.retriever as retriever_module
    import src.tools as web_search_module

    settings = isolated_settings(web_search_enabled=True)
    calls = []
    retriever_tool = object()
    web_search_tool = object()

    def fake_build_retriever_tool(resolved_settings, rebuild: bool = False):
        calls.append(("retriever", resolved_settings, rebuild))
        return retriever_tool

    def fake_build_web_search_tool(resolved_settings):
        calls.append(("web_search", resolved_settings))
        return web_search_tool

    monkeypatch.setattr(
        retriever_module,
        "build_retriever_tool",
        fake_build_retriever_tool,
    )
    monkeypatch.setattr(
        web_search_module,
        "build_web_search_tool",
        fake_build_web_search_tool,
    )

    tools = _resolve_tools(settings, GraphProviders(), rebuild_vectorstore=True)

    assert tools == [retriever_tool, web_search_tool]
    assert calls == [
        ("retriever", settings, True),
        ("web_search", settings),
    ]


def test_resolve_tools_adds_enabled_agent_tools(monkeypatch, isolated_settings):
    import src.core.retriever as retriever_module
    import src.tools as tools_module
    import src.tools as web_search_module

    settings = isolated_settings(
        web_search_enabled=False,
        weather_enabled=True,
        stock_enabled=True,
        currency_enabled=True,
        wikipedia_enabled=True,
    )
    retriever_tool = object()
    weather_tool = object()
    stock_tool = object()
    currency_tool = object()
    wikipedia_tool = object()
    calls = []

    monkeypatch.setattr(
        retriever_module,
        "build_retriever_tool",
        lambda _settings, rebuild=False: retriever_tool,
    )
    monkeypatch.setattr(
        web_search_module,
        "build_web_search_tool",
        lambda _settings: pytest.fail("web search tool should not be built"),
    )

    def fake_tool_builder(name, tool):
        def _build(resolved_settings):
            calls.append((name, resolved_settings))
            return tool

        return _build

    monkeypatch.setattr(
        tools_module,
        "build_weather_tool",
        fake_tool_builder("weather", weather_tool),
    )
    monkeypatch.setattr(
        tools_module,
        "build_stock_tool",
        fake_tool_builder("stock", stock_tool),
    )
    monkeypatch.setattr(
        tools_module,
        "build_currency_tool",
        fake_tool_builder("currency", currency_tool),
    )
    monkeypatch.setattr(
        tools_module,
        "build_wikipedia_tool",
        fake_tool_builder("wikipedia", wikipedia_tool),
    )

    tools = _resolve_tools(settings, GraphProviders(), rebuild_vectorstore=False)

    assert tools == [
        retriever_tool,
        weather_tool,
        stock_tool,
        currency_tool,
        wikipedia_tool,
    ]
    assert calls == [
        ("weather", settings),
        ("stock", settings),
        ("currency", settings),
        ("wikipedia", settings),
    ]


def test_resolve_tools_skips_web_search_when_disabled(monkeypatch, isolated_settings):
    import src.core.retriever as retriever_module
    import src.tools as web_search_module

    settings = isolated_settings(web_search_enabled=False)
    retriever_tool = object()

    monkeypatch.setattr(
        retriever_module,
        "build_retriever_tool",
        lambda _settings, rebuild=False: retriever_tool,
    )
    monkeypatch.setattr(
        web_search_module,
        "build_web_search_tool",
        lambda _settings: pytest.fail("web search tool should not be built"),
    )

    assert _resolve_tools(settings, GraphProviders(), rebuild_vectorstore=False) == [
        retriever_tool
    ]


def test_resolve_tools_preserves_provider_tools_override(isolated_settings):
    explicit_tool = object()

    tools = _resolve_tools(
        isolated_settings(web_search_enabled=True),
        GraphProviders(tools=[explicit_tool]),
        rebuild_vectorstore=True,
    )

    assert tools == [explicit_tool]


def test_resolve_tools_registers_text_editor_with_session_scope(
    monkeypatch,
    isolated_settings,
    tmp_path,
):
    import src.core.retriever as retriever_module
    import src.tools as tools_module

    settings = isolated_settings(
        web_search_enabled=False,
        file_read_enabled=True,
        text_edit_enabled=True,
    )
    retriever_tool = object()
    editor_tools = [object(), object()]
    calls = []

    monkeypatch.setattr(
        retriever_module,
        "build_retriever_tool",
        lambda _settings, rebuild=False: retriever_tool,
    )
    monkeypatch.setattr(tools_module, "build_text_file_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_word_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_excel_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_pdf_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_word_edit_tools", lambda *args, **kwargs: [])

    def fake_build_text_edit_tools(
        resolved_settings,
        *,
        session_root=None,
        thread_id="",
    ):
        calls.append((resolved_settings, session_root, thread_id))
        return editor_tools

    monkeypatch.setattr(
        tools_module,
        "build_text_edit_tools",
        fake_build_text_edit_tools,
    )
    session_root = tmp_path / "chat_uploads" / "thread-a"

    tools = _resolve_tools(
        settings,
        GraphProviders(),
        rebuild_vectorstore=False,
        session_root=session_root,
        thread_id="thread-a",
    )

    assert tools[-2:] == editor_tools
    assert calls == [(settings, session_root, "thread-a")]


def test_resolve_tools_registers_word_creator_with_session_scope(
    monkeypatch,
    isolated_settings,
    tmp_path,
):
    import src.core.retriever as retriever_module
    import src.tools as tools_module

    settings = isolated_settings(
        web_search_enabled=False,
        file_read_enabled=True,
        word_edit_enabled=True,
    )
    retriever_tool = object()
    word_tools = [object(), object(), object()]
    calls = []

    monkeypatch.setattr(
        retriever_module,
        "build_retriever_tool",
        lambda _settings, rebuild=False: retriever_tool,
    )
    monkeypatch.setattr(tools_module, "build_text_file_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_word_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_excel_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_pdf_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_text_edit_tools", lambda *args, **kwargs: [])

    def fake_build_word_edit_tools(
        resolved_settings,
        *,
        session_root=None,
        thread_id="",
    ):
        calls.append((resolved_settings, session_root, thread_id))
        return word_tools

    monkeypatch.setattr(
        tools_module,
        "build_word_edit_tools",
        fake_build_word_edit_tools,
    )
    session_root = tmp_path / "chat_uploads" / "thread-a"

    tools = _resolve_tools(
        settings,
        GraphProviders(),
        rebuild_vectorstore=False,
        session_root=session_root,
        thread_id="thread-a",
    )

    assert tools[-3:] == word_tools
    assert calls == [(settings, session_root, "thread-a")]


def test_lightweight_tools_registers_text_editor_with_session_scope(
    monkeypatch,
    isolated_settings,
    tmp_path,
):
    import src.tools as tools_module
    import src.tools as web_search_module

    settings = isolated_settings(
        file_read_enabled=True,
        text_edit_enabled=True,
    )
    web_tool = object()
    editor_tools = [object(), object()]
    calls = []
    monkeypatch.setattr(
        web_search_module,
        "build_web_search_tool",
        lambda _settings: web_tool,
    )
    monkeypatch.setattr(tools_module, "build_word_edit_tools", lambda *args, **kwargs: [])

    def fake_build_text_edit_tools(
        resolved_settings,
        *,
        session_root=None,
        thread_id="",
    ):
        calls.append((resolved_settings, session_root, thread_id))
        return editor_tools

    monkeypatch.setattr(
        tools_module,
        "build_text_edit_tools",
        fake_build_text_edit_tools,
    )
    session_root = tmp_path / "chat_uploads" / "thread-a"

    tools = _resolve_lightweight_tools(
        settings,
        GraphProviders(),
        session_root=session_root,
        thread_id="thread-a",
    )

    assert tools[0] is web_tool
    assert tools[-2:] == editor_tools
    assert calls == [(settings, session_root, "thread-a")]


def test_lightweight_tools_registers_word_creator_with_session_scope(
    monkeypatch,
    isolated_settings,
    tmp_path,
):
    import src.tools as tools_module
    import src.tools as web_search_module

    settings = isolated_settings(file_read_enabled=True, word_edit_enabled=True)
    web_tool = object()
    word_tools = [object(), object(), object()]
    calls = []
    monkeypatch.setattr(
        web_search_module,
        "build_web_search_tool",
        lambda _settings: web_tool,
    )
    monkeypatch.setattr(tools_module, "build_text_edit_tools", lambda *args, **kwargs: [])

    def fake_build_word_edit_tools(
        resolved_settings,
        *,
        session_root=None,
        thread_id="",
    ):
        calls.append((resolved_settings, session_root, thread_id))
        return word_tools

    monkeypatch.setattr(
        tools_module,
        "build_word_edit_tools",
        fake_build_word_edit_tools,
    )
    session_root = tmp_path / "chat_uploads" / "thread-a"

    tools = _resolve_lightweight_tools(
        settings,
        GraphProviders(),
        session_root=session_root,
        thread_id="thread-a",
    )

    assert tools[0] is web_tool
    assert tools[-3:] == word_tools
    assert calls == [(settings, session_root, "thread-a")]


def test_resolve_tools_registers_excel_creator_with_session_scope(
    monkeypatch,
    isolated_settings,
    tmp_path,
):
    import src.core.retriever as retriever_module
    import src.tools as tools_module

    settings = isolated_settings(
        web_search_enabled=False,
        file_read_enabled=True,
        excel_create_enabled=True,
    )
    retriever_tool = object()
    excel_tools = [object()]
    calls = []
    monkeypatch.setattr(
        retriever_module,
        "build_retriever_tool",
        lambda _settings, rebuild=False: retriever_tool,
    )
    monkeypatch.setattr(tools_module, "build_text_file_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_word_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_excel_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_pdf_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_word_edit_tools", lambda *args, **kwargs: [])
    monkeypatch.setattr(tools_module, "build_text_edit_tools", lambda *args, **kwargs: [])

    def fake_build_excel_create_tools(resolved_settings, *, session_root=None, thread_id=""):
        calls.append((resolved_settings, session_root, thread_id))
        return excel_tools

    monkeypatch.setattr(tools_module, "build_excel_create_tools", fake_build_excel_create_tools)
    session_root = tmp_path / "chat_uploads" / "thread-a"

    tools = _resolve_tools(
        settings,
        GraphProviders(),
        rebuild_vectorstore=False,
        session_root=session_root,
        thread_id="thread-a",
    )

    assert tools[-1:] == excel_tools
    assert calls == [(settings, session_root, "thread-a")]


def test_lightweight_tools_registers_excel_creator_with_session_scope(
    monkeypatch,
    isolated_settings,
    tmp_path,
):
    import src.tools as tools_module
    import src.tools as web_search_module

    settings = isolated_settings(file_read_enabled=False, excel_create_enabled=True)
    web_tool = object()
    excel_tools = [object()]
    calls = []
    monkeypatch.setattr(web_search_module, "build_web_search_tool", lambda _settings: web_tool)
    monkeypatch.setattr(tools_module, "build_word_edit_tools", lambda *args, **kwargs: [])
    monkeypatch.setattr(tools_module, "build_text_edit_tools", lambda *args, **kwargs: [])

    def fake_build_excel_create_tools(resolved_settings, *, session_root=None, thread_id=""):
        calls.append((resolved_settings, session_root, thread_id))
        return excel_tools

    monkeypatch.setattr(tools_module, "build_excel_create_tools", fake_build_excel_create_tools)
    session_root = tmp_path / "chat_uploads" / "thread-a"

    tools = _resolve_lightweight_tools(
        settings,
        GraphProviders(),
        session_root=session_root,
        thread_id="thread-a",
    )

    assert tools == [web_tool, *excel_tools]
    assert calls == [(settings, session_root, "thread-a")]


def test_both_graphs_register_powerpoint_editor_with_session_scope(
    monkeypatch,
    isolated_settings,
    tmp_path,
):
    import src.core.retriever as retriever_module
    import src.tools as tools_module
    import src.tools as web_search_module

    settings = isolated_settings(
        web_search_enabled=False,
        file_read_enabled=True,
        powerpoint_edit_enabled=True,
    )
    powerpoint_tools = [object(), object()]
    calls = []

    monkeypatch.setattr(
        retriever_module,
        "build_retriever_tool",
        lambda _settings, rebuild=False: object(),
    )
    monkeypatch.setattr(web_search_module, "build_web_search_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_text_file_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_word_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_excel_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_pdf_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_word_edit_tools", lambda *args, **kwargs: [])
    monkeypatch.setattr(tools_module, "build_text_edit_tools", lambda *args, **kwargs: [])
    monkeypatch.setattr(tools_module, "build_excel_create_tools", lambda *args, **kwargs: [])

    def fake_build_powerpoint_edit_tools(
        resolved_settings,
        *,
        session_root=None,
        thread_id="",
    ):
        calls.append((resolved_settings, session_root, thread_id))
        return powerpoint_tools

    monkeypatch.setattr(
        tools_module,
        "build_powerpoint_edit_tools",
        fake_build_powerpoint_edit_tools,
    )
    session_root = tmp_path / "chat_uploads" / "thread-a"

    full_tools = _resolve_tools(
        settings,
        GraphProviders(),
        rebuild_vectorstore=False,
        session_root=session_root,
        thread_id="thread-a",
    )
    lightweight_tools = _resolve_lightweight_tools(
        settings,
        GraphProviders(),
        session_root=session_root,
        thread_id="thread-a",
    )

    assert full_tools[-2:] == powerpoint_tools
    assert lightweight_tools[-2:] == powerpoint_tools
    assert calls == [
        (settings, session_root, "thread-a"),
        (settings, session_root, "thread-a"),
    ]


def test_both_graphs_register_excel_editor_with_session_scope(
    monkeypatch,
    isolated_settings,
    tmp_path,
):
    import src.core.retriever as retriever_module
    import src.tools as tools_module
    import src.tools as web_search_module

    settings = isolated_settings(
        web_search_enabled=False,
        file_read_enabled=True,
        excel_edit_enabled=True,
    )
    excel_edit_tools = [object(), object()]
    calls = []

    monkeypatch.setattr(
        retriever_module,
        "build_retriever_tool",
        lambda _settings, rebuild=False: object(),
    )
    monkeypatch.setattr(web_search_module, "build_web_search_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_text_file_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_word_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_excel_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_pdf_tool", lambda _settings: object())
    monkeypatch.setattr(tools_module, "build_word_edit_tools", lambda *args, **kwargs: [])
    monkeypatch.setattr(tools_module, "build_text_edit_tools", lambda *args, **kwargs: [])
    monkeypatch.setattr(tools_module, "build_excel_create_tools", lambda *args, **kwargs: [])

    def fake_build_excel_edit_tools(
        resolved_settings,
        *,
        session_root=None,
        thread_id="",
    ):
        calls.append((resolved_settings, session_root, thread_id))
        return excel_edit_tools

    monkeypatch.setattr(tools_module, "build_excel_edit_tools", fake_build_excel_edit_tools)
    session_root = tmp_path / "chat_uploads" / "thread-a"

    full_tools = _resolve_tools(
        settings,
        GraphProviders(),
        rebuild_vectorstore=False,
        session_root=session_root,
        thread_id="thread-a",
    )
    lightweight_tools = _resolve_lightweight_tools(
        settings,
        GraphProviders(),
        session_root=session_root,
        thread_id="thread-a",
    )

    assert full_tools[-2:] == excel_edit_tools
    assert lightweight_tools[-2:] == excel_edit_tools
    assert calls == [
        (settings, session_root, "thread-a"),
        (settings, session_root, "thread-a"),
    ]


def test_legacy_graph_wrappers_delegate_to_shared_builder(monkeypatch, mock_settings):
    import src.chat.graph as chat_graph
    import src.core.graph as core_graph

    qa_sentinel = object()
    chat_sentinel = object()
    captured = []

    def fake_core_builder(**kwargs):
        captured.append(kwargs)
        return qa_sentinel

    def fake_chat_builder(**kwargs):
        captured.append(kwargs)
        return chat_sentinel

    monkeypatch.setattr(core_graph, "_build_graph", fake_core_builder)
    monkeypatch.setattr(chat_graph, "_build_graph", fake_chat_builder)

    assert core_graph.build_graph(mock_settings, rebuild_vectorstore=True) is qa_sentinel
    assert chat_graph.build_chat_graph(mock_settings, rebuild_vectorstore=False) is chat_sentinel
    assert captured == [
        {
            "mode": "qa",
            "settings": mock_settings,
            "rebuild_vectorstore": True,
        },
        {
            "mode": "chat",
            "settings": mock_settings,
            "rebuild_vectorstore": False,
        },
    ]
