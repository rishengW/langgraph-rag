from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.chat.state import ChatState
from src.core.state import AgentState
from src.graph.nodes import common as common_nodes
from src.graph.nodes import (
    chat_question_resolver,
    condense_question_factory,
    format_history,
    grade_documents_factory,
    latest_user_index,
    qa_question_resolver,
)
from src.graph.state import RAGState


def test_state_aliases_preserve_old_import_paths():
    assert AgentState is RAGState
    assert ChatState is RAGState


def test_core_node_helpers_preserve_old_import_paths():
    from src.core.nodes import _question_tokens, _split_context_sentences

    assert _question_tokens("What about reinforcement learning?") == {
        "reinforcement",
        "learning",
    }
    assert _split_context_sentences("Short. A sufficiently long sentence stays available.") == [
        "A sufficiently long sentence stays available."
    ]


def test_question_resolvers_handle_qa_and_chat_shapes():
    qa_state = {"messages": [HumanMessage(content="first question")]}
    assert qa_question_resolver(qa_state) == "first question"

    chat_state = {
        "messages": [
            HumanMessage(content="old"),
            AIMessage(content="answer"),
            HumanMessage(content="raw follow-up"),
        ],
        "current_question": "standalone follow-up",
        "current_question_index": 2,
    }
    assert chat_question_resolver(chat_state) == "standalone follow-up"

    fallback_state = dict(chat_state)
    fallback_state["current_question"] = ""
    assert chat_question_resolver(fallback_state) == "raw follow-up"


def test_format_history_skips_tool_and_empty_messages():
    history = format_history(
        [
            HumanMessage(content="hello"),
            AIMessage(content=""),
            AIMessage(content="hi"),
        ]
    )
    assert history == "User: hello\nAssistant: hi"


def test_latest_user_index_finds_last_human_message():
    messages = [
        HumanMessage(content="first"),
        AIMessage(content="answer"),
        HumanMessage(content="second"),
    ]
    assert latest_user_index(messages) == 2


def test_condense_first_turn_does_not_call_llm(mock_settings):
    node = condense_question_factory(mock_settings)
    result = node({"messages": [HumanMessage(content="What is PAI?")]})

    assert result == {
        "current_question": "What is PAI?",
        "current_question_index": 0,
        "rewrite_count": 0,
    }


def test_low_relevance_generate_requires_at_least_one_keyword_match(
    monkeypatch,
    isolated_settings,
):
    settings = isolated_settings(
        allow_low_relevance_generate=True,
        min_keyword_matches=0,
        max_rewrites=2,
    )

    class FakeModel:
        def with_structured_output(self, _schema):
            return RunnableLambda(
                lambda _payload: SimpleNamespace(
                    binary_score="no",
                    explanation="irrelevant",
                )
            )

    monkeypatch.setattr(common_nodes, "new_chat_model", lambda _settings: FakeModel())

    route = grade_documents_factory(settings)(
        {
            "messages": [
                HumanMessage(content="DeepSeek latest"),
                AIMessage(content="Qwen fine tuning guide"),
            ],
            "rewrite_count": 0,
        }
    )

    assert route == "rewrite"

