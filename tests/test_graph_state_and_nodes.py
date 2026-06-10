from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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


def test_rerank_retrieved_context_prefers_lexically_relevant_chunks():
    message = ToolMessage(
        content="generic cloud setup notes\n\nPAI reinforcement learning reward model guide",
        tool_call_id="call_1",
        artifact=[
            Document(page_content="generic cloud setup notes"),
            Document(page_content="PAI reinforcement learning reward model guide"),
        ],
    )

    context = common_nodes.rerank_retrieved_context(
        "How does PAI use reinforcement learning reward models?",
        message,
    )

    assert context.split("\n\n") == [
        "PAI reinforcement learning reward model guide",
        "generic cloud setup notes",
    ]


def test_rerank_retrieved_context_supports_embedding_strategy(
    monkeypatch,
    isolated_settings,
):
    settings = isolated_settings(rerank_strategy="embedding")
    message = ToolMessage(
        content="lexical PAI reinforcement note\n\nsemantic vector match",
        tool_call_id="call_1",
        artifact=[
            Document(page_content="lexical PAI reinforcement note"),
            Document(page_content="semantic vector match"),
        ],
    )

    class FakeEmbeddings:
        def embed_query(self, _text):
            return [1.0, 0.0]

        def embed_documents(self, texts):
            vectors = {
                "lexical PAI reinforcement note": [0.0, 1.0],
                "semantic vector match": [1.0, 0.0],
            }
            return [vectors[text] for text in texts]

    monkeypatch.setattr(
        common_nodes,
        "_build_rerank_embeddings",
        lambda _settings: FakeEmbeddings(),
    )

    context = common_nodes.rerank_retrieved_context(
        "How does PAI use reinforcement learning?",
        message,
        settings,
    )

    assert context.split("\n\n") == [
        "semantic vector match",
        "lexical PAI reinforcement note",
    ]


def test_rerank_retrieved_context_hybrid_combines_scores(
    monkeypatch,
    isolated_settings,
):
    settings = isolated_settings(rerank_strategy="hybrid")
    message = ToolMessage(
        content="PAI reinforcement learning exact guide\n\nsemantic neighbor",
        tool_call_id="call_1",
    )

    class FakeEmbeddings:
        def embed_query(self, _text):
            return [1.0, 0.0]

        def embed_documents(self, _texts):
            return [[0.8, 0.2], [1.0, 0.0]]

    monkeypatch.setattr(
        common_nodes,
        "_build_rerank_embeddings",
        lambda _settings: FakeEmbeddings(),
    )

    context = common_nodes.rerank_retrieved_context(
        "PAI reinforcement learning guide",
        message,
        settings,
    )

    assert context.split("\n\n")[0] == "PAI reinforcement learning exact guide"


def test_grade_documents_uses_reranked_context_and_binary_score(
    monkeypatch,
    isolated_settings,
):
    settings = isolated_settings()
    captured_payload = {}

    class FakeModel:
        def with_structured_output(self, _schema):
            return RunnableLambda(lambda _payload: None)

    def fake_invoke_with_retry(_chain, payload, max_retries):
        captured_payload.update(payload)
        return SimpleNamespace(binary_score="yes", explanation="matched")

    monkeypatch.setattr(common_nodes, "new_chat_model", lambda _settings: FakeModel())
    monkeypatch.setattr(common_nodes, "invoke_with_retry", fake_invoke_with_retry)

    route = grade_documents_factory(settings)(
        {
            "messages": [
                HumanMessage(content="How does PAI use reinforcement learning?"),
                ToolMessage(
                    content=(
                        "unrelated installation details\n\n"
                        "PAI reinforcement learning training pipeline"
                    ),
                    tool_call_id="call_1",
                ),
            ],
            "rewrite_count": 0,
        }
    )

    assert route == "generate"
    assert captured_payload["context"].split("\n\n") == [
        "PAI reinforcement learning training pipeline",
        "unrelated installation details",
    ]


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

