from __future__ import annotations

from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from src.backend.core.state import AgentState
from src.backend.graph.nodes import (
    chat_question_resolver,
    condense_question_factory,
    format_history,
    grade_documents_factory,
    latest_user_index,
)
from src.backend.graph.nodes import common as common_nodes
from src.backend.graph.state import RAGState
from src.frontend.chat.state import ChatState


def test_state_aliases_preserve_old_import_paths():
    assert AgentState is RAGState
    assert ChatState is RAGState


def test_core_node_helpers_preserve_old_import_paths():
    from src.backend.core.nodes import _question_tokens, _split_context_sentences

    assert _question_tokens("What about reinforcement learning?") == {
        "reinforcement",
        "learning",
    }
    assert _split_context_sentences("Short. A sufficiently long sentence stays available.") == [
        "A sufficiently long sentence stays available."
    ]


def test_question_resolver_handles_chat_shapes():
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


def test_condense_followup_question_passthrough_without_history(mock_settings):
    from src.backend.graph.nodes.condense import condense_followup_question

    # No prior turns -> the raw message is returned unchanged and no LLM is hit.
    assert (
        condense_followup_question([], "Argentina and Jordan", mock_settings)
        == "Argentina and Jordan"
    )


def test_condense_followup_question_uses_history(monkeypatch, mock_settings):
    from src.backend.graph.nodes import condense as condense_module

    # Stub the chat model with a RunnableLambda so the prompt|model|parser
    # chain composes and yields our standalone question without a network call.
    fake_model = RunnableLambda(
        lambda _prompt: AIMessage(
            content="Argentina vs Jordan World Cup 2026 match result"
        )
    )
    monkeypatch.setattr(condense_module, "new_chat_model", lambda _s: fake_model)

    history = [
        HumanMessage(content="What was the Canada vs South Africa World Cup result?"),
        AIMessage(content="Canada beat South Africa 2-1 in the group stage."),
    ]
    standalone = condense_module.condense_followup_question(
        history, "Argentina and Jordan", mock_settings
    )

    assert "Argentina" in standalone and "Jordan" in standalone
    assert "World Cup" in standalone


def test_condense_standalone_followup_skips_llm(monkeypatch, mock_settings):
    from src.backend.graph.nodes import condense as condense_module

    monkeypatch.setattr(
        condense_module,
        "new_chat_model",
        lambda _settings: (_ for _ in ()).throw(AssertionError("LLM should not run")),
    )

    standalone = condense_module.condense_followup_question(
        [
            HumanMessage(content="Tell me about PAI."),
            AIMessage(content="PAI is Alibaba Cloud's AI platform."),
        ],
        "What is LangGraph?",
        mock_settings,
    )

    assert standalone == "What is LangGraph?"


def test_condense_bounds_history_without_mutating_checkpoint_messages(
    monkeypatch,
    isolated_settings,
):
    from src.backend.graph.nodes import condense as condense_module

    settings = isolated_settings(chat_context_max_turns=2, chat_context_max_chars=240)
    captured_payload = {}
    fake_model = RunnableLambda(lambda _prompt: AIMessage(content="standalone"))
    monkeypatch.setattr(condense_module, "new_chat_model", lambda _settings: fake_model)

    def fake_invoke(_chain, payload, max_retries):
        captured_payload.update(payload)
        return "standalone"

    monkeypatch.setattr(condense_module, "invoke_with_retry", fake_invoke)
    history = [
        HumanMessage(content="old-user-marker " + "x" * 80),
        AIMessage(content="old-assistant-marker " + "y" * 80),
        HumanMessage(content="recent first"),
        AIMessage(content="recent answer"),
        HumanMessage(content="recent second"),
        AIMessage(content="another recent answer"),
    ]
    original_contents = [message.content for message in history]

    result = condense_module.condense_followup_question(
        history,
        "What about its pricing?",
        settings,
    )

    assert result == "standalone"
    assert "old-user-marker" not in captured_payload["history"]
    assert "recent first" in captured_payload["history"]
    assert "recent second" in captured_payload["history"]
    assert [message.content for message in history] == original_contents


def test_bounded_chat_messages_keeps_recent_turns_and_hard_char_limit():
    messages = [
        HumanMessage(content="old question"),
        AIMessage(content="old answer"),
        HumanMessage(content="recent question"),
        AIMessage(content="recent answer"),
        ToolMessage(content="hidden tool output", tool_call_id="call-1"),
        HumanMessage(content="latest-" + "z" * 100),
    ]
    original_latest = messages[-1].content

    bounded = common_nodes.bounded_chat_messages(
        messages,
        max_turns=2,
        max_chars=60,
    )

    assert sum(len(common_nodes.message_text(message)) for message in bounded) <= 60
    assert any("latest-" in message.content for message in bounded)
    assert not any(isinstance(message, ToolMessage) for message in bounded)
    assert not any("old question" in message.content for message in bounded)
    assert messages[-1].content == original_latest


def test_bounded_chat_messages_keeps_system_note_with_its_turn():
    bounded = common_nodes.bounded_chat_messages(
        [
            SystemMessage(content="Uploaded path: chat_uploads/thread/notes.txt"),
            HumanMessage(content="Please read my notes"),
            AIMessage(content="I can do that."),
            HumanMessage(content="What did the notes say?"),
        ],
        max_turns=2,
        max_chars=500,
    )

    assert isinstance(bounded[0], SystemMessage)
    assert "chat_uploads/thread/notes.txt" in bounded[0].content


def test_chat_agent_receives_bounded_projection(monkeypatch, isolated_settings):
    settings = isolated_settings(chat_context_max_turns=1, chat_context_max_chars=100)
    captured_messages = []

    class FakeModel:
        def bind_tools(self, _tools):
            return self

    monkeypatch.setattr(common_nodes, "new_chat_model", lambda _settings: FakeModel())

    def fake_invoke(_model, messages, max_retries):
        captured_messages.extend(messages)
        return AIMessage(content="answer")

    monkeypatch.setattr(common_nodes, "invoke_with_retry", fake_invoke)
    state_messages = [
        HumanMessage(content="old question"),
        AIMessage(content="old answer"),
        HumanMessage(content="latest question"),
    ]

    common_nodes.agent_factory(
        settings,
        [],
        common_nodes.chat_question_resolver,
    )({"messages": state_messages})

    assert captured_messages[0].type == "system"
    assert [message.content for message in captured_messages[1:]] == [
        "latest question"
    ]
    assert [message.content for message in state_messages] == [
        "old question",
        "old answer",
        "latest question",
    ]


def test_chat_agent_keeps_current_tool_protocol(monkeypatch, isolated_settings):
    settings = isolated_settings(chat_context_max_turns=1, chat_context_max_chars=200)
    captured_messages = []

    class FakeModel:
        def bind_tools(self, _tools):
            return self

    monkeypatch.setattr(common_nodes, "new_chat_model", lambda _settings: FakeModel())

    def fake_invoke(_model, messages, max_retries):
        captured_messages.extend(messages)
        return AIMessage(content="It is 22 degrees.")

    monkeypatch.setattr(common_nodes, "invoke_with_retry", fake_invoke)
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "weather",
                "args": {"location": "Shanghai"},
                "id": "weather-1",
            }
        ],
    )
    tool_result = ToolMessage(
        content="Shanghai: 22 C and clear. " + "forecast detail " * 30,
        tool_call_id="weather-1",
    )
    original_tool_content = tool_result.content

    common_nodes.agent_factory(
        settings,
        [],
        common_nodes.chat_question_resolver,
    )(
        {
            "messages": [
                HumanMessage(content="old question"),
                AIMessage(content="old answer"),
                HumanMessage(content="What is the weather?"),
                tool_call,
                tool_result,
            ]
        }
    )

    projected = captured_messages[1:]
    assert [message.type for message in projected] == ["human", "ai", "tool"]
    assert projected[1].tool_calls[0]["id"] == "weather-1"
    assert projected[2].content.startswith("Shanghai: 22 C and clear")
    assert sum(len(message.content) for message in projected) <= 200
    assert tool_result.content == original_tool_content


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

    def fake_invoke_with_retry(_chain, payload, max_retries):
        captured_payload.update(payload)
        return SimpleNamespace(binary_score="yes", explanation="matched")

    monkeypatch.setattr(
        common_nodes,
        "new_structured_chat_model",
        lambda _settings, _schema: RunnableLambda(lambda _payload: None),
    )
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


def test_grade_prompt_treats_json_shape_as_literal_text():
    from src.backend.llm.prompts import GRADE_PROMPT

    rendered = GRADE_PROMPT.format(
        question="What is DeepSeek?",
        context="DeepSeek is an AI company.",
        current_date="2026-07-14",
    )

    assert '{"binary_score":"yes or no","explanation":"short reason"}' in rendered


def test_low_relevance_generate_requires_at_least_one_keyword_match(
    monkeypatch,
    isolated_settings,
):
    settings = isolated_settings(
        allow_low_relevance_generate=True,
        min_keyword_matches=0,
        max_rewrites=2,
    )

    monkeypatch.setattr(
        common_nodes,
        "new_structured_chat_model",
        lambda _settings, _schema: RunnableLambda(
            lambda _payload: SimpleNamespace(
                binary_score="no",
                explanation="irrelevant",
            )
        ),
    )

    route = grade_documents_factory(settings)(
        {
            "messages": [
                HumanMessage(content="DeepSeek latest"),
                AIMessage(content="Qwen fine tuning guide"),
            ],
            "current_question": "DeepSeek latest",
            "rewrite_count": 0,
        }
    )

    assert route == "rewrite"

