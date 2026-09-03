from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.backend.graph.nodes import fallback_answer as fallback_module


def test_fallback_uses_preserved_question_and_excludes_generated_refusal(
    isolated_settings, monkeypatch
):
    captured = {}
    monkeypatch.setattr(fallback_module, "new_chat_model", lambda _settings: "model")

    def fake_invoke(model, payload, **kwargs):
        captured.update(model=model, payload=payload, kwargs=kwargs)
        return AIMessage(content="I could not verify it live; the stable answer is 1853.")

    monkeypatch.setattr(fallback_module, "invoke_with_retry", fake_invoke)
    node = fallback_module.fallback_answer_factory(
        isolated_settings(), lambda state: state["current_question"]
    )

    result = node(
        {
            "current_question": "When was the University of Melbourne founded?",
            "messages": [
                HumanMessage(content="When was it founded?"),
                AIMessage(
                    content="I couldn't retrieve readable content from the web sources."
                ),
            ],
        }
    )

    prompt = captured["payload"][0].content
    assert "When was the University of Melbourne founded?" in prompt
    assert "retrieve readable content" not in prompt
    assert captured["model"] == "model"
    assert result["messages"][0].content.endswith("1853.")


def test_fallback_model_failure_returns_final_refusal(isolated_settings, monkeypatch):
    monkeypatch.setattr(fallback_module, "new_chat_model", lambda _settings: "model")
    monkeypatch.setattr(
        fallback_module,
        "invoke_with_retry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    node = fallback_module.fallback_answer_factory(
        isolated_settings(), lambda _state: "Original question"
    )

    result = node({"messages": [HumanMessage(content="Original question")]})

    assert "couldn't produce a reliable answer" in result["messages"][0].content
