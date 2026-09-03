from __future__ import annotations

from dataclasses import replace

from pydantic import BaseModel

from src.backend.llm.provider import build_structured_chat_model, structured_output_method


class _Result(BaseModel):
    value: str


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def with_structured_output(self, schema, **kwargs):
        self.calls.append((schema, kwargs))
        return "structured-model"


class _FakeProvider:
    def __init__(self, model: _FakeModel) -> None:
        self.model = model

    def chat_model(self, _settings):
        return self.model


def test_deepseek_structured_output_uses_tool_free_json_mode(isolated_settings):
    settings = isolated_settings(llm_provider="deepseek", deepseek_api_key="key")
    model = _FakeModel()

    result = build_structured_chat_model(settings, _Result, _FakeProvider(model))

    assert result == "structured-model"
    assert structured_output_method(settings) == "json_mode"
    assert model.calls == [(_Result, {"method": "json_mode"})]
    assert "tool_choice" not in model.calls[0][1]


def test_dashscope_keeps_library_structured_output_default(isolated_settings):
    settings = replace(isolated_settings(), llm_provider="dashscope")
    model = _FakeModel()

    build_structured_chat_model(settings, _Result, _FakeProvider(model))

    assert structured_output_method(settings) is None
    assert model.calls == [(_Result, {})]
