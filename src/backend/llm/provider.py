"""LLM provider seam used by graph nodes and future streaming adapters."""

from __future__ import annotations

from typing import Any, Protocol, cast

from langchain_community.chat_models.tongyi import ChatTongyi

from src.config import Settings


class LLMProvider(Protocol):
    """Protocol for constructing chat model instances."""

    def chat_model(self, settings: Settings) -> Any:
        """Return a chat model configured for the supplied settings."""


class DashScopeLLMProvider:
    """Default DashScope-backed LLM provider."""

    def chat_model(self, settings: Settings) -> Any:
        """Create a DashScope chat model with project-level network settings."""

        model_kwargs: dict[str, Any] = {
            "request_timeout": settings.dashscope_request_timeout,
        }
        if settings.dashscope_http_base_url:
            model_kwargs["base_address"] = settings.dashscope_http_base_url

        chat_tongyi = cast(Any, ChatTongyi)
        return chat_tongyi(
            model=settings.qwen_model,
            api_key=settings.dashscope_api_key,
            max_retries=settings.dashscope_max_retries,
            model_kwargs=model_kwargs,
        )


class DeepSeekLLMProvider:
    """DeepSeek-backed LLM provider via OpenAI-compatible ChatOpenAI."""

    def chat_model(self, settings: Settings) -> Any:
        """Create a DeepSeek chat model pointed at api.deepseek.com."""

        from langchain_openai import ChatOpenAI

        chat_openai = cast(Any, ChatOpenAI)
        return chat_openai(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            request_timeout=settings.dashscope_request_timeout,
            max_retries=settings.dashscope_max_retries,
        )


DEFAULT_LLM_PROVIDER = DashScopeLLMProvider()


def structured_output_method(settings: Settings) -> str | None:
    """Return the ``with_structured_output`` method for the active provider.

    DeepSeek's OpenAI-compatible API rejects the ``json_schema`` response
    format that ``langchain-openai`` uses by default (HTTP 400 "This
    response_format type is unavailable now). Its thinking mode also rejects
    the forced ``tool_choice`` emitted by the ``function_calling`` method.
    ``json_mode`` requests a plain JSON object without binding a tool, so it is
    compatible with both normal and thinking responses. Other providers keep
    their library default (``None``).
    """

    if settings.llm_provider.strip().lower() == "deepseek":
        return "json_mode"
    return None


def build_structured_chat_model(
    settings: Settings,
    schema: Any,
    provider: LLMProvider | None = None,
) -> Any:
    """Build a chat model bound to a structured-output schema.

    Selects a provider-appropriate structured-output method so DeepSeek does
    not hit either the unsupported ``json_schema`` response format or the
    thinking-mode ``tool_choice`` restriction.
    """

    model = build_chat_model(settings, provider)
    method = structured_output_method(settings)
    if method is not None:
        return model.with_structured_output(schema, method=method)
    return model.with_structured_output(schema)


def build_chat_model(
    settings: Settings,
    provider: LLMProvider | None = None,
) -> Any:
    """Build a chat model through the configured provider seam."""

    if provider is not None:
        return provider.chat_model(settings)

    llm_provider = settings.llm_provider.strip().lower()
    if llm_provider == "deepseek":
        return DeepSeekLLMProvider().chat_model(settings)
    return DashScopeLLMProvider().chat_model(settings)


__all__ = [
    "DEFAULT_LLM_PROVIDER",
    "DashScopeLLMProvider",
    "DeepSeekLLMProvider",
    "LLMProvider",
    "build_chat_model",
    "build_structured_chat_model",
    "structured_output_method",
]
