"""LLM provider seam used by graph nodes and future streaming adapters."""

from __future__ import annotations

from typing import Any, Protocol

from langchain_community.chat_models.tongyi import ChatTongyi

from ..config import Settings


class LLMProvider(Protocol):
    """Protocol for constructing chat model instances."""

    def chat_model(self, settings: Settings) -> Any:
        """Return a chat model configured for the supplied settings."""


class DashScopeLLMProvider:
    """Default DashScope-backed LLM provider."""

    def chat_model(self, settings: Settings) -> ChatTongyi:
        """Create a DashScope chat model with project-level network settings."""

        model_kwargs: dict[str, Any] = {
            "request_timeout": settings.dashscope_request_timeout,
        }
        if settings.dashscope_http_base_url:
            model_kwargs["base_address"] = settings.dashscope_http_base_url

        return ChatTongyi(
            model=settings.qwen_model,
            max_retries=settings.dashscope_max_retries,
            model_kwargs=model_kwargs,
        )


class DeepSeekLLMProvider:
    """DeepSeek-backed LLM provider via OpenAI-compatible ChatOpenAI."""

    def chat_model(self, settings: Settings) -> Any:
        """Create a DeepSeek chat model pointed at api.deepseek.com."""

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            request_timeout=settings.dashscope_request_timeout,
            max_retries=settings.dashscope_max_retries,
        )


DEFAULT_LLM_PROVIDER = DashScopeLLMProvider()


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
]
