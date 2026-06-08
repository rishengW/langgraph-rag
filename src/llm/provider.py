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


DEFAULT_LLM_PROVIDER = DashScopeLLMProvider()


def build_chat_model(
    settings: Settings,
    provider: LLMProvider = DEFAULT_LLM_PROVIDER,
) -> Any:
    """Build a chat model through the configured provider seam."""

    return provider.chat_model(settings)


__all__ = [
    "DEFAULT_LLM_PROVIDER",
    "DashScopeLLMProvider",
    "LLMProvider",
    "build_chat_model",
]
