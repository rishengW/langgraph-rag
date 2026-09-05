"""Chat-compatible graph node exports."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool

from src._compat import warn_deprecated_import
from src.backend.graph.nodes import (
    build_extractive_answer as _build_extractive_answer,
)
from src.backend.graph.nodes import (
    chat_question_resolver as _question_from_state,
)
from src.backend.graph.nodes import (
    condense_question_factory,
)
from src.backend.graph.nodes import (
    format_history as _format_history,
)
from src.backend.graph.nodes import (
    latest_user_index as _latest_user_index,
)
from src.backend.graph.nodes import (
    new_chat_model as _new_chat_model,
)
from src.backend.graph.nodes.common import (
    agent_factory as _shared_agent_factory,
)
from src.backend.graph.nodes.common import (
    generate_factory as _shared_generate_factory,
)
from src.backend.graph.nodes.common import (
    grade_documents_factory as _shared_grade_documents_factory,
)
from src.backend.graph.nodes.common import (
    rewrite_factory as _shared_rewrite_factory,
)
from src.backend.llm.prompts import CONDENSE_PROMPT, RAG_PROMPT
from src.config import Settings
from src.utils.retry import invoke_with_retry as _invoke_with_retry

warn_deprecated_import("src.frontend.chat.nodes", "src.backend.graph.nodes")


def agent_factory(
    settings: Settings,
    tools: list[BaseTool],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return _shared_agent_factory(settings, tools, _question_from_state)


def grade_documents_factory(settings: Settings) -> Callable[[dict[str, Any]], str]:
    return _shared_grade_documents_factory(settings, _question_from_state)


def rewrite_factory(settings: Settings) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return _shared_rewrite_factory(
        settings,
        _question_from_state,
        update_current_question=True,
    )


def generate_factory(settings: Settings) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return _shared_generate_factory(settings, _question_from_state)


__all__ = [
    "CONDENSE_PROMPT",
    "RAG_PROMPT",
    "_build_extractive_answer",
    "_format_history",
    "_invoke_with_retry",
    "_latest_user_index",
    "_new_chat_model",
    "_question_from_state",
    "agent_factory",
    "condense_question_factory",
    "generate_factory",
    "grade_documents_factory",
    "rewrite_factory",
]
