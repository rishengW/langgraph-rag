"""Chat-compatible graph node exports."""

from __future__ import annotations

from .._compat import warn_deprecated_import
from ..config import Settings
from ..graph.nodes import (
    build_extractive_answer as _build_extractive_answer,
    chat_question_resolver as _question_from_state,
    condense_question_factory,
    format_history as _format_history,
    latest_user_index as _latest_user_index,
    new_chat_model as _new_chat_model,
)
from ..graph.nodes.common import (
    agent_factory as _shared_agent_factory,
    generate_factory as _shared_generate_factory,
    grade_documents_factory as _shared_grade_documents_factory,
    rewrite_factory as _shared_rewrite_factory,
)
from ..llm.prompts import CONDENSE_PROMPT, RAG_PROMPT
from ..utils.retry import invoke_with_retry as _invoke_with_retry

warn_deprecated_import("src.chat.nodes", "src.graph.nodes")


def agent_factory(settings: Settings, tools):
    return _shared_agent_factory(settings, tools, _question_from_state)


def grade_documents_factory(settings: Settings):
    return _shared_grade_documents_factory(settings, _question_from_state)


def rewrite_factory(settings: Settings):
    return _shared_rewrite_factory(
        settings,
        _question_from_state,
        update_current_question=True,
    )


def generate_factory(settings: Settings):
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
