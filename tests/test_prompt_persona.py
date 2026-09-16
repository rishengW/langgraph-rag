"""Tests for the configurable persona style blocks and prompt builders."""

from __future__ import annotations

import pytest

from src.backend.llm.prompts import (
    AGENT_SYSTEM_PROMPT,
    RAG_PROMPT,
    TRUMP_AGENT_PERSONA,
    agent_system_prompt,
    rag_prompt,
)
from src.backend.llm.prompts import (
    TRUMP_RAG_PERSONA as _RAG_PERSONA_TEXT,
)
from src.config.loader import _coerce_setting


def test_default_prompts_keep_the_project_persona() -> None:
    assert "Donald Trump" in AGENT_SYSTEM_PROMPT
    assert "Donald Trump" in RAG_PROMPT.format(current_date="2026-01-01", question="q", context="c")


def test_agent_system_prompt_can_omit_the_persona() -> None:
    persona_free = agent_system_prompt("")

    assert "PERSONA" not in persona_free
    assert "Donald Trump" not in persona_free
    assert "{current_date}" in persona_free
    # Tool rules survive persona removal.
    assert "retrieve_source_documents" in persona_free


def test_rag_prompt_can_omit_the_persona() -> None:
    persona_free = rag_prompt("").format(current_date="2026-01-01", question="q", context="c")

    assert "Donald Trump" not in persona_free
    assert "semantic matching" in persona_free.lower()


def test_persona_blocks_match_prompt_defaults() -> None:
    assert TRUMP_AGENT_PERSONA in agent_system_prompt()
    assert _RAG_PERSONA_TEXT in rag_prompt().format(
        current_date="2026-01-01", question="q", context="c"
    )


def test_persona_style_setting_coercion_and_validation() -> None:
    assert _coerce_setting("agent_persona_style", "NONE") == "none"
    assert _coerce_setting("agent_persona_style", None, "trump") == "trump"

    with pytest.raises(ValueError, match="agent_persona_style"):
        _coerce_setting("agent_persona_style", "shakespeare")


def test_settings_field_defaults_to_trump() -> None:
    from dataclasses import fields

    from src.config import Settings

    setting = {field.name: field for field in fields(Settings)}
    assert "agent_persona_style" in setting
    assert setting["agent_persona_style"].default == "trump"
