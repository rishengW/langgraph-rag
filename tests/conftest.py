from __future__ import annotations

from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage

from src.config import Settings


@pytest.fixture
def mock_settings(tmp_path):
    return Settings(
        dashscope_api_key="test-key",
        chroma_dir=tmp_path / "chroma",
        source_urls=["https://example.com/a"],
        dashscope_max_retries=1,
    )


@pytest.fixture
def isolated_settings(mock_settings):
    def _make(**kwargs):
        return replace(mock_settings, **kwargs)

    return _make


@pytest.fixture
def ai_message():
    def _make(content: str):
        return AIMessage(content=content)

    return _make

