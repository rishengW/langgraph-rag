from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from src.config import Settings


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-mark tests by filename so unit/integration layers are selectable.

    Files named ``*_integration.py`` carry the ``integration`` marker; no
    manual decorator upkeep is needed as the layering grows.
    """

    for item in items:
        path = Path(str(item.fspath))
        name = path.stem
        if "integration" in name:
            item.add_marker(pytest.mark.integration)


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
