from __future__ import annotations

import os

from src.config.loader import load_settings, parse_optional_int, parse_urls
from src.config.settings import DEFAULT_URLS, Settings
from src.core.config import Settings as CoreSettings


def test_settings_reexport_preserves_old_import_path():
    assert CoreSettings is Settings


def test_parse_helpers():
    assert parse_urls(None) == DEFAULT_URLS
    assert parse_urls(" https://a.test, ,https://b.test ") == [
        "https://a.test",
        "https://b.test",
    ]
    assert parse_optional_int(None, 12) == 12
    assert parse_optional_int("", 12) is None
    assert parse_optional_int("42", 12) == 42


def test_load_settings_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "DASHSCOPE_API_KEY=test-key",
                "QWEN_MODEL=qwen-test",
                "SOURCE_URLS=https://a.test,https://b.test",
                "WEB_SEARCH_ENABLED=false",
                "MAX_REWRITES=0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    settings = load_settings(env_file)

    assert settings.dashscope_api_key == "test-key"
    assert settings.qwen_model == "qwen-test"
    assert settings.source_urls == ["https://a.test", "https://b.test"]
    assert settings.web_search_enabled is False
    assert settings.max_rewrites == 0
    assert os.environ["DASHSCOPE_API_KEY"] == "test-key"

