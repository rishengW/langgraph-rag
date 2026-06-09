from __future__ import annotations

import os

from src.config.loader import (
    load_settings,
    load_yaml_config,
    parse_csv_list,
    parse_optional_int,
    parse_urls,
)
from src.config.settings import (
    DEFAULT_URLS,
    DEFAULT_WEB_SEARCH_JS_FALLBACK_DOMAINS,
    Settings,
)
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
    assert parse_csv_list(" a.test, ,b.test ") == ["a.test", "b.test"]


def test_web_search_and_page_load_defaults_align_with_yaml():
    defaults = load_yaml_config("config/default.yaml")
    settings = Settings(dashscope_api_key="test-key")

    assert settings.web_search_top_k == 6
    assert defaults["web_search_top_k"] == settings.web_search_top_k
    assert settings.web_search_lightweight is True
    assert defaults["web_search_lightweight"] == settings.web_search_lightweight
    assert settings.web_search_max_page_tokens == 8000
    assert defaults["web_search_max_page_tokens"] == settings.web_search_max_page_tokens
    assert settings.web_search_min_page_chars == 200
    assert defaults["web_search_min_page_chars"] == settings.web_search_min_page_chars
    assert settings.web_search_min_page_tokens == 50
    assert defaults["web_search_min_page_tokens"] == settings.web_search_min_page_tokens
    assert settings.web_search_js_fallback_enabled is False
    assert defaults["web_search_js_fallback_enabled"] == settings.web_search_js_fallback_enabled
    assert settings.web_search_js_fallback_domains == DEFAULT_WEB_SEARCH_JS_FALLBACK_DOMAINS
    assert defaults["web_search_js_fallback_domains"] == settings.web_search_js_fallback_domains
    assert settings.web_search_js_force_domains == []
    assert defaults["web_search_js_force_domains"] == settings.web_search_js_force_domains
    assert settings.page_load_max_concurrency == 4
    assert defaults["page_load_max_concurrency"] == settings.page_load_max_concurrency
    assert settings.page_load_cache_ttl_seconds == 0
    assert defaults["page_load_cache_ttl_seconds"] == settings.page_load_cache_ttl_seconds
    assert settings.document_quality_filter_enabled is True
    assert defaults["document_quality_filter_enabled"] == settings.document_quality_filter_enabled
    assert settings.document_quality_min_text_length == 80
    assert defaults["document_quality_min_text_length"] == settings.document_quality_min_text_length
    assert settings.document_quality_min_unique_terms == 8
    assert (
        defaults["document_quality_min_unique_terms"] == settings.document_quality_min_unique_terms
    )
    assert settings.document_quality_relevance_query == ""
    assert defaults["document_quality_relevance_query"] == settings.document_quality_relevance_query
    assert settings.document_quality_query_min_overlap == 1
    assert (
        defaults["document_quality_query_min_overlap"]
        == settings.document_quality_query_min_overlap
    )


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


def test_load_yaml_config_supports_flat_values_and_lists(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "qwen_model: qwen-yaml",
                "web_search_enabled: false",
                "api_port: 9000",
                "source_urls:",
                "  - https://yaml-a.test",
                "  - https://yaml-b.test",
            ]
        ),
        encoding="utf-8",
    )

    assert load_yaml_config(config_file) == {
        "qwen_model": "qwen-yaml",
        "web_search_enabled": False,
        "api_port": 9000,
        "source_urls": ["https://yaml-a.test", "https://yaml-b.test"],
    }


def test_load_settings_precedence_cli_env_yaml_defaults(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "qwen_model: qwen-yaml",
                "api_port: 9000",
                "web_search_enabled: false",
                "source_urls:",
                "  - https://yaml.test",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("QWEN_MODEL", "qwen-env")

    settings = load_settings(
        env_file=tmp_path / ".env-missing",
        config_file=config_file,
        urls=["https://cli-url.test"],
        overrides={"api_port": 9100},
    )

    assert settings.dashscope_api_key == "env-secret"
    assert settings.qwen_model == "qwen-env"
    assert settings.api_port == 9100
    assert settings.web_search_enabled is False
    assert settings.source_urls == ["https://cli-url.test"]


def test_load_settings_accepts_page_load_max_concurrency_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("PAGE_LOAD_MAX_CONCURRENCY", "0")
    monkeypatch.setenv("PAGE_LOAD_CACHE_TTL_SECONDS", "-10")
    monkeypatch.setenv("WEB_SEARCH_MAX_PAGE_TOKENS", "-50")
    monkeypatch.setenv("WEB_SEARCH_MIN_PAGE_CHARS", "-50")
    monkeypatch.setenv("WEB_SEARCH_MIN_PAGE_TOKENS", "-1")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.page_load_max_concurrency == 1
    assert settings.page_load_cache_ttl_seconds == 0
    assert settings.web_search_max_page_tokens == 0
    assert settings.web_search_min_page_chars == 0
    assert settings.web_search_min_page_tokens == 0


def test_load_settings_accepts_lightweight_web_search_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("WEB_SEARCH_LIGHTWEIGHT", "false")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.web_search_lightweight is False


def test_load_settings_accepts_web_search_js_policy_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("WEB_SEARCH_JS_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("WEB_SEARCH_JS_FALLBACK_DOMAINS", "Example.COM,sub.test")
    monkeypatch.setenv("WEB_SEARCH_JS_FORCE_DOMAINS", "force.test")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.web_search_js_fallback_enabled is True
    assert settings.web_search_js_fallback_domains == ["example.com", "sub.test"]
    assert settings.web_search_js_force_domains == ["force.test"]


def test_load_settings_accepts_web_search_js_policy_yaml(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "web_search_js_fallback_enabled: true",
                "web_search_js_fallback_domains:",
                "  - spa.example.com",
                "  - Baike.Baidu.com",
                "web_search_js_force_domains: [force.example.com]",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=config_file)

    assert settings.web_search_js_fallback_enabled is True
    assert settings.web_search_js_fallback_domains == [
        "spa.example.com",
        "baike.baidu.com",
    ]
    assert settings.web_search_js_force_domains == ["force.example.com"]


def test_load_settings_accepts_web_search_readability_thresholds_from_yaml(
    tmp_path,
    monkeypatch,
):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "web_search_min_page_chars: 123",
                "web_search_min_page_tokens: 17",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=config_file)

    assert settings.web_search_min_page_chars == 123
    assert settings.web_search_min_page_tokens == 17


def test_load_settings_accepts_page_load_cache_ttl_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("PAGE_LOAD_CACHE_TTL_SECONDS", "45")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.page_load_cache_ttl_seconds == 45


def test_load_settings_accepts_document_quality_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("DOCUMENT_QUALITY_FILTER_ENABLED", "false")
    monkeypatch.setenv("DOCUMENT_QUALITY_MIN_TEXT_LENGTH", "-1")
    monkeypatch.setenv("DOCUMENT_QUALITY_MIN_UNIQUE_TERMS", "0")
    monkeypatch.setenv("DOCUMENT_QUALITY_RELEVANCE_QUERY", "langgraph retrieval")
    monkeypatch.setenv("DOCUMENT_QUALITY_QUERY_MIN_OVERLAP", "-5")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.document_quality_filter_enabled is False
    assert settings.document_quality_min_text_length == 0
    assert settings.document_quality_min_unique_terms == 0
    assert settings.document_quality_relevance_query == "langgraph retrieval"
    assert settings.document_quality_query_min_overlap == 0
