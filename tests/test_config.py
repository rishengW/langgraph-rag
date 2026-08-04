from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from src.config.loader import (
    _settings_defaults,
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

    assert settings.chat_context_max_turns == 8
    assert defaults["chat_context_max_turns"] == settings.chat_context_max_turns
    assert settings.chat_context_max_chars == 12000
    assert defaults["chat_context_max_chars"] == settings.chat_context_max_chars
    assert settings.web_search_top_k == 6
    assert defaults["web_search_top_k"] == settings.web_search_top_k
    assert settings.web_search_providers == []
    assert defaults["web_search_providers"] == settings.web_search_providers
    assert settings.web_search_provider_fanout == 2
    assert defaults["web_search_provider_fanout"] == settings.web_search_provider_fanout
    assert settings.web_search_provider_timeout_seconds == 8
    assert (
        defaults["web_search_provider_timeout_seconds"]
        == settings.web_search_provider_timeout_seconds
    )
    assert settings.web_search_api_timeout_seconds == 20
    assert defaults["web_search_api_timeout_seconds"] == settings.web_search_api_timeout_seconds
    assert settings.web_search_deadline_seconds == 30
    assert defaults["web_search_deadline_seconds"] == settings.web_search_deadline_seconds
    assert settings.web_search_llm_query_rewrite_enabled is False
    assert (
        defaults["web_search_llm_query_rewrite_enabled"]
        == settings.web_search_llm_query_rewrite_enabled
    )
    assert settings.web_search_min_url_score == 45
    assert defaults["web_search_min_url_score"] == settings.web_search_min_url_score
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
    assert settings.weather_enabled is False
    assert defaults["weather_enabled"] == settings.weather_enabled
    assert settings.stock_enabled is False
    assert defaults["stock_enabled"] == settings.stock_enabled
    assert settings.currency_enabled is False
    assert defaults["currency_enabled"] == settings.currency_enabled
    assert settings.wikipedia_enabled is False
    assert defaults["wikipedia_enabled"] == settings.wikipedia_enabled
    assert settings.wikipedia_max_summary_chars == 1500
    assert defaults["wikipedia_max_summary_chars"] == settings.wikipedia_max_summary_chars
    assert defaults["wikipedia_user_agent"] == settings.wikipedia_user_agent
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
    assert settings.document_quality_min_similarity == 0.5
    assert defaults["document_quality_min_similarity"] == settings.document_quality_min_similarity
    assert settings.document_quality_recency_bias_days == 365
    assert (
        defaults["document_quality_recency_bias_days"]
        == settings.document_quality_recency_bias_days
    )
    assert settings.rerank_strategy == "lexical"
    assert defaults["rerank_strategy"] == settings.rerank_strategy
    assert settings.amap_api_timeout_seconds == 10
    assert defaults["amap_api_timeout_seconds"] == settings.amap_api_timeout_seconds
    assert "amap_web_service_key" not in defaults
    assert "amap_js_api_key" not in defaults
    assert "amap_js_security_code" not in defaults


def test_amap_secret_settings_are_env_only_and_timeout_is_configurable(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "amap_web_service_key: yaml-secret",
                "amap_js_api_key: yaml-js-secret",
                "amap_js_security_code: yaml-security-secret",
                "amap_api_timeout_seconds: 0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("AMAP_WEB_SERVICE_KEY", "env-amap-web")
    monkeypatch.setenv("AMAP_JS_API_KEY", "env-amap-js")
    monkeypatch.setenv("AMAP_JS_SECURITY_CODE", "env-amap-security")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=config_file)
    defaults = _settings_defaults()

    assert settings.amap_web_service_key == "env-amap-web"
    assert settings.amap_js_api_key == "env-amap-js"
    assert settings.amap_js_security_code == "env-amap-security"
    assert settings.amap_api_timeout_seconds == 1
    assert "amap_web_service_key" not in defaults
    assert "amap_js_api_key" not in defaults
    assert "amap_js_security_code" not in defaults
    assert defaults["amap_api_timeout_seconds"] == 10


def test_env_example_documents_amap_settings():
    env_example = Path(".env.example").read_text(encoding="utf-8")

    for name in (
        "AMAP_WEB_SERVICE_KEY",
        "AMAP_JS_API_KEY",
        "AMAP_JS_SECURITY_CODE",
        "AMAP_API_TIMEOUT_SECONDS",
    ):
        assert name in env_example, f"{name} missing from .env.example"


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
    monkeypatch.setenv("WEB_SEARCH_MIN_URL_SCORE", "-10")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("WEB_SEARCH_DEADLINE_SECONDS", "-10")
    monkeypatch.setenv("AMAP_API_TIMEOUT_SECONDS", "0")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.page_load_max_concurrency == 1
    assert settings.page_load_cache_ttl_seconds == 0
    assert settings.web_search_max_page_tokens == 0
    assert settings.web_search_min_page_chars == 0
    assert settings.web_search_min_page_tokens == 0
    assert settings.web_search_min_url_score == 0
    assert settings.web_search_provider_timeout_seconds == 1
    assert settings.web_search_deadline_seconds == 1
    assert settings.amap_api_timeout_seconds == 1


def test_load_settings_accepts_and_clamps_chat_context_bounds(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("CHAT_CONTEXT_MAX_TURNS", "0")
    monkeypatch.setenv("CHAT_CONTEXT_MAX_CHARS", "-50")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.chat_context_max_turns == 1
    assert settings.chat_context_max_chars == 1


def test_load_settings_accepts_lightweight_web_search_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("WEB_SEARCH_LIGHTWEIGHT", "false")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.web_search_lightweight is False


def test_load_settings_accepts_llm_query_rewrite_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("WEB_SEARCH_LLM_QUERY_REWRITE_ENABLED", "true")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.web_search_llm_query_rewrite_enabled is True


def test_load_settings_accepts_web_search_js_policy_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("WEB_SEARCH_JS_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("WEB_SEARCH_JS_FALLBACK_DOMAINS", "Example.COM,sub.test")
    monkeypatch.setenv("WEB_SEARCH_JS_FORCE_DOMAINS", "force.test")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.web_search_js_fallback_enabled is True
    assert settings.web_search_js_fallback_domains == ["example.com", "sub.test"]
    assert settings.web_search_js_force_domains == ["force.test"]


def test_load_settings_accepts_agent_tool_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("WEATHER_ENABLED", "true")
    monkeypatch.setenv("STOCK_ENABLED", "true")
    monkeypatch.setenv("CURRENCY_ENABLED", "true")
    monkeypatch.setenv("WIKIPEDIA_ENABLED", "true")
    monkeypatch.setenv("WIKIPEDIA_MAX_SUMMARY_CHARS", "-50")
    monkeypatch.setenv("WIKIPEDIA_USER_AGENT", "test-agent/1.0 (contact: tests)")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.weather_enabled is True
    assert settings.stock_enabled is True
    assert settings.currency_enabled is True
    assert settings.wikipedia_enabled is True
    assert settings.wikipedia_max_summary_chars == 0
    assert settings.wikipedia_user_agent == "test-agent/1.0 (contact: tests)"


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
                "web_search_min_url_score: 70",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=config_file)

    assert settings.web_search_min_page_chars == 123
    assert settings.web_search_min_page_tokens == 17
    assert settings.web_search_min_url_score == 70


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
    monkeypatch.setenv("DOCUMENT_QUALITY_MIN_SIMILARITY", "0.75")
    monkeypatch.setenv("DOCUMENT_QUALITY_RECENCY_BIAS_DAYS", "-7")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.document_quality_filter_enabled is False
    assert settings.document_quality_min_text_length == 0
    assert settings.document_quality_min_unique_terms == 0
    assert settings.document_quality_relevance_query == "langgraph retrieval"
    assert settings.document_quality_query_min_overlap == 0
    assert settings.document_quality_min_similarity == 0.75
    assert settings.document_quality_recency_bias_days == 0


def test_load_settings_accepts_rerank_strategy_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("RERANK_STRATEGY", "HyBrId")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.rerank_strategy == "hybrid"


def test_load_settings_deepseek_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.llm_provider == "deepseek"
    assert settings.deepseek_api_key == "deepseek-key"
    assert settings.deepseek_model == "deepseek-v4-pro"
    assert settings.deepseek_base_url == "https://api.deepseek.com"


def test_load_settings_deepseek_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")

    import pytest as pytest_mod

    with pytest_mod.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        load_settings(env_file=tmp_path / ".env-missing", config_file=None)


def test_load_settings_defaults_to_dashscope(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.llm_provider == "dashscope"
    assert settings.deepseek_api_key == ""


# ---- long-term memory settings ---------------------------------------------

MEMORY_SETTING_NAMES = (
    "memory_enabled",
    "memory_store_path",
    "memory_max_records",
    "memory_max_record_chars",
    "memory_recall_top_k",
    "memory_context_max_chars",
    "memory_default_scope",
    "memory_auto_recall_enabled",
)


def test_memory_defaults_align_with_yaml():
    defaults = load_yaml_config("config/default.yaml")
    settings = Settings(dashscope_api_key="test-key")

    assert settings.memory_enabled is False
    assert settings.memory_store_path == ""
    assert settings.memory_max_records == 500
    assert settings.memory_max_record_chars == 1000
    assert settings.memory_recall_top_k == 5
    assert settings.memory_context_max_chars == 2000
    assert settings.memory_default_scope == "global"
    assert settings.memory_auto_recall_enabled is True

    for name in MEMORY_SETTING_NAMES:
        assert name in defaults, f"{name} missing from config/default.yaml"
        assert defaults[name] == getattr(settings, name), name


def test_memory_settings_are_documented():
    env_example = Path(".env.example").read_text(encoding="utf-8")
    yaml_text = Path("config/default.yaml").read_text(encoding="utf-8")

    for name in MEMORY_SETTING_NAMES:
        assert name.upper() in env_example, f"{name.upper()} missing from .env.example"
        assert name in yaml_text, f"{name} missing from config/default.yaml"


def test_load_settings_accepts_memory_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("MEMORY_ENABLED", "true")
    monkeypatch.setenv("MEMORY_STORE_PATH", "custom/memories.json")
    monkeypatch.setenv("MEMORY_MAX_RECORDS", "42")
    monkeypatch.setenv("MEMORY_MAX_RECORD_CHARS", "256")
    monkeypatch.setenv("MEMORY_RECALL_TOP_K", "3")
    monkeypatch.setenv("MEMORY_CONTEXT_MAX_CHARS", "900")
    monkeypatch.setenv("MEMORY_DEFAULT_SCOPE", "  SESSION ")
    monkeypatch.setenv("MEMORY_AUTO_RECALL_ENABLED", "false")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.memory_enabled is True
    assert settings.memory_store_path == "custom/memories.json"
    assert settings.memory_max_records == 42
    assert settings.memory_max_record_chars == 256
    assert settings.memory_recall_top_k == 3
    assert settings.memory_context_max_chars == 900
    assert settings.memory_default_scope == "session"
    assert settings.memory_auto_recall_enabled is False


@pytest.mark.parametrize("raw,expected", [
    ("true", True),
    ("1", True),
    ("yes", True),
    ("on", True),
    ("TRUE", True),
    ("false", False),
    ("0", False),
    ("maybe", False),
    ("", False),
])
def test_memory_boolean_accepted_values(tmp_path, monkeypatch, raw, expected):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("MEMORY_ENABLED", raw)

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.memory_enabled is expected


@pytest.mark.parametrize("env_name,raw,low,high", [
    ("MEMORY_MAX_RECORDS", "0", 1, 10000),
    ("MEMORY_MAX_RECORDS", "10001", 1, 10000),
    ("MEMORY_MAX_RECORD_CHARS", "0", 1, 10000),
    ("MEMORY_MAX_RECORD_CHARS", "10001", 1, 10000),
    ("MEMORY_RECALL_TOP_K", "0", 1, 50),
    ("MEMORY_RECALL_TOP_K", "51", 1, 50),
    ("MEMORY_CONTEXT_MAX_CHARS", "-1", 1, 20000),
    ("MEMORY_CONTEXT_MAX_CHARS", "20001", 1, 20000),
])
def test_load_settings_rejects_out_of_range_memory_ints(
    tmp_path, monkeypatch, env_name, raw, low, high
):
    """Out-of-range memory bounds fail the load instead of being clamped."""

    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv(env_name, raw)

    with pytest.raises(ValueError) as excinfo:
        load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    message = str(excinfo.value)
    assert env_name.lower() in message
    assert raw.lstrip("-") in message
    assert str(low) in message and str(high) in message


@pytest.mark.parametrize("env_name", [
    "MEMORY_MAX_RECORDS",
    "MEMORY_MAX_RECORD_CHARS",
    "MEMORY_RECALL_TOP_K",
    "MEMORY_CONTEXT_MAX_CHARS",
])
def test_load_settings_rejects_non_integer_memory_ints(tmp_path, monkeypatch, env_name):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv(env_name, "not-a-number")

    with pytest.raises(ValueError) as excinfo:
        load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    message = str(excinfo.value)
    assert env_name.lower() in message
    assert "not-a-number" in message


def test_load_settings_rejects_invalid_memory_default_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("MEMORY_DEFAULT_SCOPE", "team")

    with pytest.raises(ValueError) as excinfo:
        load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    message = str(excinfo.value)
    assert "memory_default_scope" in message
    assert "global" in message and "session" in message


def test_memory_recall_top_k_may_exceed_max_records(tmp_path, monkeypatch):
    """A top_k above the cap is allowed; recall simply cannot return more."""

    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("MEMORY_MAX_RECORDS", "2")
    monkeypatch.setenv("MEMORY_RECALL_TOP_K", "10")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.memory_max_records == 2
    assert settings.memory_recall_top_k == 10


def test_memory_settings_from_yaml(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "memory_enabled: true",
                "memory_max_records: 77",
                "memory_default_scope: session",
                "memory_auto_recall_enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=config_file)

    assert settings.memory_enabled is True
    assert settings.memory_max_records == 77
    assert settings.memory_default_scope == "session"
    assert settings.memory_auto_recall_enabled is False


# ---- automatic memory extraction settings ----------------------------------

EXTRACTION_SETTING_NAMES = (
    "memory_extraction_enabled",
    "memory_extraction_on_session_start",
    "memory_extraction_turn_interval",
    "memory_extraction_max_candidates",
    "memory_extraction_max_transcript_chars",
    "memory_extraction_timeout_seconds",
    "memory_extraction_max_concurrency",
    "memory_extraction_max_session_age_hours",
)

EXTRACTION_INT_RANGES = {
    "MEMORY_EXTRACTION_TURN_INTERVAL": (1, 1000),
    "MEMORY_EXTRACTION_MAX_CANDIDATES": (1, 20),
    "MEMORY_EXTRACTION_MAX_TRANSCRIPT_CHARS": (200, 100000),
    "MEMORY_EXTRACTION_TIMEOUT_SECONDS": (1, 600),
    "MEMORY_EXTRACTION_MAX_CONCURRENCY": (1, 16),
    "MEMORY_EXTRACTION_MAX_SESSION_AGE_HOURS": (1, 8760),
}


def test_extraction_defaults_align_with_yaml():
    defaults = load_yaml_config("config/default.yaml")
    settings = Settings(dashscope_api_key="test-key")

    assert settings.memory_extraction_enabled is False
    assert settings.memory_extraction_on_session_start is True
    assert settings.memory_extraction_turn_interval == 10
    assert settings.memory_extraction_max_candidates == 5
    assert settings.memory_extraction_max_transcript_chars == 8000
    assert settings.memory_extraction_timeout_seconds == 60
    assert settings.memory_extraction_max_concurrency == 2
    assert settings.memory_extraction_max_session_age_hours == 168

    for name in EXTRACTION_SETTING_NAMES:
        assert name in defaults, f"{name} missing from config/default.yaml"
        assert defaults[name] == getattr(settings, name), name


def test_extraction_settings_are_documented():
    env_example = Path(".env.example").read_text(encoding="utf-8")
    yaml_text = Path("config/default.yaml").read_text(encoding="utf-8")

    for name in EXTRACTION_SETTING_NAMES:
        assert name.upper() in env_example, f"{name.upper()} missing from .env.example"
        assert name in yaml_text, f"{name} missing from config/default.yaml"


def test_env_example_documents_extraction_ranges():
    env_example = Path(".env.example").read_text(encoding="utf-8")

    for env_name, (low, high) in EXTRACTION_INT_RANGES.items():
        assert f"{low}-{high}" in env_example, f"range for {env_name} not documented"


def test_load_settings_accepts_extraction_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "true")
    monkeypatch.setenv("MEMORY_EXTRACTION_ON_SESSION_START", "false")
    monkeypatch.setenv("MEMORY_EXTRACTION_TURN_INTERVAL", "4")
    monkeypatch.setenv("MEMORY_EXTRACTION_MAX_CANDIDATES", "7")
    monkeypatch.setenv("MEMORY_EXTRACTION_MAX_TRANSCRIPT_CHARS", "1200")
    monkeypatch.setenv("MEMORY_EXTRACTION_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("MEMORY_EXTRACTION_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("MEMORY_EXTRACTION_MAX_SESSION_AGE_HOURS", "24")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.memory_extraction_enabled is True
    assert settings.memory_extraction_on_session_start is False
    assert settings.memory_extraction_turn_interval == 4
    assert settings.memory_extraction_max_candidates == 7
    assert settings.memory_extraction_max_transcript_chars == 1200
    assert settings.memory_extraction_timeout_seconds == 15
    assert settings.memory_extraction_max_concurrency == 3
    assert settings.memory_extraction_max_session_age_hours == 24


@pytest.mark.parametrize("env_name", [
    "MEMORY_EXTRACTION_ENABLED",
    "MEMORY_EXTRACTION_ON_SESSION_START",
])
@pytest.mark.parametrize("raw,expected", [
    ("true", True),
    ("1", True),
    ("YES", True),
    ("on", True),
    ("false", False),
    ("0", False),
    ("maybe", False),
    ("", False),
])
def test_extraction_boolean_accepted_values(
    tmp_path, monkeypatch, env_name, raw, expected
):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv(env_name, raw)

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert getattr(settings, env_name.lower()) is expected


@pytest.mark.parametrize("env_name,low,high", [
    (name, low, high) for name, (low, high) in EXTRACTION_INT_RANGES.items()
])
def test_load_settings_rejects_out_of_range_extraction_ints(
    tmp_path, monkeypatch, env_name, low, high
):
    """Out-of-range extraction bounds fail the load instead of being clamped."""

    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")

    for offending in (str(low - 1), str(high + 1)):
        monkeypatch.setenv(env_name, offending)

        with pytest.raises(ValueError) as excinfo:
            load_settings(env_file=tmp_path / ".env-missing", config_file=None)

        message = str(excinfo.value)
        assert env_name.lower() in message
        assert offending.lstrip("-") in message
        assert str(low) in message and str(high) in message


@pytest.mark.parametrize("env_name", sorted(EXTRACTION_INT_RANGES))
def test_load_settings_rejects_non_integer_extraction_ints(
    tmp_path, monkeypatch, env_name
):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv(env_name, "not-a-number")

    with pytest.raises(ValueError) as excinfo:
        load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    message = str(excinfo.value)
    assert env_name.lower() in message
    assert "not-a-number" in message


def test_transcript_and_record_char_bounds_are_independent(tmp_path, monkeypatch):
    """No cross-field constraint: the two caps bound different things."""

    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")
    monkeypatch.setenv("MEMORY_EXTRACTION_MAX_TRANSCRIPT_CHARS", "200")
    monkeypatch.setenv("MEMORY_MAX_RECORD_CHARS", "10000")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=None)

    assert settings.memory_extraction_max_transcript_chars == 200
    assert settings.memory_max_record_chars == 10000


def test_extraction_settings_from_yaml(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "memory_extraction_enabled: true",
                "memory_extraction_on_session_start: false",
                "memory_extraction_turn_interval: 3",
                "memory_extraction_max_concurrency: 1",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-secret")

    settings = load_settings(env_file=tmp_path / ".env-missing", config_file=config_file)

    assert settings.memory_extraction_enabled is True
    assert settings.memory_extraction_on_session_start is False
    assert settings.memory_extraction_turn_interval == 3
    assert settings.memory_extraction_max_concurrency == 1


def test_no_extraction_model_setting_exists():
    """A dedicated extraction model is out of scope for this version."""

    names = {field.name for field in dataclasses.fields(Settings)}
    assert not {n for n in names if n.startswith("memory_extraction") and "model" in n}
