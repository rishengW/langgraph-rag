from __future__ import annotations

import hashlib
import os
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ..utils.networking import (
    DEFAULT_DASHSCOPE_HTTP_BASE_URL,
    configure_dashscope_base_url,
    ensure_user_agent,
    parse_dashscope_base_url,
)
from .settings import DEFAULT_URLS, Settings

DEFAULT_CONFIG_DIR = Path("config")
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "default.yaml"
DEFAULT_ENVIRONMENT = "development"

SETTING_ENV_NAMES = {
    "qwen_model": "QWEN_MODEL",
    "embedding_model": "EMBEDDING_MODEL",
    "embedding_dimension": "EMBEDDING_DIMENSION",
    "embedding_batch_size": "EMBEDDING_BATCH_SIZE",
    "chroma_dir": "CHROMA_DIR",
    "collection_name": "COLLECTION_NAME",
    "chunk_size": "CHUNK_SIZE",
    "chunk_overlap": "CHUNK_OVERLAP",
    "source_urls": "SOURCE_URLS",
    "langchain_tracing_v2": "LANGCHAIN_TRACING_V2",
    "langchain_api_key": "LANGCHAIN_API_KEY",
    "langchain_project": "LANGCHAIN_PROJECT",
    "api_key": "API_KEY",
    "api_host": "API_HOST",
    "api_port": "API_PORT",
    "cors_allow_origins": "CORS_ALLOW_ORIGINS",
    "allow_low_relevance_generate": "ALLOW_LOW_RELEVANCE_GENERATE",
    "min_keyword_matches": "MIN_KEYWORD_MATCHES",
    "max_rewrites": "MAX_REWRITES",
    "chat_context_max_turns": "CHAT_CONTEXT_MAX_TURNS",
    "chat_context_max_chars": "CHAT_CONTEXT_MAX_CHARS",
    "web_search_enabled": "WEB_SEARCH_ENABLED",
    "web_search_llm_query_rewrite_enabled": "WEB_SEARCH_LLM_QUERY_REWRITE_ENABLED",
    "web_search_provider": "WEB_SEARCH_PROVIDER",
    "web_search_providers": "WEB_SEARCH_PROVIDERS",
    "web_search_provider_fanout": "WEB_SEARCH_PROVIDER_FANOUT",
    "web_search_max_results": "WEB_SEARCH_MAX_RESULTS",
    "web_search_provider_timeout_seconds": "WEB_SEARCH_PROVIDER_TIMEOUT_SECONDS",
    "web_search_api_timeout_seconds": "WEB_SEARCH_API_TIMEOUT_SECONDS",
    "web_search_deadline_seconds": "WEB_SEARCH_DEADLINE_SECONDS",
    "web_search_top_k": "WEB_SEARCH_TOP_K",
    "web_search_min_url_score": "WEB_SEARCH_MIN_URL_SCORE",
    "web_search_region": "WEB_SEARCH_REGION",
    "web_search_timelimit": "WEB_SEARCH_TIMELIMIT",
    "web_search_verify_ssl": "WEB_SEARCH_VERIFY_SSL",
    "web_search_lightweight": "WEB_SEARCH_LIGHTWEIGHT",
    "web_search_max_page_tokens": "WEB_SEARCH_MAX_PAGE_TOKENS",
    "web_search_min_page_chars": "WEB_SEARCH_MIN_PAGE_CHARS",
    "web_search_min_page_tokens": "WEB_SEARCH_MIN_PAGE_TOKENS",
    "web_search_js_fallback_enabled": "WEB_SEARCH_JS_FALLBACK_ENABLED",
    "web_search_js_fallback_domains": "WEB_SEARCH_JS_FALLBACK_DOMAINS",
    "web_search_js_force_domains": "WEB_SEARCH_JS_FORCE_DOMAINS",
    "web_search_js_retry_budget": "WEB_SEARCH_JS_RETRY_BUDGET",
    "web_search_structure_filter_enabled": "WEB_SEARCH_STRUCTURE_FILTER_ENABLED",
    "web_search_max_link_density": "WEB_SEARCH_MAX_LINK_DENSITY",
    "web_search_min_content_words": "WEB_SEARCH_MIN_CONTENT_WORDS",
    "web_search_semantic_filter_enabled": "WEB_SEARCH_SEMANTIC_FILTER_ENABLED",
    "web_search_semantic_model": "WEB_SEARCH_SEMANTIC_MODEL",
    "web_search_semantic_min_similarity": "WEB_SEARCH_SEMANTIC_MIN_SIMILARITY",
    "web_search_domain_reputation_enabled": "WEB_SEARCH_DOMAIN_REPUTATION_ENABLED",
    "web_search_domain_reputation_min_samples": "WEB_SEARCH_DOMAIN_REPUTATION_MIN_SAMPLES",
    "serper_api_key": "SERPER_API_KEY",
    "brave_search_api_key": "BRAVE_SEARCH_API_KEY",
    "tavily_api_key": "TAVILY_API_KEY",
    "bing_search_api_key": "BING_SEARCH_API_KEY",
    "bing_search_endpoint": "BING_SEARCH_ENDPOINT",
    "weather_enabled": "WEATHER_ENABLED",
    "stock_enabled": "STOCK_ENABLED",
    "currency_enabled": "CURRENCY_ENABLED",
    "wikipedia_enabled": "WIKIPEDIA_ENABLED",
    "directions_enabled": "DIRECTIONS_ENABLED",
    "map_enabled": "MAP_ENABLED",
    "math_enabled": "MATH_ENABLED",
    "statistics_enabled": "STATISTICS_ENABLED",
    "linalg_enabled": "LINALG_ENABLED",
    "number_theory_enabled": "NUMBER_THEORY_ENABLED",
    "datetime_enabled": "DATETIME_ENABLED",
    "summarize_url_enabled": "SUMMARIZE_URL_ENABLED",
    "file_read_enabled": "FILE_READ_ENABLED",
    "file_read_root": "FILE_READ_ROOT",
    "file_read_max_bytes": "FILE_READ_MAX_BYTES",
    "memory_enabled": "MEMORY_ENABLED",
    "memory_store_path": "MEMORY_STORE_PATH",
    "memory_max_records": "MEMORY_MAX_RECORDS",
    "memory_max_record_chars": "MEMORY_MAX_RECORD_CHARS",
    "memory_recall_top_k": "MEMORY_RECALL_TOP_K",
    "memory_context_max_chars": "MEMORY_CONTEXT_MAX_CHARS",
    "memory_default_scope": "MEMORY_DEFAULT_SCOPE",
    "memory_auto_recall_enabled": "MEMORY_AUTO_RECALL_ENABLED",
    "memory_extraction_enabled": "MEMORY_EXTRACTION_ENABLED",
    "memory_extraction_on_session_start": "MEMORY_EXTRACTION_ON_SESSION_START",
    "memory_extraction_turn_interval": "MEMORY_EXTRACTION_TURN_INTERVAL",
    "memory_extraction_max_candidates": "MEMORY_EXTRACTION_MAX_CANDIDATES",
    "memory_extraction_max_transcript_chars": "MEMORY_EXTRACTION_MAX_TRANSCRIPT_CHARS",
    "memory_extraction_timeout_seconds": "MEMORY_EXTRACTION_TIMEOUT_SECONDS",
    "memory_extraction_max_concurrency": "MEMORY_EXTRACTION_MAX_CONCURRENCY",
    "memory_extraction_max_session_age_hours": "MEMORY_EXTRACTION_MAX_SESSION_AGE_HOURS",
    "wikipedia_max_summary_chars": "WIKIPEDIA_MAX_SUMMARY_CHARS",
    "wikipedia_user_agent": "WIKIPEDIA_USER_AGENT",
    "page_load_timeout": "PAGE_LOAD_TIMEOUT",
    "page_load_max_concurrency": "PAGE_LOAD_MAX_CONCURRENCY",
    "page_load_cache_ttl_seconds": "PAGE_LOAD_CACHE_TTL_SECONDS",
    "document_quality_filter_enabled": "DOCUMENT_QUALITY_FILTER_ENABLED",
    "document_quality_min_text_length": "DOCUMENT_QUALITY_MIN_TEXT_LENGTH",
    "document_quality_min_unique_terms": "DOCUMENT_QUALITY_MIN_UNIQUE_TERMS",
    "document_quality_relevance_query": "DOCUMENT_QUALITY_RELEVANCE_QUERY",
    "document_quality_query_min_overlap": "DOCUMENT_QUALITY_QUERY_MIN_OVERLAP",
    "document_quality_min_similarity": "DOCUMENT_QUALITY_MIN_SIMILARITY",
    "document_quality_recency_bias_days": "DOCUMENT_QUALITY_RECENCY_BIAS_DAYS",
    "rerank_strategy": "RERANK_STRATEGY",
    "llm_provider": "LLM_PROVIDER",
    "deepseek_model": "DEEPSEEK_MODEL",
    "deepseek_base_url": "DEEPSEEK_BASE_URL",
    "dashscope_request_timeout": "DASHSCOPE_REQUEST_TIMEOUT",
    "dashscope_max_retries": "DASHSCOPE_MAX_RETRIES",
    "dashscope_http_base_url": "DASHSCOPE_HTTP_BASE_URL",
}


# Long-term memory bounds. Unlike the other integer settings, these fail the
# load instead of clamping: a nonsensical cap should surface at startup rather
# than silently become 1. Same policy as ``rerank_strategy`` below.
_MEMORY_INT_RANGES: dict[str, tuple[int, int]] = {
    "memory_max_records": (1, 10_000),
    "memory_max_record_chars": (1, 10_000),
    "memory_recall_top_k": (1, 50),
    "memory_context_max_chars": (1, 20_000),
    "memory_extraction_turn_interval": (1, 1_000),
    "memory_extraction_max_candidates": (1, 20),
    "memory_extraction_max_transcript_chars": (200, 100_000),
    "memory_extraction_timeout_seconds": (1, 600),
    "memory_extraction_max_concurrency": (1, 16),
    "memory_extraction_max_session_age_hours": (1, 8_760),
}

MEMORY_SCOPES = ("global", "session")


def parse_urls(raw_value: str | None) -> list[str]:
    if not raw_value:
        return DEFAULT_URLS.copy()

    urls = [url.strip() for url in raw_value.split(",") if url.strip()]
    return urls or DEFAULT_URLS.copy()


def parse_csv_list(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_optional_int(raw_value: str | None, default: int | None) -> int | None:
    if raw_value is None:
        return default
    value = raw_value.strip()
    if not value:
        return None
    return int(value)


def parse_bool(raw_value: Any, default: bool = False) -> bool:
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() in ("true", "1", "yes", "on")


def _strip_yaml_comment(line: str) -> str:
    in_quote: str | None = None
    result: list[str] = []
    for char in line:
        if char in ("'", '"'):
            in_quote = None if in_quote == char else char if in_quote is None else in_quote
        if char == "#" and in_quote is None:
            break
        result.append(char)
    return "".join(result).rstrip()


def _parse_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value in ("null", "None", "~"):
        return None
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_yaml_scalar(item.strip()) for item in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_yaml_config(config_file: str | Path | None = DEFAULT_CONFIG_FILE) -> dict[str, Any]:
    """Load a small flat YAML config file without adding a runtime dependency."""

    if config_file is None:
        return {}

    path = Path(config_file)
    if not path.exists():
        return {}

    data: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_yaml_comment(raw_line)
        if not line.strip():
            continue

        stripped = line.strip()
        if current_list_key and stripped.startswith("- "):
            data.setdefault(current_list_key, []).append(_parse_yaml_scalar(stripped[2:]))
            continue

        current_list_key = None
        if ":" not in stripped:
            raise ValueError(f"Invalid YAML config line in {path}: {raw_line!r}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid empty YAML config key in {path}: {raw_line!r}")
        if raw_value.strip():
            data[key] = _parse_yaml_scalar(raw_value)
        else:
            data[key] = []
            current_list_key = key

    return data


def _settings_defaults() -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for item in fields(Settings):
        if item.name in ("dashscope_api_key", "deepseek_api_key"):
            continue
        if item.default is not MISSING:
            defaults[item.name] = item.default
        elif item.default_factory is not MISSING:
            default_factory = item.default_factory
            if callable(default_factory):
                defaults[item.name] = default_factory()
    return defaults


def _coerce_setting(name: str, value: Any, default: Any = None) -> Any:
    if name == "source_urls":
        if isinstance(value, list):
            return [str(url).strip() for url in value if str(url).strip()]
        return parse_urls(None if value is None else str(value))
    if name == "cors_allow_origins":
        if isinstance(value, list):
            return [str(url).strip() for url in value if str(url).strip()]
        if value is None:
            return list(default or [])
        return [url.strip() for url in str(value).split(",") if url.strip()]
    if name in (
        "web_search_js_fallback_domains",
        "web_search_js_force_domains",
        "web_search_providers",
    ):
        if isinstance(value, list):
            return [str(item).strip().lower() for item in value if str(item).strip()]
        if value is None:
            return list(default or [])
        items = [item.lower() for item in parse_csv_list(str(value))]
        return items if items else list(default or [])
    if name in ("chroma_dir",):
        return Path(str(value)).expanduser()
    if name in _MEMORY_INT_RANGES:
        low, high = _MEMORY_INT_RANGES[name]
        try:
            parsed = int(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} must be an integer between {low} and {high}; got {value!r}"
            ) from None
        if not low <= parsed <= high:
            raise ValueError(f"{name} must be between {low} and {high}; got {parsed}")
        return parsed
    if name == "memory_default_scope":
        scope = str(value).strip().casefold() or str(default or "global")
        if scope not in MEMORY_SCOPES:
            raise ValueError(
                "memory_default_scope must be one of: "
                f"{', '.join(MEMORY_SCOPES)}; got {value!r}"
            )
        return scope
    if name in (
        "embedding_dimension",
        "embedding_batch_size",
        "chunk_size",
        "chunk_overlap",
        "api_port",
        "min_keyword_matches",
        "max_rewrites",
        "chat_context_max_turns",
        "chat_context_max_chars",
        "web_search_max_results",
        "web_search_provider_fanout",
        "web_search_provider_timeout_seconds",
        "web_search_api_timeout_seconds",
        "web_search_deadline_seconds",
        "web_search_top_k",
        "web_search_min_url_score",
        "web_search_max_page_tokens",
        "web_search_min_page_chars",
        "web_search_min_page_tokens",
        "web_search_js_retry_budget",
        "web_search_min_content_words",
        "web_search_domain_reputation_min_samples",
        "page_load_timeout",
        "page_load_max_concurrency",
        "page_load_cache_ttl_seconds",
        "document_quality_min_text_length",
        "document_quality_min_unique_terms",
        "document_quality_query_min_overlap",
        "document_quality_recency_bias_days",
        "wikipedia_max_summary_chars",
        "dashscope_request_timeout",
        "dashscope_max_retries",
        "file_read_max_bytes",
    ):
        if name == "embedding_dimension":
            return parse_optional_int(None if value is None else str(value), default)
        parsed = int(value)
        if name in (
            "max_rewrites",
            "web_search_top_k",
            "web_search_min_url_score",
            "web_search_max_page_tokens",
            "web_search_min_page_chars",
            "web_search_min_page_tokens",
            "web_search_js_retry_budget",
            "web_search_min_content_words",
            "web_search_domain_reputation_min_samples",
            "document_quality_min_text_length",
            "document_quality_min_unique_terms",
            "document_quality_query_min_overlap",
            "document_quality_recency_bias_days",
            "wikipedia_max_summary_chars",
        ):
            return max(0, parsed)
        if name in ("chat_context_max_turns", "chat_context_max_chars"):
            return max(1, parsed)
        if name in (
            "page_load_timeout",
            "page_load_max_concurrency",
            "web_search_provider_fanout",
            "web_search_provider_timeout_seconds",
            "web_search_api_timeout_seconds",
            "web_search_deadline_seconds",
            "dashscope_max_retries",
            "file_read_max_bytes",
        ):
            return max(1, parsed)
        if name == "page_load_cache_ttl_seconds":
            return max(0, parsed)
        return parsed
    if name in (
        "document_quality_min_similarity",
        "web_search_max_link_density",
        "web_search_semantic_min_similarity",
    ):
        return max(0.0, min(1.0, float(value)))
    if name in (
        "allow_low_relevance_generate",
        "web_search_enabled",
        "web_search_llm_query_rewrite_enabled",
        "web_search_verify_ssl",
        "web_search_lightweight",
        "web_search_js_fallback_enabled",
        "web_search_structure_filter_enabled",
        "web_search_semantic_filter_enabled",
        "web_search_domain_reputation_enabled",
        "weather_enabled",
        "stock_enabled",
        "currency_enabled",
        "wikipedia_enabled",
        "directions_enabled",
        "map_enabled",
        "math_enabled",
        "statistics_enabled",
        "linalg_enabled",
        "number_theory_enabled",
        "datetime_enabled",
        "summarize_url_enabled",
        "file_read_enabled",
        "memory_enabled",
        "memory_auto_recall_enabled",
        "memory_extraction_enabled",
        "memory_extraction_on_session_start",
        "document_quality_filter_enabled",
    ):
        return parse_bool(value, bool(default))
    if name in ("web_search_provider",):
        return str(value).strip().lower() or str(default)
    if name == "rerank_strategy":
        strategy = str(value).strip().lower() or str(default or "lexical")
        if strategy not in ("lexical", "embedding", "hybrid"):
            raise ValueError("rerank_strategy must be one of: lexical, embedding, hybrid")
        return strategy
    if name in ("web_search_timelimit",):
        text = "" if value is None else str(value).strip()
        return text or None
    if name == "dashscope_http_base_url":
        return parse_dashscope_base_url(None if value is None else str(value))
    if value is None:
        return default
    return str(value).strip() or default


def _is_default_config_file(config_file: str | Path) -> bool:
    return Path(config_file).as_posix() == DEFAULT_CONFIG_FILE.as_posix()


def _environment_config_file() -> Path:
    environment = os.getenv("RAG_ENV", DEFAULT_ENVIRONMENT).strip().lower()
    return DEFAULT_CONFIG_DIR / f"{environment}.yaml"


def _selected_config_files(config_file: str | Path | None) -> list[Path]:
    if config_file is None:
        return []
    if not _is_default_config_file(config_file):
        return [Path(config_file)]

    paths = [DEFAULT_CONFIG_FILE]
    environment_path = _environment_config_file()
    if environment_path != DEFAULT_CONFIG_FILE:
        paths.append(environment_path)
    return paths


def load_selected_yaml_config(
    config_file: str | Path | None = DEFAULT_CONFIG_FILE,
) -> dict[str, Any]:
    """Load default YAML plus the active RAG_ENV overlay when applicable."""

    merged: dict[str, Any] = {}
    for path in _selected_config_files(config_file):
        merged.update(load_yaml_config(path))
    return merged


def load_cors_allow_origins(config_file: str | Path | None = DEFAULT_CONFIG_FILE) -> list[str]:
    """Load CORS origins without requiring secret-bearing runtime settings."""

    values = _settings_defaults()
    for key, value in load_selected_yaml_config(config_file).items():
        if key in values:
            values[key] = _coerce_setting(key, value, values.get(key))
    _apply_env_overrides(values)
    origins = values.get("cors_allow_origins", [])
    return list(origins) if isinstance(origins, list) else []


def _apply_env_overrides(values: dict[str, Any]) -> None:
    for name, env_name in SETTING_ENV_NAMES.items():
        raw_value = os.getenv(env_name)
        if raw_value is not None:
            values[name] = _coerce_setting(name, raw_value, values.get(name))


def apply_runtime_environment(settings: Settings) -> None:
    """Apply process globals required by legacy LangChain/DashScope clients."""

    os.environ["DASHSCOPE_API_KEY"] = settings.dashscope_api_key
    configure_dashscope_base_url(settings.dashscope_http_base_url)
    os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2

    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    if settings.deepseek_api_key:
        os.environ["DEEPSEEK_API_KEY"] = settings.deepseek_api_key

    ensure_user_agent()


def secret_fingerprint(secret: str) -> str:
    """Return a non-sensitive fingerprint for checking which secret was loaded."""

    value = (secret or "").strip()
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10] if value else "none"
    preview = "***" if len(value) <= 8 else f"{value[:3]}...{value[-4:]}"
    return f"{preview} (len={len(value)}, sha256={digest})"


def load_settings(
    env_file: str | Path = ".env",
    urls: list[str] | None = None,
    config_file: str | Path | None = DEFAULT_CONFIG_FILE,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load settings with CLI > env > YAML > built-in default precedence."""

    load_dotenv(env_file)

    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not dashscope_api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is missing. Copy .env.example to .env and add your key."
        )

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()

    values = _settings_defaults()
    for key, value in load_selected_yaml_config(config_file).items():
        if key in values:
            values[key] = _coerce_setting(key, value, values.get(key))

    _apply_env_overrides(values)

    if urls is not None:
        values["source_urls"] = list(urls)

    for key, value in (overrides or {}).items():
        if key in values and value is not None:
            values[key] = _coerce_setting(key, value, values.get(key))

    settings = Settings(
        dashscope_api_key=dashscope_api_key,
        deepseek_api_key=deepseek_api_key,
        **values,
    )

    llm_provider = (values.get("llm_provider") or "dashscope").strip().lower()
    if llm_provider == "deepseek" and not deepseek_api_key:
        raise RuntimeError(
            "LLM_PROVIDER is set to 'deepseek' but DEEPSEEK_API_KEY is missing. "
            "Copy .env.example to .env and add your DeepSeek API key."
        )

    apply_runtime_environment(settings)
    return settings


__all__ = [
    "DEFAULT_DASHSCOPE_HTTP_BASE_URL",
    "DEFAULT_CONFIG_DIR",
    "DEFAULT_CONFIG_FILE",
    "DEFAULT_ENVIRONMENT",
    "MEMORY_SCOPES",
    "apply_runtime_environment",
    "load_cors_allow_origins",
    "load_settings",
    "load_selected_yaml_config",
    "load_yaml_config",
    "parse_bool",
    "parse_csv_list",
    "parse_optional_int",
    "parse_urls",
    "secret_fingerprint",
]
