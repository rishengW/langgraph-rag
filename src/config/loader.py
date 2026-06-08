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
    "web_search_enabled": "WEB_SEARCH_ENABLED",
    "web_search_provider": "WEB_SEARCH_PROVIDER",
    "web_search_max_results": "WEB_SEARCH_MAX_RESULTS",
    "web_search_top_k": "WEB_SEARCH_TOP_K",
    "web_search_region": "WEB_SEARCH_REGION",
    "web_search_timelimit": "WEB_SEARCH_TIMELIMIT",
    "web_search_verify_ssl": "WEB_SEARCH_VERIFY_SSL",
    "page_load_timeout": "PAGE_LOAD_TIMEOUT",
    "dashscope_request_timeout": "DASHSCOPE_REQUEST_TIMEOUT",
    "dashscope_max_retries": "DASHSCOPE_MAX_RETRIES",
    "dashscope_http_base_url": "DASHSCOPE_HTTP_BASE_URL",
}


def parse_urls(raw_value: str | None) -> list[str]:
    if not raw_value:
        return DEFAULT_URLS.copy()

    urls = [url.strip() for url in raw_value.split(",") if url.strip()]
    return urls or DEFAULT_URLS.copy()


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
        if item.name == "dashscope_api_key":
            continue
        if item.default is not MISSING:
            defaults[item.name] = item.default
        elif item.default_factory is not MISSING:  # type: ignore[attr-defined]
            defaults[item.name] = item.default_factory()  # type: ignore[misc]
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
    if name in ("chroma_dir",):
        return Path(str(value)).expanduser()
    if name in (
        "embedding_dimension",
        "embedding_batch_size",
        "chunk_size",
        "chunk_overlap",
        "api_port",
        "min_keyword_matches",
        "max_rewrites",
        "web_search_max_results",
        "web_search_top_k",
        "page_load_timeout",
        "dashscope_request_timeout",
        "dashscope_max_retries",
    ):
        if name == "embedding_dimension":
            return parse_optional_int(None if value is None else str(value), default)
        parsed = int(value)
        if name in ("max_rewrites", "web_search_top_k"):
            return max(0, parsed)
        if name in ("page_load_timeout", "dashscope_max_retries"):
            return max(1, parsed)
        return parsed
    if name in (
        "allow_low_relevance_generate",
        "web_search_enabled",
        "web_search_verify_ssl",
    ):
        return parse_bool(value, bool(default))
    if name in ("web_search_provider",):
        return str(value).strip().lower() or str(default)
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


def load_selected_yaml_config(config_file: str | Path | None = DEFAULT_CONFIG_FILE) -> dict[str, Any]:
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

    ensure_user_agent()


def secret_fingerprint(secret: str) -> str:
    """Return a non-sensitive fingerprint for checking which secret was loaded."""

    value = (secret or "").strip()
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10] if value else "none"
    if len(value) <= 8:
        preview = "***"
    else:
        preview = f"{value[:3]}...{value[-4:]}"
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

    settings = Settings(dashscope_api_key=dashscope_api_key, **values)

    apply_runtime_environment(settings)
    return settings


__all__ = [
    "DEFAULT_DASHSCOPE_HTTP_BASE_URL",
    "DEFAULT_CONFIG_DIR",
    "DEFAULT_CONFIG_FILE",
    "DEFAULT_ENVIRONMENT",
    "apply_runtime_environment",
    "load_cors_allow_origins",
    "load_settings",
    "load_selected_yaml_config",
    "load_yaml_config",
    "parse_bool",
    "parse_optional_int",
    "parse_urls",
    "secret_fingerprint",
]

