from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

from ..utils.networking import (
    DEFAULT_DASHSCOPE_HTTP_BASE_URL,
    configure_dashscope_base_url,
    ensure_user_agent,
    parse_dashscope_base_url,
)
from .settings import DEFAULT_URLS, Settings


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


def load_settings(env_file: str | Path = ".env", urls: list[str] | None = None) -> Settings:
    """Load settings from `.env` and apply legacy runtime environment values."""

    load_dotenv(env_file)

    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not dashscope_api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is missing. Copy .env.example to .env and add your key."
        )

    if urls is None:
        urls = parse_urls(os.getenv("SOURCE_URLS"))

    settings = Settings(
        dashscope_api_key=dashscope_api_key,
        qwen_model=os.getenv("QWEN_MODEL", "qwen-plus").strip() or "qwen-plus",
        embedding_model=(
            os.getenv("EMBEDDING_MODEL", "text-embedding-v4").strip()
            or "text-embedding-v4"
        ),
        embedding_dimension=parse_optional_int(os.getenv("EMBEDDING_DIMENSION"), 1024),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "10")),
        chroma_dir=Path(os.getenv("CHROMA_DIR", ".chroma")).expanduser(),
        collection_name=os.getenv("COLLECTION_NAME", "rag-chroma").strip() or "rag-chroma",
        chunk_size=int(os.getenv("CHUNK_SIZE", "100")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        source_urls=urls,
        langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "false").strip() or "false",
        langchain_api_key=os.getenv("LANGCHAIN_API_KEY", "").strip(),
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "only-subcribers").strip()
        or "only-subcribers",
        api_host=os.getenv("API_HOST", "127.0.0.1").strip() or "127.0.0.1",
        api_port=int(os.getenv("API_PORT", "8000")),
        allow_low_relevance_generate=(
            os.getenv("ALLOW_LOW_RELEVANCE_GENERATE", "false").lower() in ("true", "1", "yes")
        ),
        min_keyword_matches=int(os.getenv("MIN_KEYWORD_MATCHES", "2")),
        max_rewrites=max(0, int(os.getenv("MAX_REWRITES", "2"))),
        web_search_enabled=(
            os.getenv("WEB_SEARCH_ENABLED", "true").lower() in ("true", "1", "yes")
        ),
        web_search_provider=(
            os.getenv("WEB_SEARCH_PROVIDER", "baidu").strip().lower() or "baidu"
        ),
        web_search_max_results=int(os.getenv("WEB_SEARCH_MAX_RESULTS", "20")),
        web_search_top_k=max(0, int(os.getenv("WEB_SEARCH_TOP_K", "3"))),
        web_search_region=os.getenv("WEB_SEARCH_REGION", "wt-wt").strip() or "wt-wt",
        web_search_timelimit=(os.getenv("WEB_SEARCH_TIMELIMIT", "").strip() or None),
        web_search_verify_ssl=(
            os.getenv("WEB_SEARCH_VERIFY_SSL", "true").lower() in ("true", "1", "yes")
        ),
        page_load_timeout=max(1, int(os.getenv("PAGE_LOAD_TIMEOUT", "15"))),
        dashscope_request_timeout=int(os.getenv("DASHSCOPE_REQUEST_TIMEOUT", "120")),
        dashscope_max_retries=max(1, int(os.getenv("DASHSCOPE_MAX_RETRIES", "3"))),
        dashscope_http_base_url=parse_dashscope_base_url(os.getenv("DASHSCOPE_HTTP_BASE_URL")),
    )

    apply_runtime_environment(settings)
    return settings


__all__ = [
    "DEFAULT_DASHSCOPE_HTTP_BASE_URL",
    "apply_runtime_environment",
    "load_settings",
    "parse_optional_int",
    "parse_urls",
    "secret_fingerprint",
]

