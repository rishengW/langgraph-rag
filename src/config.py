from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_DASHSCOPE_HTTP_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"

DEFAULT_URLS = [
    "https://help.aliyun.com/zh/pai/user-guide/use-pai-model-guidelines-in-claude-code?spm=a2c4g.11186623.help-menu-30347.d_3_2_4.3d354f74THB0pK&scm=20140722.H_2990607._.OR_help-T_cn~zh-V_1",
    "https://help.aliyun.com/zh/pai/user-guide/llm-fine-tuning-experience?spm=5176.30275541.aillm.1.17b42f3dcoHCyA&scm=20140722.S_%E6%88%91%E8%AE%B0%E5%BE%97%E6%9C%89%E4%B8%AA%E6%96%87%E6%A1%A3%E9%A1%B5%E9%9D%A2%E4%B8%93%E9%97%A8%E5%B0%86%E5%BE%AE%E8%B0%83%E7%9A%84%E5%90%84%E4%B8%AA%E6%96%B9%E6%B3%95%E7%9A%84%E5%9C%A8%E5%93%AA%E9%87%8C%E6%B2%A1%E6%89%BE%E5%88%B0._.RL_%E6%88%91%E8%AE%B0%E5%BE%97%E6%9C%89%E4%B8%AA%E6%96%87%E6%A1%A3%E9%A1%B5%E9%9D%A2%E4%B8%93%E9%97%A8%E5%B0%86%E5%BE%AE%E8%B0%83%E7%9A%84%E5%90%84%E4%B8%AA%E6%96%B9%E6%B3%95%E7%9A%84%E5%9C%A8%E5%93%AA%E9%87%8C%E6%B2%A1%E6%89%BE%E5%88%B0-LOC_aillm-OR_chat-V_3-RC_llm",
]


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    dashscope_api_key: str
    qwen_model: str = "qwen-plus"
    embedding_model: str = "text-embedding-v4"
    embedding_dimension: int | None = 1024
    embedding_batch_size: int = 10
    chroma_dir: Path = Path(".chroma")
    collection_name: str = "rag-chroma"
    chunk_size: int = 100
    chunk_overlap: int = 50
    source_urls: list[str] = field(default_factory=lambda: DEFAULT_URLS.copy())
    langchain_tracing_v2: str = "false"
    langchain_api_key: str = ""
    langchain_project: str = "rag-langgraph-local"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    allow_low_relevance_generate: bool = False
    min_keyword_matches: int = 2
    web_search_enabled: bool = True
    web_search_provider: str = "duckduckgo"
    web_search_max_results: int = 5
    web_search_region: str = "wt-wt"
    web_search_timelimit: str | None = None
    web_search_verify_ssl: bool = True
    dashscope_request_timeout: int = 120
    dashscope_max_retries: int = 3
    dashscope_http_base_url: str = ""


def _parse_urls(raw_value: str | None) -> list[str]:
    if not raw_value:
        return DEFAULT_URLS.copy()

    urls = [url.strip() for url in raw_value.split(",") if url.strip()]
    return urls or DEFAULT_URLS.copy()


def _parse_optional_int(raw_value: str | None, default: int | None) -> int | None:
    if raw_value is None:
        return default
    value = raw_value.strip()
    if not value:
        return None
    return int(value)


def _parse_dashscope_base_url(raw_value: str | None) -> str:
    value = (raw_value or "").strip().rstrip("/")
    if not value:
        return ""
    if not value.startswith(("http://", "https://")):
        raise RuntimeError(
            "DASHSCOPE_HTTP_BASE_URL must start with http:// or https://. "
            f"Got: {value!r}"
        )
    return value


def _configure_dashscope_base_url(base_url: str) -> None:
    if base_url:
        os.environ["DASHSCOPE_HTTP_BASE_URL"] = base_url
        effective_url = base_url
    else:
        os.environ.pop("DASHSCOPE_HTTP_BASE_URL", None)
        effective_url = DEFAULT_DASHSCOPE_HTTP_BASE_URL

    try:
        import dashscope

        dashscope.base_http_api_url = effective_url
    except Exception:
        pass


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
    """Load settings from `.env` and export values needed by LangChain clients.
    
    Args:
        env_file: Path to the .env file to load.
        urls: Optional list of source URLs. If provided, overrides env vars and defaults.
    """

    load_dotenv(env_file)

    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not dashscope_api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is missing. Copy .env.example to .env and add your key."
        )

    # Use provided urls, or fall back to env var, or use defaults
    if urls is None:
        urls = _parse_urls(os.getenv("SOURCE_URLS"))

    settings = Settings(
        dashscope_api_key=dashscope_api_key,
        qwen_model=os.getenv("QWEN_MODEL", "qwen-plus").strip() or "qwen-plus",
        embedding_model=(
            os.getenv("EMBEDDING_MODEL", "text-embedding-v4").strip()
            or "text-embedding-v4"
        ),
        embedding_dimension=_parse_optional_int(os.getenv("EMBEDDING_DIMENSION"), 1024),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "10")),
        chroma_dir=Path(os.getenv("CHROMA_DIR", ".chroma")).expanduser(),
        collection_name=os.getenv("COLLECTION_NAME", "rag-chroma").strip() or "rag-chroma",
        chunk_size=int(os.getenv("CHUNK_SIZE", "100")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        source_urls=urls,
        langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "false").strip() or "false",
        langchain_api_key=os.getenv("LANGCHAIN_API_KEY", "").strip(),
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "rag-langgraph-local").strip()
        or "rag-langgraph-local",
        api_host=os.getenv("API_HOST", "127.0.0.1").strip() or "127.0.0.1",
        api_port=int(os.getenv("API_PORT", "8000")),
        allow_low_relevance_generate=(
            os.getenv("ALLOW_LOW_RELEVANCE_GENERATE", "false").lower() in ("true", "1", "yes")
        ),
        min_keyword_matches=int(os.getenv("MIN_KEYWORD_MATCHES", "2")),
        web_search_enabled=(
            os.getenv("WEB_SEARCH_ENABLED", "true").lower() in ("true", "1", "yes")
        ),
        web_search_provider=(
            os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo").strip().lower() or "duckduckgo"
        ),
        web_search_max_results=int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5")),
        web_search_region=os.getenv("WEB_SEARCH_REGION", "wt-wt").strip() or "wt-wt",
        web_search_timelimit=(
            os.getenv("WEB_SEARCH_TIMELIMIT", "").strip() or None
        ),
        web_search_verify_ssl=(
            os.getenv("WEB_SEARCH_VERIFY_SSL", "true").lower() in ("true", "1", "yes")
        ),
        dashscope_request_timeout=int(os.getenv("DASHSCOPE_REQUEST_TIMEOUT", "120")),
        dashscope_max_retries=max(1, int(os.getenv("DASHSCOPE_MAX_RETRIES", "3"))),
        dashscope_http_base_url=_parse_dashscope_base_url(
            os.getenv("DASHSCOPE_HTTP_BASE_URL")
        ),
    )

    os.environ["DASHSCOPE_API_KEY"] = settings.dashscope_api_key
    _configure_dashscope_base_url(settings.dashscope_http_base_url)
    os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2

    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    # WebBaseLoader emits a warning if no user agent is set.
    os.environ.setdefault("USER_AGENT", "rag-langgraph-local/1.0")

    return settings
