\
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    dashscope_api_key: str
    qwen_model: str = "qwen-vl-max"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    chroma_dir: Path = Path(".chroma")
    collection_name: str = "rag-chroma"
    chunk_size: int = 100
    chunk_overlap: int = 50
    source_urls: list[str] = field(default_factory=lambda: DEFAULT_URLS.copy())
    langchain_tracing_v2: str = "false"
    langchain_api_key: str = ""
    langchain_project: str = "rag-langgraph-local"


def _parse_urls(raw_value: str | None) -> list[str]:
    if not raw_value:
        return DEFAULT_URLS.copy()

    urls = [url.strip() for url in raw_value.split(",") if url.strip()]
    return urls or DEFAULT_URLS.copy()


def load_settings(env_file: str | Path = ".env") -> Settings:
    """Load settings from `.env` and export values needed by LangChain clients."""

    load_dotenv(env_file)

    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not dashscope_api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is missing. Copy .env.example to .env and add your key."
        )

    settings = Settings(
        dashscope_api_key=dashscope_api_key,
        qwen_model=os.getenv("QWEN_MODEL", "qwen-vl-max").strip() or "qwen-vl-max",
        embedding_model=(
            os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2").strip()
            or "sentence-transformers/all-mpnet-base-v2"
        ),
        chroma_dir=Path(os.getenv("CHROMA_DIR", ".chroma")).expanduser(),
        collection_name=os.getenv("COLLECTION_NAME", "rag-chroma").strip() or "rag-chroma",
        chunk_size=int(os.getenv("CHUNK_SIZE", "100")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        source_urls=_parse_urls(os.getenv("SOURCE_URLS")),
        langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "false").strip() or "false",
        langchain_api_key=os.getenv("LANGCHAIN_API_KEY", "").strip(),
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "rag-langgraph-local").strip()
        or "rag-langgraph-local",
    )

    os.environ["DASHSCOPE_API_KEY"] = settings.dashscope_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2

    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    # WebBaseLoader emits a warning if no user agent is set.
    os.environ.setdefault("USER_AGENT", "rag-langgraph-local/1.0")

    return settings
