from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_URLS = [
    "https://help.aliyun.com/zh/pai/user-guide/use-pai-model-guidelines-in-claude-code?spm=a2c4g.11186623.help-menu-30347.d_3_2_4.3d354f74THB0pK&scm=20140722.H_2990607._.OR_help-T_cn~zh-V_1",
    "https://help.aliyun.com/zh/pai/user-guide/llm-fine-tuning-experience?spm=5176.30275541.aillm.1.17b42f3dcoHCyA&scm=20140722.S_%E6%88%91%E8%AE%B0%E5%BE%97%E6%9C%89%E4%B8%AA%E6%96%87%E6%A1%A3%E9%A1%B5%E9%9D%A2%E4%B8%93%E9%97%A8%E5%B0%86%E5%BE%AE%E8%B0%83%E7%9A%84%E5%90%84%E4%B8%AA%E6%96%B9%E6%B3%95%E7%9A%84%E5%9C%A8%E5%93%AA%E9%87%8C%E6%B2%A1%E6%89%BE%E5%88%B0._.RL_%E6%88%91%E8%AE%B0%E5%BE%97%E6%9C%89%E4%B8%AA%E6%96%87%E6%A1%A3%E9%A1%B5%E9%9D%A2%E4%B8%93%E9%97%A8%E5%B0%86%E5%BE%AE%E8%B0%83%E7%9A%84%E5%90%84%E4%B8%AA%E6%96%B9%E6%B3%95%E7%9A%84%E5%9C%A8%E5%93%AA%E9%87%8C%E6%B2%A1%E6%89%BE%E5%88%B0-LOC_aillm-OR_chat-V_3-RC_llm",
]


@dataclass(frozen=True)
class Settings:
    """Runtime configuration values.

    This dataclass is intentionally pure: it only represents values and does
    not read from, or write to, process environment variables.
    """

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
    langchain_project: str = "only-subcribers"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    allow_low_relevance_generate: bool = False
    min_keyword_matches: int = 2
    max_rewrites: int = 2
    web_search_enabled: bool = True
    web_search_provider: str = "baidu"
    web_search_max_results: int = 20
    web_search_top_k: int = 3
    web_search_region: str = "wt-wt"
    web_search_timelimit: str | None = None
    web_search_verify_ssl: bool = True
    page_load_timeout: int = 15
    dashscope_request_timeout: int = 120
    dashscope_max_retries: int = 3
    dashscope_http_base_url: str = ""

