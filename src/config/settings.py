from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_URLS = [
    "https://help.aliyun.com/zh/pai/user-guide/use-pai-model-guidelines-in-claude-code?spm=a2c4g.11186623.help-menu-30347.d_3_2_4.3d354f74THB0pK&scm=20140722.H_2990607._.OR_help-T_cn~zh-V_1",
    "https://help.aliyun.com/zh/pai/user-guide/llm-fine-tuning-experience?spm=5176.30275541.aillm.1.17b42f3dcoHCyA&scm=20140722.S_%E6%88%91%E8%AE%B0%E5%BE%97%E6%9C%89%E4%B8%AA%E6%96%87%E6%A1%A3%E9%A1%B5%E9%9D%A2%E4%B8%93%E9%97%A8%E5%B0%86%E5%BE%AE%E8%B0%83%E7%9A%84%E5%90%84%E4%B8%AA%E6%96%B9%E6%B3%95%E7%9A%84%E5%9C%A8%E5%93%AA%E9%87%8C%E6%B2%A1%E6%89%BE%E5%88%B0._.RL_%E6%88%91%E8%AE%B0%E5%BE%97%E6%9C%89%E4%B8%AA%E6%96%87%E6%A1%A3%E9%A1%B5%E9%9D%A2%E4%B8%93%E9%97%A8%E5%B0%86%E5%BE%AE%E8%B0%83%E7%9A%84%E5%90%84%E4%B8%AA%E6%96%B9%E6%B3%95%E7%9A%84%E5%9C%A8%E5%93%AA%E9%87%8C%E6%B2%A1%E6%89%BE%E5%88%B0-LOC_aillm-OR_chat-V_3-RC_llm",
]

DEFAULT_WEB_SEARCH_JS_FALLBACK_DOMAINS = [
    "baike.baidu.com",
    "zhuanlan.zhihu.com",
    "apps.microsoft.com",
    "deepseek.net",
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
    llm_provider: str = "dashscope"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_base_url: str = "https://api.deepseek.com"
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
    api_key: str = ""
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_allow_origins: list[str] = field(default_factory=list)
    # Trusted network identity. Request bodies and arbitrary identity headers
    # never override these authentication-boundary values.
    api_principal_id: str = "api-key-client"
    api_tenant_id: str = ""
    # Process-local per-principal and per-tenant quotas for the supported
    # single-instance topology. Shared quota state is a multi-replica prerequisite.
    quota_principal_requests_per_minute: int = 120
    quota_tenant_requests_per_minute: int = 600
    quota_principal_concurrent_calls: int = 4
    quota_tenant_concurrent_calls: int = 16
    quota_principal_searches_per_minute: int = 30
    quota_tenant_searches_per_minute: int = 120
    quota_principal_tokens_per_minute: int = 250_000
    quota_tenant_tokens_per_minute: int = 1_000_000
    quota_principal_tool_calls_per_minute: int = 240
    quota_tenant_tool_calls_per_minute: int = 1_000
    quota_principal_retries_per_minute: int = 600
    quota_tenant_retries_per_minute: int = 3_000
    quota_principal_cost_units_per_minute: int = 500_000
    quota_tenant_cost_units_per_minute: int = 2_000_000
    quota_max_tracked_identities: int = 4_096
    allow_low_relevance_generate: bool = False
    min_keyword_matches: int = 2
    max_rewrites: int = 2
    # Keep the persisted chat transcript complete while bounding the subset
    # sent to conversational LLM calls on each turn.
    chat_context_max_turns: int = 8
    chat_context_max_chars: int = 12_000
    # Optional general planning/reflection layer. Disabled by default so the
    # existing chat graphs keep their latency and transcript behavior.
    planning_enabled: bool = False
    planning_max_subgoals: int = 4
    planning_max_reflection_retries: int = 1
    planning_critic_threshold: float = 0.7
    web_search_enabled: bool = True
    web_search_llm_query_rewrite_enabled: bool = False
    web_search_provider: str = "bing"
    # Optional priority list. Configured key-backed providers are
    # automatically placed first for Mandarin queries even when this is empty.
    web_search_providers: list[str] = field(default_factory=list)
    web_search_provider_fanout: int = 2
    web_search_max_results: int = 20
    web_search_provider_timeout_seconds: int = 8
    web_search_api_timeout_seconds: int = 20
    web_search_deadline_seconds: int = 30
    serper_api_key: str = ""
    brave_search_api_key: str = ""
    tavily_api_key: str = ""
    bing_search_api_key: str = ""
    bing_search_endpoint: str = "https://api.bing.microsoft.com/v7.0/search"
    amap_web_service_key: str = ""
    amap_js_api_key: str = ""
    amap_js_security_code: str = ""
    amap_api_timeout_seconds: int = 10
    # REFACTOR: Use one canonical web-search fetch default and expose URL load concurrency.
    web_search_top_k: int = 6
    web_search_min_url_score: int = 45
    web_search_region: str = "wt-wt"
    web_search_timelimit: str | None = None
    web_search_verify_ssl: bool = True
    web_search_lightweight: bool = True
    web_search_max_page_tokens: int = 8000
    web_search_min_page_chars: int = 200
    web_search_min_page_tokens: int = 50
    web_search_js_fallback_enabled: bool = False
    web_search_js_fallback_domains: list[str] = field(
        default_factory=lambda: DEFAULT_WEB_SEARCH_JS_FALLBACK_DOMAINS.copy()
    )
    web_search_js_force_domains: list[str] = field(default_factory=list)
    # Bound browser renders per fetch batch. The JS retry is triggered by an
    # unreadable HTTP result rather than by a domain list, so it needs its own
    # budget to keep a bad turn from spending the whole deadline in Chromium.
    web_search_js_retry_budget: int = 2
    # REFACTOR: Dynamic structural page filtering replaces URL-shape guessing.
    web_search_structure_filter_enabled: bool = True
    web_search_max_link_density: float = 0.5
    web_search_min_content_words: int = 60
    # REFACTOR: Optional semantic (embedding) relevance layered on top of the
    # deterministic lexical gates. Disabled by default: it loads a local
    # sentence-transformers model on first use.
    web_search_semantic_filter_enabled: bool = False
    web_search_semantic_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    web_search_semantic_min_similarity: float = 0.35
    # REFACTOR: Adaptive per-domain reputation prior learned from fetch outcomes.
    web_search_domain_reputation_enabled: bool = True
    web_search_domain_reputation_min_samples: int = 5
    weather_enabled: bool = False
    stock_enabled: bool = False
    currency_enabled: bool = False
    wikipedia_enabled: bool = False
    directions_enabled: bool = False
    map_enabled: bool = False
    math_enabled: bool = False
    statistics_enabled: bool = False
    linalg_enabled: bool = False
    number_theory_enabled: bool = False
    datetime_enabled: bool = False
    summarize_url_enabled: bool = False
    file_read_enabled: bool = False
    file_read_root: str = "."
    file_read_max_bytes: int = 5_000_000
    # Word .docx creation and editing. Off by default and opt-in on top of
    # file_read_enabled. Writes are confined to the current chat session's
    # upload directory and never overwrite an existing file.
    word_edit_enabled: bool = False
    powerpoint_edit_enabled: bool = False
    # Excel .xlsx creation. Uses a configured Node runtime and node_modules
    # directory containing @oai/artifact-tool; output stays session-scoped.
    excel_create_enabled: bool = False
    excel_node_executable: str = "node"
    excel_node_modules_path: str = ""
    # Excel .xlsx editing. openpyxl-based; needs file_read_enabled. Like the
    # other editors it writes a session-scoped copy and never overwrites the
    # uploaded source.
    excel_edit_enabled: bool = False
    # Plain-text .txt creation and editing. Like Word editing, this is an opt-in write tool
    # layered on top of file_read_enabled. Edits are session-scoped and always
    # produce a new file rather than overwriting the uploaded source.
    text_edit_enabled: bool = False
    # Markdown .md creation and editing. Same session-scoped, never-overwrite
    # model as the .txt editor, layered on top of file_read_enabled.
    markdown_edit_enabled: bool = False
    # CSV .csv creation and editing. Standard-library csv module for quoting;
    # needs file_read_enabled. Same session-scoped, never-overwrite model as the
    # .txt editor, layered on top of file_read_enabled.
    csv_edit_enabled: bool = False
    # Chat-agent long-term memory. Disabled by default so existing
    # deployments keep their current tool set and touch no memory file.
    memory_enabled: bool = False
    # Empty means the default path memory/long_term_memory.json, resolved
    # against the process working directory.
    memory_store_path: str = ""
    memory_max_records: int = 500
    memory_max_record_chars: int = 1000
    memory_recall_top_k: int = 5
    memory_context_max_chars: int = 2000
    memory_default_scope: str = "global"
    memory_auto_recall_enabled: bool = True
    # Automatic memory extraction (self-updating memory). Off by default: when
    # off there is no extra LLM call, checkpoint read, or memory write.
    memory_extraction_enabled: bool = False
    memory_extraction_on_session_start: bool = True
    memory_extraction_turn_interval: int = 10
    memory_extraction_max_candidates: int = 5
    memory_extraction_max_transcript_chars: int = 8000
    memory_extraction_timeout_seconds: int = 60
    memory_extraction_max_concurrency: int = 2
    memory_extraction_max_session_age_hours: int = 168
    wikipedia_max_summary_chars: int = 1500
    wikipedia_user_agent: str = "langgraph-rag/1.0 (contact: configure WIKIPEDIA_USER_AGENT)"
    page_load_timeout: int = 15
    page_load_max_concurrency: int = 4
    page_load_cache_ttl_seconds: int = 0
    # REFACTOR: Pre-index document quality filtering defaults to conservative checks.
    document_quality_filter_enabled: bool = True
    document_quality_min_text_length: int = 80
    document_quality_min_unique_terms: int = 8
    document_quality_relevance_query: str = ""
    document_quality_query_min_overlap: int = 1
    document_quality_min_similarity: float = 0.5
    document_quality_recency_bias_days: int = 365
    # REFACTOR: Optional post-retrieval re-ranking strategy.
    rerank_strategy: str = "lexical"
    dashscope_request_timeout: int = 120
    dashscope_max_retries: int = 3
    dashscope_http_base_url: str = ""
