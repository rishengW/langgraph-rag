---
name: config-architect
description: >
  Use this skill whenever working on configuration — the Settings dataclass,
  YAML config loading, environment-variable overrides, environment overlays
  (development/staging/production), secrets handling, or CLI > env > YAML >
  defaults precedence. Trigger on mentions of Settings, load_settings, .env,
  YAML, config/default.yaml, RAG_ENV, environment variables, or config drift.
---

# Config Architect — src/config/

Domain: runtime configuration values, loading precedence, environment overlays,
runtime side-effects.
Parent: `SKILL.md` (root). Siblings: `src/llm/SKILL.md`, `src/graph/SKILL.md`,
`src/rag/SKILL.md`, `src/web_search/SKILL.md`, `src/api/SKILL.md`,
`src/sessions/SKILL.md`, `src/tools/SKILL.md`, `src/utils/SKILL.md`.

## Quick Reference

| Fact | Value |
|---|---|
| Public dataclass | `Settings` (frozen, ~60 fields) |
| Loader entry point | `load_settings(env_file=".env", urls=None, config_file=..., overrides=None)` |
| Precedence | `overrides` (CLI) > env vars > YAML config file (+ env overlay) > built-in defaults |
| Default YAML | `config/default.yaml` |
| Environment overlay | `config/{RAG_ENV}.yaml` (default `development`) |
| Required secret | `DASHSCOPE_API_KEY` (env only, never YAML) |
| Optional secret | `DEEPSEEK_API_KEY` (required only when `LLM_PROVIDER=deepseek`) |
| Side-effects after load | `os.environ` mutations for `DASHSCOPE_API_KEY`, `LANGCHAIN_*`, `DEEPSEEK_API_KEY`, `USER_AGENT`; SDK base-URL config |

## File Map

```
src/config/
├── __init__.py     # Re-exports
├── settings.py     # @dataclass(frozen=True) Settings — pure values, no I/O
└── loader.py       # load_settings, YAML parsing, env mapping, apply_runtime_environment
```

`config/` (at repo root, separate from the Python package):

```
config/
├── default.yaml        # Cross-environment baseline
├── development.yaml    # Overlay for RAG_ENV=development (default)
├── staging.yaml        # Overlay for RAG_ENV=staging
└── production.yaml     # Overlay for RAG_ENV=production
```

## Loading Precedence

```
1. CLI / programmatic overrides   (highest — passed via overrides= or urls=)
2. Environment variables           (overrides anything from YAML or defaults)
3. config/default.yaml + RAG_ENV overlay  (config/{RAG_ENV}.yaml merges over default)
4. Settings dataclass defaults     (lowest)
```

Secrets are always read from env vars and never from YAML. The loader will
**fail at startup** if `DASHSCOPE_API_KEY` is missing, or if `LLM_PROVIDER=deepseek`
is set without `DEEPSEEK_API_KEY`.

## Environment Variable Map

Every flat field on `Settings` has a corresponding env var name in
`SETTING_ENV_NAMES` (~50 entries). Pattern: `snake_case` → `UPPER_SNAKE_CASE`.

Examples:

| Setting field | Env var |
|---|---|
| `qwen_model` | `QWEN_MODEL` |
| `embedding_model` | `EMBEDDING_MODEL` |
| `web_search_provider` | `WEB_SEARCH_PROVIDER` |
| `web_search_top_k` | `WEB_SEARCH_TOP_K` |
| `page_load_timeout` | `PAGE_LOAD_TIMEOUT` |
| `llm_provider` | `LLM_PROVIDER` |
| `deepseek_model` | `DEEPSEEK_MODEL` |
| `cors_allow_origins` | `CORS_ALLOW_ORIGINS` |

**Adding a new setting requires three places to stay in sync:**

1. Field on `Settings` in `settings.py` with a sensible default.
2. Entry in `SETTING_ENV_NAMES` in `loader.py` mapping the snake_case name to its env var.
3. Coercion branch in `_coerce_setting` if the type isn't a plain string (int, bool, list, Path, etc.).
4. Sample line in `.env.example` and `config/default.yaml` for discoverability.
5. README env-var table update (if user-facing).

## Settings Groups (by Concern)

| Concern | Fields |
|---|---|
| LLM provider | `llm_provider`, `qwen_model`, `deepseek_model`, `deepseek_base_url`, `dashscope_request_timeout`, `dashscope_max_retries`, `dashscope_http_base_url` |
| Embeddings | `embedding_model`, `embedding_dimension`, `embedding_batch_size` |
| Vector store | `chroma_dir`, `collection_name`, `chunk_size`, `chunk_overlap` |
| Sources | `source_urls` |
| Web search | `web_search_enabled`, `web_search_provider`, `web_search_max_results`, `web_search_top_k`, `web_search_min_url_score`, `web_search_region`, `web_search_timelimit`, `web_search_verify_ssl`, `web_search_lightweight`, `web_search_max_page_tokens`, `web_search_min_page_chars`, `web_search_min_page_tokens`, `web_search_js_*` |
| Page load | `page_load_timeout`, `page_load_max_concurrency`, `page_load_cache_ttl_seconds` |
| Document quality | `document_quality_filter_enabled`, `document_quality_min_text_length`, `document_quality_min_unique_terms`, `document_quality_relevance_query`, `document_quality_query_min_overlap`, `document_quality_min_similarity`, `document_quality_recency_bias_days`, `rerank_strategy` |
| Optional tools | `weather_enabled`, `stock_enabled`, `currency_enabled`, `wikipedia_enabled`, `wikipedia_max_summary_chars`, `wikipedia_user_agent` |
| Retrieval tuning | `allow_low_relevance_generate`, `min_keyword_matches`, `max_rewrites` |
| API server | `api_key`, `api_host`, `api_port`, `cors_allow_origins` |
| LangSmith tracing | `langchain_tracing_v2`, `langchain_api_key`, `langchain_project` |
| Secrets | `dashscope_api_key` (required), `deepseek_api_key` (conditional) |

## Runtime Side-Effects: `apply_runtime_environment`

`load_settings` ends by calling `apply_runtime_environment(settings)`, which:

- Writes `DASHSCOPE_API_KEY` and (if set) `DEEPSEEK_API_KEY` to `os.environ` so SDKs find them.
- Calls `configure_dashscope_base_url(...)` to set both `os.environ["DASHSCOPE_HTTP_BASE_URL"]` and `dashscope.base_http_api_url`.
- Writes `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` when tracing is enabled.
- Ensures `USER_AGENT` env var is set so LangChain's `WebBaseLoader` doesn't warn.

This is the only place in the project that should mutate `os.environ`. Other
modules read settings, never the environment directly.

## Helper Functions

| Function | Purpose |
|---|---|
| `parse_urls(raw)` | Comma-separated URL string → list, falls back to `DEFAULT_URLS` |
| `parse_csv_list(raw)` | Comma-separated string → list of stripped non-empty items |
| `parse_bool(raw, default)` | Accepts `true/1/yes/on` (case-insensitive); used for boolean env vars |
| `parse_optional_int(raw, default)` | Empty string → `None`; otherwise `int(...)` |
| `load_yaml_config(path)` | Tiny built-in flat-YAML parser (no PyYAML runtime dependency) |
| `load_selected_yaml_config(config_file)` | Default YAML + `config/{RAG_ENV}.yaml` overlay merged |
| `load_cors_allow_origins(config_file)` | Load CORS list without needing `DASHSCOPE_API_KEY` (used pre-secret) |
| `secret_fingerprint(secret)` | Returns `"sk-...abcd (len=N, sha256=10chars)"` for safe key-loaded logging |

## YAML Format

`load_yaml_config` parses a deliberately small subset of YAML:

- Flat key-value pairs only: `key: value`
- Scalars: `true`/`false`, `null`/`None`/`~`, integers, floats, single/double-quoted strings, unquoted strings.
- Inline lists: `key: [a, b, c]`
- Block lists with `- ` items beneath a list key.
- Lines starting with `#` are comments; trailing comments outside quotes are stripped.

There is no runtime PyYAML dependency — keeps the loader auditable and
deployment-light. If you need nested mappings, that's a deliberate gap to
revisit.

## Environment Overlay

```
RAG_ENV=development   ← default; loads config/default.yaml + config/development.yaml
RAG_ENV=staging       ← loads config/default.yaml + config/staging.yaml
RAG_ENV=production    ← loads config/default.yaml + config/production.yaml
```

Overlay is applied only when the caller is using the default config file. If
a custom `config_file=...` is passed, no overlay is applied — the explicit
file is the sole source.

## Known Issues

| # | Issue | Severity | Location | Fix |
|---|---|---|---|---|
| 1 | `load_settings` mutates `os.environ` as a side effect | Medium | `apply_runtime_environment` | Pure separation requires updating every SDK call site to pass keys explicitly; not worth the churn yet |
| 2 | Tiny built-in YAML parser only supports flat scalars and lists | Low | `load_yaml_config` | Document the restriction; switch to PyYAML if nested config is ever needed |
| 3 | Adding a new setting requires updating 5 places (dataclass, env map, coercion, .env.example, default.yaml) | Medium | Multiple | Could be reduced with field metadata, but the current explicitness is auditable |
| 4 | No schema validation beyond `_coerce_setting` (e.g., port range, URL syntax) | Low | `_coerce_setting` | Add range/syntax validation for high-risk fields |

## Refactoring To-Do List

- [ ] **Setting groups as nested dataclasses** — `Settings.llm`, `Settings.web_search`, `Settings.embedding` instead of 60 flat fields. Was planned for Phase 2; deferred.
- [ ] **Schema validation** — port range, URL syntax, enum validation for `llm_provider`, `web_search_provider`, `rerank_strategy`.
- [ ] **Settings diff helper** — print the non-default fields at startup for debugging.
- [ ] **Test the full precedence chain** — verify CLI > env > YAML > default with a single comprehensive test matrix.

## Testing Strategy

| Test | Approach |
|---|---|
| YAML loading | Write a small YAML file in a tmp dir; assert each scalar/list parses correctly |
| Env var override | Set env var; assert it beats the YAML value |
| Overrides over env | Pass `overrides={...}`; assert it beats env vars |
| RAG_ENV overlay | Set `RAG_ENV=staging`; assert staging.yaml overlays default.yaml |
| Missing secret | Unset `DASHSCOPE_API_KEY`; assert `load_settings()` raises |
| DeepSeek without key | Set `LLM_PROVIDER=deepseek` without `DEEPSEEK_API_KEY`; assert error |
| Tests live in | `tests/test_config.py` |

## Dependencies

- `src/utils/networking.py` — `parse_dashscope_base_url`, `configure_dashscope_base_url`, `ensure_user_agent`.
- External: `python-dotenv` for `.env` loading.
- Every other `src/*` package depends on `Settings`; this module has no inbound dependencies on application code.
