---
name: current-process
description: Current project process snapshot — refactoring status, test state, branch, known issues
metadata:
  type: project
  updated: 2026-06-09
---

# Current Process

Last updated: 2026-06-09

## Active Branch

`web-developed` (ahead of `main` by 19 commits)

## Refactoring Status

Phase 1/2/3 extraction and interface work is complete. The codebase has been
reorganized into functional scopes under `src/`:

| Scope | Path | Status |
|---|---|---|
| Config/DI | `src/config/` | Stable |
| Graph engine | `src/graph/` | Stable |
| RAG pipeline | `src/rag/` | Stable |
| Web search | `src/web_search/` | Stable (3 providers) |
| LLM provider | `src/llm/` | Stable |
| API layer | `src/api/` | Stable |
| Sessions | `src/sessions/` | Stable |
| QA app | `src/qa/` | Stable |
| Chat app | `src/chat/` | Stable |
| Legacy compat | `src/core/` | Deprecated shims |

Legacy `src/core/*` and `src/chat/*` import paths are preserved as
compatibility facades that emit deprecation warnings.

### Completed Implementation Passes

**Phase 1 Baseline:**
- pytest harness and focused offline tests
- Shared utilities for retry, networking, and URL parsing
- Pure settings/config loading in `src/config/`
- Unified graph state and shared graph node implementations
- LLM prompt templates extracted to `src/llm/prompts.py`

**Phase 2 Interfaces:**
- Web search: provider protocol, Baidu/Bing/DuckDuckGo providers, factory, discovery
- RAG: retriever/embedding protocols, `ChromaRetriever`, embedding modules, document loading
- Sessions: session models/registry, TTL cleanup, isolated Chroma, SQLite metadata/checkpoint storage
- Graph: unified builder, edge helpers, typed events, executor wrapper, metrics
- API: app-state DI, shared models, typed error handlers, SSE streaming, auth/CORS

**Phase 3 Company-Readiness:**
- YAML config support with environment overlays (development/staging/production)
- Typed `RAGError` hierarchy and shared FastAPI error handlers
- Typed graph events via `GraphExecutor` (node lifecycle, retriever, grader)
- LLM provider seam in `src/llm/provider.py`
- SSE endpoints: `POST /query/stream`, `POST /chat/{thread_id}/message/stream`
- In-process metrics and `GET /metrics` on both QA and chat apps
- SQLite-backed session metadata storage with schema versioning
- SQLite checkpoint persistence for chat threads via `SQLiteMemorySaver`
- Deprecation warnings on legacy import paths
- CI workflow (`.github/workflows/ci.yml`), pre-commit, Docker, docker-compose
- API-key auth, CORS config, `/ready` endpoint
- Team docs: ARCHITECTURE, CHANGELOG, CONTRIBUTING, SECURITY, COMPANY_READINESS_GAPS

## Test Status

**130 total tests — 130 passed**

Latest verification (web-answer grounding + date-aware prompt pass):
`.venv\Scripts\python.exe -m pytest -p no:warnings` → 130 passed;
`python -m compileall src tests` passed.

### Answer-Quality Remediation (web search → answer)

A user-reported failure ("2026 DeepSeek model" query answered from ~2024
training data) was traced to three issues in the lightweight web-answer path
and fixed:

| Issue | Status | Evidence |
|---|---|---|
| `web_answer` preferred the agent's redundant in-graph `live_web_search` URLs over the session's curated discovered URLs | Fixed | `_extract_source_urls` in `src/graph/nodes/web_answer.py` now prefers `state["source_urls"]` → `settings.source_urls` → tool-message URLs (fallback only). |
| Ungrounded answers when no page had readable text (empty context → model used training data) | Fixed | `web_answer` skips the LLM and returns a grounded refusal listing attempted URLs when no fetched page has readable text. |
| Answers anchored to training cutoff (~2024); model dismissed post-cutoff facts | Fixed | Added current date + source-trust instructions to `src/web_search/prompt_builder.py` and `RAG_PROMPT` in `src/llm/prompts.py`; `{current_date}` bound from `date.today()` in the generate node. |
| Longer instructional header starved source content in the token budget | Fixed | `build_web_search_prompt` excludes the fixed preamble from the per-source budget. |

Remaining note: the agent still runs a redundant in-graph `live_web_search`
each turn even when the session already has curated URLs. It is now harmless
(results no longer win) but is a future efficiency optimization.

### Earlier 2026-06-09 first-wave fixes

Previously failing tests fixed in the 2026-06-09 first-wave pass:

1. **`test_chat_unknown_thread_uses_typed_not_found_response`**
   - File: `tests/test_api_errors.py`
   - Fixed by moving config/graph-lock resolution after the unknown-thread
     session lookup in `src/chat/api.py`.

2. **`test_sqlite_memory_saver_restores_graph_state_after_reopen`**
   - File: `tests/test_sessions.py`
   - Fixed by making `SQLiteMemorySaver` tolerate installed LangGraph saver
     variants that do not expose a `blobs` backing map.

Test files (15 total):
`tests/conftest.py`, `tests/test_api_dependencies.py`, `tests/test_api_errors.py`,
`tests/test_api_security_readiness.py`, `tests/test_api_streaming.py`,
`tests/test_compat.py`, `tests/test_config.py`, `tests/test_graph_builder.py`,
`tests/test_graph_executor.py`, `tests/test_graph_state_and_nodes.py`,
`tests/test_rag_interfaces.py`, `tests/test_retry.py`, `tests/test_sessions.py`,
`tests/test_urls.py`, `tests/test_web_search_providers.py`

Latest verification:
`python -m pytest -q` passed with 113 tests; `python -m compileall src tests`
passed; `git diff --check` passed with line-ending warnings only.

## Verification Commands

```bash
python -m pytest -q                  # Run all tests
python -m compileall src tests       # Syntax check
git diff --check                     # Whitespace check
```

## Web Search — Remediation Status

A code review (`docs/code-review-web-search.md`) and problem investigation
(`PROBLEMS_DETECTED.md`) identified web-search performance and answer-quality
gaps. The 2026-06-09 RAGRefactorDeveloper waves remediated the actionable code
items while keeping the implementation deterministic and dependency-free.

| Area | Status | Evidence |
|---|---|---|
| Sequential URL loading | Complete | `src/rag/document_loader.py` now uses bounded concurrent loading and preserves source order. |
| Per-URL timeout compounding | Mitigated | `page_load_max_concurrency` limits wall-clock impact while preserving `page_load_timeout`. |
| No HTTP/content caching | Complete | `page_load_cache_ttl_seconds` enables opt-in in-process caching for successful loads. |
| Poor pre-index content quality | Complete | `src/rag/document_quality.py` filters empty, short, boilerplate, low-signal, and optionally query-mismatched documents before splitting/embedding. |
| No post-retrieval re-ranking | Complete | `src/graph/nodes/common.py` re-ranks retrieved chunks deterministically before grading/generation. |
| Weak URL quality gate/provider fallback | Complete | `src/web_search/common.py` scores/deduplicates/filter URLs and `src/web_search/discovery.py` falls back when provider output has no usable URLs. |
| No Bing recency filter | Complete | `src/web_search/bing.py` maps `d/day`, `w/week`, and `m/month` timelimits to Bing freshness filters. |
| Missing runtime agent web-search tool | Complete | `src/web_search/tool.py` exposes `live_web_search`, and `src/graph/builder.py` adds it when `web_search_enabled` is true. |
| `web_search_top_k` config drift | Complete | Canonical default is 6 across built-in settings, YAML, `.env.example`, and README. |

Remaining architectural tradeoff: web-discovered source sets still rebuild a
single-use Chroma index and re-embed loaded chunks. The new filtering, caching,
and runtime URL tool reduce avoidable latency and poor inputs, but incremental
or shared web-search vectorstore reuse remains a future optimization.

## Web Search Providers

Three providers implemented behind a common `WebSearchProvider` protocol:

| Provider | Module | Notes |
|---|---|---|
| Bing | `src/web_search/bing.py` | Default; HTML scraping with anti-bot detection |
| Baidu | `src/web_search/baidu.py` | HTML scraping with redirect resolution |
| DuckDuckGo | `src/web_search/duckduckgo.py` | DDGS package + HTML fallback; supports `timelimit` |

Provider fallback chain: `bing → baidu → duckduckgo`
Cooldown on captcha/verification: 300 seconds

## Remaining Company-Readiness Work

Tracked in `COMPANY_READINESS_GAPS.md`:

- Install dev dependencies in CI for ruff, mypy, coverage gates
- Dependency scanning (`pip-audit` or Dependabot)
- Structured JSON logging and request/session correlation IDs
- API versioning under `/api/v1/`
- Deployment-specific infrastructure
- Rate limiting and session export endpoint

## Config Drift Notes

`web_search_top_k` drift is resolved. The canonical default is now `6` across
`src/config/settings.py`, `config/default.yaml`, `.env.example`, and README.

## Active Refactor Work Coordination

Started: 2026-06-09

Parent thread owns process-document updates for `memory/current-process.md` and
`memory/refactor-daily-forms.md` to avoid concurrent write conflicts.

Parallel RAGRefactorDeveloper workers:

| Scientist name | Agent id | Tool nickname | Work stream | Dependency |
|---|---|---|---|---|
| Ada Lovelace | `019eaa2c-67f9-7823-8862-9daf50bb86f2` | Avicenna | Fix chat unknown-thread typed 404 test | Complete and integrated |
| Isaac Newton | `019eaa2c-b1cf-77b2-99d8-9c4a313e779e` | Dewey | Fix `SQLiteMemorySaver` LangGraph compatibility test | Complete and integrated |
| Grace Hopper | `019eaa2d-1517-7502-baad-e4d8a8726788` | Goodall | Parallelize web URL loading in `src/rag/document_loader.py` | Complete and integrated |
| Marie Curie | `019eaa35-994d-7d51-ae8a-335b20751962` | Faraday | Align web-search config defaults and plumb URL-load concurrency setting | Complete and integrated |
| Katherine Johnson | `019eaa35-eb26-7760-aee8-57d1e8650fba` | Dirac | Add runtime LangChain web-search tool and graph wiring | Complete and integrated |
| Emmy Noether | `019eaa36-450f-78e2-8669-8379f5d8a2ee` | Feynman | Add provider-result quality filtering and provider fallback | Complete and integrated |
| Nikola Tesla | `019eaa40-493f-7752-a151-2920bd7aba43` | Averroes | Add in-process source document caching | Complete and integrated |
| Rosalind Franklin | `019eaa40-98a3-7001-9d3e-14997bece7e7` | Lorentz | Add Bing timelimit/recency support | Complete and integrated |
| Lise Meitner | `019eaa4a-b12f-7352-8568-01fe4e67eff7` | Archimedes | Add pre-index document content quality/relevance filtering | Complete and integrated |
| Alan Turing | `019eaa4b-07d4-7692-b7a1-39d178d97523` | Erdos | Add deterministic post-retrieval re-ranking | Complete and integrated |

Completed execution order:

1. Fixed the two baseline regressions and verified the suite.
2. Parallelized URL loading, then aligned config defaults and concurrency.
3. Added runtime web-search tooling and provider-result quality fallback.
4. Added opt-in in-process source document caching and Bing timelimit support.
5. Added pre-index document quality filtering and post-retrieval re-ranking.
6. Updated README and process files after all code work stabilized.

First-wave completed worker verification:

- Ada Lovelace changed `src/chat/api.py`; focused chat 404/API error and
  streaming tests passed in the worker.
- Isaac Newton changed `src/sessions/checkpoint.py`; focused SQLite checkpoint
  and full session tests passed in the worker.
- Grace Hopper changed `src/rag/document_loader.py` and
  `tests/test_rag_interfaces.py`; focused RAG interface tests passed in the
  worker, while `ruff` was unavailable.
- Parent integrated the first-wave patches and verified the suite: `python -m
  pytest -q` passed with 87 tests; `python -m compileall src tests` passed;
  `git diff --check` passed with line-ending warnings only.
