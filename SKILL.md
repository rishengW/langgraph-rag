---
name: system-architect
description: >
  Use this skill whenever working on the overall architecture, module boundaries,
  configuration management, or deployment of the langgraph-rag project. Covers
  directory layout, dependency management, settings design, and cross-cutting
  concerns. Trigger on mentions of project structure, config, settings, .env,
  deployment, module organization, or when planning multi-module changes.
---

# System Architect -- langgraph-rag

Domain: overall architecture, configuration, module boundaries, deployment.
Sub-architects: `src/graph/SKILL.md`, `src/rag/SKILL.md`, `src/web_search/SKILL.md`, `src/api/SKILL.md`, `src/sessions/SKILL.md`.

## Quick Reference

| Fact | Value |
|---|---|
| Project | `langgraph-rag` ("only Subcribers") |
| Language | Python 3.11+ |
| LLM Provider | DashScope (Alibaba Tongyi/Qwen) -- `qwen-plus` |
| Embeddings | DashScope `text-embedding-v4` (1024-dim) + HuggingFace fallback |
| Vector Store | Chroma (SQLite, local `.chroma/` directory) |
| Framework | LangGraph 0.2-0.7, LangChain 0.3.x |
| Web Server | FastAPI + uvicorn |
| Web Search | Baidu HTML scraping + DuckDuckGo (ddgs + HTML fallback) |
| Config | `.env` via python-dotenv (YAML support planned: Phase 2) |
| Test Framework | pytest (planned: Phase 1) |

## Architecture Map (Functional Scopes)

```
                         Entry Points
  python -m src.qa.main query|serve    (port 8000)
  python -m src.chat.main chat|serve   (port 8001)

  +-----------------------+     +---------------------------+
  |     src/api/           |     |     src/sessions/          |
  |  FastAPI + CLI         |     |  Condense + MemorySaver   |
  |  (api/SKILL.md)             |  |  (sessions/SKILL.md)         |
  +-----------+-----------+     +-----------+---------------+
              |                             |
              +-------------+---------------+
                            |
              +-------------+---------------+
              |      src/graph/              |
              |  Nodes + State + Executor    |
              |  (graph/SKILL.md)            |
              +-------------+---------------+
                            |
        +-------------------+-------------------+
        |                                       |
  +-----+--------+                    +---------+------+
  |  src/rag/     |                    | src/web_search/ |
  |  Embeddings   |                    |  Baidu + DDG    |
  |  + Chroma     |                    |  URL Discovery  |
  |  (rag/SKILL.md)                   |  (web_search/SKILL.md)    |
  +--------------+                    +----------------+
```

### Module Responsibilities

| Module | Responsibility | Architect Skill File |
|--------|---------------|---------------------|
| `src/graph/` | RAG pipeline — graph topology, node factories, state management, question resolution, graph execution | `src/graph/SKILL.md` |
| `src/rag/` | Knowledge retrieval — embeddings, Chroma vectorstore, document loading, text splitting, retriever | `src/rag/SKILL.md` |
| `src/web_search/` | Web search — URL discovery, Baidu/DDG providers, result normalization, noise filtering | `src/web_search/SKILL.md` |
| `src/api/` | API & CLI — FastAPI endpoints, request/response models, DI, SSE transport, CLI commands | `src/api/SKILL.md` |
| `src/sessions/` | Conversation engine — condense, session registry, MemorySaver, TTL, persistence | `src/sessions/SKILL.md` |

Refer to each sub-architect skill for detailed file maps, flows, known issues, and refactoring to-do lists within that scope.

## Current Source Tree (Overview)

```
src/
+-- core/           # Shared RAG engine — being refactored into functional modules below
+-- qa/             # Single-shot Q&A — API migrating to src/api/
+-- chat/           # Multi-turn chat — API to src/api/, sessions to src/sessions/
+-- graph/          # [TARGET] RAG pipeline — nodes, state, executor
+-- rag/            # [TARGET] Knowledge retrieval — embeddings, Chroma, retriever
+-- web_search/     # [TARGET] Web search — Baidu, DDG, URL discovery
+-- api/            # [TARGET] API & CLI — FastAPI routers, models, DI, SSE
+-- sessions/       # [TARGET] Conversation engine — sessions, condense, persistence
```

**Line counts:** core/ = ~1700 LOC, qa/ = ~700 LOC, chat/ = ~1300 LOC.
Detailed file maps are in each functional-scope SKILL.md.

## Configuration Landscape

### Loading

```
CLI args > env vars > YAML config file (planned Phase 2) > built-in defaults
```

Secrets (API keys) always come from env vars, never from YAML.

### Current: Settings dataclass (42 flat fields)

Groups of related fields:

| Concern | Fields |
|---|---|
| Auth | `dashscope_api_key` |
| Model | `qwen_model`, `embedding_model`, `embedding_dimension`, `embedding_batch_size` |
| Chroma | `chroma_dir`, `collection_name`, `chunk_size`, `chunk_overlap` |
| Web Search | `web_search_enabled`, `web_search_provider`, `web_search_max_results`, `web_search_top_k`, `web_search_region`, `web_search_timelimit`, `web_search_verify_ssl` |
| Network | `page_load_timeout`, `dashscope_request_timeout`, `dashscope_max_retries`, `dashscope_http_base_url` |
| Tracing | `langchain_tracing_v2`, `langchain_api_key`, `langchain_project` |
| API | `api_host`, `api_port`, `source_urls` |
| Tuning | `allow_low_relevance_generate`, `min_keyword_matches`, `max_rewrites` |

### Target Phase 2: namespaced sub-models

```python
@dataclass(frozen=True)
class Settings:
    dashscope_api_key: str              # secret: env only
    llm: LLMConfig                       # model, timeout, retries
    embedding: EmbeddingConfig            # model, dimension, batch_size
    chroma: ChromaConfig                  # dir, collection
    retrieval: RetrievalConfig            # chunk, overlap, keywords, rewrites
    search: SearchConfig                  # provider, max_results, top_k
    tracing: TracingConfig                # langsmith
    server: ServerConfig                  # host, port
    source_urls: list[str]
```

## Dependency Groups

```
requirements.txt
+-- Core Stack:    langchain, langgraph, chromadb, dashscope
+-- Web:           fastapi, uvicorn, pydantic
+-- Web Search:    beautifulsoup4, ddgs
+-- Optional:      sentence-transformers, tiktoken
+-- Dev (planned): pytest, pytest-asyncio, httpx, mypy, ruff
```

## Cross-Cutting Concerns

### 1. Error Handling Strategy

Current state: inconsistent -- `RuntimeError` everywhere. Target (Phase 2):

```
RAGError (base, code="RAG_ERROR")
+-- ConfigurationError    ("CONFIGURATION_ERROR")
+-- LLMUnavailableError   ("LLM_UNAVAILABLE")
+-- RetrieverError        ("RETRIEVER_ERROR")
+-- WebSearchError        ("WEB_SEARCH_ERROR")
+-- DocumentLoadError     ("DOCUMENT_LOAD_ERROR")
+-- AllSourcesFailedError ("ALL_SOURCES_FAILED")
```

FastAPI error handlers map `RAGError.code` to HTTP status codes.

### 2. Logging Strategy

Current: `print()` for graph node entry/exit, `logging.getLogger()` sparsely used.

Target:
- **Phase 1**: Replace all `print()` with `logger.info()`
- **Phase 2**: Structured JSON logging with `LogContext` (request_id, session_id, source_mode, rewrite_count)
- **Phase 3**: Metrics collector for per-node latency/error counters

### 3. Async Strategy

Current: synchronous everywhere. Chat API wraps in `asyncio.to_thread()`.

Target:
- **Phase 2**: `AsyncRAGExecutor` wraps `graph.stream()` in thread pool
- **Phase 3**: `httpx.AsyncClient` for web search; SSE streaming endpoints
- LangGraph remains sync (native async LangGraph is a future iteration)

### 4. Global State Elimination

Current: `_graph`, `_settings`, `_rebuild_lock` as module-level globals in `qa/api.py`.

Target (Phase 2):
- `app.state.graph` -- FastAPI `request.app.state`
- `app.state.config` -- FastAPI `request.app.state`
- `app.state.rebuild_lock` -- per-app asyncio.Lock
- Injected via `fastapi.Depends()`

## Known Issues (Cross-Module)

| # | Issue | Severity | Notes |
|---|---|---|---|
| 1 | `load_settings()` mutates `os.environ` as side effect | High | Split into pure loader + env setter (Phase 1) |
| 2 | `DASHSCOPE_API_KEY` written to `os.environ` by config loader | High | Should be injected, not globally set |
| 3 | No `pyproject.toml` or build config | Medium | Add in Phase 1 |
| 4 | Requirements not grouped by concern | Low | Regroup in Phase 1 |
| 5 | `SSL_FIX.md` documents workarounds for Windows cert issues | Low | Should be integrated into setup docs |
| 6 | No test infrastructure | Critical | Add in Phase 1 (pytest + conftest) |

## Refactoring Target Summary

See `REFACTORING_PLAN.md` for the full 1465-line architectural plan. New modules planned:

| New Module | Purpose | Phase |
|---|---|---|
| `src/config/` | Pure Settings + YAML loader (no env side-effects) | 1-2 |
| `src/graph/` | Unified graph builder + nodes + state + executor | 1-2 |
| `src/llm/` | LLM provider protocol + DashScope impl + prompts + retry | 1-2 |
| `src/rag/` | Retriever protocol + Chroma impl + embeddings + doc loader | 2 |
| `src/web_search/` | Search provider protocol + Baidu/DDG impls | 2 |
| `src/api/` | Unified FastAPI routers, DI, models (deprecate qa/api.py, chat/api.py) | 2-3 |
| `src/events/` | Typed event system for streaming | 2 |
| `src/sessions/` | Session registry with optional SQLite persistence | 2-3 |
| `src/utils/` | Shared retry, URL parsing, networking utilities | 1 |

## Phase Rollout Strategy

Each phase is a separate branch, independently mergeable and revertible:

```
main
+-- refactor/phase-1-extract      (3-5 days, low risk)
+-- refactor/phase-2-interfaces   (5-8 days, medium risk, based on phase-1)
+-- refactor/phase-3-streaming    (5-10 days, medium-high risk, based on phase-2)
```

Backward compatibility: `src/core/` and old import paths remain as thin re-exports through Phase 3. Deprecation warnings added in Phase 3.

## Refactoring To-Do List

> Source: [`REFACTORING_PLAN.md`](./REFACTORING_PLAN.md). Each item maps to a specific step in the roadmap.

### Phase 1 — Extract Components (Low Risk, 3-5 days)

- [ ] **Add `pyproject.toml`** — project metadata + pytest configuration (Step 1.1)
- [ ] **Create test harness skeleton** — `tests/conftest.py` with mock fixtures for settings, LLM, retriever (Step 1.1)
- [ ] **Create `src/utils/`** — `retry.py`, `networking.py`, `urls.py` (Steps 1.2, 1.7)
- [ ] **Replace all `print()` with `logging`** across all modules (Step 1.3)
  - [ ] `core/nodes.py` — 7 print statements at lines 232, 279, 285, 289, 294, 301
  - [ ] `chat/nodes.py` — equivalent print statements
- [ ] **Split `core/config.py`** — `src/config/settings.py` (pure dataclass) + `src/config/loader.py` (env reading, no `os.environ` mutation) (Step 1.4)
- [ ] **Reorganize `requirements.txt`** into dependency groups: Core Stack, Web, Web Search, Optional, Dev (Step 1.1)
- [ ] **Extract prompt templates** — `RAG_PROMPT`, `CONDENSE_PROMPT`, grade prompt → `src/llm/prompts.py` (Step 1.6)

### Phase 2 — Interfaces & Abstractions (Medium Risk, 5-8 days)

- [ ] **Implement typed error hierarchy** — `RAGError` base + 6 subclasses, FastAPI error handlers map codes to HTTP status (Step 2.6)
- [ ] **Structured JSON logging** — `LogContext` with `request_id`, `session_id`, `source_mode`, `rewrite_count` (Section 3.5)
- [ ] **Create `src/api/`** — unified FastAPI layer with dependency injection (Step 2.3)
  - [ ] `dependencies.py` — `get_graph()`, `get_config()`, `get_session_registry()`, `get_rebuild_lock()`
  - [ ] `routers/qa.py`, `routers/chat.py`
  - [ ] `models/qa_models.py`, `models/chat_models.py`
- [ ] **YAML config support** — `config/default.yaml` + updated `config/loader.py` with priority: CLI > env > YAML > defaults (Step 2.5)
- [ ] **Create `src/events/`** — typed event system: `NodeStartEvent`, `NodeEndEvent`, `TokenEvent`, `RetrieverResultEvent`, `GraderDecisionEvent`, `ErrorEvent`, `DoneEvent` (Step 2.4)

### Phase 3 — Streaming & Optimization (Medium-High Risk, 5-7 days)

- [ ] **SSE streaming endpoints** — `POST /query/stream`, `POST /chat/{id}/message/stream` (Step 3.1)
- [ ] **Metrics collector** — per-node latency/error counters + `GET /metrics` endpoint (Step 3.3)
- [ ] **Deprecation shims** — old `src/core/`, `src/qa/`, `src/chat/` import paths with `DeprecationWarning` (Step 3.5)
- [ ] **Update `README.md`** with new project structure and import paths

### Exit Criteria

| Phase | Key Gating Criteria |
|-------|-------------------|
| Phase 1 | All CLI commands produce identical output; >80% test coverage on extracted modules; zero `print()` in graph code |
| Phase 2 | Graph builds from mock providers; TestClient passes with no shared state issues; YAML + env merge works; error responses include typed codes |
| Phase 3 | Browser shows real-time tokens; sessions survive restart; old imports emit `DeprecationWarning` |

## References

- `REFACTORING_PLAN.md` -- full architectural plan (1465 lines)
- `README.md` -- user-facing documentation
- `.env.example` -- annotated environment variables
- `SSL_FIX.md` -- Windows SSL workarounds
- `src/graph/SKILL.md` — RAG pipeline architect
- `src/rag/SKILL.md` — Knowledge retrieval architect
- `src/web_search/SKILL.md` — Web search architect
- `src/api/SKILL.md` — API interface architect
- `src/sessions/SKILL.md` — Session engine architect
