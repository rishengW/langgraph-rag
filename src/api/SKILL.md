---
name: api-interface-architect
description: >
  Use this skill whenever working on the FastAPI application layer, CLI entry
  points, request/response models, dependency injection, SSE streaming transport,
  or endpoint design. Covers the HTTP interface for both QA and Chat apps, the
  CLI subcommands, URL parsing, global state elimination, and the target
  src/api/ module. Trigger on mentions of FastAPI, endpoints, /query, /chat,
  /health, SSE, streaming, request models, response models, CLI, or API refactoring.
---

# API Interface Architect — src/api/

Domain: FastAPI application, CLI entry points, request/response models, SSE transport, dependency injection.
Parent: `SKILL.md` (root). Siblings: `src/graph/SKILL.md`, `src/rag/SKILL.md`, `src/web_search/SKILL.md`, `src/sessions/SKILL.md`.

## Quick Reference

| Fact | Value |
|---|---|
| Framework | FastAPI 0.110-0.120 |
| Server | uvicorn 0.27+ |
| QA port | 8000 |
| Chat port | 8001 |
| API style | REST + planned SSE streaming |
| Request validation | Pydantic v2 |
| Static files | Served via `StaticFiles` middleware |
| CLI | argparse with subcommands (query, serve, rebuild, chat) |
| Source files (current) | `qa/main.py`, `qa/api.py`, `chat/main.py`, `chat/api.py` |
| Target location | `src/api/` |

## File Map (Current)

```
src/qa/
+-- main.py                 # CLI entry point + arg parsing — 225 LOC
+-- api.py                  # FastAPI app, 3 endpoints, global _graph/_settings state — 337 LOC
+-- draw_graph.py           # Graph visualization utility — 24 LOC
+-- static/
    +-- index.html          # QA web UI
    +-- script.js           # QA frontend logic

src/chat/
+-- main.py                 # CLI: serve + chat REPL subcommands — 171 LOC
+-- api.py                  # FastAPI app: /chat CRUD + /health + / — 362 LOC
+-- static/
    +-- index.html          # Chat UI
    +-- script.js           # Chat frontend logic
```

### Detailed File Responsibilities

| File | Responsibility | Key Symbols | LOC |
|------|---------------|-------------|-----|
| `qa/main.py` | CLI arg parser, `query`/`serve`/`rebuild` subcommands, backward-compat arg handling | `parse_args`, `cmd_query`, `cmd_serve`, `cmd_rebuild` | 225 |
| `qa/api.py` | FastAPI app, 3 endpoints, URL parsing, rebuild locking, per-request graph rebuild | `QueryRequest`, `QueryResponse`, `_graph`, `_settings`, `app` | 337 |
| `chat/main.py` | CLI, `serve`/`chat` subcommands, REPL loop | `cmd_serve`, `cmd_chat`, `_repl_loop` | 171 |
| `chat/api.py` | FastAPI app for chat, per-session graph building, 5 endpoints | `StartChatRequest`, `MessageRequest`, `app` | 362 |

## Endpoint Map

### QA App (port 8000)

| Method | Path | Purpose | Handler Logic |
|--------|------|---------|---------------|
| `GET` | `/` | Serve HTML UI | `FileResponse("static/index.html")` |
| `GET` | `/health` | Liveness check | Returns `{"status": "healthy"}` |
| `POST` | `/query` | Execute RAG query | See flow below |

### Chat App (port 8001)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Serve chat UI |
| `GET` | `/health` | Liveness check |
| `POST` | `/chat` | Start new chat session |
| `POST` | `/chat/{id}/message` | Send message turn |
| `GET` | `/chat/{id}/history` | Fetch conversation transcript |
| `DELETE` | `/chat/{id}` | Delete session |

### Planned SSE Streaming (Phase 3)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/query/stream` | SSE streaming query |
| `POST` | `/chat/{id}/message/stream` | SSE streaming chat turn |

## QA Endpoint Flow (POST /query)

```
1. Parse and validate QueryRequest (question, urls, rebuild, web_search, debug)
2. Determine source_mode:
   a. urls provided -> "explicit"
   b. web_search enabled -> discover_urls_from_web()
   c. otherwise -> "defaults" (use _settings.source_urls)
3. If rebuild requested:
   a. Acquire _rebuild_lock (async)
   b. Rebuild graph from scratch
   c. Restore old graph on failure (complex try/finally)
4. Run run_rag_query(graph, question, debug)
5. Return QueryResponse(answer, error, success, messages, source_urls, source_mode, source_note)
```

## Request/Response Models

### QA

```python
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    urls: str | list[str] | None = None
    rebuild: bool = False
    web_search: bool = True
    debug: bool = False

class QueryResponse(BaseModel):
    answer: str | None
    error: str | None
    success: bool = True
    messages: list[str] | None          # debug mode only
    source_urls: list[str] | None
    source_mode: str | None             # "explicit" | "web_search" | "defaults"
    source_note: str | None             # "web_search_failed" | "web_search_no_results" | etc.
```

### Chat

```python
class StartChatRequest(BaseModel):
    urls: str | list[str] | None
    web_search: bool = True
    seed_question: str | None

class StartChatResponse(BaseModel):
    thread_id: str
    source_urls: list[str]
    source_mode: str
    source_note: str | None

class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1)

class MessageResponse(BaseModel):
    thread_id: str
    answer: str
    error: str | None

class HistoryTurn(BaseModel):
    role: str     # "user" | "assistant" | "tool"
    content: str

class HistoryResponse(BaseModel):
    thread_id: str
    turns: list[HistoryTurn]
    source_urls: list[str]
    source_mode: str
```

## CLI Entry Points

### QA CLI (`python -m src.qa.main`)

| Subcommand | Purpose | Flags |
|------------|---------|-------|
| `query` | Run a single RAG query from CLI | `question`, `--urls`, `--rebuild`, `--web-search` |
| `serve` | Start the FastAPI server | `--port` (default: 8000), `--host` |
| `rebuild` | Rebuild Chroma vector database and exit | `--urls` |

Backward-compatible syntax also supported:
```bash
python -m src.qa.main "What is X?" --urls "a,b" --rebuild
```

### Chat CLI (`python -m src.chat.main`)

| Subcommand | Purpose | Flags |
|------------|---------|-------|
| `serve` | Start the FastAPI server | `--port` (default: 8001), `--host` |
| `chat` | Interactive terminal REPL | `--urls`, `--web-search` |

## Current Architecture Issues

### 1. Global Mutable State (Critical)

```python
# src/qa/api.py:66-70 — ANTI-PATTERN
_graph = None          # module-level global
_settings = None       # module-level global
_rebuild_lock: asyncio.Lock = asyncio.Lock()
```

**Problems:**
- Concurrent requests race on `_graph` reads vs rebuilds
- Lock only protects rebuild, not the read path
- Impossible to unit-test without mocking globals
- Graph rebuild failure leaves `_graph = None` permanently
- Complex try/finally restore logic for failed rebuilds (lines 264-274)

### 2. URL Parsing Duplication (Medium)

Both `qa/api.py:_parse_request_urls()` and `chat/api.py:_parse_urls()` implement identical logic:
- Handle `str` (comma-separated), `list[str]`, `None`
- Strip whitespace, filter empties

**Fix (Phase 1):** Extract to `src/utils/urls.py`.

### 3. Per-Request Graph Rebuild (Medium)

When URLs change, the QA API rebuilds the entire graph (including Chroma index) inside the request handler. This:
- Blocks the event loop during long indexing
- Duplicates rebuild logic from `graph_executor.py`
- Has complex try/finally restore logic (api.py:264-274)

**Fix (Phase 2):** Separate "build index" from "build graph."

### 4. CLI Complexity (Low)

`src/qa/main.py:parse_args()` has backward-compatibility logic (~50 lines) that manually parses unknown args to reconstruct old-style commands.

## Target Architecture — src/api/

```
src/api/
+-- __init__.py
+-- dependencies.py        # FastAPI Depends() providers (get_graph, get_config, get_registry)
+-- routers/
    +-- __init__.py
    +-- qa.py              # QA endpoints: /, /health, /query
    +-- chat.py            # Chat endpoints: /chat CRUD + /health
    +-- stream.py          # SSE endpoints: /query/stream, /chat/{id}/message/stream
+-- models/
    +-- __init__.py
    +-- qa_models.py       # QueryRequest, QueryResponse
    +-- chat_models.py     # All chat models
    +-- events.py          # StreamEvent, typed event models
+-- static/
    +-- qa/
        +-- index.html
        +-- script.js
    +-- chat/
        +-- index.html
        +-- script.js
```

### Dependency Injection Design (Phase 2)

```python
# src/api/dependencies.py
from fastapi import Request

async def get_config(request: Request) -> AppConfig:
    return request.app.state.config

async def get_qa_graph(request: Request) -> CompiledGraph:
    if request.app.state.qa_graph is None:
        request.app.state.qa_graph = build_graph(
            mode="qa",
            config=request.app.state.config,
        )
    return request.app.state.qa_graph

async def get_rebuild_lock(request: Request) -> asyncio.Lock:
    return request.app.state.rebuild_lock

async def get_session_registry(request: Request) -> ChatSessionRegistry:
    return request.app.state.session_registry
```

### SSE Streaming Design (Phase 3)

```
Client (EventSource)                    FastAPI Server
      |                                      |
      +-- POST /query/stream --------------->|
      |  {"question": "...", ...}            |
      |                                      +-- build graph
      |                                      +-- start async executor
      |<---- event: node_start --------------+  {node: "agent"}
      |<---- event: node_end ----------------+  {node: "agent"}
      |<---- event: node_start --------------+  {node: "retrieve"}
      |<---- event: retriever_result --------+  {num_docs: 4}
      |<---- event: node_end ----------------+  {node: "retrieve"}
      |<---- event: node_start --------------+  {node: "grade_documents"}
      |<---- event: grader_decision ---------+  {score: "yes"}
      |<---- event: node_end ----------------+  {node: "grade_documents"}
      |<---- event: node_start --------------+  {node: "generate"}
      |<---- event: token -------------------+  {token: "The"}
      |<---- event: token -------------------+  {token: " article"}
      |<---- event: node_end ----------------+  {node: "generate"}
      |<---- event: done --------------------+  {answer: "The article discusses..."}
```

```python
class StreamEvent(BaseModel):
    event: str                        # "node_start" | "node_end" | "token" | "error" | "done"
    node: str | None                  # graph node name
    data: str | dict | None           # payload
```

## Known Issues — API Layer

| # | Issue | Severity | Location | Fix Phase |
|---|-------|----------|----------|-----------|
| 1 | Module-level globals `_graph`, `_settings`, `_rebuild_lock` | Critical | `qa/api.py:66-70` | Phase 2 |
| 2 | URL parsing duplicated across qa/ and chat/ | Medium | Both `api.py` | Phase 1 |
| 3 | `/health` doesn't distinguish liveness from readiness | Medium | Both `api.py` | Phase 3 |
| 4 | No rate limiting | Medium | All endpoints | Phase 3 |
| 5 | Per-request graph rebuild blocks event loop | Medium | `qa/api.py:248-278` | Phase 2 |
| 6 | Failed rebuild restores globals in complex try/finally | Medium | `qa/api.py:264-274` | Phase 2 |
| 7 | No request ID / correlation ID in responses | Low | All endpoints | Phase 2 |
| 8 | Static files embedded in qa/ and chat/ separately | Low | Both `static/` | Phase 2 (unify) |
| 9 | CLI backward-compat arg parsing is fragile (~50 lines) | Low | `qa/main.py:89-133` | Phase 2 |
| 10 | Chat history serialization skips tool messages (intentional but undocumented) | Low | `chat/api.py:_serialize_messages()` | Document |

## Refactoring To-Do List

> Source: [`REFACTORING_PLAN.md`](../../REFACTORING_PLAN.md). API/CLI scope items.

### Phase 1 — Extract Without Behavioral Change

- [ ] **1.7 Consolidate URL parsing** — extract `_parse_request_urls()` from `qa/api.py` → `src/utils/urls.py`
  - [ ] Replace both `qa/api.py` and `chat/api.py` with shared version
  - [ ] Verify: null, empty string, list, CSV string all handled identically
- [ ] **1.7 Clean CLI arg parsing** — remove ~50 lines of backward-compat logic; deprecate old-style syntax

### Phase 2 — Interfaces & Abstractions

- [ ] **2.3 Remove global mutable state** — eliminate `_graph`, `_settings`, `_rebuild_lock` globals
  - [ ] Create `src/api/dependencies.py` with FastAPI `Depends()` providers
  - [ ] Move to `request.app.state`: `graph`, `config`, `rebuild_lock`, `session_registry`
  - [ ] Verify: multiple concurrent requests have no shared state pollution
- [ ] **2.3 Separate graph build from Chroma index** — decouple slow index from fast graph construction
- [ ] **2.5 Add YAML configuration support**
  - [ ] Create `config/default.yaml` with all current defaults
  - [ ] Load priority: CLI args > env vars > YAML file > built-in defaults
  - [ ] Support `--config` CLI flag; keep backward compat with `.env`-only
- [ ] **2.6 Error handlers** — map `RAGError` codes to HTTP status codes in FastAPI exception handlers
- [ ] **2.7 Unify static files** — `src/api/static/qa/` + `src/api/static/chat/`

### Phase 3 — Streaming & Deprecation

- [ ] **3.1 SSE streaming endpoints**
  - [ ] `POST /query/stream` — `StreamingResponse` with `text/event-stream`
  - [ ] `POST /chat/{id}/message/stream` — see `src/sessions/SKILL.md`
  - [ ] Emit `StreamEvent` for: `node_start`, `token`, `retriever_result`, `grader_decision`, `node_end`, `done`
  - [ ] Update both frontend JS files — replace `fetch()` with `EventSource`
- [ ] **3.3 Add `/metrics` endpoint** — per-node latency/error counters
- [ ] **3.5 Deprecation shims**
  - [ ] `src/qa/api.py` → thin wrapper calling `src/api/routers/qa.py`
  - [ ] `src/chat/api.py` → thin wrapper calling `src/api/routers/chat.py`
  - [ ] Old import paths emit `DeprecationWarning`; remain functional through Phase 3
- [ ] **3.5 Rate limiting** — `slowapi` middleware on all endpoints
- [ ] **3.5 Request ID / correlation ID** — inject `X-Request-ID` into all responses
- [ ] **3.5 Health endpoints** — distinguish `/health` (liveness) from `/ready` (readiness with graph loaded)

## Testing Strategy

| Test | Approach | Phase |
|------|----------|-------|
| Endpoint contract tests | `TestClient` with mock graph, verify status codes + response shape | Phase 1 |
| Request validation | Invalid JSON, missing fields, type errors → 422 | Phase 1 |
| Health/readiness | Mock graph state, verify HTTP codes | Phase 2 |
| Global state isolation | Concurrent requests, verify no cross-contamination | Phase 2 |
| SSE event format | `TestClient.stream()` → parse SSE lines, verify event order | Phase 3 |
| Rate limiting | Hammer endpoint, verify 429 responses | Phase 3 |
| Error response format | Trigger each `RAGError` type, verify code + message | Phase 2 |
| CLI regression | All old CLI commands produce identical output | Phase 1 |

## Dependencies

- `src/graph/` — graph builder + executor (see `src/graph/SKILL.md`)
- `src/sessions/` — ChatSessionRegistry for chat endpoints (see `src/sessions/SKILL.md`)
- `src/config/` — AppConfig for DI
- External: `fastapi`, `uvicorn`, `pydantic`, `slowapi` (Phase 3)
