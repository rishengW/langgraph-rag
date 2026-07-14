---
name: rag-operations-runbook
description: >
  Use this skill whenever running, verifying, troubleshooting, or operating the
  langgraph-rag QA or Chat applications. Trigger on requests to start the app,
  run QA/chat queries, test endpoints, debug health/readiness/metrics, inspect
  full vs lightweight graph behavior, diagnose Chroma/session persistence, back
  up or restore local state, or run project quality gates. This is the hands-on
  runbook for operational tasks, not architecture redesign.
---

# RAG Operations Runbook — langgraph-rag

Domain: running, verifying, troubleshooting, and operating the local LangGraph RAG applications.

Related project skills:

- `../../SKILL.md` — system architecture and cross-cutting concerns
- `../../src/graph/SKILL.md` — graph topology and node behavior
- `../../src/rag/SKILL.md` — Chroma, embeddings, indexing, retriever lifecycle
- `../../src/web_search/SKILL.md` — web search providers and live-search pipeline
- `../../src/api/SKILL.md` — FastAPI, SSE, request/response models
- `../../src/sessions/SKILL.md` — chat sessions, SQLite metadata, checkpoints

## Quick Reference

| Task | Command or path |
|---|---|
| Install runtime deps | `python -m pip install -r requirements.txt` |
| Install dev deps | `python -m pip install -r requirements-dev.txt` |
| Run QA server | `python -m src.qa.main serve --host 127.0.0.1 --port 8000` |
| Run Chat server | `python -m src.chat.main serve --host 127.0.0.1 --port 8001` |
| One-off QA query | `python -m src.qa.main query "Your question"` |
| Chat REPL | `python -m src.chat.main chat` |
| Tests | `python -m pytest -q` |
| Lint | `ruff check .` |
| Type check | `mypy src/` |
| Coverage gate | `python -m pytest --tb=short --cov=src --cov-report=term --cov-fail-under=70` |
| Whitespace check | `git diff --check` |
| Main docs | `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md` |

## Before Operating

1. Confirm the user wants to run or inspect the local app, not redesign it.
2. Preserve public CLI and HTTP contracts documented in `README.md`.
3. Do not print, commit, or expose real API keys. Secrets belong in `.env` or environment variables only.
4. Prefer read-only diagnostics before rebuilding indexes, deleting state, or changing config.
5. If running network-backed behavior, explain that live web search and LLM calls depend on provider availability and local API keys.

## Environment Setup

Use Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Playwright is already pinned in `requirements.txt`. When JavaScript fallback is
enabled, install its browser runtime separately:

```powershell
python -m playwright install chromium
```

Required/important environment variables:

| Variable | Purpose |
|---|---|
| `DASHSCOPE_API_KEY` | Required for DashScope/Qwen and DashScope embeddings. |
| `DEEPSEEK_API_KEY` | Optional; required only when `LLM_PROVIDER=deepseek`. |
| `LLM_PROVIDER` | `dashscope` or `deepseek`. |
| `API_KEY` | Optional bearer token for protected mutation endpoints. |
| `RAG_ENV` | Selects `config/{RAG_ENV}.yaml`; defaults to `development`. |
| `WEB_SEARCH_ENABLED` | Enables live web search. |
| `WEB_SEARCH_PROVIDER` | `bing`, `baidu`, or `duckduckgo`. |
| `WEB_SEARCH_LIGHTWEIGHT` | Use lightweight graph for web-search-sourced answers. |
| `WEB_SEARCH_LLM_QUERY_REWRITE_ENABLED` | Optional extra LLM query rewrite; deterministic cleanup is the default. |
| `CHAT_CONTEXT_MAX_TURNS` | Maximum recent turns projected into chat model calls. |
| `CHAT_CONTEXT_MAX_CHARS` | Character budget for the model-side projection; checkpoints remain complete. |

Configuration precedence is:

```text
CLI flags > process environment / .env > YAML config > built-in defaults
```

Read `config/default.yaml` plus the active overlay (`config/development.yaml`, `config/staging.yaml`, or `config/production.yaml`) when diagnosing config behavior.

## Running the Applications

### QA app

Start the QA web/API server:

```powershell
python -m src.qa.main serve --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Run a terminal query:

```powershell
python -m src.qa.main query "What does this source say about fine-tuning?" --rebuild
```

Use explicit sources:

```powershell
python -m src.qa.main query "Your question" --urls "https://example.com,https://another.com" --rebuild
```

### Chat app

Start the chat web/API server:

```powershell
python -m src.chat.main serve --host 127.0.0.1 --port 8001
```

Open:

```text
http://127.0.0.1:8001
```

Run the terminal REPL:

```powershell
python -m src.chat.main chat --urls "https://example.com,https://another.com"
python -m src.chat.main chat --seed-question "latest model releases 2026"
```

## Operational Endpoints

### QA endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Browser UI |
| `GET` | `/health` | Liveness/readiness response |
| `GET` | `/ready` | Readiness with app-state checks |
| `GET` | `/metrics` | In-process graph metrics |
| `POST` | `/query` | Batch QA request |
| `POST` | `/query/stream` | SSE graph event stream |

### Chat endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Browser UI |
| `GET` | `/health` | Liveness/session-count response |
| `GET` | `/ready` | Readiness with session-registry checks |
| `GET` | `/metrics` | In-process graph metrics |
| `POST` | `/chat` | Create a thread |
| `POST` | `/chat/{thread_id}/message` | Send a batch chat turn |
| `POST` | `/chat/{thread_id}/message/stream` | SSE chat turn stream |
| `GET` | `/chat/{thread_id}/history` | Read checkpoint transcript |
| `DELETE` | `/chat/{thread_id}` | Delete a session |

When `API_KEY` is set, mutation endpoints require:

```text
Authorization: Bearer <API_KEY>
```

Health, readiness, metrics, and read-only endpoints are intended for platform checks.

## Choosing the Graph Path

The app has two graph modes.

### Full graph

Used for configured/default URLs and explicit URL indexing.

```text
START → agent → retrieve → grade → generate → END
                 ↑                    ↓
                 └── rewrite ←────────┘
```

Chat mode adds `condense` before `agent` and uses checkpoint persistence.

Full graph uses:

- Chroma vector store
- embeddings
- retriever tool
- document relevance grading
- query rewriting
- RAG answer generation

Use full graph diagnostics when the problem involves Chroma, embeddings, retriever results, relevance grading, or rewrite loops.

### Lightweight web-search graph

Used when web search discovers sources and `WEB_SEARCH_LIGHTWEIGHT=true`.

```text
START → agent ──direct answer──► END
          │
          └─live_web_search→ decompose → bounded search → merge → web_answer → END
```

The graph owns the lightweight chat search and runs at most six queries with
three concurrent provider calls. Merge ranking prioritizes provider
title/snippet relevance and URL quality before overlap and provider rank.
Predominantly Chinese queries prefer Baidu → Bing → DuckDuckGo; other queries
start with the configured provider and fall back through Bing, Baidu, and
DuckDuckGo. Deterministic query cleanup is the default, with the optional LLM
rewrite gated by `WEB_SEARCH_LLM_QUERY_REWRITE_ENABLED`.

On failed readable or relevant extraction, the lightweight path expands queries
once and retries. If that second search/fetch attempt fails, `fallback_answer`
makes one tool-free model call with a live-verification caveat and then ends.

Use lightweight diagnostics when the problem involves live web search, URL discovery, page fetching, readable text extraction, source merge/ranking, or grounded web answers.

## Troubleshooting Playbooks

### App will not start

1. Check the exact command and working directory.
2. Confirm dependencies are installed in the active virtual environment.
3. Check `.env` for required API keys without displaying secret values.
4. Validate config files under `config/` if `RAG_ENV` or `--config` is used.
5. If import errors appear, run:

```powershell
python -m compileall src tests
```

6. If type/config errors appear, inspect `src/config/settings.py` and `src/config/loader.py`.

### `/health` works but answers fail

1. Check whether the failing endpoint is QA or Chat.
2. Check whether the request uses default sources, explicit URLs, or web search.
3. Inspect `/metrics` for graph node errors or unusual counts.
4. For full graph failures, check Chroma/indexing and retriever setup.
5. For lightweight failures, check web search provider behavior and content fetching.
6. Verify provider API keys and network access.

### Web search returns poor or no answers

1. Confirm `WEB_SEARCH_ENABLED=true`.
2. Confirm selected provider: `WEB_SEARCH_PROVIDER=bing|baidu|duckduckgo`.
3. Check whether lightweight mode is enabled with `WEB_SEARCH_LIGHTWEIGHT=true`.
4. Remember that predominantly Chinese queries automatically prefer Baidu, then Bing and DuckDuckGo; other queries start with the configured provider.
5. For low-text or JS-heavy pages, inspect:
   - `src/web_search/content_fetcher.py`
   - `src/web_search/fetch_policy.py`
   - `src/web_search/playwright_loader.py`
6. For JS fallback, confirm `playwright` is installed from `requirements.txt` and run `python -m playwright install chromium` once.
7. Inspect result title/snippet relevance and fetched-page relevance before lowering `WEB_SEARCH_MIN_URL_SCORE`; off-topic URLs should be rejected rather than merely moved down.

Do not weaken readability guards just to force an answer. Passing shell text or unreadable boilerplate to the LLM causes worse grounded answers.

### Chroma or indexing problems

1. Determine which Chroma path is in use:
   - global/default: `.chroma/`
   - isolated chat session: `.chroma/chat/<thread_id>/`
2. Check whether `embedding_config.json` matches the configured embedding model and dimension.
3. Rebuild only when needed; rebuilding can be slow and may make network/API calls.
4. On Windows file-lock errors, stop running servers before deleting or replacing `.chroma/`.
5. Consult `src/rag/SKILL.md` before changing retriever, embedding, or document-loader code.

### Chat session/history problems

1. Identify the `thread_id`.
2. Check whether session metadata is in `.chroma/chat/sessions.sqlite3`.
3. Check whether checkpoint state is in `.chroma/chat/checkpoints.sqlite3`.
4. Confirm the server was started normally so `SQLiteMemorySaver` is active.
5. For explicit/web-search sessions, confirm isolated Chroma path exists under `.chroma/chat/<thread_id>/`.
6. Distinguish persistence from model context: the checkpoint/history remains complete even though each model call sees only the bounded recent projection configured by `CHAT_CONTEXT_MAX_TURNS` and `CHAT_CONTEXT_MAX_CHARS`.
7. Consult `src/sessions/SKILL.md` before changing registry, isolation, SQLite, or checkpoint behavior.

### Streaming/SSE problems

1. Compare batch endpoint behavior to stream endpoint behavior first.
2. QA stream: `POST /query/stream`.
3. Chat stream: `POST /chat/{thread_id}/message/stream`.
4. Check event formatting in `src/api/streaming.py`.
5. Check typed graph events in `src/graph/events.py`.
6. Check executor behavior in `src/graph/executor.py`.
7. Stream failures that do not affect batch answers are usually transport/event formatting issues, not RAG-quality issues.

### API auth problems

1. If `API_KEY` is unset, local development mutation endpoints are open.
2. If `API_KEY` is set, protected endpoints require `Authorization: Bearer <API_KEY>`.
3. Check `src/api/auth.py` for auth behavior.
4. Do not log real bearer tokens.

## Backup and Restore

Stop QA and Chat servers before copying Chroma or SQLite state. Copying a live Chroma directory can capture SQLite files while handles are open.

Backup example:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
New-Item -ItemType Directory -Force -Path "backups\$stamp"
Copy-Item -Recurse -Force ".chroma" "backups\$stamp\.chroma"
```

If chat sessions are important, ensure these files are included:

```text
.chroma/chat/sessions.sqlite3
.chroma/chat/checkpoints.sqlite3
```

Restore procedure:

1. Stop running servers.
2. Move current `.chroma/` aside instead of deleting it immediately.
3. Copy backup `.chroma/` into the project root.
4. Restore session SQLite files under `.chroma/chat/` if they were backed up separately.
5. Start the app.
6. Verify `/health`, `/ready`, `/metrics`, QA query, chat session creation, and chat history retrieval.

If embedding metadata does not match current settings, a rebuild on next retriever build is expected and safer than querying incompatible vectors.

## Verification Gates

For source changes, run the relevant fastest check first, then broaden:

```powershell
python -m pytest -q
ruff check .
mypy src/
python -m pytest --tb=short --cov=src --cov-report=term --cov-fail-under=70
python -m compileall src tests
git diff --check
```

Use focused tests while iterating, then the broader gates before reporting completion.

Examples:

| Change area | Focused checks |
|---|---|
| URL parsing | `python -m pytest tests/test_urls.py -q` |
| Retry utilities | `python -m pytest tests/test_retry.py -q` |
| API auth/readiness | `python -m pytest tests/test_api_security_readiness.py -q` |
| API streaming | `python -m pytest tests/test_api_streaming.py -q` |
| Graph builder/state/nodes | `python -m pytest tests/test_graph_builder.py tests/test_graph_state_and_nodes.py -q` |
| Web search primitives/providers | `python -m pytest tests/test_web_search_lightweight_primitives.py tests/test_web_search_providers.py -q` |
| Sessions | `python -m pytest tests/test_sessions.py -q` |

Report skipped checks honestly. If tests fail, include the failing command and the relevant error summary.

## Operational Change Rules

- Prefer configuration changes over code changes when diagnosing environment-specific behavior.
- Prefer restarting the app over deleting state when the symptom may be a stale process.
- Back up `.chroma/` before destructive Chroma or session operations.
- Keep default-source chat sessions on the shared global Chroma store; explicit and web-search sessions should remain isolated by thread.
- Keep lightweight web-search answers grounded in fetched readable page text. If no readable text is available, preserve the fallback/caveat behavior.
- Do not change public API schemas or CLI flags as part of an operational fix unless the user explicitly asks for a migration.

## When to Consult Other Skills

| Symptom or task | Consult |
|---|---|
| Graph topology, node routing, rewrite/grade logic | `../../src/graph/SKILL.md` |
| Chroma, embeddings, document loading, retriever rebuilds | `../../src/rag/SKILL.md` |
| Search providers, URL discovery, JS fallback, fetch quality | `../../src/web_search/SKILL.md` |
| Endpoint contracts, auth, SSE formatting, dependency injection | `../../src/api/SKILL.md` |
| Chat session lifecycle, SQLite metadata, checkpoints | `../../src/sessions/SKILL.md` |
| Cross-module architecture or deployment docs | `../../SKILL.md` |
