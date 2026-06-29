# Only Subscribers

A local LangGraph retrieval-augmented generation project with two FastAPI apps:

- **QA** (`src/qa`): single-shot question answering on configured URLs, custom URLs, or web-search results.
- **Chat** (`src/chat`): multi-turn chat with per-thread source sets, persisted session metadata, and SQLite-backed LangGraph checkpoints.

The project supports two LangGraph workflows:

- **Full graph**: agent → retrieve (Chroma) → grade → generate, with query rewrite on low-relevance grades.
- **Lightweight graph**: agent → decompose → web_search → merge → web_answer, with one-shot conditional expansion on retrieval failure and a training-data agent fallback. Skips Chroma, embeddings, grading, and rewriting entirely.

The project started as a Python extraction of a Jupyter notebook; it is now organized as a reusable codebase with YAML configuration, typed graph events, SSE streaming, health/readiness/metrics endpoints, optional API-key auth, Docker support, CI quality gates, and company-readiness documentation.

## Project Structure

```text
langgraph-rag/
|-- .github/workflows/        # CI
|-- config/                   # YAML config (default + per-environment overlays)
|-- src/
|   |-- api/                  # Shared FastAPI helpers (auth, CORS, errors, streaming, dependencies)
|   |-- chat/                 # Multi-turn chat app (API, UI, entry point)
|   |-- config/               # Settings dataclass + YAML/env loader
|   |-- core/                 # Deprecated re-exports (redirect to src/graph, src/rag, etc.)
|   |-- errors.py             # Typed RAG exceptions
|   |-- graph/                # LangGraph builder, state, edges, executor, metrics
|   |   |-- nodes/            # Node factories (agent, condense, decompose, expand, generate, grade, merge, rewrite, web_answer)
|   |-- llm/                  # LLM provider seam (DashScope, DeepSeek) + prompt templates
|   |-- qa/                   # Single-shot QA app (API, CLI, UI)
|   |-- rag/                  # Chroma retriever, embeddings (DashScope, HuggingFace), document loader/quality
|   |-- sessions/             # Chat session registry, SQLite metadata + checkpoint persistence
|   |-- tools/                # Optional agent tools (weather, stock, currency, Wikipedia) + shared HTTP helper
|   |-- utils/                # Retry, networking, URL parsing helpers
|   |-- web_search/           # Live web search providers (Bing, Baidu, DuckDuckGo), discovery, content fetching, JS fallback
|-- tests/                    # Offline-focused pytest suite
|-- ARCHITECTURE.md           # System architecture and operational notes
|-- COMPANY_READINESS_GAPS.md # Completed and remaining company-readiness work
|-- CONTRIBUTING.md           # Local development workflow and quality gates
|-- SECURITY.md               # Secret handling and API security behavior
|-- CHANGELOG.md              # Notable project changes
|-- Dockerfile
|-- docker-compose.yml
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install runtime dependencies:

```powershell
python -m pip install -r requirements.txt
```

For local development and CI-equivalent checks:

```powershell
python -m pip install -r requirements-dev.txt
```

Create a local environment file:

```powershell
Copy-Item .env.example .env       # Windows
cp .env.example .env              # macOS / Linux
```

Set your API keys in `.env`:

```text
DASHSCOPE_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here    # optional, only if LLM_PROVIDER=deepseek
```

## Configuration

Settings are loaded with this precedence:

1. CLI flags (`--urls`, `--rebuild`, `--host`, `--port`)
2. Process environment variables and `.env`
3. YAML config files
4. Built-in defaults in `src/config/settings.py`

By default the app reads `config/default.yaml` and overlays `config/{RAG_ENV}.yaml`. `RAG_ENV` defaults to `development`; `staging` and `production` overlays are included. Secrets stay outside YAML.

Key settings:

| Env variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `dashscope` | `dashscope` or `deepseek` |
| `QWEN_MODEL` | `qwen-plus` | DashScope model name |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | DeepSeek model name |
| `EMBEDDING_MODEL` | `text-embedding-v4` | Embedding model |
| `WEB_SEARCH_ENABLED` | `true` | Enable live web search |
| `WEB_SEARCH_PROVIDER` | `bing` | `bing`, `baidu`, or `duckduckgo` |
| `WEB_SEARCH_LIGHTWEIGHT` | `true` | Use lightweight graph for web search |
| `WEB_SEARCH_TOP_K` | `6` | Top-K URLs after merge ranking |
| `WEB_SEARCH_MIN_URL_SCORE` | `45` | URL quality threshold (0-100) |
| `WEB_SEARCH_JS_FALLBACK_ENABLED` | `false` | Opt-in Playwright Chromium retry for JS-rendered domains |
| `WEATHER_ENABLED` | `false` | Open-Meteo weather/forecast tool |
| `STOCK_ENABLED` | `false` | yfinance stock-quote tool |
| `CURRENCY_ENABLED` | `false` | Frankfurter currency-conversion tool |
| `WIKIPEDIA_ENABLED` | `false` | MediaWiki summary tool |
| `WIKIPEDIA_USER_AGENT` | `langgraph-rag/1.0 (configure)` | Required when `WIKIPEDIA_ENABLED=true` |
| `CHROMA_DIR` | `.chroma` | Vector store location |
| `API_KEY` | — | API auth key (open when unset) |
| `RERANK_STRATEGY` | `lexical` | `lexical`, `embedding`, or `hybrid` |

## LangGraph Architecture

### Full Graph (QA and Chat)

```
START → agent → retrieve → grade → generate → END
                 ↑                     ↓
                 └── rewrite ←─────────┘ (on "not relevant")
```

- **agent**: LLM with bound tools (Chroma retriever + optional live web search). A system prompt (`AGENT_SYSTEM_PROMPT`) guides the model to answer directly from its own knowledge for math, general knowledge, coding, and chitchat — only calling tools when external or up-to-date information is needed.
- **retrieve**: `ToolNode` executes the selected tool (Chroma vector search or live web search).
- **grade**: LLM grades retrieved context relevance. Routes to `generate` if relevant, `rewrite` if not (up to `max_rewrites` limit).
- **rewrite**: LLM rewrites the query with semantic intent clarification, then loops back to `agent`.
- **generate**: LLM synthesizes a final answer from the retrieved context using `RAG_PROMPT`.
- **condense** (chat only): Standalone question extraction from conversation history using `CONDENSE_PROMPT`.

### Lightweight Graph (Web Search)

```
START → agent ──(no tool)──► END                                    (direct answer)
          │
          └─(live_web_search)─► decompose → web_search → merge → web_answer ─► END   (grounded)
                                                                       │
                                                          (no readable, !expanded)
                                                                       │
                                                                       ▼
                                                          expand → web_search → merge → web_answer ─► END   (grounded retry)
                                                                                              │
                                                                                  (no readable, expanded)
                                                                                              │
                                                                                              ▼
                                                                                           agent ─► END   (training-data fallback)
```

Skips Chroma, embeddings, grading, and rewriting. The agent either answers directly (system prompt steers it away from tools for stable factual questions like founding dates and capitals) or calls the live web search tool, which kicks off a `decompose → web_search → merge → web_answer` chain:

- **decompose** splits a compound question into 1-3 atomic sub-questions (passthrough for already-atomic Qs).
- **web_search** fans out one keyword query per sub-question.
- **merge** dedupes URLs by canonical form and ranks by `(hit_count, best_provider_rank)`, keeping `web_search_top_k`.
- **web_answer** fetches the merged URLs, extracts readable text, and prompts the LLM to synthesize a grounded answer.

When `web_answer` produces no readable content and expansion has not yet fired, the post-`web_answer` edge routes to **expand**, which produces k=3 keyword paraphrases per sub-question. The graph re-enters `web_search → merge → web_answer` with the broader query set; the merge stage combines the new URLs with the first attempt's URLs rather than discarding them. A second failure routes to the **agent** so the LLM can answer from training data with a "couldn't verify against the live web" caveat. Each fallback step is bounded by a one-shot flag (`expansion_attempted`) and an attempt counter (`web_answer_attempts`, capped at `WEB_ANSWER_FALLBACK_MAX_ATTEMPTS=3`) so the graph cannot loop indefinitely.

### Graph Nodes

| Node | Factory | Prompt |
|---|---|---|
| agent | `agent_factory` | `AGENT_SYSTEM_PROMPT` |
| retrieve | `ToolNode` | — |
| grade | `grade_documents_factory` | `GRADE_PROMPT` |
| rewrite | `rewrite_factory` | inline prompt |
| generate | `generate_factory` | `RAG_PROMPT` |
| condense | `condense_question_factory` | `CONDENSE_PROMPT` |
| decompose | `decompose_factory` | inline prompt (1-3 sub-questions, passthrough on atomic) |
| expand | `expand_factory` | inline prompt (k=3 paraphrases per sub-question) |
| merge | `merge_factory` | — (pure code: dedupe + rank URLs) |
| web_answer | `web_answer_factory` | `build_web_search_prompt` |

### Agent Tools

The agent can be given any combination of these tools via per-tool config flags. All optional tools default to off; the retriever is always present in the full graph and `live_web_search` defaults to on.

| Tool | Module | Config flag | Notes |
|---|---|---|---|
| `retrieve_source_documents` | `src/rag/chroma_retriever.py` | always on (full graph) | Chroma vectorstore retrieval |
| `live_web_search` | `src/web_search/tool.py` | `WEB_SEARCH_ENABLED=true` (default) | Bing/Baidu/DuckDuckGo HTML scraping |
| `get_weather` | `src/tools/weather.py` | `WEATHER_ENABLED=true` | Open-Meteo forecast (city or coordinates), no API key |
| `get_stock_quote` | `src/tools/stock.py` | `STOCK_ENABLED=true` | yfinance / Yahoo Finance, no API key |
| `convert_currency` | `src/tools/currency.py` | `CURRENCY_ENABLED=true` | Frankfurter API (201 currencies), no API key |
| `search_wikipedia` | `src/tools/wikipedia_tool.py` | `WIKIPEDIA_ENABLED=true` | MediaWiki API summary + URL, set `WIKIPEDIA_USER_AGENT` |

When the agent calls a non-web-search tool in the lightweight graph (weather, stock, currency, Wikipedia), the post-tool edge routes back to the agent so it can synthesize the structured tool output into a final answer — bypassing the `decompose → web_search → merge → web_answer` chain that's specific to `live_web_search`.

### LLM Provider Seam

`src/llm/provider.py` implements a provider protocol with two backends:

- **DashScopeLLMProvider** — ChatTongyi with DashScope models (default: `qwen-plus`)
- **DeepSeekLLMProvider** — ChatOpenAI pointed at `api.deepseek.com` (default: `deepseek-v4-pro`)

### Embedding Providers

`src/rag/embeddings.py` supports:

- **DashScopeEmbeddings** — Tongyi text embeddings (default: `text-embedding-v4`)
- **HuggingFaceEmbeddingModel** — Local HuggingFace embedding models

## Running QA

Ask a question from the terminal:

```powershell
python -m src.qa.main query "What does this source say about fine-tuning?" --rebuild
```

Use custom sources:

```powershell
python -m src.qa.main query "Your question" --urls "https://example.com,https://another.com" --rebuild
```

Start the QA web/API server:

```powershell
python -m src.qa.main serve --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`.

API endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Browser UI |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness with app-state checks |
| `GET` | `/metrics` | In-process graph metrics |
| `POST` | `/query` | Non-streaming answer |
| `POST` | `/query/stream` | SSE graph event stream |

## Running Chat

Start the chat web/API server:

```powershell
python -m src.chat.main serve --host 127.0.0.1 --port 8001
```

Then open `http://127.0.0.1:8001`.

Use the terminal REPL:

```powershell
python -m src.chat.main chat --urls "https://example.com,https://another.com"
python -m src.chat.main chat --seed-question "latest model releases 2026"
```

API endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Browser UI |
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness with session-registry checks |
| `GET` | `/metrics` | In-process graph metrics |
| `POST` | `/chat` | Create a thread |
| `POST` | `/chat/{id}/message` | Send a turn |
| `POST` | `/chat/{id}/message/stream` | SSE stream: node events + per-token answer deltas (`?tokens=false` for node events only) |
| `GET` | `/chat/{id}/history` | Read transcript |
| `DELETE` | `/chat/{id}` | Delete a thread |

## Persistence

The default Chroma vector store lives under `.chroma/`.

Chat uses additional persisted state:

- `.chroma/chat/<thread_id>/`: isolated Chroma store for per-thread sources
- `.chroma/chat/sessions.sqlite3`: session metadata via `SQLiteStorage`
- `.chroma/chat/checkpoints.sqlite3`: LangGraph checkpoints via `SQLiteMemorySaver`

Chat sessions survive app restarts. Sessions with the global default source set share the global Chroma collection; sessions with explicit URLs or web-discovered sources get isolated per-thread stores. The `SQLiteMemorySaver` checkpointer is thread-safe under concurrent `/chat` requests for the same thread (uses a reentrant lock around the inherited `MemorySaver` mutations and the SQLite snapshot).

## Web Search

Live web search is provided by three providers in `src/web_search/`:

- **Bing** (`BingWebSearch`) — HTML scraping with optional `d`/`w`/`m` recency filter
- **Baidu** (`BaiduWebSearch`) — HTML scraping with captcha detection
- **DuckDuckGo** (`DuckDuckGoWebSearch`) — HTML scraping

The default provider order is Bing → Baidu → DuckDuckGo, with automatic fallback on failure. Search queries are LLM-rewritten into keyword form before being sent to the provider (with mechanical filler-word stripping as a fallback when the LLM is unavailable).

A multi-stage quality pipeline runs before the agent sees URLs:

- **URL scoring** (`web_search_min_url_score`, default 45) drops low-quality result patterns (search/login/tag/file/feed pages) and rewards hostname/path matches against query keywords.
- **Provider-result dedup** by canonical host/path.
- **Pre-index document filtering** drops short/empty/boilerplate/low-signal pages, with optional embedding similarity gate against the question (`document_quality_relevance_query`) and configurable recency bias from extracted publication dates.
- **Post-retrieval re-ranking** scores chunks by query/document overlap, frequency, and phrase matches; `RERANK_STRATEGY` switches between lexical (default), embedding, or hybrid.
- **Optional JS-capable fallback** (`WEB_SEARCH_JS_FALLBACK_ENABLED`) retries known JS-only domains (`baike.baidu.com`, `zhuanlan.zhihu.com`, `apps.microsoft.com`, `deepseek.net` by default) through a lazy headless Chromium adapter when the HTTP loader returns empty/insufficient text. Off by default; requires installing `playwright` and a Chromium runtime.

When `web_search_lightweight` is enabled (default), the lightweight graph routes the agent's tool call through `decompose → web_search → merge → web_answer` as described above, with one-shot conditional expansion and a training-data agent fallback as the final safety net.

## Streaming

Both apps expose SSE streaming endpoints, but they stream at different granularities:

- **QA `/query/stream`**: node lifecycle events (`node_start`, `node_end`, retriever/grader summaries) plus a final `done` event carrying the full answer.
- **Chat `/chat/{id}/message/stream`**: the same node lifecycle events, and — by default — per-token `token` events streamed from the answer-producing nodes (`generate`, `web_answer`, `agent`) as the LLM generates them, followed by a final `done` event. Pass `?tokens=false` to fall back to node-events-only streaming.

Token streaming uses LangGraph's combined `stream_mode=["updates", "messages"]`. Only genuine streaming chunks (`AIMessageChunk`) are forwarded; the aggregated final message a node returns is dropped so the answer is not duplicated. Tokens from internal structured-output calls (decompose, expand, grade, condense, rewrite) are filtered out so they never leak into the user-visible answer.

The browser chat UI consumes the token stream and renders the answer incrementally. The terminal REPL (`python -m src.chat.main chat`) also streams tokens to stdout as they arrive. Both fall back to the final `done` answer when a provider does not emit token chunks.

## Auth, CORS, and Security

Local development is open when `API_KEY` is unset. When set, mutation endpoints require:

```text
Authorization: Bearer <API_KEY>
```

Protected endpoints: `POST /query`, `POST /query/stream`, `POST /chat`, `POST /chat/{id}/message`, `POST /chat/{id}/message/stream`, `DELETE /chat/{id}`.

CORS is configured via `cors_allow_origins` in YAML or `CORS_ALLOW_ORIGINS` env var. Health, readiness, metrics, and read-only endpoints remain available for platform checks.

## Docker

```powershell
$env:DASHSCOPE_API_KEY = "your_key"
$env:API_KEY = "optional_key"
docker compose up --build
```

Starts QA on `http://127.0.0.1:8000` and Chat on `http://127.0.0.1:8001` with named volumes for `.chroma` and session data.

## Verification

```powershell
python -m pytest -q                         # Run tests (current baseline: 220 passed)
ruff check .                                # Lint
mypy src/                                   # Type check
python -m pytest --tb=short --cov=src --cov-report=term --cov-fail-under=70
python -m compileall src tests              # Syntax check
git diff --check                            # Whitespace check
```

## Notes

- The chat model is configurable via `LLM_PROVIDER` — DashScope (`qwen-plus`) or DeepSeek (`deepseek-v4-pro`). Embeddings always use DashScope/Tongyi unless `embedding_model` is set to a HuggingFace model.
- Existing Chroma stores with incompatible embedding metadata are rebuilt automatically on startup.
- Web search defaults to Bing with Baidu and DuckDuckGo fallback. If Baidu returns a captcha/verification page, it is temporarily skipped. Set `WEB_SEARCH_PROVIDER` to pin a single provider.
- The agent system prompt (`AGENT_SYSTEM_PROMPT` in `src/llm/prompts.py`) tells the model to answer directly when tools aren't needed — covering math, general knowledge, programming concepts, definitions, well-established stable facts (founding dates, capitals, public figures), and chitchat — so the graph avoids unnecessary retrieval/rewrite cycles.
- Reranking (`RERANK_STRATEGY`) defaults to lexical (keyword-based); `embedding` uses cosine similarity against embedding vectors; `hybrid` combines both.
- Optional agent tools (`weather`, `stock`, `currency`, `wikipedia`) are off by default. Enable them via the per-tool `_ENABLED` flag in `.env`. None require an API key; only `WIKIPEDIA_USER_AGENT` should be customized for shared deployments.
- The lightweight graph's conditional expansion fires only on web-search retrieval failure — single-keyword questions that get a readable page back take the fast path with one search and one LLM call. Compound questions that decompose into multiple sub-Qs still take the fast path; expansion only fires when no fetched page yields readable text.
