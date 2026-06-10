# LangGraph RAG

A local LangGraph retrieval-augmented generation project with two FastAPI apps:

- **QA** (`src/qa`): single-shot question answering on configured URLs, custom URLs, or web-search results.
- **Chat** (`src/chat`): multi-turn chat with per-thread source sets, persisted session metadata, and SQLite-backed LangGraph checkpoints.

The project supports two LangGraph workflows:

- **Full graph**: agent → retrieve (Chroma) → grade → generate, with query rewrite on low-relevance grades.
- **Lightweight graph**: agent → live web search → direct web answer. Skips Chroma, embeddings, grading, and rewriting entirely.

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
|   |   |-- nodes/            # Node factories (agent, rewrite, grade, generate, condense, web_answer)
|   |-- llm/                  # LLM provider seam (DashScope, DeepSeek) + prompt templates
|   |-- qa/                   # Single-shot QA app (API, CLI, UI)
|   |-- rag/                  # Chroma retriever, embeddings (DashScope, HuggingFace), document loader/quality
|   |-- sessions/             # Chat session registry, SQLite metadata + checkpoint persistence
|   |-- utils/                # Retry, networking, URL parsing helpers
|   |-- web_search/           # Live web search providers (Bing, Baidu, DuckDuckGo), discovery, content fetching
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
| `WEB_SEARCH_LIGHTWEIGHT` | — | Use lightweight graph for web search |
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
START → agent → web_search → web_answer → END
           ↓                     
           └── END (when agent answers directly)
```

Skips Chroma, embeddings, grading, and rewriting. The agent either answers directly (system prompt steers it away from tools for simple questions) or calls the live web search tool. `web_answer` fetches and reads the discovered pages, then prompts the LLM to synthesize an answer grounded in the fetched content.

### Graph Nodes

| Node | Factory | Prompt |
|---|---|---|
| agent | `agent_factory` | `AGENT_SYSTEM_PROMPT` |
| retrieve | `ToolNode` | — |
| grade | `grade_documents_factory` | `GRADE_PROMPT` |
| rewrite | `rewrite_factory` | inline prompt |
| generate | `generate_factory` | `RAG_PROMPT` |
| condense | `condense_question_factory` | `CONDENSE_PROMPT` |
| web_answer | `web_answer_factory` | `build_web_search_prompt` |

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
| `POST` | `/chat/{id}/message/stream` | SSE graph event stream |
| `GET` | `/chat/{id}/history` | Read transcript |
| `DELETE` | `/chat/{id}` | Delete a thread |

## Persistence

The default Chroma vector store lives under `.chroma/`.

Chat uses additional persisted state:

- `.chroma/chat/<thread_id>/`: isolated Chroma store for per-thread sources
- `.chroma/chat/sessions.sqlite3`: session metadata via `SQLiteStorage`
- `.chroma/chat/checkpoints.sqlite3`: LangGraph checkpoints via `SQLiteMemorySaver`

Chat sessions survive app restarts. Sessions with the global default source set share the global Chroma collection; sessions with explicit URLs or web-discovered sources get isolated per-thread stores.

## Web Search

Live web search is provided by three providers in `src/web_search/`:

- **Bing** (`BingWebSearch`) — HTML scraping
- **Baidu** (`BaiduWebSearch`) — HTML scraping with captcha detection
- **DuckDuckGo** (`DuckDuckGoWebSearch`) — HTML scraping

The default provider order is Bing → Baidu → DuckDuckGo, with automatic fallback on failure. Results are deduplicated and ranked.

When `web_search_lightweight` is enabled, the lightweight graph bypasses Chroma entirely and feeds fetched page content directly to the LLM for answer synthesis.

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
python -m pytest -q                         # Run tests
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
- The agent system prompt (`AGENT_SYSTEM_PROMPT` in `src/llm/prompts.py`) tells the model to answer directly when tools aren't needed — covering math, general knowledge, programming concepts, definitions, and chitchat — so the graph avoids unnecessary retrieval/rewrite cycles.
- Reranking (`RERANK_STRATEGY`) defaults to lexical (keyword-based); `embedding` uses cosine similarity against embedding vectors; `hybrid` combines both.
