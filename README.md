# LangGraph RAG

A local LangGraph retrieval-augmented generation project with two FastAPI apps:

- `src.qa`: single-shot question answering on configured URLs, custom URLs, or web-search results.
- `src.chat`: multi-turn chat with per-thread source sets, persisted session metadata, and persisted LangGraph checkpoints.

The project started as a Python extraction of `rag-langgraph.ipynb`; it is now organized as a reusable codebase with YAML configuration, typed graph events, SSE streaming, health/readiness/metrics endpoints, optional API-key auth, Docker support, CI quality gates, and company-readiness documentation.

## Security Note

The original notebook contained hard-coded API keys. They were removed. Put local secrets in `.env` or process environment variables, and rotate any key that was previously committed, shared, or pasted into notebooks.

Never commit these values:

```text
DASHSCOPE_API_KEY
API_KEY
LANGCHAIN_API_KEY
```

## Project Structure

```text
langgraph-rag/
|-- config/                  # Default YAML plus RAG_ENV overlays
|-- memory/                  # Project process and refactor tracking notes
|-- src/
|   |-- api/                 # Shared FastAPI auth, errors, readiness, metrics helpers
|   |-- chat/                # Multi-turn chat app, API, UI, graph wiring
|   |-- config/              # Settings loading from YAML, env, .env, CLI flags
|   |-- core/                # Shared RAG compatibility surface
|   |-- graph/               # LangGraph builder, typed events, executor, metrics
|   |-- qa/                  # Single-shot QA app, API, UI, CLI
|   |-- rag/                 # Retrieval, indexing, embeddings, web loading/search
|   |-- sessions/            # Session registry, SQLite metadata, checkpoint persistence
|   `-- llm/                 # Model and prompt helpers
|-- tests/                   # Offline-focused pytest suite
|-- .github/workflows/       # CI
|-- ARCHITECTURE.md
|-- COMPANY_READINESS_GAPS.md
|-- CONTRIBUTING.md
|-- SECURITY.md
|-- Dockerfile
`-- docker-compose.yml
```

## Setup

Create and activate a virtual environment from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install runtime dependencies:

```powershell
python -m pip install -r requirements.txt
```

If pip reports `No matching distribution found for langchain<0.4,>=0.3.0`,
force the project venv to use PyPI explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements.txt
```

If a previously-used venv contains newer LangChain packages, reinstall the
project-pinned dependency set and verify it:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall --index-url https://pypi.org/simple -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

For local development and CI-equivalent checks, install dev dependencies too:

```powershell
python -m pip install -r requirements-dev.txt
```

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

Set your DashScope key in `.env`:

```text
DASHSCOPE_API_KEY=your_real_key_here
```

Optional runtime settings commonly changed in `.env`:

```text
RAG_ENV=development
API_KEY=
LLM_PROVIDER=dashscope
DEEPSEEK_API_KEY=your_key_here
QWEN_MODEL=qwen-plus
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSION=1024
CHROMA_DIR=.chroma
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=bing
WEB_SEARCH_TOP_K=6
PAGE_LOAD_MAX_CONCURRENCY=4
PAGE_LOAD_CACHE_TTL_SECONDS=0
DOCUMENT_QUALITY_FILTER_ENABLED=true
```

## Configuration

Settings are loaded with this precedence:

1. CLI flags, such as `--urls`, `--rebuild`, `--host`, or `--port`
2. Process environment variables and `.env`
3. YAML config files
4. Built-in defaults

By default, the app reads `config/default.yaml` and overlays `config/{RAG_ENV}.yaml`. `RAG_ENV` defaults to `development`; `staging` and `production` overlays are included. Secrets stay outside YAML.

You can also pass a custom YAML file to the app commands:

```powershell
python -m src.qa.main serve --config config\staging.yaml
```

## Running QA

Ask a question from the terminal:

```powershell
python -m src.qa.main query "What does this source say about fine-tuning?" --rebuild
```

Use custom sources:

```powershell
python -m src.qa.main query "Your question here" --urls "https://example.com,https://another.com" --rebuild
```

Start the QA web/API server:

```powershell
python -m src.qa.main serve --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`.

Important QA endpoints:

- `GET /`: browser UI
- `GET /health`: liveness
- `GET /ready`: readiness with local app-state checks
- `GET /metrics`: in-process graph metrics snapshot
- `POST /query`: non-streaming answer
- `POST /query/stream`: SSE graph event stream

Example request:

```json
{
  "question": "Your question here",
  "urls": "https://example.com,https://another.com",
  "web_search": true,
  "rebuild": false
}
```

## Running Chat

Start the chat web/API server:

```powershell
python -m src.chat.main serve --host 127.0.0.1 --port 8001
```

Then open `http://127.0.0.1:8001`.

Use the terminal REPL:

```powershell
python -m src.chat.main chat --urls "https://example.com,https://another.com"
```

Or seed a chat with an initial web search. Later turns refresh web-search
sources automatically unless explicit URLs were provided:

```powershell
python -m src.chat.main chat --seed-question "Qwen fine-tuning best practices"
```

Important chat endpoints:

- `GET /`: browser UI
- `GET /health`: liveness
- `GET /ready`: readiness with local session-registry checks
- `GET /metrics`: in-process graph metrics snapshot
- `POST /chat`: create a thread
- `POST /chat/{thread_id}/message`: send a turn
- `POST /chat/{thread_id}/message/stream`: SSE graph event stream for a turn
- `GET /chat/{thread_id}/history`: read transcript
- `DELETE /chat/{thread_id}`: delete a thread

## Persistence

The default Chroma vector store lives under `.chroma/`.

Chat uses additional persisted state:

- `.chroma/chat/<thread_id>/`: isolated Chroma store for thread-specific sources
- `.chroma/chat/sessions.sqlite3`: session metadata and history
- `.chroma/chat/checkpoints.sqlite3`: LangGraph checkpoint state via `SQLiteMemorySaver`

That means normal chat sessions can survive an app restart. Sessions that use the global default source set share the global Chroma collection; sessions with explicit URLs or web-discovered sources get isolated per-thread stores.

## Auth, CORS, And Readiness

Local development remains open when `API_KEY` is unset.

When `API_KEY` is set, mutation endpoints require:

```text
Authorization: Bearer <API_KEY>
```

Protected mutation endpoints include `POST /query`, `POST /query/stream`, `POST /chat`, `POST /chat/{thread_id}/message`, `POST /chat/{thread_id}/message/stream`, and `DELETE /chat/{thread_id}`.

CORS is configured through `cors_allow_origins` in YAML or `CORS_ALLOW_ORIGINS` in the environment. `/health`, `/ready`, `/metrics`, and read-only endpoints remain available for platform checks and observability.

## Docker

Build and run both apps:

```powershell
docker compose up --build
```

The compose file starts:

- QA on `http://127.0.0.1:8000`
- Chat on `http://127.0.0.1:8001`

It mounts named volumes for `.chroma` and session data. Pass secrets through your shell or `.env` before starting compose:

```powershell
$env:DASHSCOPE_API_KEY = "your_real_key_here"
$env:API_KEY = "optional_gateway_key"
docker compose up --build
```

## Verification

Run the local test suite:

```powershell
python -m pytest -q
```

The latest integrated local verification passed with 113 tests.

With dev dependencies installed, run CI-style checks:

```powershell
ruff check .
mypy src/
python -m pytest --tb=short --cov=src --cov-report=term --cov-fail-under=70
python -m compileall src tests
git diff --check
```

`git diff --check` may report line-ending warnings on Windows; those are separate from whitespace errors.

## Documentation

- `ARCHITECTURE.md`: current system architecture and operational notes
- `CONTRIBUTING.md`: local development workflow and quality gates
- `SECURITY.md`: secret handling and API security behavior
- `CHANGELOG.md`: notable project changes
- `COMPANY_READINESS_GAPS.md`: completed and remaining company-readiness work
- `memory/current-process.md`: latest recorded process checkpoint
- `memory/refactor-daily-forms.md`: chronological refactor/process log

## Current Readiness Snapshot

Completed foundations include YAML environment overlays, typed errors, typed graph events, SSE streaming, metrics, readiness checks, API-key auth, CORS config, CI, Docker, dev tooling, SQLite session metadata, and SQLite LangGraph checkpoint persistence.

Remaining work called out in `COMPANY_READINESS_GAPS.md` includes dependency/security scanning, fuller structured logging and request IDs, API versioning, rate limiting/session export, deployment-specific infrastructure, and broader public API documentation.

## Notes

- This project still calls DashScope/Tongyi for embeddings. The chat model is configurable: DashScope (`qwen-plus`, default) or DeepSeek (`deepseek-v4-pro`). Set `LLM_PROVIDER=deepseek` and `DEEPSEEK_API_KEY` in `.env` to switch.
- The default chat model is `qwen-plus`; the DeepSeek option uses `deepseek-v4-pro`.
- The default embedding model is `text-embedding-v4`; existing Chroma stores with incompatible embedding metadata are rebuilt automatically.
- Web search defaults to Bing, then falls back through Baidu and DuckDuckGo. If Baidu returns a verification/captcha page, discovery temporarily skips Baidu during the fallback pass. Set `WEB_SEARCH_PROVIDER=baidu` or `WEB_SEARCH_PROVIDER=duckduckgo` if preferred.
