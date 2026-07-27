# Only Subscribers

A local LangGraph retrieval-augmented generation project with two FastAPI apps:

- **QA** (`src/qa`): single-shot question answering on configured URLs, custom URLs, or web-search results.
- **Chat** (`src/chat`): multi-turn chat with per-thread source sets, persisted session metadata, and SQLite-backed LangGraph checkpoints.

The project supports two LangGraph workflows:

- **Full graph**: agent → retrieve (Chroma) → grade → generate, with query rewrite on low-relevance grades.
- **Lightweight graph**: agent → decompose → bounded parallel search → relevance-aware merge → web_answer, with one expanded retry followed by a tool-free training-data fallback. Skips Chroma, embeddings, and grading; search-query LLM rewriting is optional and disabled by default.

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
|   |   |-- nodes/            # Node factories (agent, condense, decompose, bounded search, expand, generate, grade, merge, rewrite, web_answer)
|   |-- llm/                  # LLM provider seam (DashScope, DeepSeek) + prompt templates
|   |-- qa/                   # Single-shot QA app (API, CLI, UI)
|   |-- rag/                  # Chroma retriever, embeddings (DashScope, HuggingFace), document loader/quality
|   |-- sessions/             # Chat session registry, SQLite metadata + checkpoint persistence
|   |-- tools/                # Optional agent tools (weather, stock, currency, Wikipedia, directions, map, math, statistics, linear algebra, number theory, datetime, summarize-url, file readers) + shared HTTP helper
|   |-- utils/                # Retry, networking, URL parsing helpers
|   |-- web_search/           # Search APIs + HTML fallbacks, discovery, ranking, fetching (HTML/PDF/JS), structural + semantic filtering, domain reputation
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

Playwright is included in the runtime requirements. Install its Chromium binary only if you enable `WEB_SEARCH_JS_FALLBACK_ENABLED` (the browser retry is skipped silently without it):

```powershell
python -m playwright install chromium
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
| `CHAT_CONTEXT_MAX_TURNS` | `8` | Recent chat turns projected into model calls |
| `CHAT_CONTEXT_MAX_CHARS` | `12000` | Hard character budget for the model-side chat projection |
| `WEB_SEARCH_ENABLED` | `true` | Enable live web search |
| `WEB_SEARCH_LLM_QUERY_REWRITE_ENABLED` | `false` | Compatibility opt-in for an extra search-query LLM call |
| `WEB_SEARCH_PROVIDER` | `bing` | Backward-compatible primary/fallback provider |
| `WEB_SEARCH_PROVIDERS` | empty | Optional comma-separated provider priority list |
| `WEB_SEARCH_PROVIDER_FANOUT` | `2` | Providers queried concurrently in each discovery stage |
| `WEB_SEARCH_PROVIDER_TIMEOUT_SECONDS` | `8` | Network timeout for each provider request |
| `WEB_SEARCH_API_TIMEOUT_SECONDS` | `20` | Request timeout for key-backed search APIs |
| `WEB_SEARCH_DEADLINE_SECONDS` | `30` | Overall provider-discovery deadline |
| `SERPER_API_KEY` | empty | Enables Serper and prioritizes it for Mandarin search |
| `BRAVE_SEARCH_API_KEY` | empty | Enables Brave Search API |
| `TAVILY_API_KEY` | empty | Enables Tavily Search API |
| `BING_SEARCH_API_KEY` | empty | Enables the configured official Bing Search API endpoint |
| `WEB_SEARCH_LIGHTWEIGHT` | `true` | Use lightweight graph for web search |
| `WEB_SEARCH_TOP_K` | `6` | Top-K URLs after merge ranking |
| `WEB_SEARCH_MIN_URL_SCORE` | `45` | URL quality threshold (0-100) |
| `WEB_SEARCH_JS_FALLBACK_ENABLED` | `false` | Opt-in Playwright Chromium retry for unreadable pages |
| `WEB_SEARCH_JS_RETRY_BUDGET` | `2` | Maximum browser renders per fetch batch |
| `WEB_SEARCH_STRUCTURE_FILTER_ENABLED` | `true` | Drop listing/login/thin pages by measuring the fetched page |
| `WEB_SEARCH_MAX_LINK_DENSITY` | `0.5` | Anchor-text share above which a page counts as a listing |
| `WEB_SEARCH_MIN_CONTENT_WORDS` | `60` | Content units below which a page is too thin to ground an answer |
| `WEB_SEARCH_SEMANTIC_FILTER_ENABLED` | `false` | Opt-in embedding similarity that can only add recall |
| `WEB_SEARCH_SEMANTIC_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Local sentence-transformers model for similarity |
| `WEB_SEARCH_SEMANTIC_MIN_SIMILARITY` | `0.35` | Cosine similarity required for a bonus or rescue |
| `WEB_SEARCH_DOMAIN_REPUTATION_ENABLED` | `true` | Learn a per-domain ranking prior from fetch outcomes |
| `WEB_SEARCH_DOMAIN_REPUTATION_MIN_SAMPLES` | `5` | Observations before a domain's reputation counts |
| `WEATHER_ENABLED` | `false` | Open-Meteo weather/forecast tool |
| `STOCK_ENABLED` | `false` | yfinance stock-quote tool |
| `CURRENCY_ENABLED` | `false` | Frankfurter currency-conversion tool |
| `WIKIPEDIA_ENABLED` | `false` | MediaWiki summary tool |
| `WIKIPEDIA_USER_AGENT` | `langgraph-rag/1.0 (configure)` | Required when `WIKIPEDIA_ENABLED=true` |
| `DIRECTIONS_ENABLED` | `false` | OSRM route/distance/time tool |
| `MAP_ENABLED` | `false` | Open-Meteo + OpenStreetMap place locator tool |
| `MATH_ENABLED` | `false` | SymPy symbolic math tool (calculus, algebra, limits, series) |
| `STATISTICS_ENABLED` | `false` | Descriptive statistics tool (stdlib) |
| `LINALG_ENABLED` | `false` | Linear algebra / matrix tool (SymPy) |
| `NUMBER_THEORY_ENABLED` | `false` | Number theory tool: factors, primes, GCD/LCM, bases (SymPy) |
| `DATETIME_ENABLED` | `false` | Date math / timezone tool (stdlib) |
| `SUMMARIZE_URL_ENABLED` | `false` | Single-URL fetch + summarize tool |
| `FILE_READ_ENABLED` | `false` | Local file-reading tools (.txt/.md/.log/.csv, .docx, .xlsx, .pdf) + chat uploads |
| `FILE_READ_ROOT` | `.` | Root directory the file tools and uploads are confined to |
| `FILE_READ_MAX_BYTES` | `5000000` | Maximum readable/uploadable file size in bytes |
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
          └─(live_web_search)─► decompose → search_queries → merge → web_answer ─► END   (grounded)
                                                                       │
                                                          (no readable, !expanded)
                                                                       │
                                                                       ▼
                                                          expand → search_queries → merge → web_answer ─► END   (grounded retry)
                                                                                              │
                                                                                  (no readable, expanded)
                                                                                              │
                                                                                              ▼
                                                                                             END   (grounded refusal)
```

Skips Chroma, embeddings, and grading. The agent either answers directly (system prompt steers it away from tools for stable factual questions like founding dates and capitals) or calls the live web search tool, which kicks off a `decompose → search_queries → merge → web_answer` chain:

- **decompose** splits a compound question into 1-3 atomic sub-questions; atomic questions bypass the decomposition LLM.
- **search_queries** executes up to 6 distinct queries with at most 3 concurrent provider calls; this bounded fan-out is owned by the graph, so chat does not run a duplicate preliminary search or rebuild the graph per turn.
- **merge** dedupes URLs by canonical form and ranks provider title/snippet relevance and overall quality before cross-query overlap and provider rank, keeping `web_search_top_k`.
- **web_answer** fetches the merged URLs (HTML, PDF, or a bounded browser render), then admits pages through readability, structural measurement, query relevance, publication-date, duplicate, and evidence gates before prompting the LLM for a grounded answer. Each page's outcome is recorded against its domain for future ranking.

When `web_answer` produces no readable, relevant content and expansion has not yet fired, the post-`web_answer` edge routes to **expand**, which produces bounded keyword paraphrases per sub-question. The graph re-enters `search_queries → merge → web_answer` with the broader query set; the merge stage combines the new URLs with the first attempt's URLs rather than discarding them. A second failure terminates with the grounded refusal. Web-search mode never substitutes an answer from model training knowledge when no source survives admission.

Provider queries use deterministic keyword cleanup by default. Relative time wording is resolved to a concrete year, so "who wins the world cup this year" searches for the current year instead of returning evergreen all-time list pages; `last year` and `next year` (and `今年`/`去年`/`明年`) resolve the same way, and an explicit year in the question always wins. Set `WEB_SEARCH_LLM_QUERY_REWRITE_ENABLED=true` only when an additional LLM rewrite call is worth its latency and cost; it is not required for ordinary chat search.

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
| search_queries | `search_queries_factory` | — (bounded concurrent provider calls) |
| merge | `merge_factory` | — (pure code: relevance-aware dedupe + rank) |
| web_answer | `web_answer_factory` | `build_web_search_prompt` |
| fallback_answer | `fallback_answer_factory` | inline tool-free fallback prompt |

### Agent Tools

The agent can be given any combination of these tools via per-tool config flags. All optional tools default to off; the retriever is always present in the full graph and `live_web_search` defaults to on.

| Tool | Module | Config flag | Notes |
|---|---|---|---|
| `retrieve_source_documents` | `src/rag/chroma_retriever.py` | always on (full graph) | Chroma vectorstore retrieval |
| `live_web_search` | `src/web_search/tool.py` | `WEB_SEARCH_ENABLED=true` (default) | Serper/Brave/Tavily/Bing APIs plus Bing/Baidu/DuckDuckGo HTML fallbacks |
| `get_weather` | `src/tools/weather.py` | `WEATHER_ENABLED=true` | Open-Meteo forecast (city or coordinates), no API key |
| `get_stock_quote` | `src/tools/stock.py` | `STOCK_ENABLED=true` | yfinance / Yahoo Finance, no API key |
| `convert_currency` | `src/tools/currency.py` | `CURRENCY_ENABLED=true` | Frankfurter API (201 currencies), no API key |
| `search_wikipedia` | `src/tools/wikipedia_tool.py` | `WIKIPEDIA_ENABLED=true` | MediaWiki API summary + URL, set `WIKIPEDIA_USER_AGENT` |
| `get_directions` | `src/tools/directions.py` | `DIRECTIONS_ENABLED=true` | OSRM route/distance/time between two places, no API key |
| `find_on_map` | `src/tools/map_tool.py` | `MAP_ENABLED=true` | Open-Meteo geocoding + OpenStreetMap link, no API key |
| `solve_math` | `src/tools/math_tool.py` | `MATH_ENABLED=true` | SymPy derivatives/integrals/solve/simplify/limits/series, no API key |
| `compute_statistics` | `src/tools/statistics_tool.py` | `STATISTICS_ENABLED=true` | Mean/median/mode/variance/stdev/quartiles (stdlib), no API key |
| `linear_algebra` | `src/tools/linalg_tool.py` | `LINALG_ENABLED=true` | Matrix det/inverse/transpose/multiply/eigenvalues, solve Ax=b (SymPy) |
| `number_theory` | `src/tools/number_theory_tool.py` | `NUMBER_THEORY_ENABLED=true` | Factorization, primality, GCD/LCM, base conversion (SymPy) |
| `calculate_datetime` | `src/tools/datetime_tool.py` | `DATETIME_ENABLED=true` | Date math, timezone conversion, weekdays (stdlib), no API key |
| `summarize_url` | `src/tools/summarize_tool.py` | `SUMMARIZE_URL_ENABLED=true` | Fetch one URL and summarize it (reuses web-search fetcher + LLM) |
| `read_text_file` | `src/tools/text_file.py` | `FILE_READ_ENABLED=true` | Read .txt/.md/.log/.csv from `FILE_READ_ROOT` |
| `read_word_document` | `src/tools/word_file.py` | `FILE_READ_ENABLED=true` | Extract text from a .docx in `FILE_READ_ROOT` |
| `read_excel_spreadsheet` | `src/tools/excel_file.py` | `FILE_READ_ENABLED=true` | Read .xlsx rows from `FILE_READ_ROOT` (needs `openpyxl`) |
| `read_pdf` | `src/tools/pdf_file.py` | `FILE_READ_ENABLED=true` | Extract text from a .pdf in `FILE_READ_ROOT` (needs `pypdf`) |

When the agent calls a non-web-search tool in the lightweight graph (weather, stock, currency, Wikipedia, directions, map, math, statistics, linear algebra, number theory, datetime, summarize-url, or a file reader), the post-tool edge routes back to the agent so it can synthesize the structured tool output into a final answer — bypassing the `decompose → search_queries → merge → web_answer` chain that's specific to `live_web_search`.

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
| `POST` | `/chat/{id}/upload` | Upload files (.txt/.md/.log/.csv, .docx, .xlsx, .pdf) for the thread's file tools |
| `GET` | `/chat/{id}/history` | Read transcript |
| `DELETE` | `/chat/{id}` | Delete a thread (also removes the thread's uploaded files) |

## Persistence

The default Chroma vector store lives under `.chroma/`.

Chat uses additional persisted state:

- `.chroma/chat/<thread_id>/`: isolated Chroma store for per-thread sources
- `.chroma/chat/sessions.sqlite3`: session metadata via `SQLiteStorage`
- `.chroma/chat/checkpoints.sqlite3`: LangGraph checkpoints via `SQLiteMemorySaver`
- `.chroma/web-search/reputation.sqlite3`: rolling per-domain fetch outcomes used as a ranking prior (safe to delete; it rebuilds itself)

Chat sessions survive app restarts. Sessions with explicit URLs use isolated per-thread stores. Lightweight web chat compiles once, discovers sources inside the graph, and persists the latest URLs as session metadata without rebuilding Chroma or the graph. Checkpoints retain the complete transcript for history and restart recovery, while model calls receive only a bounded recent projection controlled by `CHAT_CONTEXT_MAX_TURNS` and `CHAT_CONTEXT_MAX_CHARS`. The `SQLiteMemorySaver` checkpointer is thread-safe under concurrent `/chat` requests for the same thread (uses a reentrant lock around the inherited `MemorySaver` mutations and the SQLite snapshot).

## File Uploads and Reading

Chat mode can read local files through four agent tools (`read_text_file`, `read_word_document`, `read_excel_spreadsheet`, `read_pdf`), and the chat UI can upload files for those tools to read. The whole feature is gated by `FILE_READ_ENABLED` (default `false`).

### Enabling

```text
FILE_READ_ENABLED=true
FILE_READ_ROOT=./uploads        # directory the tools and uploads are confined to
FILE_READ_MAX_BYTES=5000000     # per-file size cap
```

Set `FILE_READ_ROOT` to a dedicated directory rather than the project root so the tools and uploads are sandboxed away from source and config files. The Excel tool also requires `openpyxl` (already pinned in `requirements.txt`).

### Supported types

| Tool | Extensions |
|---|---|
| `read_text_file` | `.txt`, `.md`, `.log`, `.csv` |
| `read_word_document` | `.docx` (legacy binary `.doc` is not supported) |
| `read_excel_spreadsheet` | `.xlsx` (legacy `.xls` is not supported) |
| `read_pdf` | `.pdf` (scanned/image-only PDFs without a text layer cannot be read) |

### Upload → read flow

1. In the chat UI, click the 📎 button and pick one or more files. The browser POSTs them to `POST /chat/{id}/upload` as `multipart/form-data`.
2. Each file is saved under `<FILE_READ_ROOT>/chat_uploads/<thread_id>/<name>`. The response lists saved files (with tool-ready relative paths) and per-file errors. Accepted files show a chip; rejected files show a red error chip (and the error banner when the whole upload fails).
3. On the next message turn, the server injects a `SystemMessage` listing the uploaded files' exact paths so the LLM knows what to pass to the file tools. The note is announced once per new file (tracked on the session) and is filtered out of the user-visible transcript.
4. Ask the assistant to read or summarize a file by name; the model calls the matching file tool with the full path from the injected note.

### Security model

- Uploads and reads require `FILE_READ_ENABLED=true`; otherwise both are refused.
- All paths resolve under `FILE_READ_ROOT`; `../` traversal outside the root is denied.
- Sensitive files (`.env`, private keys, `credentials`, etc.) are always refused even inside the root.
- File type is allowlisted and size is capped (`FILE_READ_MAX_BYTES`) at both the upload endpoint and the tools.
- Uploads land in per-thread directories, so threads cannot read each other's files. Deleting a thread (`DELETE /chat/{id}`) also removes its upload directory.
- The upload endpoint respects `API_KEY` auth like other mutation endpoints.

The filesystem is the source of truth for uploads, so a server restart preserves uploaded files; the next turn re-announces the full current set to the model.

## Web Search

Live web search supports key-backed APIs and HTML fallbacks in `src/web_search/`:

- **Serper** (`SerperWebSearch`) - supported Google-results JSON API
- **Brave** (`BraveWebSearch`) - supported Brave Search JSON API
- **Tavily** (`TavilyWebSearch`) - supported Tavily JSON API
- **Bing API** (`BingApiWebSearch`) - configured Microsoft Bing Search endpoint
- **Bing** (`BingWebSearch`) - HTML fallback with optional recency filter
- **Baidu** (`BaiduWebSearch`) - HTML fallback with CAPTCHA detection
- **DuckDuckGo** (`DuckDuckGoWebSearch`) - bounded HTML fallback

Configured API providers are automatically prioritized for predominantly Chinese queries. Discovery runs ordered provider stages with `web_search_provider_fanout=2`, independent adapter timeouts, a shared overall deadline, and per-provider circuit breakers. A stage ends provider fallback only when it yields at least two usable URLs, so a single weak hit no longer suppresses every remaining provider; thinner stages keep their URLs and the next stage merges more recall on top. If no API key is configured, Mandarin search falls back to concurrent Baidu/Bing HTML discovery and then DuckDuckGo. An explicitly injected provider remains pinned for tests and integrations.

### Filtering pipeline

Hard rejections are reserved for URLs that can never be sources (auth and search paths, unresolved search-engine redirects, doorway scripts, unparseable binaries). Everything else is scored, measured, or demoted, so a relevant page is not lost to a guess about its URL shape.

- **Pre-fetch admission and ranking** combines provider title/snippet relevance, identifiers, quoted titles, `site:` constraints, intent evidence, language, source authority, requested years, count evidence, and URL quality. Hard constraint mismatches are rejected. Owner-name lookalike hosts and tag/category/author listing paths are *demoted* rather than removed, because substring and path guesses also hit genuine first-party sources.
- **Structural page filtering** (`WEB_SEARCH_STRUCTURE_FILTER_ENABLED`, on by default) measures the page that was actually fetched — anchor-text share, anchors per 100 words, and content volume — and removes index/tag listings, login and enable-JavaScript shells, and thin SEO pages. Unmeasurable pages abstain instead of being rejected. This is what replaces URL-pattern guessing; see `src/web_search/page_structure.py`.
- **Optional semantic relevance** (`WEB_SEARCH_SEMANTIC_FILTER_ENABLED`, off by default) adds cosine similarity from a local sentence-transformers model. It can only add recall: a bounded bonus for strong matches, a rescue at the gate floor for results that lexical scoring filtered out, and a second chance for pages the lexical page gate rejected. It never lowers a lexical score or bypasses the year, quantity, and typed-evidence gates. First use downloads the model.
- **Adaptive domain reputation** (`WEB_SEARCH_DOMAIN_REPUTATION_ENABLED`, on by default) records per-domain fetch outcomes (grounded, rejected, unreachable) in `CHROMA_DIR/web-search/reputation.sqlite3` and feeds a bounded ranking prior back into merge. It stays neutral until a domain reaches `WEB_SEARCH_DOMAIN_REPUTATION_MIN_SAMPLES`, and never rejects a URL on its own.
- **Provider-result dedup** by canonical host/path, plus per-domain diversity in merge.
- **Document fetching** loads HTML concurrently and extracts `.pdf` sources with `pypdf` (bounded to 30 pages / 20 MB), so official notices and vendor whitepapers stay eligible. Scanned PDFs without a text layer fail like any other unreadable page.
- **Optional JS-capable fallback** (`WEB_SEARCH_JS_FALLBACK_ENABLED`) retries through a lazy headless Chromium adapter whenever an HTTP fetch returned no readable text or a login/enable-JavaScript shell — the trigger is the measurement, not a domain list. It is bounded by `WEB_SEARCH_JS_RETRY_BUDGET`, and the configured domains (`baike.baidu.com`, `zhuanlan.zhihu.com`, `apps.microsoft.com`, `deepseek.net` by default) only get priority inside that budget. Off by default; requires `playwright` plus a Chromium runtime.
- **Pre-index document filtering** (full graph only) drops short/empty/boilerplate/low-signal pages, with an optional embedding similarity gate against the question (`document_quality_relevance_query`) and configurable recency bias from extracted publication dates.
- **Post-retrieval re-ranking** (full graph only) scores chunks by query/document overlap, frequency, and phrase matches; `RERANK_STRATEGY` switches between lexical (default), embedding, or hybrid.

When `web_search_lightweight` is enabled (default), the lightweight graph routes the agent's tool call through `decompose → search_queries → merge → web_answer` as described above, with one-shot conditional expansion and a grounded refusal when no evidence survives. Chat uses this as the sole web-search owner; it does not perform a preliminary provider search or recompile the graph for each turn.

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
python -m pytest -q                         # Run tests (current baseline: 455 passed)
ruff check .                                # Lint
mypy src/                                   # Type check
python -m pytest --tb=short --cov=src --cov-report=term --cov-fail-under=70
python -m compileall src tests              # Syntax check
git diff --check                            # Whitespace check
```

## Notes

- The chat model is configurable via `LLM_PROVIDER` — DashScope (`qwen-plus`) or DeepSeek (`deepseek-v4-pro`). Embeddings always use DashScope/Tongyi unless `embedding_model` is set to a HuggingFace model.
- Existing Chroma stores with incompatible embedding metadata are rebuilt automatically on startup.
- Key-backed search APIs are preferred for Mandarin when configured. Otherwise search uses the bounded Bing/Baidu/DuckDuckGo HTML fallbacks. CAPTCHA and repeated network failures open temporary provider circuits instead of blocking every expanded query.
- Noise filtering is measured rather than pattern-matched: structural page assessment and the learned domain prior are on by default, and semantic similarity is available as an opt-in recall layer. The tradeoff is that structural filtering needs a fetch first, so a noisy URL still costs one concurrent request.
- The agent system prompt (`AGENT_SYSTEM_PROMPT` in `src/llm/prompts.py`) tells the model to answer directly when tools aren't needed — covering math, general knowledge, programming concepts, definitions, well-established stable facts (founding dates, capitals, public figures), and chitchat — so the graph avoids unnecessary retrieval/rewrite cycles.
- Reranking (`RERANK_STRATEGY`) defaults to lexical (keyword-based); `embedding` uses cosine similarity against embedding vectors; `hybrid` combines both.
- Optional agent tools (`weather`, `stock`, `currency`, `wikipedia`, `directions`, `map`, `math`, `statistics`, `linalg`, `number_theory`, `datetime`, `summarize_url`, and the file readers) are off by default. Enable them via the per-tool `_ENABLED` flag in `.env` (the four file readers share `FILE_READ_ENABLED`). None require an API key; only `WIKIPEDIA_USER_AGENT` should be customized for shared deployments, and the file tools should have `FILE_READ_ROOT` pointed at a dedicated directory.
- The lightweight graph's conditional expansion fires only on web-search retrieval failure — single-keyword questions that get a readable, relevant page back take the fast path with one search and one LLM call. Compound questions that decompose into multiple sub-Qs still take the fast path; expansion only fires when no fetched page yields usable evidence.
