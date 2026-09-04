# Only Subscribers

A local LangGraph retrieval-augmented generation project with a multi-turn chat
FastAPI app:

- **Chat** (`src/frontend/chat`): multi-turn chat with per-thread source sets, persisted session metadata, and SQLite-backed LangGraph checkpoints.

The project started as a Python extraction of a Jupyter notebook; it is now organized as a reusable codebase with YAML configuration, typed graph events, SSE streaming, health/readiness/metrics endpoints, optional API-key auth, Docker support, CI quality gates, and company-readiness documentation.

The same stateless RAG engine that grounds the chat app is also exposed outside the web app as stateless MCP tools (`rag_ask`, `rag_web_search_answer`); see [MCP RAG tools](#mcp-rag-tools).

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
|   |-- adapters/mcp_server/  # Separate stateless inbound MCP server (stdio or authenticated Streamable HTTP)
|   |-- api/                  # Shared FastAPI helpers (auth, CORS, errors, streaming, dependencies)
|   |-- chat/                 # Multi-turn chat app (API, UI, entry point)
|   |-- config/               # Settings dataclass + YAML/env loader
|   |-- core/                 # Deprecated re-exports (redirect to src/backend/graph, src/backend/rag, etc.)
|   |-- errors.py             # Typed RAG exceptions
|   |-- graph/                # LangGraph builder, state, edges, executor, metrics
|   |   |-- nodes/            # Node factories (agent, condense, decompose, bounded search, expand, generate, grade, merge, rewrite, web_answer)
|   |-- llm/                  # LLM provider seam (DashScope, DeepSeek) + prompt templates
|   |-- rag/                  # Chroma retriever, embeddings (DashScope, HuggingFace), document loader/quality
|   |-- sessions/             # Chat session registry, SQLite metadata + checkpoint persistence
|   |-- tools/                # Optional agent tools (weather, stock, currency, Wikipedia, directions, map, math, statistics, linear algebra, number theory, datetime, summarize-url, live web search, file readers, document editors, long-term memory) + shared HTTP and geocoding helpers
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

To enable the AMap-backed map and directions tools and render their map cards
in chat, also configure an AMap Web Service key plus a JS API key and security
code. The Web Service key and security code remain server-side; only the JS API
key is returned to the browser.

```text
AMAP_WEB_SERVICE_KEY=your_amap_web_service_key_here
AMAP_JS_API_KEY=your_amap_js_api_key_here
AMAP_JS_SECURITY_CODE=your_amap_js_security_code_here
MAP_ENABLED=true
DIRECTIONS_ENABLED=true
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
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek-compatible endpoint; override for a proxy or gateway |
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
| `DIRECTIONS_ENABLED` | `false` | AMap route/distance/time tool; requires `AMAP_WEB_SERVICE_KEY` |
| `MAP_ENABLED` | `false` | AMap POI/address/district locator; requires `AMAP_WEB_SERVICE_KEY` |
| `AMAP_WEB_SERVICE_KEY` | - | Server-only AMap Web Service key used by map and directions tools |
| `AMAP_JS_API_KEY` | - | Browser-visible AMap JS API key used for chat map cards |
| `AMAP_JS_SECURITY_CODE` | - | Server-only JS security code injected by the same-origin AMap proxy |
| `AMAP_API_TIMEOUT_SECONDS` | `10` | Timeout for AMap Web Service and proxy requests (minimum 1 second) |
| `MATH_ENABLED` | `false` | SymPy symbolic math tool (calculus, algebra, limits, series) |
| `STATISTICS_ENABLED` | `false` | Descriptive statistics tool (stdlib) |
| `LINALG_ENABLED` | `false` | Linear algebra / matrix tool (SymPy) |
| `NUMBER_THEORY_ENABLED` | `false` | Number theory tool: factors, primes, GCD/LCM, bases (SymPy) |
| `DATETIME_ENABLED` | `false` | Date math / timezone tool (stdlib) |
| `SUMMARIZE_URL_ENABLED` | `false` | Single-URL fetch + summarize tool |
| `FILE_READ_ENABLED` | `false` | Local file-reading tools (.txt/.md/.log/.csv, .docx, .xlsx, .pdf) + chat uploads; `.pptx` uploads also require `POWERPOINT_EDIT_ENABLED` |
| `FILE_READ_ROOT` | `.` | Root directory the file tools and uploads are confined to |
| `FILE_READ_MAX_BYTES` | `5000000` | Maximum readable/uploadable file size in bytes |
| `WORD_EDIT_ENABLED` | `false` | Word .docx creation/editing; needs FILE_READ_ENABLED too; files stay in session uploads |
| `POWERPOINT_EDIT_ENABLED` | `false` | PowerPoint .pptx inspection/editing; needs FILE_READ_ENABLED too; files stay in session uploads |
| `EXCEL_CREATE_ENABLED` | `false` | Excel .xlsx creation; needs FILE_READ_ENABLED and the artifact-tool Node runtime |
| `EXCEL_EDIT_ENABLED` | `false` | Excel .xlsx editing (openpyxl); needs FILE_READ_ENABLED too; files stay in session uploads |
| `EXCEL_NODE_EXECUTABLE` | `node` | Loader-provided Node.js executable used for Excel creation |
| `EXCEL_NODE_MODULES_PATH` | - | Loader-provided `node_modules` directory containing `@oai/artifact-tool` |
| `TEXT_EDIT_ENABLED` | `false` | Plain-text .txt creation/editing; needs FILE_READ_ENABLED too; files stay in session uploads |
| `MARKDOWN_EDIT_ENABLED` | `false` | Markdown .md creation/editing; needs FILE_READ_ENABLED too; files stay in session uploads |
| `TYPESCRIPT_EDIT_ENABLED` | `false` | TypeScript .ts/.tsx creation/editing; needs FILE_READ_ENABLED too; files stay in session uploads |
| `JSON_EDIT_ENABLED` | `false` | JSON .json creation and structured path-based editing; needs FILE_READ_ENABLED too |
| `JSONL_EDIT_ENABLED` | `false` | JSONL .jsonl creation/editing (every line must parse as JSON); needs FILE_READ_ENABLED too |
| `R_EDIT_ENABLED` | `false` | R .r creation/editing; needs FILE_READ_ENABLED too |
| `RUST_EDIT_ENABLED` | `false` | Rust .rs creation/editing; needs FILE_READ_ENABLED too |
| `GO_EDIT_ENABLED` | `false` | Go .go creation/editing; needs FILE_READ_ENABLED too |
| `SQL_EDIT_ENABLED` | `false` | SQL .sql creation/editing (MySQL/PostgreSQL/SQLite scripts); needs FILE_READ_ENABLED too |
| `PHP_EDIT_ENABLED` | `false` | PHP .php creation/editing; needs FILE_READ_ENABLED too |
| `RUBY_EDIT_ENABLED` | `false` | Ruby .rb creation/editing; needs FILE_READ_ENABLED too |
| `LATEX_EDIT_ENABLED` | `false` | LaTeX .tex creation/editing; needs FILE_READ_ENABLED too |
| `PROLOG_EDIT_ENABLED` | `false` | Prolog .pl creation/editing; needs FILE_READ_ENABLED too |
| `HASKELL_EDIT_ENABLED` | `false` | Haskell .hs creation/editing; needs FILE_READ_ENABLED too |
| `LUA_EDIT_ENABLED` | `false` | Lua .lua creation/editing; needs FILE_READ_ENABLED too |
| `JULIA_EDIT_ENABLED` | `false` | Julia .jl creation/editing; needs FILE_READ_ENABLED too |
| `SHELL_EDIT_ENABLED` | `false` | Shell script .sh/.bash creation/editing; needs FILE_READ_ENABLED too |
| `MATLAB_EDIT_ENABLED` | `false` | MATLAB .m creation/editing; needs FILE_READ_ENABLED too |
| `GROOVY_EDIT_ENABLED` | `false` | Groovy .groovy creation/editing; needs FILE_READ_ENABLED too |
| `SWIFT_EDIT_ENABLED` | `false` | Swift .swift creation/editing; needs FILE_READ_ENABLED too |
| `LOG_EDIT_ENABLED` | `false` | Log .log creation/editing; needs FILE_READ_ENABLED too |
| `CHROMA_DIR` | `.chroma` | Vector store location |
| `RAG_ENV` | `development` | Runtime environment; `production` activates fail-closed deployment topology checks |
| `RAG_WORKER_COUNT` | `1` | Declared production worker count; must remain `1` while local state or locks are authoritative |
| `RAG_REPLICA_COUNT` | `1` | Declared production replica count; must remain `1` while local state or locks are authoritative |
| `API_KEY` | — | API auth key for mutation endpoints via `Authorization: Bearer`; open when unset |
| `API_HOST` | `127.0.0.1` | Bind address for the Chat server |
| `API_PORT` | `8001` | Listen port for the Chat server |
| `CORS_ALLOW_ORIGINS` | empty | Comma-separated browser origins allowed to call the API |
| `MCP_ENABLED` | `false` | Enable only the separately launched inbound MCP process |
| `MCP_TRANSPORT` | `stdio` | `stdio` or stateless Streamable HTTP (`http`) |
| `MCP_ENVIRONMENT` | `development` | `development`, `staging`, or `production`; controls fail-closed HTTP policy |
| `MCP_AUTH_SECRET_ENV` | `API_KEY` | Environment-variable reference for the transitional HTTP bearer key |
| `MCP_ALLOWED_HOSTS` | empty | Exact HTTP Host allowlist; required outside development |
| `MCP_ALLOWED_ORIGINS` | empty | Exact browser Origin allowlist; absent Origin remains valid for non-browser clients |
| `MCP_DEADLINE_SECONDS` | `120` | Mandatory whole-call server deadline (bounded to 600 seconds) |
| `MCP_MAX_SEARCH_RESULTS` | `10` | Maximum discovered web-search URLs admitted before source validation and invocation |
| `MCP_MAX_CONCURRENCY` | `4` | Process-local concurrent MCP calls for the supported single instance |
| `RERANK_STRATEGY` | `lexical` | `lexical`, `embedding`, or `hybrid` |
| `MEMORY_ENABLED` | `false` | Enable long-term memory tools (save/recall/forget) and auto-recall injection |
| `MEMORY_STORE_PATH` | - | Memory store file path; defaults to `memory/long_term_memory.json` relative to working directory |
| `MEMORY_MAX_RECORDS` | `500` | Total records kept across all scopes (1–10000); oldest by last-recall time evicted first |
| `MEMORY_MAX_RECORD_CHARS` | `1000` | Maximum characters per memory record (1–10000) |
| `MEMORY_RECALL_TOP_K` | `5` | Records returned per recall call (1–50) |
| `MEMORY_CONTEXT_MAX_CHARS` | `2000` | Character cap on memory text injected into each prompt (1–20000) |
| `MEMORY_DEFAULT_SCOPE` | `global` | Default scope: `global` (all sessions) or `session` (this conversation only) |
| `MEMORY_AUTO_RECALL_ENABLED` | `true` | Prepend relevant memories to each chat turn automatically |
| `MEMORY_EXTRACTION_ENABLED` | `false` | Auto-extract durable facts from finished sessions and every `MEMORY_EXTRACTION_TURN_INTERVAL` turns |
| `MEMORY_EXTRACTION_ON_SESSION_START` | `true` | Also extract from the previous session when a new session starts |
| `MEMORY_EXTRACTION_TURN_INTERVAL` | `10` | Turns per extraction round (1–1000) |
| `MEMORY_EXTRACTION_MAX_CANDIDATES` | `5` | Maximum memories accepted from one extraction round (1–20) |
| `MEMORY_EXTRACTION_MAX_TRANSCRIPT_CHARS` | `8000` | Character cap on transcript excerpt sent to one extraction (200–100000) |
| `MEMORY_EXTRACTION_TIMEOUT_SECONDS` | `60` | Seconds to wait for the extraction LLM call (1–600) |
| `MEMORY_EXTRACTION_MAX_CONCURRENCY` | `2` | Extractions allowed to run at once (1–16); excess requests are dropped |
| `MEMORY_EXTRACTION_MAX_SESSION_AGE_HOURS` | `168` | Skip session-start extraction for sessions idle longer than this (1–8760 hours) |

## LangGraph Architecture

### Full Graph

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

The agent can be given any combination of these tools via per-tool config flags. All optional tools default to off; the retriever is always present in the full graph and `live_web_search` defaults to on. The document editors are additionally session-scoped: they are only offered in a chat session that has an upload directory, so sessionless callers (the stateless MCP RAG tools) expose none of them regardless of the flags.

| Tool | Module | Config flag | Notes |
|---|---|---|---|
| `retrieve_source_documents` | `src/backend/rag/chroma_retriever.py` | always on (full graph) | Chroma vectorstore retrieval |
| `live_web_search` | `src/backend/tools/web_search.py` | `WEB_SEARCH_ENABLED=true` (default) | Serper/Brave/Tavily/Bing APIs plus Bing/Baidu/DuckDuckGo HTML fallbacks |
| `get_weather` | `src/backend/tools/weather.py` | `WEATHER_ENABLED=true` | Open-Meteo forecast (city or coordinates), no API key |
| `get_stock_quote` | `src/backend/tools/stock.py` | `STOCK_ENABLED=true` | yfinance / Yahoo Finance, no API key |
| `convert_currency` | `src/backend/tools/currency.py` | `CURRENCY_ENABLED=true` | Frankfurter API (201 currencies), no API key |
| `search_wikipedia` | `src/backend/tools/wikipedia_tool.py` | `WIKIPEDIA_ENABLED=true` | MediaWiki API summary + URL, set `WIKIPEDIA_USER_AGENT` |
| `get_directions` | `src/backend/tools/directions.py` | `DIRECTIONS_ENABLED=true` | AMap driving/walking/cycling route, distance, time, and GCJ-02 map artifact; requires `AMAP_WEB_SERVICE_KEY` |
| `find_on_map` | `src/backend/tools/map_tool.py` | `MAP_ENABLED=true` | AMap POI/address/district lookup, GCJ-02 coordinates, and map artifact; requires `AMAP_WEB_SERVICE_KEY` |
| `solve_math` | `src/backend/tools/math_tool.py` | `MATH_ENABLED=true` | SymPy derivatives/integrals/solve/simplify/limits/series, no API key |
| `compute_statistics` | `src/backend/tools/statistics_tool.py` | `STATISTICS_ENABLED=true` | Mean/median/mode/variance/stdev/quartiles (stdlib), no API key |
| `linear_algebra` | `src/backend/tools/linalg_tool.py` | `LINALG_ENABLED=true` | Matrix det/inverse/transpose/multiply/eigenvalues, solve Ax=b (SymPy) |
| `number_theory` | `src/backend/tools/number_theory_tool.py` | `NUMBER_THEORY_ENABLED=true` | Factorization, primality, GCD/LCM, base conversion (SymPy) |
| `calculate_datetime` | `src/backend/tools/datetime_tool.py` | `DATETIME_ENABLED=true` | Date math, timezone conversion, weekdays (stdlib), no API key |
| `summarize_url` | `src/backend/tools/summarize_tool.py` | `SUMMARIZE_URL_ENABLED=true` | Fetch one URL and summarize it (reuses web-search fetcher + LLM) |
| `read_text_file` | `src/backend/tools/text_file.py` | `FILE_READ_ENABLED=true` | Read .txt/.md/.log/.csv from `FILE_READ_ROOT` |
| `read_markdown_file` | `src/backend/tools/markdown_file.py` | `FILE_READ_ENABLED=true` | Read a Markdown .md from `FILE_READ_ROOT` (including session uploads) |
| `read_typescript_file` | `src/backend/tools/typescript_file.py` | `FILE_READ_ENABLED=true` | Read a TypeScript .ts/.tsx from `FILE_READ_ROOT` (including session uploads) |
| `read_json_file` | `src/backend/tools/json_file.py` | `FILE_READ_ENABLED=true` | Read a .json or .jsonl from `FILE_READ_ROOT` (including session uploads) |
| `read_r_file` | `src/backend/tools/r_file.py` | `FILE_READ_ENABLED=true` | Read an R .r file from `FILE_READ_ROOT` (including session uploads) |
| `read_rust_file` | `src/backend/tools/rust_file.py` | `FILE_READ_ENABLED=true` | Read a Rust .rs file from `FILE_READ_ROOT` (including session uploads) |
| `read_go_file` | `src/backend/tools/go_file.py` | `FILE_READ_ENABLED=true` | Read a Go .go file from `FILE_READ_ROOT` (including session uploads) |
| `read_groovy_file` | `src/backend/tools/groovy_file.py` | `FILE_READ_ENABLED=true` | Read a Groovy .groovy file from `FILE_READ_ROOT` (including session uploads) |
| `read_sql_file` | `src/backend/tools/sql_file.py` | `FILE_READ_ENABLED=true` | Read a SQL .sql script from `FILE_READ_ROOT` (including session uploads) |
| `read_swift_file` | `src/backend/tools/swift_file.py` | `FILE_READ_ENABLED=true` | Read a Swift .swift file from `FILE_READ_ROOT` (including session uploads) |
| `read_log_file` | `src/backend/tools/log_file.py` | `FILE_READ_ENABLED=true` | Read a .log file from `FILE_READ_ROOT` (including session uploads) |
| `read_php_file` | `src/backend/tools/php_file.py` | `FILE_READ_ENABLED=true` | Read a PHP .php file from `FILE_READ_ROOT` (including session uploads) |
| `read_ruby_file` | `src/backend/tools/ruby_file.py` | `FILE_READ_ENABLED=true` | Read a Ruby .rb file from `FILE_READ_ROOT` (including session uploads) |
| `read_latex_file` | `src/backend/tools/latex_file.py` | `FILE_READ_ENABLED=true` | Read a LaTeX .tex file from `FILE_READ_ROOT` (including session uploads) |
| `read_prolog_file` | `src/backend/tools/prolog_file.py` | `FILE_READ_ENABLED=true` | Read a Prolog .pl file from `FILE_READ_ROOT` (including session uploads) |
| `read_haskell_file` | `src/backend/tools/haskell_file.py` | `FILE_READ_ENABLED=true` | Read a Haskell .hs file from `FILE_READ_ROOT` (including session uploads) |
| `read_lua_file` | `src/backend/tools/lua_file.py` | `FILE_READ_ENABLED=true` | Read a Lua .lua file from `FILE_READ_ROOT` (including session uploads) |
| `read_julia_file` | `src/backend/tools/julia_file.py` | `FILE_READ_ENABLED=true` | Read a Julia .jl file from `FILE_READ_ROOT` (including session uploads) |
| `read_shell_file` | `src/backend/tools/shell_file.py` | `FILE_READ_ENABLED=true` | Read a shell .sh/.bash script from `FILE_READ_ROOT` (including session uploads) |
| `read_matlab_file` | `src/backend/tools/matlab_file.py` | `FILE_READ_ENABLED=true` | Read a MATLAB .m file from `FILE_READ_ROOT` (including session uploads) |
| `inspect_zip_file` | `src/backend/tools/zip_file.py` | `FILE_READ_ENABLED=true` | List entries (with sizes) of a .zip archive from `FILE_READ_ROOT` (including session uploads) |
| `read_zip_entry` | `src/backend/tools/zip_file.py` | `FILE_READ_ENABLED=true` | Read one text entry from inside a .zip without extracting it; bomb-guarded, binary entries described not shown |
| `read_word_document` | `src/backend/tools/word_file.py` | `FILE_READ_ENABLED=true` | Extract text from a .docx in `FILE_READ_ROOT` |
| `read_excel_spreadsheet` | `src/backend/tools/excel_file.py` | `FILE_READ_ENABLED=true` | Read .xlsx rows from `FILE_READ_ROOT` (needs `openpyxl`) |
| `create_excel_spreadsheet` | `src/backend/tools/excel_create.py` | `FILE_READ_ENABLED=true` and `EXCEL_CREATE_ENABLED=true` | Create a styled, formula-capable .xlsx workbook in the current session |
| `inspect_excel_spreadsheet` | `src/backend/tools/excel_edit.py` | `FILE_READ_ENABLED=true` and `EXCEL_EDIT_ENABLED=true` | List worksheets, cells, values, and formulas of a session-uploaded .xlsx |
| `edit_excel_spreadsheet` | `src/backend/tools/excel_edit.py` | `FILE_READ_ENABLED=true` and `EXCEL_EDIT_ENABLED=true` | Apply expected-value-checked cell, formula, row/column, worksheet, and format edits to a session-uploaded .xlsx; creates a new file |
| `read_pdf` | `src/backend/tools/pdf_file.py` | `FILE_READ_ENABLED=true` | Extract text from a .pdf in `FILE_READ_ROOT` (needs `pypdf`) |
| `create_word_document` | `src/backend/tools/word_edit.py` | `FILE_READ_ENABLED=true` and `WORD_EDIT_ENABLED=true` | Create a formatted .docx in the current session with headings, lists, and tables |
| `inspect_word_document` | `src/backend/tools/word_edit.py` | `FILE_READ_ENABLED=true` and `WORD_EDIT_ENABLED=true` | List numbered paragraphs and table cells of a session-uploaded .docx |
| `edit_word_document` | `src/backend/tools/word_edit.py` | `FILE_READ_ENABLED=true` and `WORD_EDIT_ENABLED=true` | Apply structured edits to a session-uploaded .docx; creates a new file |
| `inspect_powerpoint` | `src/backend/tools/powerpoint_edit.py` | `FILE_READ_ENABLED=true` and `POWERPOINT_EDIT_ENABLED=true` | List slides, shape paths, text, and table cells of a session-uploaded .pptx (needs `python-pptx`) |
| `edit_powerpoint` | `src/backend/tools/powerpoint_edit.py` | `FILE_READ_ENABLED=true` and `POWERPOINT_EDIT_ENABLED=true` | Apply expected-text-checked text/table edits to a session-uploaded .pptx; creates a new file |
| `inspect_text_file` | `src/backend/tools/text_edit.py` | `FILE_READ_ENABLED=true` and `TEXT_EDIT_ENABLED=true` | List numbered lines and text-format metadata for a session-uploaded .txt |
| `edit_text_file` | `src/backend/tools/text_edit.py` | `FILE_READ_ENABLED=true` and `TEXT_EDIT_ENABLED=true` | Apply structured line edits to a session-uploaded .txt; creates a new file |
| `create_text_file` | `src/backend/tools/text_edit.py` | `FILE_READ_ENABLED=true` and `TEXT_EDIT_ENABLED=true` | Create a new UTF-8 .txt in the current chat session |
| `inspect_markdown_file` | `src/backend/tools/markdown_edit.py` | `FILE_READ_ENABLED=true` and `MARKDOWN_EDIT_ENABLED=true` | List numbered lines and text-format metadata for a session-uploaded .md |
| `edit_markdown_file` | `src/backend/tools/markdown_edit.py` | `FILE_READ_ENABLED=true` and `MARKDOWN_EDIT_ENABLED=true` | Apply structured line edits to a session-uploaded .md; creates a new file |
| `create_markdown_file` | `src/backend/tools/markdown_edit.py` | `FILE_READ_ENABLED=true` and `MARKDOWN_EDIT_ENABLED=true` | Create a new UTF-8 Markdown .md in the current chat session |
| `inspect_typescript_file` | `src/backend/tools/typescript_edit.py` | `FILE_READ_ENABLED=true` and `TYPESCRIPT_EDIT_ENABLED=true` | List numbered lines and text-format metadata for a session-uploaded .ts/.tsx |
| `edit_typescript_file` | `src/backend/tools/typescript_edit.py` | `FILE_READ_ENABLED=true` and `TYPESCRIPT_EDIT_ENABLED=true` | Apply structured line edits to a session-uploaded .ts/.tsx; creates a new file |
| `create_typescript_file` | `src/backend/tools/typescript_edit.py` | `FILE_READ_ENABLED=true` and `TYPESCRIPT_EDIT_ENABLED=true` | Create a new UTF-8 TypeScript .ts/.tsx in the current chat session |
| `inspect_json_file` | `src/backend/tools/json_edit.py` | `FILE_READ_ENABLED=true` and `JSON_EDIT_ENABLED=true` | List documented paths and values of a session-uploaded .json |
| `edit_json_file` | `src/backend/tools/json_edit.py` | `FILE_READ_ENABLED=true` and `JSON_EDIT_ENABLED=true` | Apply expected-checked path-based set/delete/append edits to a session-uploaded .json; creates a new file |
| `create_json_file` | `src/backend/tools/json_edit.py` | `FILE_READ_ENABLED=true` and `JSON_EDIT_ENABLED=true` | Create a new UTF-8 .json (must parse as valid JSON) in the current chat session |
| `inspect_jsonl_file` | `src/backend/tools/jsonl_edit.py` | `FILE_READ_ENABLED=true` and `JSONL_EDIT_ENABLED=true` | List numbered lines of a session-uploaded .jsonl; every line is validated JSON |
| `edit_jsonl_file` | `src/backend/tools/jsonl_edit.py` | `FILE_READ_ENABLED=true` and `JSONL_EDIT_ENABLED=true` | Apply structured line edits to a session-uploaded .jsonl; written lines must parse as JSON; creates a new file |
| `create_jsonl_file` | `src/backend/tools/jsonl_edit.py` | `FILE_READ_ENABLED=true` and `JSONL_EDIT_ENABLED=true` | Create a new UTF-8 .jsonl in the current chat session (every line must parse as JSON) |
| `inspect_r_file` / `edit_r_file` / `create_r_file` | `src/backend/tools/r_edit.py` | `FILE_READ_ENABLED=true` and `R_EDIT_ENABLED=true` | Line-oriented create/inspect/edit for session-uploaded .r files; creates a new file |
| `inspect_rust_file` / `edit_rust_file` / `create_rust_file` | `src/backend/tools/rust_edit.py` | `FILE_READ_ENABLED=true` and `RUST_EDIT_ENABLED=true` | Line-oriented create/inspect/edit for session-uploaded .rs files; creates a new file |
| `inspect_go_file` / `edit_go_file` / `create_go_file` | `src/backend/tools/go_edit.py` | `FILE_READ_ENABLED=true` and `GO_EDIT_ENABLED=true` | Line-oriented create/inspect/edit for session-uploaded .go files; creates a new file |
| `inspect_sql_file` / `edit_sql_file` / `create_sql_file` | `src/backend/tools/sql_edit.py` | `FILE_READ_ENABLED=true` and `SQL_EDIT_ENABLED=true` | Line-oriented create/inspect/edit for session-uploaded .sql scripts (MySQL/PostgreSQL/SQLite); creates a new file |
| `inspect_php_file` / `edit_php_file` / `create_php_file` | `src/backend/tools/php_edit.py` | `FILE_READ_ENABLED=true` and `PHP_EDIT_ENABLED=true` | Line-oriented create/inspect/edit for session-uploaded .php files; creates a new file |
| `inspect_ruby_file` / `edit_ruby_file` / `create_ruby_file` | `src/backend/tools/ruby_edit.py` | `FILE_READ_ENABLED=true` and `RUBY_EDIT_ENABLED=true` | Line-oriented create/inspect/edit for session-uploaded .rb files; creates a new file |
| `inspect_latex_file` / `edit_latex_file` / `create_latex_file` | `src/backend/tools/latex_edit.py` | `FILE_READ_ENABLED=true` and `LATEX_EDIT_ENABLED=true` | Line-oriented create/inspect/edit for session-uploaded .tex files; creates a new file |
| `inspect_prolog_file` / `edit_prolog_file` / `create_prolog_file` | `src/backend/tools/prolog_edit.py` | `FILE_READ_ENABLED=true` and `PROLOG_EDIT_ENABLED=true` | Line-oriented create/inspect/edit for session-uploaded .pl files; creates a new file |
| `inspect_haskell_file` / `edit_haskell_file` / `create_haskell_file` | `src/backend/tools/haskell_edit.py` | `FILE_READ_ENABLED=true` and `HASKELL_EDIT_ENABLED=true` | Line-oriented create/inspect/edit for session-uploaded .hs files; creates a new file |
| `inspect_lua_file` / `edit_lua_file` / `create_lua_file` | `src/backend/tools/lua_edit.py` | `FILE_READ_ENABLED=true` and `LUA_EDIT_ENABLED=true` | Line-oriented create/inspect/edit for session-uploaded .lua files; creates a new file |
| `inspect_julia_file` / `edit_julia_file` / `create_julia_file` | `src/backend/tools/julia_edit.py` | `FILE_READ_ENABLED=true` and `JULIA_EDIT_ENABLED=true` | Line-oriented create/inspect/edit for session-uploaded .jl files; creates a new file |
| `inspect_shell_file` / `edit_shell_file` / `create_shell_file` | `src/backend/tools/shell_edit.py` | `FILE_READ_ENABLED=true` and `SHELL_EDIT_ENABLED=true` | Line-oriented create/inspect/edit for session-uploaded .sh/.bash scripts; creates a new file |
| `inspect_matlab_file` / `edit_matlab_file` / `create_matlab_file` | `src/backend/tools/matlab_edit.py` | `FILE_READ_ENABLED=true` and `MATLAB_EDIT_ENABLED=true` | Line-oriented create/inspect/edit for session-uploaded .m files; creates a new file |
| `inspect_groovy_file` / `edit_groovy_file` / `create_groovy_file` | `src/backend/tools/groovy_edit.py` | `FILE_READ_ENABLED=true` and `GROOVY_EDIT_ENABLED=true` | Line-oriented create/inspect/edit for session-uploaded .groovy files; creates a new file |
| `inspect_swift_file` / `edit_swift_file` / `create_swift_file` | `src/backend/tools/swift_edit.py` | `FILE_READ_ENABLED=true` and `SWIFT_EDIT_ENABLED=true` | Line-oriented create/inspect/edit for session-uploaded .swift files; creates a new file |
| `inspect_log_file` / `edit_log_file` / `create_log_file` | `src/backend/tools/log_edit.py` | `FILE_READ_ENABLED=true` and `LOG_EDIT_ENABLED=true` | Line-oriented create/inspect/edit for session-uploaded .log files; creates a new file |
| `save_memory` | `src/backend/tools/memory_tool.py` | `MEMORY_ENABLED=true` | Remember a durable fact, preference, or task the user states about themselves; stores with optional category, tags, and scope |
| `recall_memory` | `src/backend/tools/memory_tool.py` | `MEMORY_ENABLED=true` | Look up what is already remembered about the user by keyword; used before answering questions about the user not covered in the current conversation |
| `forget_memory` | `src/backend/tools/memory_tool.py` | `MEMORY_ENABLED=true` | Delete stored memories by id or keyword when the user asks to forget something |

When the agent calls a non-web-search tool in the lightweight graph (weather, stock, currency, Wikipedia, directions, map, math, statistics, linear algebra, number theory, datetime, summarize-url, or a file reader), the post-tool edge routes back to the agent so it can synthesize the structured tool output into a final answer — bypassing the `decompose → search_queries → merge → web_answer` chain that's specific to `live_web_search`.

### LLM Provider Seam

`src/backend/llm/provider.py` implements a provider protocol with two backends:

- **DashScopeLLMProvider** — ChatTongyi with DashScope models (default: `qwen-plus`)
- **DeepSeekLLMProvider** — ChatOpenAI pointed at `api.deepseek.com` (default: `deepseek-v4-pro`)

### Embedding Providers

`src/backend/rag/embeddings.py` supports:

- **DashScopeEmbeddings** — Tongyi text embeddings (default: `text-embedding-v4`)
- **HuggingFaceEmbeddingModel** — Local HuggingFace embedding models

## Inbound MCP Server

The inbound MCP process is separate from both FastAPI applications and publishes exactly two stateless tools: `rag_ask` (configured defaults or caller-supplied HTTPS sources) and `rag_web_search_answer` (forced existing web discovery). It never advertises chat sessions, files, editors, memory mutation, administration, or internal graph tools.

For a local MCP client, set `MCP_ENABLED=true` and keep `MCP_TRANSPORT=stdio`, then configure the client to launch:

```text
<repo>/.venv/Scripts/python.exe -m src.frontend.adapters.mcp_server.main
```

On macOS/Linux use `<repo>/.venv/bin/python`. Standard output is reserved for MCP protocol frames; diagnostics and bounded audit metadata go to standard error. The subprocess receives the explicit `local-process` principal.

For Streamable HTTP, select `MCP_TRANSPORT=http`. Staging and production fail before socket bind unless the environment variable named by `MCP_AUTH_SECRET_ENV` contains a bearer key, `MCP_PUBLIC_BASE_URL` is HTTPS, and `MCP_ALLOWED_HOSTS` contains exact deployment hosts. Anonymous HTTP requires both `MCP_ENVIRONMENT=development` and `MCP_ALLOW_ANONYMOUS_HTTP=true`. The endpoint defaults to `http://127.0.0.1:8002/mcp`; send the shared key as `Authorization: Bearer <key>`.

HTTP uses the SDK's stateless ASGI application with exact Host/Origin checks and request-body bounds. Caller-supplied and discovered source hosts are resolved before application invocation, and every resolved address must be public. The current legacy document loaders may resolve again and follow redirects without connection-address pinning, so this release does **not** claim connection-boundary DNS-rebinding protection for outbound source fetching. Deploy restricted egress and an allowlisting proxy; a future fetcher must pin validated addresses and independently validate each redirect before this limitation can be removed. Downstream synchronous graph/provider calls also cannot always be force-cancelled after the adapter deadline fires, although cancellation propagates through asynchronous seams.

## Running Chat

Start the chat web/API server:

```powershell
python -m src.frontend.chat.main serve --host 127.0.0.1 --port 8001
```

Then open `http://127.0.0.1:8001`.

Use the terminal REPL:

```powershell
python -m src.frontend.chat.main chat --urls "https://example.com,https://another.com"
python -m src.frontend.chat.main chat --seed-question "latest model releases 2026"
```

API endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Browser UI |
| `GET` | `/health` | Public dependency-free liveness probe |
| `GET` | `/ready` | Public bounded readiness status |
| `GET` | `/admin/health/dependencies` | API-key-protected dependency diagnostics |
| `GET` | `/metrics` | In-process graph metrics |
| `POST` | `/chat` | Create a thread |
| `POST` | `/chat/{id}/message` | Send a turn |
| `POST` | `/chat/{id}/message/stream` | SSE stream: node events + per-token answer deltas (`?tokens=false` for node events only) |
| `POST` | `/chat/{id}/upload` | Upload files (.txt/.md/.log/.csv, .docx, .xlsx, .pdf; .pptx when PowerPoint editing is enabled) for the thread's file tools |
| `GET` | `/chat/{id}/files/{filename}` | Download a session-scoped file (uploaded source or created/edited artifact) |
| `GET` | `/chat/{id}/history` | Read transcript |
| `DELETE` | `/chat/{id}` | Delete a thread (also removes the thread's uploaded files) |
| `GET` | `/chat/config` | Returns AMap client configuration (key, timeout) to the browser |
| `GET` | `/_AMapService/{proxied_path}` | Proxies AMap JS API requests (avoids CORS); uses `AMAP_JS_SECURITY_CODE` server-side |

## Persistence

The default Chroma vector store lives under `.chroma/`.

Chat uses additional persisted state:

- `.chroma/chat/<thread_id>/`: isolated Chroma store for per-thread sources
- `.chroma/chat/sessions.sqlite3`: session metadata via `SQLiteStorage`
- `.chroma/chat/checkpoints.sqlite3`: LangGraph checkpoints via `SQLiteMemorySaver`
- `.chroma/web-search/reputation.sqlite3`: rolling per-domain fetch outcomes used as a ranking prior (safe to delete; it rebuilds itself)

Chat sessions survive app restarts. Sessions with explicit URLs use isolated per-thread stores. Lightweight web chat compiles once, discovers sources inside the graph, and persists the latest URLs as session metadata without rebuilding Chroma or the graph. Checkpoints retain the complete transcript for history and restart recovery, while model calls receive only a bounded recent projection controlled by `CHAT_CONTEXT_MAX_TURNS` and `CHAT_CONTEXT_MAX_CHARS`. The `SQLiteMemorySaver` checkpointer is thread-safe under concurrent `/chat` requests for the same thread (uses a reentrant lock around the inherited `MemorySaver` mutations and the SQLite snapshot).

## File Uploads and Reading

Chat mode can read local files through five agent tools (`read_text_file`, `read_markdown_file`, `read_word_document`, `read_excel_spreadsheet`, `read_pdf`). When PowerPoint editing is enabled, the chat UI can also upload `.pptx` files for the `inspect_powerpoint` and `edit_powerpoint` tools. File uploads are gated by `FILE_READ_ENABLED` (default `false`), and `.pptx` uploads additionally require `POWERPOINT_EDIT_ENABLED=true`.

### Enabling

```text
FILE_READ_ENABLED=true
FILE_READ_ROOT=./uploads        # directory the tools and uploads are confined to
FILE_READ_MAX_BYTES=5000000     # per-file size cap
WORD_EDIT_ENABLED=true          # optional session-scoped .docx creation/editing
POWERPOINT_EDIT_ENABLED=true    # optional session-scoped .pptx inspection/editing
EXCEL_CREATE_ENABLED=true       # optional session-scoped .xlsx creation
EXCEL_NODE_EXECUTABLE=/path/to/loader/node
EXCEL_NODE_MODULES_PATH=/path/to/loader/node_modules
TEXT_EDIT_ENABLED=true          # optional session-scoped .txt creation/editing
MARKDOWN_EDIT_ENABLED=true      # optional session-scoped .md creation/editing
TYPESCRIPT_EDIT_ENABLED=true    # optional session-scoped .ts/.tsx creation/editing
JSON_EDIT_ENABLED=true          # optional session-scoped .json creation/editing
JSONL_EDIT_ENABLED=true         # optional session-scoped .jsonl creation/editing
R_EDIT_ENABLED=true             # optional session-scoped .r creation/editing
RUST_EDIT_ENABLED=true          # optional session-scoped .rs creation/editing
GO_EDIT_ENABLED=true            # optional session-scoped .go creation/editing
SQL_EDIT_ENABLED=true           # optional session-scoped .sql creation/editing
PHP_EDIT_ENABLED=true           # optional session-scoped .php creation/editing
RUBY_EDIT_ENABLED=true          # optional session-scoped .rb creation/editing
LATEX_EDIT_ENABLED=true         # optional session-scoped .tex creation/editing
PROLOG_EDIT_ENABLED=true        # optional session-scoped .pl creation/editing
HASKELL_EDIT_ENABLED=true       # optional session-scoped .hs creation/editing
LUA_EDIT_ENABLED=true           # optional session-scoped .lua creation/editing
JULIA_EDIT_ENABLED=true         # optional session-scoped .jl creation/editing
SHELL_EDIT_ENABLED=true         # optional session-scoped .sh/.bash creation/editing
MATLAB_EDIT_ENABLED=true        # optional session-scoped .m creation/editing
GROOVY_EDIT_ENABLED=true        # optional session-scoped .groovy creation/editing
SWIFT_EDIT_ENABLED=true         # optional session-scoped .swift creation/editing
LOG_EDIT_ENABLED=true           # optional session-scoped .log creation/editing
```

Set `FILE_READ_ROOT` to a dedicated directory rather than the project root so the tools and uploads are sandboxed away from source and config files. The Excel reader requires `openpyxl` (already pinned in `requirements.txt`); the creator uses the configured artifact-tool Node runtime.

### Supported types

| Tool | Extensions |
|---|---|
| `read_text_file` | `.txt`, `.md`, `.log`, `.csv` |
| `read_markdown_file` | `.md` |
| `read_word_document` | `.docx` (legacy binary `.doc` is not supported) |
| `read_excel_spreadsheet` | `.xlsx` (legacy `.xls` is not supported) |
| `read_pdf` | `.pdf` (scanned/image-only PDFs without a text layer cannot be read) |
| `inspect_powerpoint` / `edit_powerpoint` | `.pptx` (session-scoped inspection/editing when enabled) |

`read_pdf` is for local files only and refuses URLs. PDFs found on the web are handled by the web-search fetch path and the `summarize_url` tool instead, which download and extract them with the same `pypdf` backend.

### PowerPoint editing

PowerPoint editing requires both `FILE_READ_ENABLED=true` and `POWERPOINT_EDIT_ENABLED=true`. Upload a `.pptx` file, ask the agent to call `inspect_powerpoint`, and use the returned slide and shape locations when requesting an edit.

Shape locations are zero-based paths. A top-level shape is addressed as `shape_path: [0]`; a text shape inside the third top-level group is addressed as `shape_path: [2, 0]`. `inspect_powerpoint` also lists table cells with their row and column indexes.

`edit_powerpoint` supports `replace_text`, `append_text`, `delete_text`, and `replace_table_cell`. Every operation must include the current `expected_text` from inspection, which prevents edits against stale content. Text replacement preserves the target's existing first-run and paragraph styling where available. Each successful edit is written to a new downloadable `.pptx` in the current session; the uploaded source is never overwritten.

### Upload → read flow

1. In the chat UI, click the 📎 button and pick one or more files. The browser POSTs them to `POST /chat/{id}/upload` as `multipart/form-data`.
2. Each file is saved under `<FILE_READ_ROOT>/chat_uploads/<thread_id>/<name>`. The response lists saved files (with tool-ready relative paths) and per-file errors. Accepted files show a chip; rejected files show a red error chip (and the error banner when the whole upload fails).
3. On the next message turn, the server injects a `SystemMessage` listing the uploaded files' exact paths so the LLM knows what to pass to the file tools. The note is announced once per new file (tracked on the session) and is filtered out of the user-visible transcript.
4. Ask the assistant to read or summarize a supported file by name. For a `.pptx`, ask it to inspect the presentation first; when an edit is explicitly requested, it uses the inspection result to call `edit_powerpoint` with the exact path and `expected_text` values.

### Security model

- Uploads and reads require `FILE_READ_ENABLED=true`; otherwise both are refused.
- Word creation and edits additionally require `WORD_EDIT_ENABLED=true`; PowerPoint uploads and edits require `POWERPOINT_EDIT_ENABLED=true`; Excel creation requires `EXCEL_CREATE_ENABLED=true`; text creation and edits require `TEXT_EDIT_ENABLED=true`; Markdown creation and edits require `MARKDOWN_EDIT_ENABLED=true`; TypeScript creation and edits require `TYPESCRIPT_EDIT_ENABLED=true`; JSON requires `JSON_EDIT_ENABLED=true`, JSONL `JSONL_EDIT_ENABLED=true`, and R/Rust/Go/SQL `R_EDIT_ENABLED=true` / `RUST_EDIT_ENABLED=true` / `GO_EDIT_ENABLED=true` / `SQL_EDIT_ENABLED=true`; PHP `PHP_EDIT_ENABLED=true`; Ruby `RUBY_EDIT_ENABLED=true`; LaTeX `LATEX_EDIT_ENABLED=true`; Prolog `PROLOG_EDIT_ENABLED=true`; Haskell `HASKELL_EDIT_ENABLED=true`; Lua `LUA_EDIT_ENABLED=true`; Julia `JULIA_EDIT_ENABLED=true`; shell scripts `SHELL_EDIT_ENABLED=true`; MATLAB `MATLAB_EDIT_ENABLED=true`; Groovy `GROOVY_EDIT_ENABLED=true`; Swift `SWIFT_EDIT_ENABLED=true`; log files `LOG_EDIT_ENABLED=true`. All writes are confined to the current thread and never overwrite an existing file.
- Excel creation uses `@oai/artifact-tool` from the loader-provided Node runtime. Set `EXCEL_NODE_EXECUTABLE` and `EXCEL_NODE_MODULES_PATH` to those loader paths; the tool creates a task-local dependency junction, validates the exported workbook, scans formula errors, and renders every worksheet before publication.
- All paths resolve under `FILE_READ_ROOT`; `../` traversal outside the root is denied.
- Sensitive files (`.env`, private keys, `credentials`, etc.) are always refused even inside the root.
- File type is allowlisted and size is capped (`FILE_READ_MAX_BYTES`) at both the upload endpoint and the tools.
- Uploads land in per-thread directories, so threads cannot read each other's files. Deleting a thread (`DELETE /chat/{id}`) also removes its upload directory.
- The upload endpoint respects `API_KEY` auth like other mutation endpoints.

The filesystem is the source of truth for uploads, so a server restart preserves uploaded files; the next turn re-announces the full current set to the model.

## Long-term Memory

Long-term memory gives the chat agent a persistent store of facts, preferences, and tasks about the user that survive across sessions. It is completely opt-in and off by default.

### Configuration

```text
MEMORY_ENABLED=true
MEMORY_STORE_PATH=                    # blank → memory/long_term_memory.json (git-ignored)
MEMORY_DEFAULT_SCOPE=global           # "global" (all sessions) or "session" (this conversation only)
MEMORY_AUTO_RECALL_ENABLED=true       # prepend relevant memories to each chat turn
```

When enabled, three agent-callable tools appear:

| Tool | Purpose |
|---|---|
| `save_memory` | Store a durable fact, preference, or task with optional category, tags, and scope |
| `recall_memory` | Look up remembered content by keyword before answering a question about the user |
| `forget_memory` | Delete a memory by id or by keyword when the user asks to forget something |

### Automatic extraction (self-updating memory)

When `MEMORY_EXTRACTION_ENABLED=true`, the agent distils durable facts from finished sessions and from every `MEMORY_EXTRACTION_TURN_INTERVAL` turns without being asked. This costs one extra LLM call per extraction round.

```text
MEMORY_EXTRACTION_ENABLED=true
MEMORY_EXTRACTION_TURN_INTERVAL=10
MEMORY_EXTRACTION_MAX_CANDIDATES=5
MEMORY_EXTRACTION_MAX_TRANSCRIPT_CHARS=8000
MEMORY_EXTRACTION_TIMEOUT_SECONDS=60
MEMORY_EXTRACTION_MAX_CONURRENCY=2
MEMORY_EXTRACTION_MAX_SESSION_AGE_HOURS=168
```

Each extraction round runs in parallel (bounded by `MEMORY_EXTRACTION_MAX_CONCURRENCY`) and is capped in parallelism; excess requests are dropped rather than queued. Extracted content passes through a watermark so the model can distinguish recalled material from user input.

### Scope and eviction

Memories carry a scope (`global` or `session`). Global memories are shared across all threads and persist to the store file; session-scoped memories are deleted when their thread is deleted. The store keeps at most `MEMORY_MAX_RECORDS` records total — the oldest by last-recall time are evicted first once the cap is reached.

### Security model

- Memory is entirely off when `MEMORY_ENABLED=false`; no memory file is read or created.
- The store file path resolves relative to the working directory; a `..` segment is refused.
- The store file is git-ignored by default.
- Tool results that indicate a failure are prefixed with `MEMORY_ERROR:` so the agent loop continues rather than crashing.
- Per-turn call budgets prevent runaway tool loops.

## Web Search

Live web search supports key-backed APIs and HTML fallbacks in `src/backend/web_search/`:

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
- **Structural page filtering** (`WEB_SEARCH_STRUCTURE_FILTER_ENABLED`, on by default) measures the page that was actually fetched — anchor-text share, anchors per 100 words, and content volume — and removes index/tag listings, login and enable-JavaScript shells, and thin SEO pages. Unmeasurable pages abstain instead of being rejected. This is what replaces URL-pattern guessing; see `src/backend/web_search/page_structure.py`.
- **Optional semantic relevance** (`WEB_SEARCH_SEMANTIC_FILTER_ENABLED`, off by default) adds cosine similarity from a local sentence-transformers model. It can only add recall: a bounded bonus for strong matches, a rescue at the gate floor for results that lexical scoring filtered out, and a second chance for pages the lexical page gate rejected. It never lowers a lexical score or bypasses the year, quantity, and typed-evidence gates. First use downloads the model.
- **Adaptive domain reputation** (`WEB_SEARCH_DOMAIN_REPUTATION_ENABLED`, on by default) records per-domain fetch outcomes (grounded, rejected, unreachable) in `CHROMA_DIR/web-search/reputation.sqlite3` and feeds a bounded ranking prior back into merge. It stays neutral until a domain reaches `WEB_SEARCH_DOMAIN_REPUTATION_MIN_SAMPLES`, and never rejects a URL on its own.
- **Provider-result dedup** by canonical host/path, plus per-domain diversity in merge.
- **Document fetching** loads HTML concurrently and extracts PDF sources with `pypdf` (bounded to 30 pages / 20 MB), so official notices and vendor whitepapers stay eligible. PDFs are recognized both by a `.pdf` path and by the `%PDF-` response header, which covers extension-less endpoints like `arxiv.org/pdf/1706.03762`. Scanned PDFs without a text layer fail like any other unreadable page.
- **Optional JS-capable fallback** (`WEB_SEARCH_JS_FALLBACK_ENABLED`) retries through a lazy headless Chromium adapter whenever an HTTP fetch returned no readable text or a login/enable-JavaScript shell — the trigger is the measurement, not a domain list. It is bounded by `WEB_SEARCH_JS_RETRY_BUDGET`, and the configured domains (`baike.baidu.com`, `zhuanlan.zhihu.com`, `apps.microsoft.com`, `deepseek.net` by default) only get priority inside that budget. Off by default; requires `playwright` plus a Chromium runtime.
- **Pre-index document filtering** (full graph only) drops short/empty/boilerplate/low-signal pages, with an optional embedding similarity gate against the question (`document_quality_relevance_query`) and configurable recency bias from extracted publication dates.
- **Post-retrieval re-ranking** (full graph only) scores chunks by query/document overlap, frequency, and phrase matches; `RERANK_STRATEGY` switches between lexical (default), embedding, or hybrid.

When `web_search_lightweight` is enabled (default), the lightweight graph routes the agent's tool call through `decompose → search_queries → merge → web_answer` as described above, with one-shot conditional expansion and a grounded refusal when no evidence survives. Chat uses this as the sole web-search owner; it does not perform a preliminary provider search or recompile the graph for each turn.

## Answer citations

Sources are cited by URL. The prompts state this explicitly and forbid invented reference markers, because the source blocks handed to the model carry no numbers, IDs, or line numbers.

Models trained on transcripts from other tool-augmented assistants sometimes reproduce those assistants' internal citation syntax anyway — for example `【199†L91-L126】` (a source index, a dagger, and a line range) or `[oaicite:0]`. Those markers reference nothing in this project, cannot be verified, and signal that the model is improvising attribution.

`src/backend/llm/sanitize.py` removes them from user-facing text in two places:

- `strip_citation_artifacts` runs on the final answer in `web_answer`, `generate`, `fallback_answer`, and on direct agent replies (tool-call carriers are left untouched).
- `CitationArtifactFilter` does the same for SSE token streams, holding back any tail that could still become a marker and flushing it when the marker completes or is ruled out.

URLs, markdown links, ordinary brackets like `[1]` or `[sic]`, and a lone `†` are all preserved.

## Streaming

The chat app exposes an SSE streaming endpoint:

- **Chat `/chat/{id}/message/stream`**: node lifecycle events (`node_start`, `node_end`, retriever/grader summaries), and — by default — per-token `token` events streamed from the answer-producing nodes (`generate`, `web_answer`, `agent`) as the LLM generates them, followed by a final `done` event. Pass `?tokens=false` to fall back to node-events-only streaming.

Token streaming uses LangGraph's combined `stream_mode=["updates", "messages"]`. Only genuine streaming chunks (`AIMessageChunk`) are forwarded; the aggregated final message a node returns is dropped so the answer is not duplicated. Tokens from internal structured-output calls (decompose, expand, grade, condense, rewrite) are filtered out so they never leak into the user-visible answer.

Streamed tokens also pass through `CitationArtifactFilter` (`src/backend/llm/sanitize.py`), which buffers partial text so a fabricated citation marker split across chunks is still removed. See [Answer citations](#answer-citations).

The browser chat UI consumes the token stream and renders the answer incrementally. The terminal REPL (`python -m src.frontend.chat.main chat`) also streams tokens to stdout as they arrive. Both fall back to the final `done` answer when a provider does not emit token chunks.

## Auth, CORS, and Security

Local development is open when `API_KEY` is unset. When set, mutation endpoints require:

```text
Authorization: Bearer <API_KEY>
```

Protected endpoints: `POST /chat`, `POST /chat/{id}/message`, `POST /chat/{id}/message/stream`, `DELETE /chat/{id}`, and `GET /admin/health/dependencies`. The administrative health route is hidden with `404` when no API key is configured.

CORS is configured via `cors_allow_origins` in YAML or `CORS_ALLOW_ORIGINS` env var. Public liveness and readiness expose only bounded status; dependency names and states are available only through the authenticated administrative route.

## Docker

```powershell
$env:DASHSCOPE_API_KEY = "your_key"
$env:API_KEY = "optional_key"
docker compose up --build
```

Starts Chat on `http://127.0.0.1:8001` with named volumes for `.chroma` and session data.

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
- Key-backed search APIs are preferred for Mandarin when configured. Otherwise search uses the bounded Bing/Baidu/DuckDuckGo HTML fallbacks. CAPTCHA and repeated network failures open temporary provider circuits instead of blocking every expanded query.
- Noise filtering is measured rather than pattern-matched: structural page assessment and the learned domain prior are on by default, and semantic similarity is available as an opt-in recall layer. The tradeoff is that structural filtering needs a fetch first, so a noisy URL still costs one concurrent request.
- The agent system prompt (`AGENT_SYSTEM_PROMPT` in `src/backend/llm/prompts.py`) tells the model to answer directly when tools aren't needed — covering math, general knowledge, programming concepts, definitions, well-established stable facts (founding dates, capitals, public figures), and chitchat — so the graph avoids unnecessary retrieval/rewrite cycles.
- Reranking (`RERANK_STRATEGY`) defaults to lexical (keyword-based); `embedding` uses cosine similarity against embedding vectors; `hybrid` combines both.
- Location questions route to `find_on_map` before `live_web_search`. Place lookup goes through the shared geocoder in `src/backend/tools/_geocoding.py`: request wording is stripped ("在地图上找出上海的位置" → "上海"), then AMap resolves the place in three fallback stages — POI text search first, address geocoding if that yields no confident match, administrative district lookup last — and the candidates are deduped and ranked by match score. Coordinates come back in GCJ-02, which the map artifact renders directly. Requires `AMAP_WEB_SERVICE_KEY`; with no key configured the geocoder returns no candidates. Since text-similarity geocoding can return a neighbouring or same-named place, results below the confidence threshold are labelled `APPROXIMATE MATCH` and same-name ties are labelled `AMBIGUOUS`; the agent is instructed to verify those with a web search rather than assert them.
- Optional agent tools (`weather`, `stock`, `currency`, `wikipedia`, `directions`, `map`, `math`, `statistics`, `linalg`, `number_theory`, `datetime`, `summarize_url`, and the file readers) are off by default. Enable them via the per-tool `_ENABLED` flag in `.env` (the five file readers — .txt, .md, .docx, .xlsx, .pdf — share `FILE_READ_ENABLED`). Most need no API key; the exceptions are `get_directions` and `find_on_map`, which require `AMAP_WEB_SERVICE_KEY`. `WIKIPEDIA_USER_AGENT` should be customized for shared deployments, and the file tools should have `FILE_READ_ROOT` pointed at a dedicated directory.
- The lightweight graph's conditional expansion fires only on web-search retrieval failure — single-keyword questions that get a readable, relevant page back take the fast path with one search and one LLM call. Compound questions that decompose into multiple sub-Qs still take the fast path; expansion only fires when no fetched page yields usable evidence.
