---
name: qa-app-architect
description: >
  Use this skill whenever working on the single-shot Q&A application — the
  FastAPI app at port 8000, the CLI commands query/serve/rebuild, the static
  web UI under src/qa/static/, the /query endpoint, or the legacy
  positional-question CLI compatibility layer. Trigger on mentions of qa.api,
  qa.main, /query, port 8000, single-shot, or the QA static UI.
---

# QA App Architect — src/qa/

Domain: the single-shot question-answer interface — one question in, one
answer out, no memory between calls.
Parent: `SKILL.md` (root). Siblings: `src/chat/SKILL.md`, `src/api/SKILL.md`,
`src/graph/SKILL.md`.

## Quick Reference

| Fact | Value |
|---|---|
| Module path | `src/qa/` |
| CLI entry | `python -m src.qa.main {query,serve,rebuild}` |
| Default port | 8000 |
| Default host | 127.0.0.1 (overridable via `--host` or `API_HOST`) |
| HTTP endpoints | `GET /`, `GET /health`, `GET /ready`, `POST /query`, `POST /query/stream`, `GET /metrics` |
| Static UI | `src/qa/static/index.html` + `script.js` (served at `/` and `/static/*`) |
| Memory | None — each request rebuilds state |
| Source URLs | Per-request: explicit `urls`, web-searched, or configured defaults |

## File Map

```
src/qa/
├── __init__.py        # Package docstring
├── api.py             # FastAPI app: create_app() + endpoints + lifespan
├── main.py            # CLI entry: query/serve/rebuild + backward-compat positional path
├── draw_graph.py      # Utility: render the QA LangGraph to a PNG via Mermaid
└── static/
    ├── index.html     # Single-page form (question + sources + checkboxes)
    └── script.js      # Form submission + answer rendering, source-mode hint
```

## Two Entry Surfaces

### CLI

```bash
python -m src.qa.main query "What is fine-tuning?"          # Single question
python -m src.qa.main query "..." --urls "https://a,https://b"
python -m src.qa.main query "..." --rebuild                 # Rebuild Chroma first
python -m src.qa.main query "..." --no-web-search           # Skip web discovery

python -m src.qa.main serve                                 # FastAPI on :8000
python -m src.qa.main serve --port 5000 --host 0.0.0.0
python -m src.qa.main serve --reload                        # Dev hot-reload
python -m src.qa.main serve --rebuild                       # Rebuild on startup

python -m src.qa.main --rebuild                             # Rebuild only, then exit
```

The CLI also accepts a **legacy positional question** (`python -m src.qa.main "question"`) for backward compatibility with the pre-subcommand interface. `parse_args` reconstructs a `query` namespace when it sees an unknown positional argument.

### HTTP

```
GET  /                  → serves static/index.html
GET  /static/script.js  → static asset
GET  /health            → liveness probe
GET  /ready             → readiness with local checks
GET  /metrics           → in-process GraphExecutor metrics snapshot
POST /query             → one question, one answer (JSON)
POST /query/stream      → SSE stream of typed graph events + final answer
```

### `/query` Contract

```json
{
  "question": "...",                 // required, min length 1
  "urls": null,                       // null | string (CSV) | list[string]
  "web_search": true,                 // discover URLs when urls is empty
  "rebuild": false,                   // rebuild Chroma before answering
  "debug": false                      // include intermediate messages in response
}
```

Response includes the answer, list of source URLs used, source mode
(`explicit | web_search | defaults`), and an optional source note when web
search fell back to defaults.

## Source Resolution per Request

```
request.urls supplied?
  ├── yes  → source_mode = "explicit"
  └── no   → web search enabled and allowed?
              ├── yes → discover_urls_from_web
              │          ├── results found → source_mode = "web_search"
              │          └── empty/failed → source_mode = "defaults", source_note set
              └── no  → source_mode = "defaults"
```

When sources change between requests, the QA app's `/query` handler rebuilds
the graph (with the new retriever) under an `asyncio.Lock` so concurrent
requests don't trample each other's Chroma directories.

## Lifespan

`create_app(rebuild_db, api_host, api_port)` registers a lifespan context that:

1. Loads `Settings` (via `src/config/load_settings`).
2. Builds the default graph (with `rebuild_vectorstore=rebuild_db`).
3. Stores `graph`, `settings`, and `rebuild_lock` on `app.state`.
4. Yields to serve traffic.

A failed rebuild during `/query` restores the previous graph (regression fixed
earlier in the project — see `memory/refactor-daily-forms.md`).

## Web UI

Single-page form:

- Textarea for the question.
- Optional comma-separated source URLs.
- Web-search checkbox (enabled by default).
- Rebuild checkbox (off by default).
- Debug checkbox.

The page sends a `POST /query`, renders the answer in a result panel, and
shows a one-line source-mode hint (e.g., "Sources: Web search · 6 URLs").
All model output is rendered via `escapeHtml` then `innerHTML`, so HTML/JS
in the answer is rendered as text, not executed.

## Lightweight vs Heavy Web-Search Path

When `web_search_lightweight=True` (default) and sources came from web search,
the CLI uses `build_lightweight_graph` instead of `build_graph`. The
lightweight graph:

- Skips Chroma embedding and retrieval.
- Fetches each URL, extracts readable text, builds a direct-answer prompt.
- Sends a single LLM call with all fetched content as context.

Trade-off: faster for one-shot questions over a small fixed source set, but
no semantic search — the model sees raw page text. Heavy path is preferred
when sources are stable across many questions (the persistent Chroma index
amortizes embedding cost).

## Known Issues

| # | Issue | Severity | Location | Fix |
|---|---|---|---|---|
| 1 | Legacy positional CLI path was previously dead due to argparse subparser rejection | Fixed | `main.py` parse_args | Reconstructs args from `parse_known_args` unknowns |
| 2 | A single `/query` failure during rebuild used to leave `app.state.graph = None` permanently | Fixed | `api.py` /query | Snapshots previous graph and restores on failure |
| 3 | `request.urls.strip()` crashed when a list was sent (schema allows both) | Fixed | `api.py` /query | Uses already-parsed `urls` list for explicit-mode detection |
| 4 | The lightweight path's `web_answer` used to prefer redundant in-graph tool URLs over the curated session URLs | Fixed | (cross-cutting) | Priority reordered in `src/graph/nodes/web_answer.py` |
| 5 | UI escapes HTML on output but accepts arbitrary URLs as input without scheme validation | Low | `api.py` and `script.js` | `parse_url_input` doesn't validate schemes; `web_search.common.normalize_urls` filters non-HTTP later in the pipeline |
| 6 | No streaming-aware UI for `/query/stream` yet | Low | `static/script.js` | SSE endpoint exists but the UI uses `POST /query` (non-streaming) |

## Refactoring To-Do List

- [ ] **Remove legacy positional CLI path** in a major bump — clean break to `python -m src.qa.main query "..."`.
- [ ] **Switch UI to `/query/stream`** with progressive token rendering.
- [ ] **Surface tool-call traces** in the debug panel (currently the UI only shows the final answer + sources).
- [ ] **Add a UI affordance for the lightweight vs heavy path** so users can see which mode answered.

## Testing Strategy

| Test | Approach |
|---|---|
| `parse_args` modes | `query --rebuild`, `serve --port`, `--rebuild` only, legacy positional |
| Lifespan | `app.router.lifespan_context(app)` populates `app.state.graph`/`settings` |
| `/query` happy path | TestClient with stubbed graph; assert answer and `source_mode=explicit` |
| `/query` rebuild failure | Force `build_graph` to raise; assert previous graph restored |
| `/query` with list URLs | Send `{"urls": ["https://a"]}`; assert no AttributeError |
| Static dir | `src/qa/static/index.html` and `script.js` resolve via `Path(__file__).parent / "static"` |
| Tests live in | `tests/test_api_*.py`, `tests/test_graph_builder.py` |

## Dependencies

- `src/config/Settings` and `load_settings`.
- `src/graph/build_graph`, `src/graph/build_lightweight_graph`, `src/graph/executor.GraphExecutor`.
- `src/api/*` — shared FastAPI dependency injection, error handlers, SSE streaming, request/response models.
- `src/core/graph_executor.run_rag_query` — orchestrates the legacy/non-streaming flow.
- External: `fastapi`, `uvicorn`.
