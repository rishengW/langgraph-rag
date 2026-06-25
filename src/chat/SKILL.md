---
name: chat-app-architect
description: >
  Use this skill whenever working on the multi-turn chat application — the
  FastAPI app at port 8001, the chat REPL CLI, the static chat UI under
  src/chat/static/, the /chat endpoints, the session registry, the chat
  graph wrapper that adds CONDENSE and MemorySaver checkpointing, or the
  per-turn web-search refresh. Trigger on mentions of chat.api, chat.main,
  /chat, port 8001, multi-turn, session, MemorySaver, condense, or thread_id.
---

# Chat App Architect — src/chat/

Domain: multi-turn conversation interface with per-thread memory, optional
per-turn source refresh, and SQLite-backed checkpoint persistence.
Parent: `SKILL.md` (root). Siblings: `src/qa/SKILL.md`, `src/api/SKILL.md`,
`src/graph/SKILL.md`, `src/sessions/SKILL.md`.

## Quick Reference

| Fact | Value |
|---|---|
| Module path | `src/chat/` |
| CLI entry | `python -m src.chat.main {serve,chat}` |
| Default port | 8001 |
| Default host | 127.0.0.1 (overridable via `--host` or `CHAT_API_HOST`) |
| HTTP endpoints | `POST /chat`, `POST /chat/{thread_id}/message`, `POST /chat/{thread_id}/message/stream`, `GET /chat/{thread_id}/history`, `DELETE /chat/{thread_id}`, `GET /health`, `GET /ready`, `GET /metrics` |
| Static UI | `src/chat/static/index.html` + `script.js` (chat-style transcript) |
| Memory | LangGraph `MemorySaver` per thread, optionally backed by `SQLiteMemorySaver` |
| Session metadata | Optional `SQLiteStorage` under `.chroma/chat/sessions.sqlite3` |
| Source persistence | Per-thread Chroma at `.chroma/chat/<thread_id>/` for isolated sessions; shared store for "defaults" sessions |

## File Map

```
src/chat/
├── __init__.py     # Package docstring (deprecated re-export shim notice)
├── api.py          # FastAPI app: create_app() + endpoints + lifespan + session restore
├── main.py        # CLI: serve + chat REPL with per-turn web-search refresh
├── graph.py        # build_chat_graph wrapper around src/graph/builder
├── nodes.py        # Re-exports condense/agent/grade/rewrite/generate from src/graph/nodes
├── sessions.py     # Re-exports ChatSession/ChatSessionRegistry from src/sessions
├── state.py        # Re-exports ChatState from src/graph/state
└── static/
    ├── index.html  # Chat-style transcript with composer + start screen
    └── script.js   # localStorage-backed thread restoration + streaming-friendly client
```

> Several files here are deprecation shims that re-export from canonical
> modules (`src/graph/`, `src/sessions/`, `src/api/`). Emit
> `DeprecationWarning` when imported. Treat the canonical modules as the
> source of truth.

## Two Entry Surfaces

### CLI

```bash
python -m src.chat.main serve                              # FastAPI on :8001
python -m src.chat.main serve --port 9000 --reload
python -m src.chat.main chat                               # Terminal REPL
python -m src.chat.main chat --urls "https://a,https://b"  # Skip web search
python -m src.chat.main chat --seed-question "PAI fine-tuning"  # Seed search
```

### HTTP

```
POST   /chat                              → start a thread, returns thread_id + source info
POST   /chat/{tid}/message                → user turn, returns assistant reply
POST   /chat/{tid}/message/stream         → SSE: token stream + typed graph events
GET    /chat/{tid}/history                → full transcript for UI reload
DELETE /chat/{tid}                        → drop session and Chroma cleanup
GET    /health                            → liveness + session count
GET    /ready                             → local-only readiness check
GET    /metrics                           → in-process executor metrics
```

## Differences from `src/qa/`

| Concern | QA (`src/qa/`) | Chat (`src/chat/`) |
|---|---|---|
| Memory | None | LangGraph `MemorySaver` per `thread_id`; optional SQLite persistence |
| Sources | Per-request, can change every call | Fixed-at-start by default; per-turn refresh when sources came from web search and source_mode != "explicit" |
| Multi-turn rewrite | None | `condense_question` node runs at the start of every turn |
| Endpoint shape | `POST /query` (single shot) | `POST /chat` (start) + `POST /chat/{tid}/message` (turn) + `GET /chat/{tid}/history` (transcript) |
| State key | None | `thread_id` (UUID hex; passed in `config={"configurable": {"thread_id": ...}}` to LangGraph) |
| Lifespan | Builds default graph once | Loads settings + session registry + checkpointer; restores persisted sessions |
| Default port | 8000 | 8001 |

## Session Lifecycle

```
POST /chat
  ↓ resolve sources: explicit URLs | web search | defaults
  ↓ build per-thread Settings (isolated Chroma dir if not "defaults")
  ↓ build graph (lightweight if web_search + lightweight enabled, else heavy with retriever)
  ↓ register ChatSession in ChatSessionRegistry
  ↓ return thread_id

POST /chat/{tid}/message
  ↓ look up session (404 if missing)
  ↓ if source_mode != "explicit" and web_search_enabled: refresh sources from web search
  ↓ snapshot prev message count (for current-turn answer isolation)
  ↓ graph.invoke({"messages": [HumanMessage(message)]}, {"configurable": {"thread_id": tid}})
  ↓ scan new messages for the last AI message with non-empty content (skip tool-call carriers)
  ↓ return answer

GET /chat/{tid}/history
  ↓ graph.get_state(config) → snapshot.values["messages"]
  ↓ serialize: user/assistant turns only, no tool messages

DELETE /chat/{tid}
  ↓ registry.delete(tid)  (also removes the per-thread Chroma dir for isolated sessions)
```

## Chat-Specific Graph

`build_chat_graph(settings, rebuild_vectorstore, checkpointer)` calls
`src/graph/builder.build_graph(mode="chat", ...)`. The chat graph adds two
things over the QA graph:

1. **`condense_question` node** runs first, rewriting the latest user turn
   into a standalone question using prior history. Sets
   `state["current_question"]`. Without this, the retriever embedding
   would lose conversational context ("what about that?" becomes a noisy
   vector lookup).
2. **`MemorySaver` checkpointer** attached at compile time. Conversation
   state is keyed by `thread_id` in `config["configurable"]["thread_id"]`.

## Per-Turn Web Search Refresh

The chat app re-runs web search on **every user turn** (when the session
wasn't started from explicit URLs and `web_search_enabled=True`). If the new
results differ from the current session URLs, the graph is rebuilt with the
new sources before invoking. This keeps multi-turn chats current as the
topic shifts.

- Skipped when `source_mode == "explicit"` (user pinned the sources).
- Failures are logged and the previous sources are kept (no hard fail).
- Rebuilding the graph is locked via `graph_factory_lock` from `src/api/dependencies.py`.

## Session Persistence

Lifespan creates two SQLite stores under `.chroma/chat/`:

| Store | Path | Contents |
|---|---|---|
| `SQLiteStorage` | `.chroma/chat/sessions.sqlite3` | `SessionMetadata`: thread_id, source URLs, source mode, isolated_chroma flag, created_at, schema_version |
| `SQLiteMemorySaver` | `.chroma/chat/checkpoints.sqlite3` | LangGraph checkpoint blobs (full conversation state) |

On startup, lifespan calls `_restore_persisted_sessions` which:

1. Walks `SQLiteStorage.list_metadata()`.
2. For each, builds the per-session Settings + graph (lightweight or heavy).
3. Calls `registry.restore(...)` to wire the graph back into the active registry.

This means chat survives server restart: open `/chat/{tid}/history` for an
old thread and the transcript comes back.

## Web UI

- **Start screen**: textarea for first question + optional URLs + web-search checkbox + "Start chat" button.
- **Transcript view**: scrolling user/assistant bubbles, sticky composer at the bottom, "New chat" button in the header.
- **localStorage**: `thread_id` persists across page reloads; `script.js` calls `GET /chat/{tid}/history` on `DOMContentLoaded` to restore the conversation.
- **Optimistic UX**: "Thinking..." placeholder while waiting on the LLM; replaced when the response arrives. Errors render in a red banner without dropping the transcript.

## Known Issues

| # | Issue | Severity | Location | Fix |
|---|---|---|---|---|
| 1 | Chat 404 used to be shadowed by 503 when config/lock dependencies were resolved before the session lookup | Fixed | `api.py` post_message | Look up the session first, resolve config after |
| 2 | `SQLiteMemorySaver` failed to restore on newer LangGraph (no `blobs` attribute) | Fixed | `src/sessions/checkpoint.py` | Tolerate variants that don't expose `blobs` |
| 3 | Per-turn web search adds latency to every message; not skippable per-turn from the UI | Medium | `api.py` _refresh_session_sources_from_web | Surface a per-turn `refresh=false` opt-out |
| 4 | Session memory is unbounded; chats with hundreds of turns grow the `MemorySaver` per-thread state | Medium | LangGraph internal | Window the message history before invoke; or use LangGraph's planned summarization checkpoint |
| 5 | Streaming endpoint exists but the static UI doesn't use it | Low | `static/script.js` | Switch to SSE for progressive token rendering |
| 6 | The session refresh happens unconditionally on every turn for web-search sessions, even if the question is a clarification on prior content | Medium | api.py | Add a heuristic (or LLM-judged) refresh trigger based on whether the question references current information |

## Refactoring To-Do List

- [ ] **Window message history** before `graph.invoke` so long sessions don't OOM the checkpointer.
- [ ] **Per-turn refresh opt-out** in the request body and UI.
- [ ] **SSE streaming UI** — switch `script.js` to consume `/chat/{tid}/message/stream`.
- [ ] **Session export endpoint** — `GET /chat/{tid}/export` returns the transcript as Markdown or JSON.
- [ ] **TTL eviction** — purge sessions idle for longer than `chat_session_ttl_seconds` (already tracked in `src/sessions/`).
- [ ] **Remove deprecation shims** in `src/chat/{state,nodes,sessions,graph}.py` once consumers migrate to canonical paths.

## Testing Strategy

| Test | Approach |
|---|---|
| Lifespan: settings + registry + restored sessions | Stub `SQLiteStorage.list_metadata()`; assert registry contains the persisted thread_id |
| `POST /chat` with explicit URLs | Assert source_mode == "explicit", isolated Chroma path created |
| `POST /chat` with web_search → fallback to defaults | Stub `discover_urls_from_web` to raise; assert source_mode == "defaults" + source_note == "web_search_failed" |
| `POST /chat/{tid}/message` happy path | Stub graph.invoke; assert assistant reply returned, tool-call carriers skipped |
| Unknown thread → typed 404 | `POST /chat/missing/message` returns `ResourceNotFoundError` not 503 |
| `GET /history` after multiple turns | Round-trip user/assistant turns from `graph.get_state` |
| Per-turn refresh skipped for explicit sources | `source_mode == "explicit"`; assert no `discover_urls_from_web` call |
| Tests live in | `tests/test_api_errors.py`, `tests/test_api_streaming.py`, `tests/test_sessions.py` |

## Dependencies

- `src/config/Settings` and `load_settings`.
- `src/graph/builder` — `build_graph(mode="chat")`, `build_lightweight_graph(mode="chat")`, `build_memory_saver`.
- `src/sessions/` — `ChatSession`, `ChatSessionRegistry`, `SQLiteStorage`, `SQLiteMemorySaver`, `_settings_for_session`.
- `src/api/` — auth, CORS, error handlers, streaming, dependencies, request/response models.
- `src/core/web_search` (deprecated re-export) → `src/web_search.discover_urls_from_web`.
- External: `fastapi`, `uvicorn`, `langchain_core.messages.HumanMessage`.
