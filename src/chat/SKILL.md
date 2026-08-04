---
name: chat-app-architect
description: >
  Use this skill whenever working on the multi-turn chat application — the
  FastAPI app at port 8001, the chat REPL CLI, the static chat UI under
  src/chat/static/, the /chat endpoints, the session registry, the chat
  graph wrapper that adds CONDENSE and MemorySaver checkpointing, or the
  graph-owned web-search path. Trigger on mentions of chat.api, chat.main,
  /chat, port 8001, multi-turn, session, MemorySaver, condense, or thread_id.
---

# Chat App Architect — src/chat/

Domain: multi-turn conversation interface with per-thread memory, graph-owned
web discovery, and SQLite-backed checkpoint persistence.
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
| Memory extraction | `memory_hooks.on_session_start` + `memory_hooks.after_turn` |
| Session metadata | Optional `SQLiteStorage` under `.chroma/chat/sessions.sqlite3` |
| Source persistence | Explicit-source Chroma under `.chroma/chat/<thread_id>/`; lightweight web URLs in session metadata |

## File Map

```
src/chat/
├── __init__.py     # Package docstring (deprecated re-export shim notice)
├── api.py          # FastAPI app: create_app() + endpoints + lifespan + session restore
├── main.py         # CLI: serve + graph-owned lightweight chat REPL
├── memory_hooks.py # Extraction runtime, watermark adapter, and trigger hooks
├── graph.py        # build_chat_graph wrapper around src/graph/builder
├── nodes.py        # Re-exports condense/agent/grade/rewrite/generate from src/graph/nodes
├── sessions.py     # Re-exports ChatSession/ChatSessionRegistry from src/sessions
├── state.py        # Re-exports ChatState from src/graph/state
└── static/
    ├── index.html  # Chat-style transcript with composer + start screen
    └── script.js   # localStorage-backed thread restoration + streaming-friendly client
```

`memory_hooks.py` owns the automatic extraction runtime, persisted watermark
adapter, previous-session trigger, and completed-round trigger.

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
| Sources | Per-request, can change every call | Explicit URLs stay fixed; lightweight web discovery runs inside the compiled graph |
| Multi-turn rewrite | None | `condense_question` runs only for context-dependent follow-ups |
| Endpoint shape | `POST /query` (single shot) | `POST /chat` (start) + `POST /chat/{tid}/message` (turn) + `GET /chat/{tid}/history` (transcript) |
| State key | None | `thread_id` (UUID hex; passed in `config={"configurable": {"thread_id": ...}}` to LangGraph) |
| Lifespan | Builds default graph once | Loads settings + session registry + checkpointer; restores persisted sessions |
| Default port | 8000 | 8001 |

## Session Lifecycle

```
POST /chat
  ↓ resolve mode: explicit URLs | graph-owned lightweight web | defaults
  ↓ build per-thread Settings (isolated Chroma dir if not "defaults")
  ↓ build graph (lightweight if web_search + lightweight enabled, else heavy with retriever)
  ↓ register ChatSession in ChatSessionRegistry
  ↓ return thread_id

POST /chat/{tid}/message
  ↓ look up session (404 if missing)
  ↓ clear prior-turn lightweight search state
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

Session creation calls `on_session_start` after registry metadata is persisted
and schedules at most one eligible previous session. Completed non-streaming,
streaming, and CLI turns call `after_turn` only after the graph checkpoint is
available; due work is handed to the background scheduler before the reply path
returns.

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

## Graph-Owned Web Search

When `web_search_lightweight=True`, chat compiles the lightweight graph once.
The agent decides whether a turn needs live information; a web tool call flows
through conditional decomposition, bounded parallel search, merge, fetch, and
answer nodes. The API clears prior-turn search state before invocation and
persists the resulting URLs as session metadata afterward without rebuilding
the graph. The heavyweight compatibility path still discovers sources before
building Chroma, but disables the graph's live-search tool to avoid a duplicate
provider call.

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

When automatic memory extraction is enabled, lifespan creates one
`ExtractionRuntime` from the shared checkpoint saver, registry, and metadata
storage. Both trigger hooks are best-effort. The daemon-thread scheduler is
shut down without joining when the app or REPL exits, so extraction cannot hold
up a response or process shutdown.

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
| 3 | Per-turn preliminary search and graph rebuild added latency to every message | Fixed | `api.py` | Lightweight chat now owns search inside one compiled graph |
| 4 | Full transcripts were sent to the model on every turn | Fixed | graph nodes | Bound the model context while retaining persisted history |
| 5 | Streaming endpoint exists but the static UI doesn't use it | Low | `static/script.js` | Switch to SSE for progressive token rendering |
| 6 | Standalone follow-ups unnecessarily invoked the condensation LLM | Fixed | condense node | Condense only when contextual signals are present |

## Refactoring To-Do List

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
- `src/memory/` provides `MemoryExtractor`, `ExtractionScheduler`, transcript slicing, and watermark helpers.
- External: `fastapi`, `uvicorn`, `langchain_core.messages.HumanMessage`.
