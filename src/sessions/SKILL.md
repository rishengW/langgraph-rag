---
name: session-engine-architect
description: >
  Use this skill whenever working on multi-turn conversation management — session
  lifecycle, condense question logic, conversation history, MemorySaver
  checkpointer, ChatSessionRegistry, TTL cleanup, Chroma isolation per session,
  or SQLite persistence. Covers the session state machine from creation through
  active use to expiration. Trigger on mentions of sessions, thread_id, condense,
  MemorySaver, ChatSession, ChatSessionRegistry, conversation history,
  multi-turn, TTL, or session persistence.
---

# Session Engine Architect — src/sessions/

Domain: multi-turn conversation sessions, condense question logic, checkpointing, session registry, persistence.
Parent: [[system-architect]]. Siblings: [[rag-pipeline-architect]], [[knowledge-retrieval-architect]], [[web-search-architect]], [[api-interface-architect]].

## Quick Reference

| Fact | Value |
|---|---|
| Session storage | `ChatSessionRegistry` — in-memory, thread-safe |
| Checkpointer | `MemorySaver` (langgraph.checkpoint.memory) |
| Extra graph node | `condense` (standalone question rewrite from conversation history) |
| State fields added | `current_question`, `current_question_index` |
| Chroma isolation | Per-session: `.chroma/chat/<thread_id>/` |
| TTL cleanup | None currently (sessions never expire) |
| Persistence | In-memory only (lost on server restart) |
| Source files (current) | `chat/sessions.py`, `chat/graph.py`, `chat/nodes.py` (condense), `chat/state.py` |
| Target location | `src/sessions/` |

## File Map (Current)

```
src/chat/
+-- sessions.py             # ChatSession + ChatSessionRegistry — 94 LOC
+-- graph.py                # build_chat_graph() — condense + MemorySaver — 93 LOC
+-- nodes.py                # condense_question_factory (plus 4 duplicated from core/) — 348 LOC
+-- state.py                # ChatState (Chat-specific fields) — 33 LOC
```

### Detailed File Responsibilities

| File | Responsibility | Key Symbols | LOC |
|------|---------------|-------------|-----|
| `sessions.py` | `ChatSession` dataclass, `ChatSessionRegistry` thread-safe in-memory map | `ChatSession`, `ChatSessionRegistry` | 94 |
| `graph.py` | Chat graph with `condense` + `MemorySaver` checkpointer | `build_chat_graph`, `_build_memory_saver` | 93 |
| `nodes.py` | `condense_question_factory` + 4 duplicated node factories | `condense_question_factory`, `_format_history`, `_latest_user_index` | 348 |
| `state.py` | `ChatState` (adds `current_question`, `current_question_index`) | `ChatState` | 33 |

## Session Lifecycle

```
CREATE ──► ACTIVE ──► (TTL expired) ──► CLEANUP
  │           │
  │           ├── message → graph.invoke() → answer
  │           ├── message → graph.invoke() → answer
  │           └── ... (multi-turn with MemorySaver checkpointing)
  │
  DELETE ──► Remove from registry + release Chroma resources
```

### ChatSession Dataclass

```python
@dataclass
class ChatSession:
    thread_id: str
    graph: Any                     # Compiled LangGraph with MemorySaver
    settings: Settings             # Per-session settings (may have isolated Chroma dir)
    source_urls: list[str]
    source_mode: str               # "explicit" | "web_search" | "defaults"
    created_at: float              # time.time()
```

### ChatSessionRegistry API

```python
class ChatSessionRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[str, ChatSession] = {}

    def create(self, graph, settings, source_urls, source_mode, thread_id=None) -> ChatSession
    def get(self, thread_id) -> ChatSession | None
    def delete(self, thread_id) -> bool
    def list_ids(self) -> list[str]
    def __len__(self) -> int
```

Thread-safe via `threading.Lock()`. In-memory only — lost on server restart.
`delete()` does NOT clean up Chroma directories (Phase 2 fix).

## Condense Node Logic

The condense node rewrites a follow-up question into a standalone question using conversation history:

```python
def condense_question(state):
    messages = list(state["messages"])
    if not messages:
        return {"current_question": "", "current_question_index": 0}

    latest_idx = _latest_user_index(messages)  # scan from end for last HumanMessage
    latest_msg = messages[latest_idx]
    latest_text = getattr(latest_msg, "content", "")

    # Turn 1 (no history before latest): return raw text
    if latest_idx == 0:
        return {
            "current_question": latest_text,
            "current_question_index": 0,
            "rewrite_count": 0,
        }

    # Turn N: format prior turns and prompt LLM
    history = _format_history(messages[:latest_idx])
    prompt = CONDENSE_PROMPT.format(history=history, question=latest_text)
    llm = _new_chat_model(settings)
    # ... invoke LLM, handle failure -> fallback to latest_text
```

**Condense prompt:**
```
Given the following conversation and a follow-up question, rewrite the
follow-up so it is a standalone question that can be understood without the
prior context. Preserve the user's intent and language. If the follow-up is
already self-contained, return it unchanged.

Conversation history:
{history}

Follow-up question:
{question}

Standalone question:
```

### Helper Functions

| Function | Purpose | Notes |
|----------|---------|-------|
| `_format_history(messages)` | Render prior turns as "User: ...\nAssistant: ..." transcript | Skips tool/function messages intentionally |
| `_latest_user_index(messages)` | Scan from end for last HumanMessage | O(n) — acceptable for in-memory but add index caching if moving to DB |

## Chroma Isolation Strategy

| Session type | Chroma location | Shared? |
|---|---|---|
| Default URLs | `.chroma/` (global) | Yes — all default sessions share |
| Explicit URLs | `.chroma/chat/<thread_id>/` | No — isolated per session |
| Web search URLs | `.chroma/chat/<thread_id>/` | No — isolated per session |

Isolation is decided at session creation:

```python
def _settings_for_session(base, urls, thread_id, isolated):
    if not isolated:
        return replace(base, source_urls=urls)  # shares global Chroma
    return replace(
        base,
        source_urls=urls,
        chroma_dir=base.chroma_dir / "chat" / thread_id,
        collection_name=f"{base.collection_name}-chat-{thread_id}",
    )
```

## Chat API Endpoint Flow (POST /chat/{id}/message)

```
1. Lookup session by thread_id from ChatSessionRegistry
2. Build config: {"configurable": {"thread_id": thread_id}}
3. Build inputs: {"messages": [HumanMessage(content=message)]}
4. Snapshot pre-turn message count (prev_count)
5. Run graph.invoke(inputs, config) via asyncio.to_thread()
6. Extract NEW messages (messages[prev_count:])
7. Find last AI message with content (skip tool-call carriers)
8. Return answer
```

**Subtlety:** Step 6 prevents echoing the user's question or a prior turn's reply. The rewrite node may append an AIMessage carrying a fallback question — filtering by new messages only avoids treating that as the answer.

### Answer Extraction

```python
answer = ""
for msg in new_messages:
    if getattr(msg, "type", None) == "ai" or msg.__class__.__name__ == "AIMessage":
        text = getattr(msg, "content", "")
        if text and text.strip():
            answer = text
            break
```

Skips tool-call carriers (AIMessages with empty content but `tool_calls` populated). Returns the first AI message with substantive content.

## Graph Construction (Chat-Specific)

### build_chat_graph() vs core/build_graph()

```python
def build_chat_graph(settings, rebuild_vectorstore=False):
    retriever_tool = build_retriever_tool(settings, rebuild=rebuild_vectorstore)
    tools = [retriever_tool]

    workflow = StateGraph(ChatState)           # <-- ChatState, not AgentState

    workflow.add_node("condense", condense_question_factory(settings))  # <-- extra node
    # ... 4 nodes shared with core (but from chat/nodes.py)
    workflow.add_edge(START, "condense")       # <-- START goes to condense, not agent
    workflow.add_edge("condense", "agent")
    # ... same conditional edges as core

    return workflow.compile(checkpointer=_build_memory_saver())  # <-- checkpointer
```

### MemorySaver Version Handling

```python
def _build_memory_saver():
    try:
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
    except ImportError:
        from langgraph.checkpoint.memory import InMemorySaver  # older path
        return InMemorySaver()
```

## Known Issues

### Priority 1 — Critical

| # | Issue | Location | Detail |
|---|---|---|---|
| 1 | **Sessions never expire** — no TTL; memory grows unboundedly | `chat/sessions.py` | Add background cleanup daemon (Phase 2) |
| 2 | **delete() doesn't clean up Chroma** — isolated session directories orphaned on disk | `chat/sessions.py:79-85` | Release Chroma + rmtree on delete (Phase 2) |

### Priority 2 — High

| # | Issue | Location | Detail |
|---|---|---|---|
| 3 | **No session persistence** — server restart loses all active chats | `chat/sessions.py` | Add SQLiteStorage backend (Phase 3) |
| 4 | **Checkpointer always MemorySaver** — no way to swap for persistent checkpointer | `chat/graph.py:_build_memory_saver()` | Accept checkpointer factory in builder (Phase 3) |
| 5 | **condense_question_factory has no tests** | `chat/nodes.py:86-134` | Add unit tests with mock LLM (Phase 1) |

### Priority 3 — Medium

| # | Issue | Detail |
|---|---|---|
| 6 | Answer extraction uses `getattr(msg, "content", "")` — fragile across LangChain versions | Standardize to `msg.content` with type guard |
| 7 | `_format_history` skips tool messages — intentional but undocumented | Add docstring |
| 8 | `_latest_user_index` scans messages linearly from end — O(n) on long conversations | Acceptable for in-memory; add index caching if migrating to DB |
| 9 | `_build_memory_saver()` has compat fallback for old LangGraph `InMemorySaver` | Remove once minimum langgraph>=0.3 is set |

## Target — src/sessions/

```
src/sessions/
+-- __init__.py
+-- registry.py            # ChatSessionRegistry with TTL cleanup + Chroma release
+-- models.py              # ChatSession dataclass
+-- storage.py             # StorageBackend Protocol
+-- in_memory.py           # InMemoryStorage (current default)
+-- sqlite.py              # SQLiteStorage (Phase 3)
```

### TTL Cleanup Design (Phase 2)

```python
class ChatSessionRegistry:
    def __init__(self, ttl_seconds: int = 3600, cleanup_interval: int = 300):
        self._ttl = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._cleanup_thread: threading.Thread | None = None

    def start_background_cleanup(self) -> None:
        """Launch a daemon thread that periodically removes stale sessions."""

    def cleanup_expired(self) -> int:
        """Remove and release Chroma resources for sessions past TTL."""
        count = 0
        now = time.time()
        with self._lock:
            expired = [
                tid for tid, s in self._sessions.items()
                if now - s.created_at > self._ttl
            ]
            for tid in expired:
                self._delete_with_cleanup(tid)
                count += 1
        return count

    def _delete_with_cleanup(self, thread_id: str) -> None:
        """Delete session + release Chroma directory if isolated."""
        session = self._sessions.pop(thread_id, None)
        if session is None:
            return
        _release_chroma_system(session.settings.chroma_dir)
        # Remove isolated Chroma dir (only for isolated sessions)
        if session.settings.chroma_dir != base_chroma_dir:
            _rmtree_with_retry(session.settings.chroma_dir)
```

### SQLite Persistence Design (Phase 3)

```sql
CREATE TABLE IF NOT EXISTS sessions (
    thread_id     TEXT PRIMARY KEY,
    source_urls   TEXT NOT NULL,       -- JSON array
    source_mode   TEXT NOT NULL,
    chroma_dir    TEXT,                -- NULL if shared
    created_at    REAL NOT NULL,
    last_active   REAL NOT NULL,
    checkpoint    BLOB                 -- LangGraph checkpoint (serialized)
);
```

**Implementation notes:**
- `StorageBackend` Protocol: `save()`, `load()`, `delete()`, `list_ids()`
- `InMemoryStorage` for dev/testing, `SQLiteStorage` for production
- Checkpointer: swap `MemorySaver` for `SqliteSaver` from `langgraph.checkpoint.sqlite`
- Migration path: no existing sessions to migrate (currently in-memory only)

## Refactoring To-Do List

> Source: [`REFACTORING_PLAN.md`](../../REFACTORING_PLAN.md). Session/chat scope items.

### Phase 1 — Extract Without Behavioral Change

- [ ] **1.5 Unify node libraries** — `condense_question_factory` stays unique to sessions, other 4 factories use shared `src/graph/nodes/` with Chat resolver
- [ ] **1.5 Merge state types** — `ChatState` + `AgentState` → `RAGState` with optional `current_question`, `current_question_index`
- [ ] **1.6 Extract `CONDENSE_PROMPT`** → `src/llm/prompts.py`
- [ ] **1.5 Add unit tests for condense node** — mock LLM returns fixed standalone question; verify turn-1 skip path

### Phase 2 — Interfaces & Abstractions

- [ ] **2.3 DI for ChatSessionRegistry** — inject via `fastapi.Depends()` instead of module-level singleton
  - [ ] `get_session_registry(request)` dependency in `src/api/dependencies.py`
- [ ] **2.7 Extract `_settings_for_session`** Chroma isolation logic → `src/sessions/`
- [ ] **2.7 Remove `_build_memory_saver()` compat fallback** — set minimum `langgraph>=0.3`
- [ ] **2.7 Migrate condense node** → `src/graph/nodes/condense.py` (shared graph, sessions-scoped logic)
- [ ] **2.3 TTL cleanup** — `ChatSessionRegistry.start_background_cleanup()` with configurable TTL
  - [ ] `cleanup_expired()` — remove stale sessions
  - [ ] `_delete_with_cleanup()` — release Chroma + rmtree isolated directories

### Phase 3 — Persistence & Streaming

- [ ] **3.2 Session persistence**
  - [ ] Define `StorageBackend` Protocol: `save()`, `load()`, `delete()`, `list_ids()`
  - [ ] `InMemoryStorage` (default, no behavior change)
  - [ ] `SQLiteStorage` with sessions table schema
  - [ ] Swap `MemorySaver` for `SqliteSaver` from `langgraph.checkpoint.sqlite`
  - [ ] Session recovery: restore active sessions on server restart
  - [ ] Verify: server restart preserves chat sessions with full history
- [ ] **3.1 SSE streaming** — `POST /chat/{id}/message/stream` (see `[[api-interface-architect]]` for transport)
- [ ] **3.5 Deprecation shim** — `src/chat/sessions.py` → re-exports from `src/sessions/`
- [ ] **3.5 Chat REPL improvements** — add progress spinner + async support

## Testing Strategy

| Test | Approach | Phase |
|------|----------|-------|
| condense node | Mock LLM returns fixed standalone question; verify state update | Phase 1 |
| condense turn-1 skip | Verify no LLM call when history is empty | Phase 1 |
| session create/get/delete | Unit tests with mock graph | Phase 1 |
| session TTL cleanup | Fast-forward time, verify expired sessions removed + Chroma dirs cleaned | Phase 2 |
| Chroma cleanup on delete | Verify directory removed for isolated sessions | Phase 2 |
| Full chat flow | Mock LLM + mock retriever, multi-turn conversation | Phase 2 |
| Session persistence | Create session → restart server → verify restoration | Phase 3 |
| Answer extraction | Verify tool-call carriers filtered, fallback question not echoed | Phase 1 |

## Dependencies

- `src/graph/` — shared node factories (see `[[rag-pipeline-architect]]`)
- `src/rag/retriever.py` — `_release_chroma_system()`, `_rmtree_with_retry()` for cleanup (see `[[knowledge-retrieval-architect]]`)
- `src/api/dependencies.py` — DI wiring (see `[[api-interface-architect]]`)
- `src/web_search/` — `discover_urls_from_web()` for web search sessions (see `[[web-search-architect]]`)
- External: `langgraph.checkpoint.memory.MemorySaver`, `langgraph.checkpoint.sqlite.SqliteSaver` (Phase 3)
