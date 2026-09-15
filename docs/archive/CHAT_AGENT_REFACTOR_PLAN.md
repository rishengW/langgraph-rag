# Chat Agent Refactor Plan — Adopting the Deer-Flow Architecture

> **Scope:** Refactor the **chat agent** of `langgraph-rag` onto the agent‑+‑middleware
> architecture used by `deer-flow-main` (ByteDance DeerFlow 2.0). The **QA mode**
> and the **lightweight web-search path** are explicitly out of scope for the
> initial migration (see §10).
>
> **Status:** Draft for review. File paths and function names are verified against
> the current `src/` tree as of 2026-09.

---

## 1. Goal & Scope

### 1.1 Goal

Replace the chat mode's **explicit `StateGraph` topology** (`condense → agent → retrieve → grade → {generate | rewrite} → END`) with a **single agent built via `langchain.agents.create_agent` + an ordered middleware chain**, mirroring how DeerFlow assembles its lead agent in `deerflow/agents/lead_agent/agent.py` and `deerflow/agents/factory.py`.

The chat agent is a multi-turn tool-calling loop — which is exactly what DeerFlow's `create_agent` + `AgentMiddleware` model is designed for. The RAG-specific logic (condense, retrieve, grade, rewrite, generate, citation cleanup) moves **from graph nodes/edges into middlewares and a retrieval tool**, without losing the hard guarantees the current graph enforces.

### 1.2 In scope

- Chat mode **heavy Chroma-backed path** (`build_graph(mode="chat")`).
- State schema, agent factory, middleware chain, checkpointer, streaming adapter.
- The chat FastAPI surface (`src/chat/api.py`) routing the heavy path to the new agent.

### 1.3 Out of scope (this migration)

- **QA mode** (`build_graph(mode="qa")`) — its single-shot explicit graph is a reasonable fit for its shape; leave untouched.
- **Lightweight web-search path** (`build_lightweight_graph`) — its `decompose → fan-out → merge → web_answer` with structural/semantic/recency filtering and grounded-refusal fallback is genuinely graph-shaped and a poor fit for the agent+middleware model. It keeps running on the existing graph during and after this migration (see §4.5, §8 Phase 5).
- **Planning / self-critique / reflection** nodes (`planner`, `subgoal_*`, `answer_self_critique`, `reflection_revise`) — gated behind `planning_enabled`; defer until the base agent is stable.
- DeerFlow subsystems with no chat-RAG analogue: sandbox, subagents, IM channels, skills loader, MCP client, Next.js frontend. None are adopted.
- DashScope/Qwen remains the LLM; Chroma remains the vector store. **No provider migration.**

---

## 2. Current Architecture (verified)

### 2.1 Two chat graph paths

The chat app selects the path at session creation (`src/chat/api.py: _build_chat_graph_for_session` vs `_build_lightweight_chat_graph_for_session`):

**Heavy path** (`src/graph/builder.py: build_graph(mode="chat")`):

```
START → condense → [planner?] → agent → (retrieve → grade → {generate | rewrite → agent}) → generate → [self_critique ⇄ reflection_revise?] → END
```

**Lightweight path** (`build_lightweight_graph(mode="chat")`):

```
START → [planner?] → agent → (decompose → execute_search_queries → merge → web_answer {→ expand → execute_search_queries → … → fallback_answer}) → END
```

This plan replaces **only the heavy path**.

### 2.2 Key files (heavy chat path)

| File | Role |
|---|---|
| `src/graph/builder.py` | `build_graph(mode="chat")` — assembles the heavy StateGraph; `GraphNodeOverrides`/`GraphProviders` DI; `_resolve_checkpointer`; `_compile_with_catalog`. |
| `src/graph/state.py` | `RAGState` TypedDict (aliased as `ChatState`): `messages` (add_messages), `current_question`, `current_question_index`, `source_urls`, `source_mode`, `rewrite_count`, web-search reset fields, planning fields. |
| `src/graph/edges.py` | `route_after_agent_with_critique`, `GRADE_EDGE_MAP`, etc. |
| `src/graph/nodes/common.py` | `agent_factory`, `grade_documents_factory` (conditional edge), `rewrite_factory`, `generate_factory`, `rerank_retrieved_context`, `bounded_chat_messages`, `build_extractive_answer`, `chat_question_resolver`. |
| `src/graph/nodes/condense.py` | `condense_question_factory`, `condense_followup_question`, `needs_condensation`. |
| `src/graph/executor.py` | `GraphExecutor` — wraps compiled graph; `stream(stream_mode=["updates","messages"])`; token filter to `ANSWER_NODES`. |
| `src/graph/events.py` | Typed `GraphEvent`s (`NodeStart/End`, `ToolStart/End`, `Token`, `RetrieverResult`, `GraderDecision`, `Artifact`, `Error`, `Done`). |
| `src/application/chat_service.py` | `ChatApplicationService` — `start`/`complete_turn`/`stream_turn`/`history`/`delete`; quota acquisition; turn lock. |
| `src/application/session_lifecycle.py` | `SessionLifecycleService` — session creation, source discovery, graph build per session. |
| `src/application/turn_execution.py` | `TurnExecutionService` — serialized execution, snapshot/rollback via checkpointer, `_graph_inputs_for_turn`, `_tool_principal_context`. |
| `src/sessions/checkpoint.py` | `SQLiteMemorySaver` — subclasses `MemorySaver`, pickles entire storage/writes/blobs maps into one SQLite row; `enforce_ownership`; `snapshot_thread`/`restore_thread`. |
| `src/sessions/sqlite.py` | `SQLiteStorage` — session metadata (schema v2). |
| `src/rag/chroma_retriever.py` | `ChromaRetriever` — owns Chroma lifecycle; `as_tool()` → `create_retriever_tool("retrieve_source_documents", …)`; per-thread isolated dirs. |
| `src/llm/provider.py` | `build_chat_model(settings)` → `ChatTongyi`; `build_structured_chat_model(settings, schema)`. |
| `src/llm/prompts.py` | `AGENT_SYSTEM_PROMPT` (tool-selection policy), `RAG_PROMPT` (synthesis), `GRADE_PROMPT`, `CONDENSE_PROMPT`. |
| `src/llm/sanitize.py` | `strip_citation_artifacts`. |
| `src/memory/recall.py` | `build_turn_messages` — memory recall + upload note → turn messages. |
| `src/chat/memory_hooks.py` | `build_extraction_runtime`, `after_turn`, `on_session_start`. |
| `src/chat/api.py` | `create_app()`; endpoints; `_build_chat_application_service`; `_graph_inputs_for_turn`; `_restore_persisted_sessions`. |
| `src/mcp/providers.py` | Tool providers → `ToolCatalogSnapshot`; `RetrieverToolProvider`, `MemoryToolProvider`, `BuiltinToolProvider`, `DocumentToolProvider`, `SessionEditingToolProvider`, `InjectedToolProvider`. |
| `src/api/streaming.py` | `format_sse(GraphEvent)`. |

### 2.3 Behaviors that must survive the refactor

These are load-bearing and must be preserved or explicitly superseded:

1. **Grade → rewrite loop with a hard budget** (`grade_documents_factory` + `rewrite_factory`, bounded by `settings.max_rewrites`). The grader uses a structured `Grade` model (`binary_score` + `explanation`), reranks via `rerank_retrieved_context` (lexical/embedding/hybrid), falls back to keyword matching when the LLM is unreachable, and forces `generate` when the rewrite budget is exhausted or `allow_low_relevance_generate` + keyword threshold is met.
2. **Condense follow-ups** (`condense_question_factory`) — rewrites a follow-up into a standalone question before the agent runs, gated by `needs_condensation`.
3. **Bounded chat history projection** (`bounded_chat_messages`) — `chat_context_max_turns` / `chat_context_max_chars`, preserving the complete current-turn tool protocol.
4. **Citation-artifact stripping** (`strip_citation_artifacts`) on direct agent answers and generated answers.
5. **Extractive-answer fallback** (`build_extractive_answer`) when DashScope is unreachable at generation time.
6. **Per-thread Chroma isolation** (`.chroma/chat/<thread_id>/`) and per-thread upload dirs (`session_upload_dir`).
7. **Memory recall + async extraction** (`build_turn_messages` + `after_turn`/`on_session_start`).
8. **Turn serialization + rollback** (`TurnExecutionService`: per-session `turn_lock`, checkpointer snapshot/restore on failure/cancellation).
9. **Ownership-enforced checkpointing** (`SQLiteMemorySaver` with `enforce_ownership=True`, `register_owner`, `snapshot_thread(owner=)`).
10. **Quotas** (`QuotaManager`: per-principal/per-tenant rolling-minute budgets; `turn_lock` serialization).

---

## 3. Target Architecture (Deer-Flow-Shaped)

### 3.1 The shape

One agent, one state schema, an ordered middleware chain, a standard checkpointer, a streaming adapter. Mirrors DeerFlow's `make_lead_agent` + `_build_middlewares` + `ThreadState` + `make_checkpointer` + `run_agent`/`StreamBridge`.

```
HTTP turn → ChatApplicationService (quota + turn_lock)
         → ChatAgentRuntime.run_agent(astream)
            → make_chat_agent(config) → create_agent(model, tools, middlewares, prompt, state_schema=ChatThreadState)
            → middleware chain flows each model turn:
               ContextWindow → Condense → Uploads → Memory → RetrievalGrader → Citation → (Title)
            → StreamBridge/GraphEvent adapter → SSE
```

### 3.2 State schema — `ChatThreadState`

Replaces `RAGState`/`ChatState` for the chat heavy path. Extends `langchain.agents.AgentState` (which already provides `messages` with the add_messages reducer), adding the channels the middlewares need. This mirrors DeerFlow's `ThreadState` extension pattern.

```python
# src/chat_agent/state.py
from typing import Annotated, TypedDict
from langchain.agents import AgentState

class ChatThreadState(AgentState):
    # AgentState already gives us `messages` (add_messages reducer).
    current_question: str                       # condense output; standalone question
    rewrite_count: int                          # retrieval-grader budget tracking
    source_urls: list[str]                      # current turn's source set
    source_mode: str                            # "explicit" | "web_search" | "defaults"
    last_graded_relevant: bool                   # retrieval-grader gate for generation
    artifacts: Annotated[list[str], merge_artifacts]   # dedup-preserving (from src/graph/artifacts.py)
    # NOTE: web-search fan-out fields stay on the lightweight path's state, not here.
```

`merge_artifacts` is reused from `src/graph/artifacts.py` (do not duplicate).

### 3.3 Agent factory — `make_chat_agent`

Two entry points, mirroring DeerFlow's `make_lead_agent` (config-driven) and `create_deerflow_agent` (config-free):

- `src/chat_agent/agent.py`
  - `create_chat_agent(model, tools, *, features: ChatRuntimeFeatures, system_prompt) -> CompiledAgent` — config-free, pure-argument; assembles the middleware chain from declarative flags. Used by tests and the in-process client.
  - `make_chat_agent(config: RunnableConfig) -> CompiledAgent` — resolves model (`build_chat_model`), tools (`compose_chat_tools`), features (from `Settings`), and system prompt (`apply_chat_prompt_template`); calls `create_chat_agent`.

`CompiledAgent` is `langchain.agents.create_agent(model, tools, middleware=middlewares, prompt=system_prompt, state_schema=ChatThreadState)`.

### 3.4 Middleware chain (heavy chat path)

Ordered, mirroring DeerFlow's `_build_middlewares`. Each is an `AgentMiddleware` subclass with `before_model` / `after_model` / `before_tool_call` / `after_tool_call` hooks.

| # | Middleware | Hook(s) | Replaces (old node/edge) | Responsibility |
|---|---|---|---|---|
| 1 | `ContextWindowMiddleware` | `before_model` | `bounded_chat_messages` | Project `messages` to `chat_context_max_turns`/`chat_context_max_chars`, preserving the complete current-turn tool protocol. |
| 2 | `CondenseMiddleware` | `before_model` | `condense` node | If `needs_condensation(latest HumanMessage)`, rewrite it to standalone via `condense_followup_question`; store result in `current_question`. |
| 3 | `UploadsMiddleware` | `before_model` | `_new_upload_context` | Inject the upload-context `SystemMessage` for not-yet-announced files (file paths for the document tools). |
| 4 | `MemoryMiddleware` | `before_model` (+ after-turn) | `build_turn_messages` (recall) + `after_turn`/`on_session_start` (extraction) | Inject recalled memory as a `SystemMessage` ahead of the upload note; queue the turn for async extraction (existing `build_extraction_runtime`). |
| 5 | `RetrievalGraderMiddleware` | `after_tool_call` (for `retrieve_source_documents`) | `grade_documents` edge + `rewrite` node | Rerank retrieved context (`rerank_retrieved_context`), grade it (`Grade` model + `GRADE_PROMPT`), and on irrelevance **rewrite + re-retrieve within the middleware** (bounded by `max_rewrites`); set `last_graded_relevant`. Falls back to keyword matching + forced-generate rules exactly as today. |
| 6 | `CitationMiddleware` | `after_model` | `strip_citation_artifacts` + `build_extractive_answer` | Strip citation artifacts from the final assistant message; on model failure, substitute `build_extractive_answer(question, context)`. |
| 7 | `TitleMiddleware` *(optional, new)* | `after_model` | — | Auto-generate a thread title after the first exchange (a DeerFlow capability the chat UI can opt into; not a regression if omitted). |

> **Ordering rationale:** ContextWindow and Condense run before the model sees the turn (Condense after ContextWindow so the rewritten user message is already bounded). Uploads and Memory inject system context. RetrievalGrader intercepts the retrieval tool result before the model re-enters. Citation cleans the model's final output. This mirrors DeerFlow's ordering discipline (data/turn setup → guardrails → model-interleaved logic → output cleanup).

### 3.5 Retrieval as a tool (unchanged)

`ChromaRetriever.as_tool()` already produces a `create_retriever_tool("retrieve_source_documents", …)`. **Keep it.** The agent calls it; `RetrievalGraderMiddleware` intercepts its result. No change to `src/rag/chroma_retriever.py`, `document_loader.py`, `embeddings.py`, `dashscope_embeddings.py`.

### 3.6 System prompt consolidation

Old graph uses two prompts: `AGENT_SYSTEM_PROMPT` (tool selection, in `agent_factory`) and `RAG_PROMPT` (synthesis, in `generate_factory`). The agent model has **one** model loop, so these consolidate into a single template:

- `src/chat_agent/prompt.py`
  - `apply_chat_prompt_template(settings, …) -> str` — combines tool-selection policy + grounded-synthesis policy + the current date anchor. New `CHAT_AGENT_SYSTEM_PROMPT` template (derive by merging `AGENT_SYSTEM_PROMPT` and `RAG_PROMPT` from `src/llm/prompts.py`; keep both old templates intact for QA mode).

### 3.7 Checkpointer — standard

Replace `SQLiteMemorySaver` (custom pickle-the-whole-map) with the standard LangGraph SQLite checkpointer, wrapped to preserve ownership enforcement:

- `src/chat_agent/checkpointer.py`
  - `make_chat_checkpointer(settings) -> BaseCheckpointSaver` — returns `AsyncSqliteSaver` (or `SqliteSaver`) pointed at `.chroma/chat/checkpoints.sqlite3`.
  - `OwnershipWrapper` — adapts the standard checkpointer to the `register_owner` / `snapshot_thread(owner=)` / `restore_thread(owner=)` contract that `TurnExecutionService` already expects (see `turn_execution.py: _snapshot`/`_rollback` and the `ownership_enforced` attribute check). This keeps §2.3 #9 intact without the pickle-row hack.

### 3.8 Streaming adapter — agent → `GraphEvent`

The chat API and `TurnExecutionService` are built around typed `GraphEvent`s (`src/graph/events.py`) and `format_sse`. Rather than rip that out, adapt the agent's `astream(stream_mode=["messages","updates"])` output to the existing `GraphEvent` types:

- `src/chat_agent/streaming.py`
  - `ChatAgentStreamAdapter` — maps LangGraph agent stream chunks → `GraphEvent`s (`NodeStart/End`, `ToolStart/End`, `Token`, `RetrieverResult`, `GraderDecision`, `Artifact`, `Error`, `Done`), reusing the existing `CitationArtifactFilter` token-boundary logic from `src/graph/executor.py`. Token events are emitted only for the agent's user-facing model calls (the agent's own responses), not for grader/condense structured-output calls — the grader/condense now run inside middlewares, so their LLM calls are internal and already filtered out by virtue of not being the agent's response stream.
  - `run_chat_agent(agent, inputs, config, *, on_event) -> None` — the equivalent of DeerFlow's `runtime/runs/worker.py: run_agent`; runs `agent.astream(...)` inside an `asyncio.Task`, publishes events through the adapter, and is driven by `TurnExecutionService`.

This keeps `format_sse`, the SSE endpoint shape, and every `GraphEvent` consumer unchanged.

---

## 4. Key Architectural Decisions

### 4.1 The grade/rewrite loop — middleware-enforced, not prompt-guided  *(decision)*

This is the single most important design choice. Two options were considered:

- **R1 (prompt-guided):** Move grade/rewrite into the system prompt + retriever tool description. Lowest effort, but loses the hard guarantee — the agent could skip retrieval or answer from stale context. **Rejected.**
- **R2 (middleware-enforced):** A `RetrievalGraderMiddleware` whose `after_tool_call` hook (for `retrieve_source_documents`) reranks, grades, and on irrelevance rewrites + re-retrieves *within the middleware*, bounded by `max_rewrites`. Reuses `grade_documents_factory`'s exact grading logic (`Grade` model, `GRADE_PROMPT`, keyword fallback, `allow_low_relevance_generate`, budget-exhausted → force generate). **Adopted.**

This preserves the §2.3 #1 guarantee structurally. The middleware owns `rewrite_count` and `last_graded_relevant`; generation (the agent's final response) is gated by the agent loop naturally — the model re-enters after the graded tool result is in context.

**Boundary case to implement explicitly:** when the rewrite budget is exhausted, the middleware must mark the (possibly-irrelevant) context as "best available" and let the model generate, exactly as the old `grade_documents` edge returns `"generate"` at `rewrite_count >= max_rewrites`.

### 4.2 Framework upgrade is a hard prerequisite  *(decision)*

`langchain.agents.create_agent` with the `AgentMiddleware` API is a **LangChain 1.x / LangGraph 1.x** feature. langgraph-rag currently pins `langgraph==0.6.11`, `langchain==0.3.30`. **The middleware architecture cannot be built on the current pins.** Phase 0 (§8) is the framework upgrade and is non-negotiable as the foundation.

### 4.3 DashScope/Qwen stays; no provider migration  *(decision)*

`build_chat_model(settings)` → `ChatTongyi` is retained. `create_agent(model=ChatTongyi(...), …)` is provider-agnostic. The only requirement is that `ChatTongyi` supports `.bind_tools()` — it already does (the current `agent_factory` binds tools to it). DeerFlow's multi-provider reflection (`resolve_class` for models) is **not** adopted in this migration; it's a possible later phase.

### 4.4 Chroma retriever stays a tool; no graph change to RAG plumbing  *(decision)*

`ChromaRetriever`, `document_loader.py`, `build_embeddings`, `DashScopeTextEmbeddings`, per-thread isolation, embedding-config auto-rebuild — all unchanged. The retriever is already exposed as a tool (`as_tool()`); the refactor just routes its result through `RetrievalGraderMiddleware` instead of through the `retrieve → grade` graph edges.

### 4.5 Lightweight web-search path keeps its graph  *(decision)*

`build_lightweight_graph(mode="chat")` remains as-is. The chat API will **route** at session creation: heavy source modes → new `create_chat_agent`; web-search-lightweight mode → existing `build_lightweight_graph`. This is already how the API branches today (`_build_chat_graph_for_session` vs `_build_lightweight_chat_graph_for_session`), so the routing seam already exists. No behavior change for web-search sessions.

### 4.6 Planning / self-critique deferred  *(decision)*

`planning_enabled` nodes (`planner`, `subgoal_*`, `answer_self_critique`, `reflection_revise`) are not migrated. Until a later phase, **planning is disabled for chat** when running on the new agent (`features.planning = False`), and the old graph remains available behind a flag for any session that needs planning. This is a known, scoped capability gap — called out in the risk register (§9).

### 4.7 Harness/App boundary discipline  *(decision, aspirational)*

DeerFlow's strongest structural convention is the **publishable-harness / unpublished-app split with a CI-enforced import firewall** (`deerflow.*` may never import `app.*`). Adopt the same discipline for the new code:

- **`src/chat_agent/`** (the "harness"): agent factory, state, middlewares, checkpointer, streaming adapter, prompt. May import `..config`, `..llm`, `..rag`, `..memory`, `..graph.artifacts`, `..graph.events`. **Must not import `..application` or `..chat.api`.**
- **`src/application/`** and **`src/chat/api.py`** (the "app"): compose `chat_agent` into the existing services and endpoints.

Add a CI test (`tests/test_chat_agent_boundary.py`) asserting `src/chat_agent/**` imports no symbol from `src/application` or `src/chat`. This is the single most valuable structural inheritance from DeerFlow.

---

## 5. Target File Layout (new)

```
src/chat_agent/
├── __init__.py                     # public re-exports
├── agent.py                        # make_chat_agent, create_chat_agent, ChatRuntimeFeatures
├── state.py                        # ChatThreadState (+ merge_artifacts re-export)
├── prompt.py                       # apply_chat_prompt_template, CHAT_AGENT_SYSTEM_PROMPT
├── checkpointer.py                 # make_chat_checkpointer, OwnershipWrapper
├── streaming.py                    # ChatAgentStreamAdapter, run_chat_agent
├── tools.py                        # compose_chat_tools(settings, session) -> list[BaseTool]
└── middleware/
    ├── __init__.py
    ├── context_window.py           # ContextWindowMiddleware
    ├── condense.py                 # CondenseMiddleware
    ├── uploads.py                  # UploadsMiddleware
    ├── memory.py                   # MemoryMiddleware
    ├── retrieval_grader.py         # RetrievalGraderMiddleware
    ├── citation.py                 # CitationMiddleware
    └── title.py                    # TitleMiddleware (optional)
```

**Modified:**
- `src/application/turn_execution.py` — `TurnExecutionService` drives `run_chat_agent` for chat-agent sessions instead of `session.graph.invoke`/`GraphExecutor.stream`; keep snapshot/rollback + `turn_lock` + `_tool_principal_context`.
- `src/application/session_lifecycle.py` — builds a `make_chat_agent`-based runtime for heavy sessions; keeps `build_lightweight_graph` for web-search-lightweight sessions.
- `src/chat/api.py` — `_build_chat_graph_for_session` → `_build_chat_agent_for_session`; `_graph_inputs_for_turn` simplified (memory/upload injection moves to middlewares, so turn inputs shrink to `{"messages": [HumanMessage(message)]}` plus the web-search reset fields when applicable).
- `src/sessions/checkpoint.py` — `SQLiteMemorySaver` retained as a compatibility alias / for the lightweight path; new heavy sessions use `make_chat_checkpointer`.

**Retired (Phase 4, after the new agent is stable and the old heavy graph has no live sessions):**
- `src/graph/builder.py` chat-mode branch of `build_graph` (keep `mode="qa"` and `build_lightweight_graph`).
- `src/graph/nodes/common.py` chat usage of `agent_factory`/`rewrite_factory`/`generate_factory` (keep for QA mode).
- `src/chat/nodes.py`, `src/chat/graph.py` facades (if they only serve the heavy path).
- `src/graph/edges.py` chat-only edges (`AGENT_EDGE_MAP`, `GRADE_EDGE_MAP` for the heavy path).

---

## 6. Middleware Specifications

Each middleware's contract, mapped from the verified old behavior. Reuse existing functions verbatim where possible — do not rewrite grading/condense/rerank logic.

### 6.1 `ContextWindowMiddleware` (before_model)

- **Reuses:** `bounded_chat_messages(messages, max_turns=settings.chat_context_max_turns, max_chars=settings.chat_context_max_chars)` from `src/graph/nodes/common.py`.
- **Hook:** `before_model` — replace `state["messages"]` with the bounded projection.
- **Constraint:** must preserve the complete current-turn tool protocol (AI tool-call carrier + matching ToolMessage) — `bounded_chat_messages` already does.

### 6.2 `CondenseMiddleware` (before_model)

- **Reuses:** `needs_condensation` and `condense_followup_question` from `src/graph/nodes/condense.py`.
- **Hook:** `before_model` — if the latest `HumanMessage` needs condensation, rewrite its content to the standalone form; set `state["current_question"]`.
- **Note:** rewriting the user message in-place is acceptable (the original remains in the checkpointer state from prior turns); the chat history endpoint already strips condense artifacts. Verify this does not double-count against `bounded_chat_messages` — run Condense **after** ContextWindow (see ordering).

### 6.3 `UploadsMiddleware` (before_model)

- **Reuses:** `list_session_uploads`, `build_upload_context_note`, `session.announced_uploads` tracking from `src/chat/uploads.py` + `src/chat/api.py: _new_upload_context`.
- **Hook:** `before_model` — prepend the upload-context `SystemMessage` for not-yet-announced files; update `announced_uploads`.
- **Gating:** `settings.file_read_enabled`.

### 6.4 `MemoryMiddleware` (before_model + after-turn)

- **Reuses:** `build_turn_messages` (recall) and `build_extraction_runtime`/`after_turn`/`on_session_start` from `src/memory/` + `src/chat/memory_hooks.py`.
- **Hook:** `before_model` — inject recalled memory as a `SystemMessage` ahead of the upload note.
- **After-turn:** the existing extraction runtime is driven by `TurnExecutionService`'s `after_turn` hook — leave that wiring in place; `MemoryMiddleware` only owns the *recall* injection.

### 6.5 `RetrievalGraderMiddleware` (after_tool_call) — the critical one

- **Reuses:** `rerank_retrieved_context`, `new_structured_chat_model(settings, Grade)`, `GRADE_PROMPT`, the `Grade` pydantic model, keyword-matching fallback, `allow_low_relevance_generate`, `min_keyword_matches`, `max_rewrites`, and the exact decision tree from `grade_documents_factory` + `rewrite_factory` (both in `src/graph/nodes/common.py`).
- **Hook:** `after_tool_call` for tool name `retrieve_source_documents`.
- **Behavior:**
  1. Rerank the retrieved `ToolMessage` content via `rerank_retrieved_context(question, tool_message, settings)`.
  2. Grade with the `Grade` model + `GRADE_PROMPT` (date-anchored); on LLM failure, fall back to keyword matching; on `allow_low_relevance_generate` + keyword threshold, mark relevant.
  3. If relevant (or budget exhausted): set `last_graded_relevant=True`; rewrite the `ToolMessage` content to the reranked context; emit a `GraderDecision` event; return control to the agent (which synthesizes the final answer).
  4. If irrelevant and budget remains: increment `rewrite_count`; run `rewrite_factory`'s rewrite logic to produce a new query; **call the retriever again within the middleware** (bounded by `max_rewrites`); re-grade. Emit `GraderDecision` + a synthetic `ToolStart/End` pair for the re-retrieve so the stream stays legible.
  5. **Budget exhausted:** mark the best-available context as relevant (force generate), exactly as the old edge returns `"generate"` at `rewrite_count >= max_rewrites`.
- **Hard guarantee preserved:** the agent never synthesizes a final answer from an ungraded retrieval — the grader runs in `after_tool_call` before the model re-enters.

### 6.6 `CitationMiddleware` (after_model)

- **Reuses:** `strip_citation_artifacts` and `build_extractive_answer` from `src/llm/sanitize.py` + `src/graph/nodes/common.py`.
- **Hook:** `after_model` — for the final assistant message (no tool calls), strip citation artifacts; on `after_model` error from the model, substitute `build_extractive_answer(question, context)` where `context` is the last graded retrieval.

### 6.7 `TitleMiddleware` (after_model) — optional

- New capability (DeerFlow has it; chat currently does not). After the first exchange, generate a short title. **Gated by a setting** (`chat_auto_title_enabled`, default `False`) so the migration introduces no behavior change unless opted in.

---

## 7. Turn Execution & Streaming (wiring)

### 7.1 `TurnExecutionService` adaptation

Today `_execute_sync`/`_stream` call `session.graph.invoke`/`GraphExecutor.stream`. For chat-agent sessions, replace with `run_chat_agent`:

- `TurnExecutionDependencies` gains an optional `agent_runner: Callable[..., AsyncIterator[GraphEvent]]` (default keeps the graph path for QA/lightweight sessions).
- `_stream` calls `agent_runner(agent, inputs, config=_thread_config(thread_id), stream_tokens=…)` and yields `GraphEvent`s through the same queue/producer pattern — the rollback/snapshot/`turn_lock`/`_tool_principal_context` plumbing is unchanged.
- `_execute_sync` (non-streaming) calls the agent's `ainvoke` and extracts the final assistant message via the existing `_last_assistant_answer` helper.

The snapshot/rollback path (`_snapshot`/`_rollback` via `checkpointer.snapshot_thread`/`restore_thread`) works unchanged because `OwnershipWrapper` (§3.7) exposes the same contract.

### 7.2 Stream adapter

`ChatAgentStreamAdapter` maps `agent.astream(stream_mode=["messages","updates"])` chunks → `GraphEvent`s. Token events emit only for the agent's user-facing AIMessage chunks (the grader/condense/rewrite LLM calls happen inside middlewares and are not part of the response stream, so the existing `ANSWER_NODES`-style filtering is structurally satisfied). Reuse `CitationArtifactFilter` for cross-boundary citation stripping.

### 7.3 Chat API changes

`src/chat/api.py`:
- `_build_chat_graph_for_session` → `_build_chat_agent_for_session(settings, checkpointer, thread_id)` returning a `ChatAgentRuntime` (agent + checkpointer + adapter) instead of a compiled graph.
- `_restore_persisted_sessions` — heavy sessions rebuild via `_build_chat_agent_for_session`; lightweight sessions unchanged.
- `_graph_inputs_for_turn` — shrinks to `{"messages": [HumanMessage(message)]}` for heavy sessions (memory/upload/condense handled by middlewares); web-search reset fields still seeded for lightweight sessions.
- Endpoints, SSE shape, auth, quotas, uploads, downloads — **unchanged**.

---

## 8. Phased Migration

Each phase is independently shippable and reversible. Run on a feature branch.

### Phase 0 — Framework upgrade (prerequisite, high risk)

**Goal:** Unpin LangGraph 0.6 / LangChain 0.3 → LangGraph 1.x / LangChain 1.x so the `create_agent` + `AgentMiddleware` API is available.

**Steps:**
1. Verify Python version compatibility (DeerFlow requires 3.12; langgraph-rag targets 3.11 — confirm LangChain 1.x supports 3.11 or raise the floor to 3.12).
2. Create `requirements.upgrade.txt` / a branch; bump `langgraph`, `langchain`, `langchain-core`, `langchain-community`, `langchain-openai`, `langchain-text-splitters`, `langchain-chroma`, `langchain-huggingface` to 1.x.
3. Run `langgraph-rag`'s existing test suite (~95 files) against the new pins; fix breakages. Expect: `StateGraph` API drift, `add_messages` import path, `ToolNode` behavior, `MemorySaver` → `InMemorySaver` rename (already shimmed in `build_memory_saver`), `create_retriever_tool` signature, `ChatTongyi` `.bind_tools` behavior.
4. Validate both chat paths and QA mode still work end-to-end on the upgraded framework **before** any architectural change.

**Acceptance:** green suite on upgraded pins; no behavior change.

**Rollback:** drop the branch; old pins untouched.

### Phase 1 — Standard checkpointer + boundary test (infra, no behavior change)

**Goal:** Replace `SQLiteMemorySaver`'s pickle-row hack with the standard SQLite checkpointer behind an ownership wrapper; establish the harness/app boundary.

**Steps:**
1. Implement `src/chat_agent/checkpointer.py: make_chat_checkpointer` + `OwnershipWrapper` exposing `register_owner` / `snapshot_thread(owner=)` / `restore_thread(owner=)` / `ownership_enforced` over the standard `SqliteSaver`/`AsyncSqliteSaver`.
2. Wire `create_app` lifespan to use `make_chat_checkpointer` for new sessions; keep `SQLiteMemorySaver` as a fallback for existing persisted sessions (migration path: reopen old checkpoints read-only or migrate on first access — decide during implementation).
3. Add `tests/test_chat_agent_boundary.py` asserting `src/chat_agent/**` imports nothing from `src/application` or `src/chat`.
4. Both chat paths still run on their existing graphs; only the checkpointer under them changed.

**Acceptance:** snapshot/rollback still works; ownership still enforced; boundary test green; suite green.

**Rollback:** revert to `SQLiteMemorySaver`.

### Phase 2 — `create_chat_agent` + middleware chain (core refactor, heavy path only)

**Goal:** Build the new agent and middlewares; run the heavy chat path on the new runtime behind a feature flag.

**Steps:**
1. Implement `state.py`, `prompt.py`, `tools.py`, `agent.py` (`create_chat_agent` + `make_chat_agent`).
2. Implement middlewares in dependency order: `ContextWindow` → `Condense` → `Uploads` → `Memory` → `RetrievalGrader` → `Citation` (→ `Title` opt-in). Each reuses the existing function from §6.
3. Implement `streaming.py: ChatAgentStreamAdapter` + `run_chat_agent`.
4. Add a `chat_agent_enabled` setting (default `False`). When `True`, `session_lifecycle` builds the new agent for heavy sessions; when `False`, the old graph.
5. Adapt `TurnExecutionService` to accept an `agent_runner` and route to it for chat-agent sessions.
6. Tests: port the existing chat-mode tests to run under `chat_agent_enabled=True`; add middleware-level unit tests for the grader decision tree (every branch: relevant, irrelevant-within-budget, irrelevant-budget-exhausted, LLM-failure → keyword fallback, `allow_low_relevance_generate`).

**Acceptance:** heavy chat path behaves identically under the new agent with `chat_agent_enabled=True` (same grade/rewrite/condense/citation behavior); lightweight path unchanged; suite green.

**Rollback:** flip `chat_agent_enabled=False`.

### Phase 3 — Promote to default

**Goal:** Make the new agent the default for heavy chat sessions.

**Steps:**
1. Flip `chat_agent_enabled` default to `True`.
2. Update `_restore_persisted_sessions` to rebuild heavy sessions via `_build_chat_agent_for_session`.
3. Soak in staging; monitor metrics (`MetricsCollector`) for parity with the old path.

**Acceptance:** no regressions over a soak period; rollback path (flag → `False`) verified.

**Rollback:** flip the flag.

### Phase 4 — Retire the old heavy-path graph code

**Goal:** Remove the dead heavy-path graph branches now that the agent is the only heavy-path runtime.

**Steps:**
1. Remove the chat-mode branch from `build_graph` (keep `mode="qa"` and `build_lightweight_graph`).
2. Remove chat-only edges (`AGENT_EDGE_MAP`, `GRADE_EDGE_MAP` heavy-path entries) and `src/chat/nodes.py` / `src/chat/graph.py` if they only served the heavy path.
3. Remove `SQLiteMemorySaver` if no path uses it (else keep for lightweight).
4. Keep `agent_factory`/`rewrite_factory`/`generate_factory`/`grade_documents_factory` if QA mode still uses them; otherwise retire.

**Acceptance:** dead code removed; suite green; no production path references the retired symbols.

**Rollback:** revert the deletion commit.

### Phase 5 (future, out of this migration's scope) — Lightweight web-search path

Assess whether `build_lightweight_graph`'s `decompose → fan-out → merge → web_answer` can be re-expressed as the agent + a `WebSearchMiddleware` (with the structural/semantic/recency filtering and grounded-refusal fallback moving into the middleware). This is a larger design effort and is explicitly deferred.

---

## 9. Risk Register

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | **Framework upgrade (Phase 0) breaks the suite in ways that are hard to fix.** | High — blocks the whole migration. | Do Phase 0 on a branch first; if breakage is severe, fall back to Option B (§11) and revisit. |
| 2 | **`RetrievalGraderMiddleware` subtly changes the grade/rewrite decision tree.** | High — silent correctness regression in answer grounding. | Port the exact `grade_documents_factory` + `rewrite_factory` logic verbatim; add a branch-coverage test per decision branch; run old-vs-new parity tests on a fixed set of questions. |
| 3 | **`ChatTongyi` (DashScope) behaves differently under `create_agent` than under the hand-rolled `agent_factory` loop.** | Medium — tool-call / streaming quirks. | Validate `.bind_tools()` + structured output under LangChain 1.x early in Phase 2; DeerFlow's `patched_*` provider pattern is a reference if quirks appear. |
| 4 | **Standard SQLite checkpointer loses `SQLiteMemorySaver`'s snapshot/restore semantics.** | Medium — turn rollback breaks. | `OwnershipWrapper` exposes the exact `snapshot_thread`/`restore_thread` contract; dedicated rollback tests in Phase 1. |
| 5 | **Condense rewriting the user message in-place corrupts history.** | Medium — wrong standalone question, bad transcript. | Condense runs after ContextWindow; verify the history endpoint still strips condense artifacts; add a transcript-parity test. |
| 6 | **Planning/self-critique capability gap for chat.** | Low/Medium — chat loses `planning_enabled` on the new path. | Documented (§4.6); keep the old graph available behind a flag for planning sessions; plan a Phase 5+ for planning migration. |
| 7 | **Persisted-checkpoint migration.** | Medium — existing in-flight chat sessions may not restore on the new checkpointer. | Phase 1 handles read-only reopen or migrate-on-access; gate Phase 3 promotion on a clean migration path. |
| 8 | **Streaming event parity.** | Medium — frontend depends on specific `GraphEvent` shapes. | `ChatAgentStreamAdapter` reuses `CitationArtifactFilter` and emits the exact `GraphEvent` types; add a streaming-parity test that asserts the event sequence for a canonical turn. |

---

## 10. Out of Scope (recap)

- QA mode, lightweight web-search path, planning/self-critique nodes, sandbox, subagents, MCP client, IM channels, skills, Next.js frontend, multi-provider reflection, DashScope→OpenAI/Anthropic migration.

---

## 11. Fallback Option (if Phase 0 fails)

If the LangChain 1.x upgrade proves too disruptive, the lighter-touch **Option B** from the earlier analysis remains viable for the chat agent alone:

- Keep the explicit graph; **borrow only** DeerFlow's: standard checkpointer (Phase 1), per-thread workspace isolation discipline, config-driven reflection for toggling chat features at runtime, and the harness/app boundary test.
- This yields ~40% of the architectural benefit at ~20% of the risk, but does **not** achieve the middleware-centric control flow that is the point of the migration.

Decision point: end of Phase 0. If Phase 0 is green, proceed to Phase 1–4. If not, re-scope to Option B.

---

## 12. Open Questions (resolve before Phase 2)

1. **Planning gap:** Is disabling `planning_enabled` for chat acceptable during the migration, or must planning be preserved on the new path from day one? (Affects Phase 2 scope materially.)
2. **Persisted-checkpoint migration:** Migrate old `SQLiteMemorySaver` rows to the standard checkpointer format on first access, or require a clean break (new sessions only)?
3. **Title middleware:** Opt-in now, or defer entirely to avoid scope creep?
4. **Python floor:** Can the project stay on 3.11, or does LangChain 1.x force 3.12? (Affects the Dockerfile and CI matrices.)
