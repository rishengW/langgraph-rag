---
name: rag-pipeline-architect
description: >
  Use this skill whenever working on the RAG execution pipeline — graph topology,
  node factories, state management, question resolution, graph execution, or
  streaming events. Covers LangGraph StateGraph assembly, the agent→retrieve→
  grade→rewrite→generate loop, unified RAGState, and the QuestionResolver strategy
  for eliminating node duplication. Trigger on mentions of graph, nodes, state,
  agent, grade, rewrite, generate, build_graph, run_rag_query, or RAGState.
---

# RAG Pipeline Architect — src/graph/

Domain: graph topology, node factories, state management, question resolution, graph execution.
Parent: `SKILL.md` (root). Siblings: `src/rag/SKILL.md`, `src/web_search/SKILL.md`, `src/api/SKILL.md`, `src/sessions/SKILL.md`.

## Quick Reference

| Fact | Value |
|---|---|
| Graph type | LangGraph `StateGraph[RAGState]` |
| Nodes | agent, retrieve (ToolNode), grade_documents, rewrite, generate |
| LLM | `ChatTongyi` (`qwen-plus`) via DashScope |
| Max rewrites | 2 (configurable via `max_rewrites`) |
| Source files (current) | `core/graph.py`, `core/nodes.py`, `core/state.py`, `core/graph_executor.py` |
| Target location | `src/graph/` |

## File Map (Current)

```
src/core/
+-- state.py                # AgentState TypedDict (messages + rewrite_count) — 23 LOC
+-- nodes.py                # 4 node factories + retry logic + extractive fallback — 429 LOC
+-- graph.py                # build_graph() — StateGraph assembly (5 nodes) — 53 LOC
+-- graph_executor.py       # run_rag_query() — convenience wrapper — 108 LOC

src/chat/
+-- state.py                # ChatState (adds current_question, current_question_index) — 33 LOC
+-- nodes.py                # 5 node factories (adds condense_question), copies core/ node logic — 348 LOC
+-- graph.py                # build_chat_graph() — condense + MemorySaver — 93 LOC
```

### Detailed File Responsibilities

| File | Responsibility | Key Symbols | LOC |
|------|---------------|-------------|-----|
| `state.py` (core) | `AgentState` TypedDict | `AgentState` | 23 |
| `state.py` (chat) | `ChatState` (adds `current_question`, `current_question_index`) | `ChatState` | 33 |
| `nodes.py` (core) | 4 node factories + retry + extractive fallback + SSL config + prompt templates | `agent_factory`, `grade_documents_factory`, `rewrite_factory`, `generate_factory`, `_invoke_with_retry` | 429 |
| `nodes.py` (chat) | 5 node factories (adds `condense_question_factory`, duplicates 4 from core/) | Same as core + `condense_question_factory` | 348 |
| `graph.py` (core) | StateGraph assembly with 5 nodes + conditional edges | `build_graph` | 53 |
| `graph.py` (chat) | Chat graph with condense + MemorySaver checkpointer | `build_chat_graph`, `_build_memory_saver` | 93 |
| `graph_executor.py` | `run_rag_query()` convenience wrapper | `run_rag_query` | 108 |

## Graph Flow

### QA Graph (`build_graph()`)

```
START -> agent -> [tools_condition]
                    +-- "tools" -> retrieve -> [grade_documents]
                    |                          +-- "generate" -> generate -> END
                    |                          +-- "rewrite" -> rewrite -> agent (loop)
                    +-- END
```

### Chat Graph (`build_chat_graph()`)

```
START -> condense -> agent -> [tools_condition]
                              +-- "tools" -> retrieve -> [grade_documents]
                              |                          +-- "generate" -> generate -> END
                              |                          +-- "rewrite" -> rewrite -> agent (loop)
                              +-- END
```

### How Chat Differs from QA

| Aspect | QA Graph | Chat Graph |
|--------|----------|------------|
| Entry | START -> agent | START -> condense -> agent |
| Question source | `messages[0].content` | `state["current_question"]` (set by condense) |
| Checkpointer | None | `MemorySaver` (per thread_id) |
| State type | `AgentState` | `ChatState` (adds current_question, current_question_index) |
| Rewrite budget reset | Never (single turn) | Per turn (condense sets rewrite_count=0) |

### Node Descriptions

| Node | Factory Function | Behavior |
|------|-----------------|----------|
| `agent` | `agent_factory(settings, tools)` | LLM (`ChatTongyi`) with bound retriever tool; decides to call tool or answer directly |
| `retrieve` | `ToolNode(tools)` | Built-in LangGraph `ToolNode` wrapping Chroma retriever tool |
| `grade_documents` | `grade_documents_factory(settings)` | Structured output LLM grader (binary yes/no + explanation), with keyword fallback |
| `rewrite` | `rewrite_factory(settings)` | LLM rewrites the query for better retrieval |
| `generate` | `generate_factory(settings)` | RAG prompt + LLM + `StrOutputParser` |

## Key Design Patterns

### 1. Factory Pattern for Nodes

All nodes are created by factory functions that capture `Settings` as a closure:

```python
def agent_factory(settings: Settings, tools) -> Callable:
    def agent(state) -> dict: ...
    return agent
```

**Limitation:** Settings is a monolithic dependency. Nodes only need subsets:
- `agent` — LLM model name, timeout, retries, tools
- `grade` — LLM, min_keyword_matches, max_rewrites
- `rewrite` — LLM
- `generate` — LLM, RAG prompt

**Target (Phase 2):** DI via protocols. Nodes receive only what they need.

### 2. Question Resolution — Root Cause of Node Duplication

QA mode: `messages[0].content` is always the user question.
Chat mode: `_question_from_state()` reads `current_question` with fallback chain.

This is why `chat/nodes.py` duplicates 4 of 5 node factories from `core/nodes.py`.

**Target (Phase 1):** Inject `QuestionResolver` callable into shared node factories:

```python
class QuestionResolver(Protocol):
    def __call__(self, state: dict) -> str: ...

# QA resolver
def qa_resolver(state) -> str:
    return _message_text(state["messages"][0])

# Chat resolver
def chat_resolver(state) -> str:
    return (state.get("current_question") or
            _message_text(state["messages"][state.get("current_question_index", -1)]))
```

### 3. Grader Decision Logic

```
LLM grades context relevance -> binary "yes"/"no"
  +-- "yes" -> generate
  +-- "no" ->
      +-- LLM failed + context present -> generate (skip rewrite)
      +-- allow_low_relevance + keyword_matches >= threshold -> generate
      +-- rewrite_count >= max_rewrites -> generate (budget exhausted)
      +-- otherwise -> rewrite
```

### 4. Fallback Chain for Generation

When the LLM is unreachable during answer generation:
1. Try LLM with retry (up to `max_retries` attempts)
2. On failure -> `_build_extractive_answer(question, context)`
3. Extractive: keyword-token matching against retrieved sentences, top-5 by relevance
4. If context is empty -> generic "I could not reach DashScope" message

### 5. LLM Model Creation

```python
# nodes.py:_new_chat_model()
def _new_chat_model(settings: Settings) -> ChatTongyi:
    model_kwargs = {
        "request_timeout": settings.dashscope_request_timeout,
    }
    if settings.dashscope_http_base_url:
        model_kwargs["base_address"] = settings.dashscope_http_base_url
    return ChatTongyi(
        model=settings.qwen_model,
        max_retries=settings.dashscope_max_retries,
        model_kwargs=model_kwargs,
    )
```

## State Definition

### Current State (Two Types)

```python
# core/state.py — QA
class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rewrite_count: int

# chat/state.py — Chat
class ChatState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rewrite_count: int
    current_question: str        # standalone question from condense node
    current_question_index: int  # index of the user message in messages
```

### Target: Unified RAGState (Phase 1)

```python
class RAGState(TypedDict, total=False):
    """Unified state for both QA and Chat modes."""

    # Core message history
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Rewrite loop control
    rewrite_count: int
    max_rewrites: int

    # Chat-specific (absent in QA mode)
    current_question: str
    current_question_index: int

    # Metadata (read-only after initial injection)
    source_urls: list[str]
    source_mode: Literal["explicit", "web_search", "defaults"]
    source_note: str | None

    # Observability
    errors: list[str]
```

## Known Issues

### Priority 1 — Fix Immediately

| # | Issue | File | Fix |
|---|-------|------|-----|
| 1 | **Node factories duplicated ~70%** between core/nodes.py and chat/nodes.py | Both 4xx-line files | Inject `QuestionResolver`; share factories |
| 2 | **`print()` statements** in graph nodes (not structured logging) | `nodes.py:232,279,285,289,294,301` | Replace with `logger.info()` |
| 3 | **AgentState + ChatState divergence** — should be one unified type | Two separate files | Merge into `RAGState` |

### Priority 2 — Structural

| # | Issue | Notes |
|---|-------|-------|
| 4 | `RAG_PROMPT` embedded in `nodes.py:211` alongside generation logic | Extract to `prompts.py` (Phase 1) |
| 5 | `_invoke_with_retry` duplicates retry logic also in `embeddings.py` | Consolidate to `utils/retry.py` (Phase 1) |
| 6 | `_message_text` and `_question_from_state` have subtle type handling | Document edge cases; add unit tests |
| 7 | `run_rag_query()` duplicates final-answer extraction logic | Extract from `graph_executor.py` |
| 8 | `_configure_ssl()` runs at module import time | Move to explicit init function |

## Refactoring Target — src/graph/

```
src/graph/
+-- __init__.py
+-- state.py              # Unified RAGState (replaces AgentState + ChatState)
+-- builder.py            # Single build_graph(mode="qa"|"chat", providers, config)
+-- executor.py           # AsyncRAGExecutor (streaming + batch)
+-- edges.py              # Conditional edge logic (grade decision, tools_condition)
+-- nodes/
    +-- __init__.py
    +-- agent.py          # agent_factory(resolver, llm_provider, tools)
    +-- condense.py       # condense_factory(resolver, llm_provider) — moved from chat
    +-- grade.py          # grade_factory(resolver, llm_provider, config)
    +-- rewrite.py        # rewrite_factory(resolver, llm_provider)
    +-- generate.py       # generate_factory(resolver, llm_provider, prompt)
```

## Refactoring To-Do List

> Source: [`REFACTORING_PLAN.md`](../../REFACTORING_PLAN.md). RAG pipeline scope items.

### Phase 1 — Extract Without Behavioral Change

- [ ] **1.5 Unify node libraries** — eliminate 4 duplicated factories between `core/nodes.py` and `chat/nodes.py`
  - [ ] Create `src/graph/nodes/agent.py`, `grade.py`, `rewrite.py`, `generate.py` as shared implementations
  - [ ] Inject `QuestionResolver: Callable[[dict], str]` into each factory
  - [ ] QA resolver reads `messages[0].content`; Chat resolver reads `state["current_question"]` with fallback
  - [ ] `core/nodes.py` → thin re-export shim with QA resolver
  - [ ] `chat/nodes.py` → thin re-export shim with Chat resolver (keeps only `condense_question_factory`)
  - [ ] Verify: both QA and chat produce identical answers to test queries
- [ ] **1.5 Create `src/graph/state.py`** — unified `RAGState` replacing `AgentState` + `ChatState`
  - [ ] `current_question` and `current_question_index` become optional fields
- [ ] **1.3 Replace `print()` with `logging`** — 7 print statements in `nodes.py` (lines 232, 279, 285, 289, 294, 301)
- [ ] **1.6 Extract prompt templates** — `RAG_PROMPT`, grade prompt template → `src/llm/prompts.py`

### Phase 2 — Interfaces & Abstractions

- [ ] **2.2 DI in graph builder** — `build_graph()` accepts `(retriever, llm, config)` not `(settings: Settings)`
  - [ ] Node factories receive only their needed dependencies, not full Settings object
  - [ ] Graph assembly becomes environment-agnostic; testable with mock providers
- [ ] **2.4 Typed event system** — wire event emission into node callbacks
  - [ ] `NodeStartEvent`, `NodeEndEvent`, `TokenEvent`, `DoneEvent` — emitted during graph execution
  - [ ] `graph/executor.py` supports two modes: `run()` (sync) and `stream()` (async generator of events)
- [ ] **2.7 Consolidate graphs** — single `build_graph(mode="qa"|"chat", providers, config)` in `src/graph/builder.py`
  - [ ] `mode="qa"` — no condense node, no checkpointer
  - [ ] `mode="chat"` — condense node + `MemorySaver` checkpointer
  - [ ] `core/graph.py` and `chat/graph.py` → thin re-export shims
- [ ] **2.7 Extract `src/graph/edges.py`** — conditional edge logic (grade decision, tools_condition)

### Phase 3 — Streaming & Optimization

- [ ] **3.1 Token streaming** — generate node emits `TokenEvent` per token
- [ ] **3.5 Deprecation shims** — `src/core/graph.py`, `src/core/nodes.py`, `src/core/state.py` → pure re-exports with `DeprecationWarning`

## Testing Strategy

| Layer | What to test | Mock strategy |
|-------|-------------|---------------|
| Node factories | Correct state updates, edge decisions | Mock LLM returns fixed responses |
| Grade logic | Relevance decision rules, budget exhaustion | Mock grader LLM |
| Graph topology | Node ordering, edge conditions, rewrite loop cap | Mock all providers |
| Graph executor | Output extraction, error handling | Mock compiled graph |
| Question resolver | QA vs Chat resolution correctness | Both state shapes |
| Full pipeline | Regression: known Q&A pairs produce identical output | Mock LLM + mock retriever |
