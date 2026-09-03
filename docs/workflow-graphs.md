# LangGraph RAG — Workflow Diagrams

## 1. Full RAG Graph (`build_graph`)

Used for **chat** with Chroma vectorstore retrieval, document grading, and query rewriting.

```mermaid
graph TD
    START((START)) --> condense["🔵 condense<br/><i>Condense chat history<br/>into standalone question</i>"]
    condense --> agent["🟢 agent<br/><i>LLM + tools<br/>Decide: answer or call tool?</i>"]

    agent -->|"tool called"| retrieve["📚 retrieve<br/><i>ToolNode: execute tool<br/>(Chroma search, web search,<br/>weather, stock, etc.)</i>"]
    agent -->|"direct answer"| END((END))

    retrieve --> grade["🔍 grade<br/><i>LLM grades relevance<br/>of retrieved docs</i>"]

    grade -->|"✅ relevant"| generate["💬 generate<br/><i>LLM synthesizes answer<br/>from ranked context</i>"]
    grade -->|"❌ not relevant<br/>& rewrites remain"| rewrite["🔄 rewrite<br/><i>LLM rephrases query<br/>to improve retrieval</i>"]

    rewrite --> agent

    generate --> END

    style START fill:#555,stroke:#333,color:#fff
    style END fill:#555,stroke:#333,color:#fff
    style condense fill:#4A90D9,stroke:#2E6BB5,color:#fff
    style agent fill:#50C878,stroke:#2E8B57,color:#fff
    style retrieve fill:#E8A317,stroke:#B8860B,color:#fff
    style grade fill:#9B59B6,stroke:#7D3C98,color:#fff
    style rewrite fill:#E74C3C,stroke:#C0392B,color:#fff
    style generate fill:#1ABC9C,stroke:#16A085,color:#fff
```

### Execution Path
```
START → condense → agent → retrieve → grade → generate → END
                      ↑                    |
                      +—— rewrite ←———————+
```

### Edge Routing Logic

| Edge | Condition | Map |
|------|-----------|-----|
| `agent` → `retrieve` | Agent called a tool | `tools_condition()` → `"tools"` |
| `agent` → `END` | Agent answered directly | `tools_condition()` → `END` |
| `grade` → `generate` | Docs relevant OR rewrite budget exhausted | `GRADE_EDGE_MAP["generate"]` |
| `grade` → `rewrite` | Docs not relevant + budget remains | `GRADE_EDGE_MAP["rewrite"]` |
| `rewrite` → `agent` | Always (fixed edge) | — |

### Rewrite Loop
The `rewrite_count` in `RAGState` limits rewrites (default `max_rewrites=1`).
When the budget is exhausted, the graph forces `grade → generate` even if documents
are low-quality. The `allow_low_relevance_generate` setting also gates this.

---

## 2. Lightweight Graph (`build_lightweight_graph`)

Used when **web_search_lightweight** is enabled. In chat mode the compiled graph
owns discovery; the API does not search or rebuild the graph before each turn.
Skips Chroma, embeddings, grading, and the full-graph retrieval rewrite loop.
Provider-query LLM rewriting remains an optional, disabled-by-default setting.

```mermaid
graph TD
    START((START)) --> agent["🟢 agent<br/><i>LLM + live_web_search tool<br/>+ optional aux tools<br/>(weather, stock, wiki, etc.)</i>"]

    agent -->|"called live_web_search"| decompose["decompose<br/><i>Atomic passthrough or<br/>1-3 sub-questions</i>"]
    agent -->|"direct answer"| END1((END))
    agent -->|"called other tool<br/>(weather, stock, etc.)"| tool_node["ToolNode"]

    decompose --> search_queries["🌐 search_queries<br/><i>≤6 queries<br/>≤3 concurrent</i>"]
    search_queries --> merge["merge<br/><i>Dedupe + relevance,<br/>quality, overlap rank</i>"]
    merge --> web_answer["📝 web_answer<br/><i>Fetch pages from URLs<br/>Build grounded prompt<br/>Synthesize answer</i>"]
    tool_node --> agent

    web_answer -->|"readable sources"| END2((END))
    web_answer -->|"first fetch failure"| expand["expand<br/><i>bounded query variants</i>"]
    expand --> search_queries
    web_answer -->|"expanded fetch failure"| fallback_answer["fallback_answer<br/><i>Tool-free model call<br/>with verification caveat</i>"]
    fallback_answer --> END3((END))

    style START fill:#555,stroke:#333,color:#fff
    style END1 fill:#555,stroke:#333,color:#fff
    style END2 fill:#555,stroke:#333,color:#fff
    style END3 fill:#555,stroke:#333,color:#fff
    style agent fill:#50C878,stroke:#2E8B57,color:#fff
    style search_queries fill:#3498DB,stroke:#2471A3,color:#fff
    style web_answer fill:#F39C12,stroke:#D68910,color:#fff
```

### Edge Routing Logic

| Edge | Condition | Map |
|------|-----------|-----|
| `agent` → `decompose` | Agent called only `live_web_search` | `route_after_lightweight_agent()` → `"decompose"` |
| `agent` → `END` | Agent answered directly | `LIGHTWEIGHT_AGENT_EDGE_MAP[END]` |
| `agent` → `web_search` | Agent called another or multiple tools | `route_after_lightweight_agent()` → `"web_search"` |
| `decompose` → `search_queries` | Fixed edge | bounded provider fan-out |
| `search_queries` → `merge` → `web_answer` | Fixed edges | relevance-aware rank, fetch, synthesize |
| `web_search` → `agent` | Tool was weather/stock/wiki/etc. | `route_after_lightweight_tool()` → `"agent"` |
| `web_answer` → `expand` | First attempt has no readable, relevant source | one bounded expanded retry |
| `web_answer` → `fallback_answer` | Expanded retry also fails | one tool-free fallback call |
| `fallback_answer` → `END` | Always (fixed edge) | prevents another tool/search loop |

The graph owns the entire lightweight chat search. It runs at most six distinct
queries with at most three provider calls in flight, then merges provider
title/snippet relevance and URL quality before overlap and provider rank. There
are exactly two search/fetch attempts: the original batch and one expanded
batch. Deterministic query cleanup is the default; optional LLM query rewriting
is enabled only by `WEB_SEARCH_LLM_QUERY_REWRITE_ENABLED=true`.

Ordinary queries try the configured provider first, followed by any missing
providers in Bing → Baidu → DuckDuckGo order. Predominantly Chinese queries
prefer Baidu → Bing → DuckDuckGo.

### Auxiliary Tool Loop
When the agent calls a non-web-search tool (weather, stock, currency, Wikipedia), the
output routes back to `agent` so the LLM sees the structured result and synthesizes
a final answer. The `web_answer` node is only used for web page grounding.

---

## 3. System Architecture Overview

```mermaid
graph TB
    subgraph "Clients"
        CLI["💻 CLI<br/>(qa query / chat repl)"]
        Browser["🌐 Browser<br/>(static UI)"]
    end

    subgraph "FastAPI Apps"
        QA_API["/query<br/>/query/stream<br/>Single-shot QA"]
        Chat_API["/chat<br/>/chat/{id}/message<br/>Multi-turn Chat"]
    end

    subgraph "Graph Layer"
        GraphExec["GraphExecutor<br/>SSE event stream"]
        FullGraph["Full Graph<br/>agent → retrieve → grade<br/>→ generate [↻ rewrite]"]
        LightGraph["Lightweight Graph<br/>agent → bounded search<br/>→ merge → web_answer"]
    end

    subgraph "Providers"
        LLM["LLM<br/>DashScope/Qwen<br/>or DeepSeek"]
        Embed["Embeddings<br/>DashScope/Tongyi<br/>or HuggingFace"]
        Search["Web Search<br/>Bing / Baidu<br/>/ DuckDuckGo"]
    end

    subgraph "Storage"
        Chroma[("Chroma<br/>Vectorstore")]
        SQLite[("SQLite<br/>Sessions +<br/>Checkpoints")]
    end

    subgraph "Tools"
        RetrieverTool["Chroma Retriever"]
        WebTool["live_web_search"]
        AuxTools["weather / stock<br/>currency / wiki"]
    end

    CLI --> QA_API
    CLI --> Chat_API
    Browser --> QA_API
    Browser --> Chat_API

    QA_API --> GraphExec
    Chat_API --> GraphExec
    GraphExec --> FullGraph
    GraphExec --> LightGraph

    FullGraph --> LLM
    FullGraph --> RetrieverTool
    FullGraph --> WebTool
    FullGraph --> AuxTools

    LightGraph --> LLM
    LightGraph --> WebTool
    LightGraph --> AuxTools

    RetrieverTool --> Chroma
    RetrieverTool --> Embed
    WebTool --> Search

    Chat_API --> SQLite
    FullGraph -.->|"chat mode<br/>checkpoints"| SQLite

    style CLI fill:#7F8C8D,stroke:#566573,color:#fff
    style Browser fill:#7F8C8D,stroke:#566573,color:#fff
    style QA_API fill:#2980B9,stroke:#1A5276,color:#fff
    style Chat_API fill:#2980B9,stroke:#1A5276,color:#fff
    style GraphExec fill:#8E44AD,stroke:#6C3483,color:#fff
    style FullGraph fill:#27AE60,stroke:#1E8449,color:#fff
    style LightGraph fill:#27AE60,stroke:#1E8449,color:#fff
    style LLM fill:#E67E22,stroke:#CA6F1E,color:#fff
    style Embed fill:#E67E22,stroke:#CA6F1E,color:#fff
    style Search fill:#E67E22,stroke:#CA6F1E,color:#fff
    style Chroma fill:#1ABC9C,stroke:#16A085,color:#fff
    style SQLite fill:#1ABC9C,stroke:#16A085,color:#fff
```

---

## 4. Tool Binding Matrix

Which tools are bound to the agent in each graph:

| Tool | Full Graph | Lightweight Graph | Requires Flag |
|------|:----------:|:-----------------:|:-------------:|
| Chroma Retriever | ✅ | ❌ | — |
| `live_web_search` | ✅ | ✅ | `web_search_enabled` |
| `get_weather` | ✅ | ✅ | `weather_enabled` |
| `get_stock_quote` | ✅ | ✅ | `stock_enabled` |
| `convert_currency` | ✅ | ✅ | `currency_enabled` |
| `search_wikipedia` | ✅ | ✅ | `wikipedia_enabled` |

---

## 5. Chat Graph Selection Logic

```mermaid
graph TD
    Request["Incoming Request<br/>(URLs + question)"] --> HasURLs{"Explicit URLs<br/>provided?"}

    HasURLs -->|"Yes"| FullViaRebuild["Full Graph<br/>(rebuild Chroma<br/>from URLs)"]
    HasURLs -->|"No"| WebEnabled{"web_search<br/>enabled?"}

    WebEnabled -->|"No"| FullDefaults["Full Graph<br/>(default Chroma<br/>collection)"]
    WebEnabled -->|"Yes"| Lightweight{"web_search<br/>lightweight?"}

    Lightweight -->|"Yes"| LightGraph["Lightweight Graph<br/>graph-owned search per turn"]
    Lightweight -->|"No"| Discover["Discover URLs<br/>before full-graph build"]

    Discover --> FoundURLs{"URLs found?"}
    FoundURLs -->|"Yes"| FullViaRebuild
    FoundURLs -->|"No"| FullDefaults

    style Request fill:#7F8C8D,stroke:#566573,color:#fff
    style HasURLs fill:#F4D03F,stroke:#D4AC0D,color:#333
    style WebEnabled fill:#F4D03F,stroke:#D4AC0D,color:#333
    style FoundURLs fill:#F4D03F,stroke:#D4AC0D,color:#333
    style Lightweight fill:#F4D03F,stroke:#D4AC0D,color:#333
    style FullViaRebuild fill:#27AE60,stroke:#1E8449,color:#fff
    style FullDefaults fill:#27AE60,stroke:#1E8449,color:#fff
    style LightGraph fill:#3498DB,stroke:#2471A3,color:#fff
```

Lightweight chat does not perform a preliminary provider search. It compiles
the graph once, searches only after the agent calls `live_web_search`, and
persists the resulting source URLs without rebuilding Chroma or the graph.

---

## 6. Chat Session Lifecycle

```mermaid
sequenceDiagram
    participant Client
    participant ChatAPI as Chat API
    participant Registry as Session Registry
    participant SQLite as SQLite Storage
    participant Graph as LangGraph

    Client->>ChatAPI: POST /chat {urls, seed_question}
    ChatAPI->>Registry: create_session()
    Registry->>SQLite: persist metadata
    ChatAPI->>Graph: build_graph(thread_id)
    ChatAPI-->>Client: {thread_id}

    loop Each turn
        Client->>ChatAPI: POST /chat/{id}/message {message}
        ChatAPI->>Registry: get_session(thread_id)
        ChatAPI->>Graph: invoke(state, config={thread_id})
        Graph->>SQLite: save checkpoint
        Graph-->>ChatAPI: answer + events
        ChatAPI-->>Client: SSE stream / JSON response
    end

    Client->>ChatAPI: DELETE /chat/{id}
    ChatAPI->>Registry: remove_session()
    ChatAPI->>SQLite: delete metadata
    ChatAPI-->>Client: 204 No Content
```

The checkpoint retains the full transcript for `/history` and restart recovery.
Only the recent projection sent to model calls is bounded by
`CHAT_CONTEXT_MAX_TURNS` and `CHAT_CONTEXT_MAX_CHARS`.
