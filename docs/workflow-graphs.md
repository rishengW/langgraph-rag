# LangGraph RAG — Workflow Diagrams

## 1. Full RAG Graph (`build_graph`)

Used for **QA** and **Chat** modes with Chroma vectorstore retrieval, document grading, and query rewriting.

```mermaid
graph TD
    START((START)) --> condense["🔵 condense<br/><i>Condense chat history<br/>into standalone question</i>"]
    condense --> agent["🟢 agent<br/><i>LLM + tools<br/>Decide: answer or call tool?</i>"]

    START -->|"QA mode<br/>(skip condense)"| agent

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

### QA Mode (no condense)
```
START → agent → retrieve → grade → generate → END
                  ↑                    |
                  +—— rewrite ←———————+
```

### Chat Mode (with condense)
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

Used when **web_search_lightweight** is enabled and sources were discovered via web search.
Skips Chroma, embeddings, grading, and rewriting entirely.

```mermaid
graph TD
    START((START)) --> agent["🟢 agent<br/><i>LLM + live_web_search tool<br/>+ optional aux tools<br/>(weather, stock, wiki, etc.)</i>"]

    agent -->|"called live_web_search"| web_search["🌐 web_search<br/><i>ToolNode: execute<br/>live_web_search<br/>→ returns URL list</i>"]
    agent -->|"direct answer"| END1((END))
    agent -->|"called other tool<br/>(weather, stock, etc.)"| web_search

    web_search -->|"tool was live_web_search"| web_answer["📝 web_answer<br/><i>Fetch pages from URLs<br/>Build grounded prompt<br/>Synthesize answer</i>"]
    web_search -->|"tool was other<br/>(weather, stock, wiki)"| agent

    web_answer --> END2((END))

    style START fill:#555,stroke:#333,color:#fff
    style END1 fill:#555,stroke:#333,color:#fff
    style END2 fill:#555,stroke:#333,color:#fff
    style agent fill:#50C878,stroke:#2E8B57,color:#fff
    style web_search fill:#3498DB,stroke:#2471A3,color:#fff
    style web_answer fill:#F39C12,stroke:#D68910,color:#fff
```

### Edge Routing Logic

| Edge | Condition | Map |
|------|-----------|-----|
| `agent` → `web_search` | Agent called any tool | `LIGHTWEIGHT_AGENT_EDGE_MAP["tools"]` |
| `agent` → `END` | Agent answered directly | `LIGHTWEIGHT_AGENT_EDGE_MAP[END]` |
| `web_search` → `web_answer` | Tool was `live_web_search` | `route_after_lightweight_tool()` → `"web_answer"` |
| `web_search` → `agent` | Tool was weather/stock/wiki/etc. | `route_after_lightweight_tool()` → `"agent"` |

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
        LightGraph["Lightweight Graph<br/>agent → web_search<br/>→ web_answer"]
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

## 5. Graph Selection Logic

```mermaid
graph TD
    Request["Incoming Request<br/>(URLs + question)"] --> HasURLs{"Explicit URLs<br/>provided?"}

    HasURLs -->|"Yes"| FullViaRebuild["Full Graph<br/>(rebuild Chroma<br/>from URLs)"]
    HasURLs -->|"No"| WebEnabled{"web_search<br/>enabled?"}

    WebEnabled -->|"No"| FullDefaults["Full Graph<br/>(default Chroma<br/>collection)"]
    WebEnabled -->|"Yes"| Discover["Discover URLs<br/>via web search"]

    Discover --> FoundURLs{"URLs found?"}
    FoundURLs -->|"Yes"| Lightweight{"web_search<br/>lightweight?"}
    FoundURLs -->|"No"| FullDefaults

    Lightweight -->|"Yes"| LightGraph["Lightweight Graph"]
    Lightweight -->|"No"| FullViaRebuild

    style Request fill:#7F8C8D,stroke:#566573,color:#fff
    style HasURLs fill:#F4D03F,stroke:#D4AC0D,color:#333
    style WebEnabled fill:#F4D03F,stroke:#D4AC0D,color:#333
    style FoundURLs fill:#F4D03F,stroke:#D4AC0D,color:#333
    style Lightweight fill:#F4D03F,stroke:#D4AC0D,color:#333
    style FullViaRebuild fill:#27AE60,stroke:#1E8449,color:#fff
    style FullDefaults fill:#27AE60,stroke:#1E8449,color:#fff
    style LightGraph fill:#3498DB,stroke:#2471A3,color:#fff
```

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
