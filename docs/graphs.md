# LangGraph RAG — Node Graphs

## Full RAG Graph

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD
    __start__([START]):::first
    agent("🟢 agent<br/>LLM decides: answer or use tool?"):::agent
    retrieve("📚 retrieve<br/>ToolNode: execute tool call"):::retrieve
    rewrite("🔄 rewrite<br/>Reformulate query"):::rewrite
    generate("💬 generate<br/>Synthesize final answer"):::generate
    __end__([END]):::last

    __start__ --> agent
    agent -.->|"direct answer"| __end__
    agent -.->|"tool called"| retrieve
    retrieve -.->|"docs relevant"| generate
    retrieve -.->|"docs irrelevant + budget left"| rewrite
    rewrite --> agent
    generate --> __end__

    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
    classDef agent fill:#50C878,stroke:#2E8B57,color:#fff
    classDef retrieve fill:#E8A317,stroke:#B8860B,color:#fff
    classDef rewrite fill:#E74C3C,stroke:#C0392B,color:#fff
    classDef generate fill:#1ABC9C,stroke:#16A085,color:#fff
```

**Flow:** `START → agent → retrieve → generate → END` (with `rewrite → agent` loop)

---

## Full RAG Graph — Chat Mode

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD
    __start__([START]):::first
    condense("🔵 condense<br/>Condense chat history<br/>into standalone question"):::condense
    agent("🟢 agent<br/>LLM decides: answer or use tool?"):::agent
    retrieve("📚 retrieve<br/>ToolNode: execute tool call"):::retrieve
    rewrite("🔄 rewrite<br/>Reformulate query"):::rewrite
    generate("💬 generate<br/>Synthesize final answer"):::generate
    __end__([END]):::last

    __start__ --> condense
    condense --> agent
    agent -.->|"direct answer"| __end__
    agent -.->|"tool called"| retrieve
    retrieve -.->|"docs relevant"| generate
    retrieve -.->|"docs irrelevant + budget left"| rewrite
    rewrite --> agent
    generate --> __end__

    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
    classDef condense fill:#4A90D9,stroke:#2E6BB5,color:#fff
    classDef agent fill:#50C878,stroke:#2E8B57,color:#fff
    classDef retrieve fill:#E8A317,stroke:#B8860B,color:#fff
    classDef rewrite fill:#E74C3C,stroke:#C0392B,color:#fff
    classDef generate fill:#1ABC9C,stroke:#16A085,color:#fff
```

**Flow:** `START → condense → agent → retrieve → generate → END` (with `rewrite → agent` loop)

---

## Lightweight Web-Search Graph

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD
    __start__([START]):::first
    agent("🟢 agent<br/>LLM + live_web_search<br/>+ optional aux tools"):::agent
    web_search("🌐 web_search<br/>ToolNode: execute tool"):::websearch
    web_answer("📝 web_answer<br/>Fetch pages, build prompt,<br/>synthesize grounded answer"):::webanswer
    __end__([END]):::last

    __start__ --> agent
    agent -.->|"direct answer"| __end__
    agent -.->|"tool called"| web_search
    web_search -.->|"live_web_search"| web_answer
    web_search -.->|"weather/stock/wiki"| agent
    web_answer --> __end__

    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
    classDef agent fill:#50C878,stroke:#2E8B57,color:#fff
    classDef websearch fill:#3498DB,stroke:#2471A3,color:#fff
    classDef webanswer fill:#F39C12,stroke:#D68910,color:#fff
```

**Flow:** `START → agent → web_search → web_answer → END`

---

## Graph Selection Decision Tree

```mermaid
flowchart TD
    A["Request: URLs + question"] --> B{"Explicit URLs?"}
    B -->|Yes| C["Full Graph<br/>(rebuild Chroma)"]
    B -->|No| D{"web_search enabled?"}
    D -->|No| E["Full Graph<br/>(default Chroma)"]
    D -->|Yes| F["Discover URLs<br/>via web search"]
    F --> G{"URLs found?"}
    G -->|Yes| H{"lightweight mode?"}
    G -->|No| E
    H -->|Yes| I["Lightweight Graph"]
    H -->|No| C
```

---

## Node Summary

| Node | Full Graph | Lightweight | Role |
|------|:----------:|:-----------:|------|
| `condense` | ✅ (chat) | ❌ | Rewrite multi-turn chat into standalone question |
| `agent` | ✅ | ✅ | LLM reasons and decides: answer directly or call tool |
| `retrieve` | ✅ | ❌ | Execute tool (Chroma search → return docs) |
| `grade` | ✅ | ❌ | LLM grades doc relevance → route to generate or rewrite |
| `rewrite` | ✅ | ❌ | LLM reformulates query when docs are irrelevant |
| `generate` | ✅ | ❌ | LLM synthesizes answer from retrieved context |
| `web_search` | ❌ | ✅ | Execute live_web_search tool → return URLs |
| `web_answer` | ❌ | ✅ | Fetch pages, build grounded prompt, synthesize |

## Edge Summary

| From | To | Graph | Condition |
|------|----|-------|-----------|
| `START` | `condense` | Full | Always |
| `START` | `agent` | Lightweight | No condense node |
| `condense` | `agent` | Full | Always |
| `agent` | `retrieve` | Full | Agent called a tool |
| `agent` | `END` | Both | Agent answered directly |
| `agent` | `web_search` | Lightweight | Agent called a tool |
| `retrieve` | `generate` | Full | Grade says docs are relevant |
| `retrieve` | `rewrite` | Full | Grade says docs NOT relevant |
| `rewrite` | `agent` | Full | Always (loop back) |
| `generate` | `END` | Full | Always |
| `web_search` | `web_answer` | Lightweight | Tool was `live_web_search` |
| `web_search` | `agent` | Lightweight | Tool was other (weather/stock/wiki) |
| `web_answer` | `END` | Lightweight | Always |
