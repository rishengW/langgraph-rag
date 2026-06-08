# Code Review: Agent Web Search Not Enabled

**Date:** 2026-06-08  
**Branch:** `web-developed`  
**Reviewer:** Claude  
**Severity:** Medium — the agent cannot answer questions outside its indexed documents or training data.

---

## Summary

The agent returns "I don't know" when asked questions that require live web lookup (e.g., "What's the date today?"). The root cause is that **web search is never exposed as a LangChain tool that the agent can invoke at runtime**. The existing web search infrastructure is only used for one-time URL discovery *before* the graph is built — it seeds the retriever with source URLs, then the agent is limited to querying that static index.

---

## Finding 1 (Critical): Missing LangChain web search tool

**File:** `src/web_search/` (all modules)  
**Severity:** Critical

### What's there

The `src/web_search/` package has a well-designed provider system:

| Module | Purpose |
|---|---|
| `protocol.py` | `WebSearchProvider` protocol — `search(query) -> list[str]` (URLs) |
| `baidu.py` | `BaiduWebSearch` — scrapes Baidu search results |
| `duckduckgo.py` | `DuckDuckGoWebSearch` — DuckDuckGo API and HTML scraping |
| `factory.py` | `get_search_provider()` — instantiates by name |
| `discovery.py` | `discover_urls_from_web()` — returns URLs for indexing |
| `common.py` | URL normalization, noise filtering, HTTP helpers |

### What's missing

**None of these classes are wrapped as LangChain tools.** The agent in this codebase uses LangChain's `bind_tools()` mechanism (`src/graph/nodes/common.py:234`):

```python
model = new_chat_model(settings).bind_tools(tools)
```

This requires each tool to be a `langchain_core.tools.BaseTool` instance (e.g., created with the `@tool` decorator or `StructuredTool.from_function()`). A plain Python class with a `search()` method that returns `list[str]` will not work.

### Impact

The agent has no mechanism to search the live web during a conversation.

---

## Finding 2 (Critical): `_resolve_tools()` only builds the retriever

**File:** `src/graph/builder.py`, lines 154–167  
**Severity:** Critical

```python
def _resolve_tools(
    settings: Settings | None,
    providers: GraphProviders,
    rebuild_vectorstore: bool,
) -> list[Any]:
    if providers.tools is not None:
        return list(providers.tools)

    if settings is None:
        return []

    from ..core.retriever import build_retriever_tool

    return [build_retriever_tool(settings, rebuild=rebuild_vectorstore)]
    #      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #      ONLY the Chroma retriever tool. No web search tool.
```

### What happens

1. When `providers.tools` is `None` (the default in production paths), this function always returns exactly **one tool**: the `retrieve_source_documents` Chroma retriever.
2. The `settings.web_search_enabled` flag (default `True`, see `src/config/settings.py:41`) is **never consulted** by this function.
3. Even if a web search tool existed, it would never reach the agent because nothing adds it to the list.

### Contrast with the settings

`src/config/settings.py` already has all the configuration needed for web search:

```python
web_search_enabled: bool = True        # line 41 — unused by _resolve_tools
web_search_provider: str = "baidu"     # line 42
web_search_max_results: int = 20       # line 43
web_search_top_k: int = 3              # line 44
web_search_region: str = "wt-wt"       # line 45
web_search_timelimit: str | None = None  # line 46
web_search_verify_ssl: bool = True     # line 47
```

These settings are consumed by `discover_urls_from_web()` (a startup-only operation) but not by any runtime agent tool.

---

## Finding 3 (Moderate): `web_search_enabled` has misleading semantics

**Files:**
- `src/config/settings.py:41`
- `src/chat/main.py:116-118`
- `src/qa/main.py:213-217`

### What it does

`web_search_enabled` controls whether the CLI/API entry points perform a **one-time web search at graph build time** to discover source URLs:

```python
# src/chat/main.py:116-118
if urls is None and args.web_search and settings.web_search_enabled and args.seed_question.strip():
    found = discover_urls_from_web(args.seed_question.strip(), settings)
```

```python
# src/qa/main.py:212-213
if urls is None and args.web_search and settings.web_search_enabled:
    discovered_urls = discover_urls_from_web(args.question, settings)
```

### What the name implies

The name `web_search_enabled` strongly implies the **agent** can search the web. A developer reading the config would reasonably expect the agent to have a web search capability when this is `True`.

### Recommendation

Either rename it to `web_search_for_source_urls` to reflect its actual scope, or (preferably) wire it up so that when `True`, the agent actually gets a web search tool.

---

## Finding 4 (Low): Agent edge routing is hardcoded to a single tool node

**File:** `src/graph/edges.py`, lines 8–11

```python
AGENT_EDGE_MAP = {
    "tools": "retrieve",
    END: END,
}
```

When the agent emits a tool call, it always routes to the `"retrieve"` node, which is a `ToolNode` containing only the retriever tool. If a web search tool is added, a second `ToolNode` (e.g., `"web_search"`) or a combined `ToolNode` must be introduced, and the edge map must be updated accordingly. LangGraph's `ToolNode` can hold multiple tools and dispatches by name, so the simplest fix is to pass all tools to a single `ToolNode`.

---

## Finding 5 (Informational): Startup-only web search design

**Files:**
- `src/web_search/discovery.py`
- `src/chat/main.py:102-131`
- `src/qa/main.py:200-233`

### Current flow

```
User starts CLI
  → discover_urls_from_web() called ONCE at startup
  → URLs indexed into Chroma vectorstore
  → Graph compiled with retriever tool pointing at that index
  → Agent can only query the index, never the live web
```

This means:
- The agent is **blind to the live web** during a conversation.
- If the indexed documents don't contain the answer, the agent falls back to its training data (which has a cutoff) or says "I don't know."
- The user's `--web-search` CLI flag and `web_search_enabled` setting control the startup step only — not runtime behavior.

---

## Recommended Fix Plan

### Step 1: Create a LangChain web search tool

In `src/web_search/`, create a new module (e.g., `src/web_search/tool.py`) that:

```python
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web for current information. Returns formatted search results."""
    ...
```

The tool should call the existing `get_search_provider()` + `discover_urls_from_web()` and then fetch + summarize the page content, returning text the LLM can use.

### Step 2: Add the tool in `_resolve_tools()`

In `src/graph/builder.py`, modify `_resolve_tools()`:

```python
def _resolve_tools(settings, providers, rebuild_vectorstore):
    if providers.tools is not None:
        return list(providers.tools)
    if settings is None:
        return []

    tools = [build_retriever_tool(settings, rebuild=rebuild_vectorstore)]

    if settings.web_search_enabled:
        from ..web_search.tool import web_search as web_search_tool
        tools.append(web_search_tool)

    return tools
```

### Step 3: Update the edge map if needed

If using a single combined `ToolNode` (recommended), no changes to `edges.py` are needed — LangGraph's `ToolNode` handles multi-tool dispatch automatically. The existing `AGENT_EDGE_MAP` (`"tools" → "retrieve"`) would continue to work if the `"retrieve"` `ToolNode` holds all tools.

Alternatively, rename the node to something more generic (e.g., `"tools"`) and update `AGENT_EDGE_MAP` accordingly.

---

## Affected Files

| File | Change needed |
|---|---|
| `src/web_search/tool.py` | **New file** — LangChain `@tool` wrapping web search |
| `src/web_search/__init__.py` | Export the new tool |
| `src/graph/builder.py:154-167` | Add web search tool when `web_search_enabled` is `True` |
| `src/graph/edges.py:8-11` | Possibly update routing if a second `ToolNode` is introduced |
| `src/graph/builder.py:107-112` | Ensure `ToolNode` receives all tools, not just retriever |

---

## Verification

After the fix, the following should work:

1. Ask the agent: *"What's the date today?"* → Agent calls `web_search` tool → returns current date.
2. Ask the agent: *"What's the latest news about AI?"* → Agent calls `web_search` tool → returns current information.
3. Ask the agent about indexed documents → Agent still calls `retrieve_source_documents` → returns document-based answer.
4. Verify that setting `web_search_enabled: false` removes the web search tool but keeps the retriever.
