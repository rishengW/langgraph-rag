---
name: web-search-refactor-architect
description: >
  Use this skill when refactoring how web search integrates with the RAG pipeline.
  Covers the mismatch between one-shot web search and the persistent-vectorstore
  architecture, proposes a lightweight "fetch → extract → prompt" path, and maps
  every module that must change. Trigger on mentions of web search refactoring,
  removing Chroma dependency for web search, speeding up web search, or
  re-architecting the search→answer pipeline.
---

# Web Search Refactor Architect — refactor/

Domain: re-architecting web search away from the heavy vector-store pipeline toward
a lightweight direct-context path, while preserving the persistent-index path for
static/docs use cases.

Parent: [[system-architect]]. Related: [[web-search-architect]], [[rag-pipeline-architect]], [[knowledge-retrieval-architect]].

## Quick Reference

| Fact | Value |
|---|---|
| Current web search cost | Dominated by embed + Chroma build for a single-use index (URL fetch is already parallel) |
| Target web search path | URL fetch + extract + direct LLM prompt (no embed, no Chroma) |
| Root problem | Web search is shoehorned into a pipeline designed for persistent document corpora |
| Key insight | For one-shot queries, the vector store adds cost but no value — the LLM can read the pages directly |
| Context budget | qwen-plus: 131K tokens, deepseek-v4-pro: similar or larger — plenty for 3–6 web pages |

> **Accuracy note (2026-06-09):** `src/rag/document_loader.py` **already** fetches
> URLs concurrently (`ThreadPoolExecutor`, `max_concurrent_loads=4`) with TTL
> caching and order preservation. Earlier drafts of this doc claimed sequential
> 60–90s fetching — that is no longer true. The real one-shot overhead is
> embedding + Chroma index build, which the lightweight path removes. The new
> `content_fetcher.py` should **reuse** `load_source_documents` rather than
> reimplement parallel fetch.

---

## The Core Problem

### What This Project Was Built For

The original RAG pipeline was designed for **static documentation URLs** — Aliyun
help pages listed in `DEFAULT_URLS`. For that use case, the flow makes sense:

```
Static URLs → Fetch once → Split → Embed → Persist in Chroma → Query many times
```

The vector store amortizes the indexing cost across many queries. Chroma is a good fit.

### What Web Search Actually Needs

When the user has no URLs, the system discovers them via web search. But those
URLs are **single-use** — they're fetched, embedded, indexed, queried once, then
the Chroma collection is discarded. Every step after fetching the page text is waste:

```
Web search → URLs → Fetch pages → Split → Embed (API calls!) → Build Chroma
→ Retrieve top-K → Grade → Generate

                    ↑ All of this is one-shot overhead ↑
```

### The Mismatch in One Diagram

```
                     STATIC DOCS PATH (keep)
                     ══════════════════════
                     Load once, index, query N times
                     Vector store amortizes cost

  Source URLs  ──→  Fetch  ──→  Split  ──→  Embed  ──→  Chroma  ──→  Retrieve  ──→  Generate
                                                              ↑
                     WEB SEARCH PATH (current — broken)       │
                     ═══════════════════════════════════      │
                     Discover URLs, then force through        │
                     the same heavy pipeline. Index is        │
                     queried ONCE then thrown away.           │

                     WEB SEARCH PATH (target — lightweight)
                     ═══════════════════════════════════════
                     Discover URLs, fetch, extract text,
                     pass directly to LLM. No embeddings.
                     No Chroma. Seconds, not minutes.

  Web Search  ──→  URLs  ──→  Parallel Fetch  ──→  Extract Text  ──→  Direct LLM Prompt  ──→  Answer
```

---

## Current Pipeline: Step-by-Step Trace

### Entry Points

| Entry | File | What happens |
|---|---|---|
| CLI query | `src/qa/main.py` (`main()`, ~L212) | Calls `discover_urls_from_web()`, then `settings_for_discovered_urls()`, then `build_graph(rebuild_vectorstore=True)` |
| API query | `src/qa/api.py` (`query`, ~L165) | Same logic, wrapped in `POST /query` handler |
| API stream | `src/qa/api.py` (`query_stream`, ~L340) | Same logic, wrapped in `POST /query/stream` |

> Note: CLI/API import `discover_urls_from_web` / `settings_for_discovered_urls`
> from `src.core.web_search`, which is a backward-compat facade re-exporting
> from `src.web_search`. Prefer symbol names over line numbers — line numbers
> drift.

### Step 1: Web Search → URL Discovery (fast — ~1-3s)

```
discover_urls_from_web()                          # src/web_search/discovery.py
  → get_search_provider("bing")                    # src/web_search/factory.py
    → BingWebSearch.search(query, max_results=20)  # src/web_search/bing.py
      → HTTP GET bing.com/search
      → BeautifulSoup parse HTML
      → Extract candidate hrefs
      → Unwrap Bing redirect URLs
  → select_top_urls(urls, top_k=6)                 # src/web_search/common.py
    → Filter noise hostnames
    → Take top-K
  → Return list[str] (6 URLs)
```

**Latency:** ~1-3 seconds (single HTTP request + HTML parse). This step is fine.

### Step 2: Rebuild Graph With Discovered URLs (HEAVY)

```
settings_for_discovered_urls(settings, urls)       # src/web_search/discovery.py
  → Creates Settings with source_urls=urls,
    chroma_dir=.chroma/web-search/<hash>            # isolated collection

build_graph(settings, rebuild_vectorstore=True)     # src/graph/builder.py
  → _resolve_tools()
    → build_retriever_tool(settings, rebuild=True) # src/rag/chroma_retriever.py
      → ChromaRetriever(settings, rebuild=True)
        → _build_langchain_retriever(rebuild=True)
```

### Step 3: `_build_langchain_retriever` — The Bottleneck

```
_build_langchain_retriever(rebuild=True)           # src/rag/chroma_retriever.py
  │
  ├─ _clear_chroma_store()                          # Delete old index
  │
  ├─ load_and_split_documents(urls, ...)            # src/rag/document_loader.py
  │   │
  │   ├─ load_source_documents(urls)                # PARALLEL — already optimized
  │   │   └─ _load_url_documents_batch(...)         # ThreadPoolExecutor
  │   │       └─ max_concurrent_loads=4, TTL cache, order preserved
  │   │                                             # 6 URLs fetched concurrently
  │   │
  │   ├─ filter_quality_documents(docs)             # src/rag/document_quality.py
  │   │       └─ Drops low-quality / boilerplate pages before indexing
  │   │
  │   └─ split_documents(docs)                      # RecursiveCharacterTextSplitter
  │                                                 # Splits into dozens of chunks
  │
  └─ Chroma.from_documents(doc_splits, ...)         # THE real one-shot overhead
      └─ Embed every chunk via DashScope API         # N chunks ÷ batch_size API calls
                                                     # + Chroma index build + persist
```

**Cost breakdown (6 URLs, typical web pages):**

| Phase | Note |
|---|---|
| URL fetching | Parallel (ThreadPoolExecutor); not the bottleneck |
| HTML parsing + quality filter + text splitting | Minor |
| Embedding (all chunks via DashScope API) | **One-shot waste** — index queried once |
| Chroma index build + persist | **One-shot waste** — collection discarded after |

The dominant avoidable cost is the embed + Chroma steps on an index that is
built, queried once, and thrown away. Fetch latency was already addressed by the
earlier parallelization work.

### Step 4: Graph Execution (lightweight after index exists)

```
graph.stream({"messages": [("user", question)]})   # src/core/graph_executor.py:67
  │
  ├─ agent → LLM decides to call retriever tool
  ├─ retrieve → ToolNode calls Chroma vector search
  ├─ grade_documents → LLM grades relevance (yes/no)
  ├─ [maybe] rewrite → LLM rewrites query, loops back to agent
  └─ generate → LLM generates answer from retrieved context
```

**Latency:** ~5-15 seconds (LLM calls). This step is fine.

---

## Target Pipeline: Lightweight Web Search Path

### Proposed Flow

```
User Question
  │
  ├─ discover_urls_from_web(question, settings)     # SAME — keep as-is (~1-3s)
  │
  ├─ fetch_pages(urls)                              # NEW — wraps existing parallel loader
  │   └─ document_loader.load_source_documents(...)  # reuse ThreadPoolExecutor + cache
  │       └─ per page: extract article text (NEW)
  │           └─ Strip nav, ads, scripts, boilerplate
  │           └─ Truncate to ~8K tokens each
  │
  ├─ build_web_search_prompt(question, pages)        # NEW — assemble context (~0s)
  │   └─ Format: "## Source: {url}\n{text}\n\n" per page
  │   └─ Add: "Answer based on these web sources. Cite sources."
  │
  └─ llm.invoke(prompt)                              # DIRECT — no retriever (~3-10s)
      └─ Return answer
```

**Target latency:** ~10-25 seconds total (vs 70-120s current)

### What Gets Removed (Web Search Path Only)

| Removed | Why |
|---|---|
| `RecursiveCharacterTextSplitter` | No chunking needed — full page text goes to LLM |
| `DashScopeTextEmbeddings` / embedding API calls | No vectors needed |
| `Chroma.from_documents()` | No vector store needed |
| `ChromaRetriever` / vector similarity search | Content is in the prompt directly |
| `grade_documents` node | LLM reads full pages — relevance grading is redundant |
| `rewrite` loop | No retrieval to improve — one-shot prompt |
| `settings_for_discovered_urls()` | No isolated Chroma collection needed |

### What Stays (Both Paths)

| Kept | Why |
|---|---|
| `discover_urls_from_web()` and all providers | URL discovery still needed |
| `select_top_urls()` / noise filtering | Still want to skip known junk domains |
| `BingWebSearch`, `BaiduWebSearch`, `DuckDuckGoWebSearch` | Providers unchanged |
| `web_search_top_k`, `web_search_region`, etc. | Config still needed |
| `live_web_search` tool (`src/web_search/tool.py`) | Already doing lightweight path for runtime agent searches |
| Static docs path (full Chroma pipeline) | Still needed for `source_urls` / `DEFAULT_URLS` |

---

## Architecture Decision: Two Paths, One Entry Point

The system should choose the pipeline based on **where the URLs came from**:

```
                    ┌─────────────────────────┐
                    │  User provides URLs?     │
                    └───────────┬─────────────┘
                                │
               ┌────────────────┼────────────────┐
               │ YES            │ NO             │
               ▼                ▼                │
    ┌──────────────────┐  ┌──────────────────────┐
    │ Explicit URLs     │  │ Web search enabled?   │
    │ (static docs)     │  └──────────┬───────────┘
    └────────┬─────────┘             │
             │              ┌────────┼────────┐
             │              │ YES    │ NO     │
             │              ▼        ▼        │
             │    ┌──────────────┐ ┌──────────────────┐
             │    │ Web Search    │ │ DEFAULT_URLS      │
             │    │ (lightweight) │ │ (static docs)     │
             │    └──────┬───────┘ └────────┬─────────┘
             │           │                  │
             ▼           ▼                  ▼
    ┌────────────────────────────┐  ┌──────────────────────────────┐
    │ HEAVY PATH                 │  │ HEAVY PATH                   │
    │ Persist → Embed → Chroma   │  │ Persist → Embed → Chroma     │
    │ → Retrieve → Generate      │  │ → Retrieve → Generate        │
    └────────────────────────────┘  └──────────────────────────────┘

    ┌──────────────────────────────────┐
    │ LIGHTWEIGHT PATH (NEW)           │
    │ Fetch → Extract → Prompt → LLM   │
    └──────────────────────────────────┘
```

**Key rule:** The heavy path is for **persistent corpora** queried many times.
The lightweight path is for **discovered URLs** used once.

### Edge Cases

| Scenario | Path | Rationale |
|---|---|---|
| User has no URLs, web search enabled | Lightweight | One-shot — don't build index |
| User has no URLs, web search disabled | Heavy (DEFAULT_URLS) | Static docs, queried repeatedly |
| User provides explicit URLs | Heavy | User is configuring a knowledge base |
| Chat session with web search | Lightweight first; cache results | Re-fetch if TTL expired; don't build Chroma |
| `live_web_search` tool at runtime | Lightweight (already implemented) | One-shot agent lookup |

---

## Files To Change

### New Files

| File | Purpose |
|---|---|
| `src/web_search/content_fetcher.py` | Parallel page fetching with text extraction |
| `src/web_search/prompt_builder.py` | Assemble direct-answer prompt from fetched pages |
| `src/graph/nodes/web_answer.py` | New `web_answer` node — calls LLM directly with page context |
| `src/graph/web_pipeline.py` | Lightweight graph: agent → web_search → web_answer (no Chroma, no embed) |
| `tests/test_web_search_pipeline.py` | Tests for the lightweight path |

### Modified Files

| File | Change |
|---|---|
| `src/qa/main.py` | Branch: web-search URLs → lightweight graph; explicit URLs → heavy graph |
| `src/qa/api.py` | Same branching logic for `POST /query` |
| `src/graph/builder.py` | Add `build_lightweight_graph()` or `mode="web_search"` |
| `src/web_search/__init__.py` | Export new `fetch_pages`, `build_web_search_prompt` |
| `src/config/settings.py` | Add `web_search_lightweight: bool = True`, `web_search_max_page_tokens: int = 8000` |
| `src/config/loader.py` | Plumb new settings through env/YAML/CLI |
| `config/default.yaml` | Add new config keys |
| `.env.example` | Document new env vars |

### Files That Stay (No Changes)

| File | Why unchanged |
|---|---|
| `src/web_search/bing.py`, `baidu.py`, `duckduckgo.py` | URL discovery providers still work as before |
| `src/web_search/common.py` | Noise filtering, URL normalization still used |
| `src/web_search/factory.py` | Provider factory still used |
| `src/web_search/discovery.py` | `discover_urls_from_web()` still used |
| `src/rag/*` | Heavy path preserved for static docs |
| `src/sessions/*` | Session logic unaffected |
| `src/api/*` | API contracts preserved |

---

## Content Extraction Design

### What `content_fetcher.py` Does

> **Reuse, don't reimplement:** parallel fetch + TTL caching already exist in
> `src/rag/document_loader.py` (`load_source_documents`, `SourceDocumentCache`).
> `content_fetcher.py` should call into that for fetching and focus on the
> *new* part — extracting clean article text and truncating to a token budget —
> rather than building another `ThreadPoolExecutor`.

```python
@dataclass
class FetchedPage:
    url: str
    title: str
    text: str           # cleaned article text
    fetch_time_ms: float
    error: str | None

def fetch_pages(
    urls: list[str],
    *,
    timeout: float = 15.0,
    max_tokens_per_page: int = 8000,
    cache_ttl_seconds: int = 300,
) -> list[FetchedPage]:
    """
    1. Fetch URLs via existing load_source_documents (parallel + cached).
    2. For each loaded document, extract main content / strip boilerplate.
    3. Normalize whitespace.
    4. Truncate to max_tokens_per_page (rough: chars ÷ 4).
    5. Return pages in original URL order.
    """
```

### What `prompt_builder.py` Does

```python
def build_web_search_prompt(
    question: str,
    pages: list[FetchedPage],
    *,
    max_total_tokens: int = 100000,
) -> str:
    """
    Assemble a prompt with the question and page contents.
    Includes source URLs for citation.
    """
    # Template:
    # "Answer the question using the web sources below. Cite sources where possible.
    #
    #  Question: {question}
    #
    #  Sources:
    #  --- Source: {url} (Title: {title}) ---
    #  {text}
    #  --- End Source ---
    #  ..."
```

---

## Graph Design: Lightweight Path

### Node: `web_search_agent`

```
web_search_agent(state):
  - If state has a tool_call for "live_web_search":
    → Run web search, fetch pages, return content in messages
  - Otherwise:
    → Return existing messages as-is
```

### Node: `web_answer`

```
web_answer(state):
  - Extract question from state
  - Extract fetched page contents from messages
  - Build prompt via build_web_search_prompt()
  - Call LLM directly (no retriever, no grade, no rewrite)
  - Return answer
```

### Graph Topology

```
START → web_search_agent → [has_tool_call?]
           │                    │
           │ NO                 │ YES
           ▼                    ▼
         web_answer        live_web_search tool
           │                    │
           ▼                    ▼
          END              web_answer → END
```

This is dramatically simpler than the heavy graph:

```
Current QA graph (heavy):   agent → retrieve → grade → [rewrite → agent] → generate
                                                       ↑ loop up to 2×    │
Lightweight graph:           web_search_agent → web_answer
```

---

## Migration Strategy

### Phase A: Add Lightweight Path (Safe — no behavior change for existing users)

1. Add `content_fetcher.py` and `prompt_builder.py`
2. Add `build_lightweight_graph()` in `src/graph/builder.py`
3. Add `web_search_lightweight` setting (default `True`)
4. Branch in `src/qa/main.py` and `src/qa/api.py`: if web search URLs + lightweight enabled → use new path
5. Keep heavy path as fallback (`web_search_lightweight=False`)

### Phase B: Validate

1. Add tests comparing lightweight vs heavy answers for same queries
2. Measure latency: lightweight should be 5-25s vs 70-120s
3. Verify source citations appear in answers
4. Run full test suite — existing heavy path tests should still pass

### Phase C: Optimize

1. Tune `max_tokens_per_page` and `max_concurrency`
2. Add streaming for the `web_answer` node
3. Consider caching fetched pages across queries (shared cache with TTL)

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Page content too large for context window | Truncate each page to `max_tokens_per_page`; drop pages that push over budget |
| LLM ignores source citations | Prompt engineering; test for citation presence |
| Web page extraction misses key content | Multi-strategy extraction (article > main > body); fallback to full text |
| Some queries genuinely need vector search | Keep heavy path as opt-out; let user set `web_search_lightweight=False` |
| JavaScript-rendered pages return empty text | Add headless browser option later (Phase C); accept limitation for now |

---

## Known Issues (Current State)

These are tracked in `PROBLEMS_DETECTED.md` and `docs/code-review-web-search.md`:

| # | Issue | Severity | Fixed by this refactor? |
|---|---|---|---|
| 1 | Sequential URL loading | — | **Already fixed** — `document_loader` fetches in parallel (`ThreadPoolExecutor`, `max_concurrent_loads=4`) |
| 2 | Full re-embedding on every search | High | Yes — no embeddings in lightweight path |
| 3 | Full Chroma rebuild for single-use index | High | Yes — no Chroma in lightweight path (this is the real win) |
| 4 | No content quality scoring before indexing | Medium | **Partially already fixed** — `document_quality.filter_quality_documents` runs in heavy path; lightweight path skips indexing |
| 5 | Web search not exposed as runtime agent tool | Medium | Already fixed by `live_web_search` in `src/web_search/tool.py` |
| 6 | No re-ranking after retrieval | Low | N/A — no retrieval in lightweight path |

---

## Refactoring To-Do List

### Phase A — Build Lightweight Path

- [ ] **A.1 Create `src/web_search/content_fetcher.py`**
  - [ ] `FetchedPage` dataclass
  - [ ] `fetch_pages()` — **delegate parallel fetch to `document_loader.load_source_documents`** (do not reimplement `ThreadPoolExecutor`)
  - [ ] `extract_text(html)` — strip boilerplate, keep article content (the new logic)
  - [ ] `estimate_tokens(text)` — rough char/4 heuristic
  - [ ] Reuse existing `SourceDocumentCache` / `page_load_cache_ttl_seconds` caching

- [ ] **A.2 Create `src/web_search/prompt_builder.py`**
  - [ ] `build_web_search_prompt(question, pages)` → str
  - [ ] Source citation formatting
  - [ ] Token budget management (truncate pages that don't fit)

- [ ] **A.3 Add lightweight graph in `src/graph/builder.py`**
  - [ ] `build_lightweight_graph(settings)` or `build_graph(mode="web_search")`
  - [ ] Nodes: agent (with live_web_search tool) → web_answer
  - [ ] No Chroma, no embeddings, no retriever, no grade, no rewrite

- [ ] **A.4 Add `web_answer` node in `src/graph/nodes/web_answer.py`**
  - [ ] `web_answer_factory(settings)` → callable
  - [ ] Assembles prompt from fetched pages, calls LLM, returns answer

- [ ] **A.5 Add config keys**
  - [ ] `web_search_lightweight: bool = True` in `src/config/settings.py`
  - [ ] `web_search_max_page_tokens: int = 8000`
  - [ ] Plumb through `src/config/loader.py`, `config/default.yaml`, `.env.example`

- [ ] **A.6 Branch in entry points**
  - [ ] `src/qa/main.py` `main()` (~L212 discovery block) — if web search + lightweight → new path
  - [ ] `src/qa/api.py` `query` (~L165) and `query_stream` (~L340) — same branching
  - [ ] `src/chat/api.py` — same for chat sessions

- [ ] **A.7 Export new public API**
  - [ ] `src/web_search/__init__.py` — export `fetch_pages`, `build_web_search_prompt`, `FetchedPage`

### Phase B — Validate

- [ ] **B.1 Add `tests/test_web_search_pipeline.py`**
  - [ ] Test `fetch_pages()` with mock HTTP responses
  - [ ] Test `build_web_search_prompt()` output format
  - [ ] Test `web_answer` node with mock LLM
  - [ ] Integration test: question → web search URLs → lightweight graph → answer
  - [ ] Verify source citations appear

- [ ] **B.2 Regression safety**
  - [ ] Existing heavy-path tests still pass
  - [ ] `web_search_lightweight=False` preserves old behavior
  - [ ] Full `python -m pytest -q` green

- [ ] **B.3 Latency measurement**
  - [ ] Log fetch duration per URL
  - [ ] Log total end-to-end time
  - [ ] Verify lightweight path completes in <30s for typical queries

### Phase C — Optimize (Future)

- [ ] **C.1 Streaming** — token-by-token SSE for web_answer node
- [ ] **C.2 Cross-query page cache** — share fetched pages across similar queries
- [ ] **C.3 Smarter extraction** — Readability-like algorithm for main content detection
- [ ] **C.4 JavaScript rendering** — optional Playwright/Puppeteer for JS-heavy pages
- [ ] **C.5 Keep heavy path as a permanent opt-out** — do NOT deprecate `web_search_lightweight=False`; the grade/rewrite loop is a genuine safety net for hard/noisy queries

---

## Dependencies

- Existing: `src/web_search/` (providers, discovery, noise filtering)
- Existing: `src/rag/document_loader.py` (parallel fetch + `SourceDocumentCache`) — reuse for `fetch_pages`
- Existing: `src/rag/document_quality.py` (boilerplate/low-quality filtering)
- Existing: `src/llm/provider.py` (LLMProvider seam)
- Existing: `src/utils/retry.py` (retry logic)
- Existing: `src/utils/networking.py` (SSL config)
- New dependency: none — uses existing `requests`/`bs4` and stdlib

---

## References

- `PROBLEMS_DETECTED.md` — 12 root causes of web search slowness and poor quality
- `docs/code-review-web-search.md` — finding that web search wasn't a LangChain tool
- `src/web_search/SKILL.md` — web search provider architecture
- `src/rag/SKILL.md` — knowledge retrieval architecture (heavy path)
- `src/graph/SKILL.md` — graph pipeline architecture
- [[system-architect]] — top-level architecture
- [[web-search-architect]] — URL discovery providers
- [[rag-pipeline-architect]] — graph topology and nodes
- [[knowledge-retrieval-architect]] — Chroma and embeddings
