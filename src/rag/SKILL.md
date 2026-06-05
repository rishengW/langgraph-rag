---
name: knowledge-retrieval-architect
description: >
  Use this skill whenever working on knowledge retrieval and indexing — embeddings,
  Chroma vectorstore lifecycle, document loading, text splitting, retriever tool
  construction, or the Retriever/EmbeddingModel Protocol abstractions. Covers the
  full document→chunk→embed→store→retrieve pipeline. Trigger on mentions of
  retriever, embeddings, Chroma, vectorstore, document loader, text splitter,
  WebBaseLoader, chunk_size, chunk_overlap, or index/rebuild.
---

# Knowledge Retrieval Architect — src/rag/

Domain: embeddings, Chroma vectorstore, document loading, text splitting, retriever.
Parent: [[system-architect]]. Siblings: [[rag-pipeline-architect]], [[web-search-architect]], [[api-interface-architect]], [[session-engine-architect]].

## Quick Reference

| Fact | Value |
|---|---|
| Embeddings | DashScope `text-embedding-v4` (1024-dim) + HuggingFace fallback |
| Vector Store | Chroma, persisted at `.chroma/` |
| Text Splitter | `RecursiveCharacterTextSplitter.from_tiktoken_encoder` |
| Chunk defaults | size=100, overlap=50 |
| Document loader | `WebBaseLoader` with configurable timeout |
| Retriever interface | LangChain `BaseRetriever` + tool wrapping |
| Source files (current) | `core/embeddings.py`, `core/retriever.py` |
| Target location | `src/rag/` |

## File Map (Current)

```
src/core/
+-- embeddings.py           # DashScopeTextEmbeddings + build_embeddings() — 151 LOC
+-- retriever.py            # build_retriever() + build_retriever_tool() + Chroma lifecycle — 294 LOC
```

### Detailed File Responsibilities

| File | Responsibility | Key Symbols | LOC |
|------|---------------|-------------|-----|
| `embeddings.py` | `DashScopeTextEmbeddings` (LangChain `Embeddings` impl), retry logic, batch ordering | `DashScopeTextEmbeddings`, `build_embeddings`, `_call_with_retry` | 151 |
| `retriever.py` | Chroma lifecycle, web doc loading, text splitting, Windows file-lock retry, retriever tool factory | `build_retriever`, `build_retriever_tool`, `_release_chroma_system`, `_rmtree_with_retry` | 294 |

## Data Flow

```
Source URLs
  │
  ▼
WebBaseLoader ──► raw HTML documents
  │
  ▼
RecursiveCharacterTextSplitter ──► chunks (size=100 tokens, overlap=50)
  │
  ▼
Embedding Model (DashScope or HuggingFace) ──► dense vectors (1024-dim)
  │
  ▼
Chroma Vectorstore ──► persisted at .chroma/
  │
  ▼
Retriever (as_tool) ──► LangChain Tool bound to graph agent
```

## Key Design Patterns

### 1. Embedding Model Selection

```python
# embeddings.py:build_embeddings()
def build_embeddings(settings: Settings) -> Embeddings:
    if is_dashscope_embedding_model(settings.embedding_model):
        return DashScopeTextEmbeddings(
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            ...
        )
    else:
        from sentence_transformers import SentenceTransformer
        return HuggingFaceEmbeddings(model_name=settings.embedding_model)
```

### 2. Chroma Lifecycle

```
build_retriever(settings, rebuild=False):
  1. Check if chroma_dir exists
  2. Check embedding config for consistency (_embedding_config_matches)
  3. If persisted -> load existing index (Chroma.from_persist_directory)
  4. If not persisted or rebuild=True -> index from source_urls
     a. Load documents via WebBaseLoader (with timeout)
     b. Split via RecursiveCharacterTextSplitter
     c. Create/update Chroma collection
     d. Save embedding config metadata (model, dimension)
  5. Return retriever object
```

### 3. Retry Logic — 2 Scattered Implementations

| Layer | Location | Retryable Errors | Backoff |
|-------|----------|-----------------|---------|
| Embedding API | `embeddings.py:_call_with_retry()` | Rate limit, 5xx, network | Exponential |
| Chroma deletion | `retriever.py:_rmtree_with_retry()` | PermissionError, OSError | Fixed delay |

**Target (Phase 1):** Consolidate into `src/utils/retry.py`.

### 4. Windows File-Lock Handling

`_release_chroma_system()` handles Windows file-lock issues by GC-collecting and deleting the Chroma system client before directory removal:

```python
def _release_chroma_system(chroma_dir: Path) -> None:
    import gc
    client = chromadb.PersistentClient(path=str(chroma_dir))
    # ... delete collection, reset system
    del client
    gc.collect()
```

### 5. Retriever Tool Construction

```python
def build_retriever_tool(settings, rebuild=False):
    retriever = build_retriever(settings, rebuild=rebuild)
    return create_retriever_tool(
        retriever,
        name="retrieve_documents",
        description="Search the knowledge base for relevant information.",
    )
```

## Known Issues

### Priority 1

| # | Issue | File | Fix |
|---|-------|------|-----|
| 1 | `_call_with_retry` duplicates pattern from `nodes.py:_invoke_with_retry` | `embeddings.py` | Consolidate to `utils/retry.py` |
| 2 | `_rmtree_with_retry` is Windows-specific complexity | `retriever.py` | Isolate behind interface (Phase 2) |
| 3 | Chroma `embedding_config.json` metadata check is fragile | `retriever.py` | Version the ser/des format |

### Priority 2

| # | Issue | Notes |
|---|-------|-------|
| 4 | WebBaseLoader timeout is per-URL; no parallel fetch | Could use `asyncio.gather` for concurrent loads |
| 5 | No embedding cache — same chunks re-embedded on every rebuild | Add LRU cache (Phase 3) |
| 6 | Text splitter uses tiktoken; `chunk_size=100` is tokens, not characters | Document this clearly |
| 7 | No `Protocol` or `ABC` for retriever/embeddings — adding a new provider requires modifying core | Define Protocols (Phase 2) |

## Refactoring Target — src/rag/

```
src/rag/
+-- __init__.py
+-- retriever.py              # Retriever Protocol
+-- chroma_retriever.py       # ChromaRetriever implementation
+-- embeddings.py             # EmbeddingModel Protocol
+-- dashscope_embeddings.py   # DashScopeTextEmbeddings implementation
+-- hf_embeddings.py          # HuggingFaceEmbeddings factory
+-- document_loader.py        # WebBaseLoader + text splitter orchestration
```

### Protocol Definitions (Phase 2)

```python
@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 4) -> list[Document]: ...
    def as_tool(self) -> BaseTool: ...
    def rebuild(self, urls: list[str] | None = None) -> None: ...

@runtime_checkable
class EmbeddingModel(Protocol):
    model_name: str
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

## Refactoring To-Do List

> Source: [`REFACTORING_PLAN.md`](../../REFACTORING_PLAN.md). Knowledge retrieval scope items.

### Phase 1 — Extract Without Behavioral Change

- [ ] **1.2 Extract retry logic** — move `_call_with_retry` from `embeddings.py` → `src/utils/retry.py`
  - [ ] Consolidate with `_invoke_with_retry` (from `nodes.py`) and `_rmtree_with_retry` (from `retriever.py`)
- [ ] **1.2 Extract SSL config** — move `_configure_ssl()` side effect → `src/utils/networking.py`
- [ ] **1.4 Split config** — embed/Chroma settings into namespaced config models (see `[[api-interface-architect]]` Phase 2.5)

### Phase 2 — Interfaces & Abstractions

- [ ] **2.1 Define `Retriever` Protocol** — `src/rag/retriever.py`
  - [ ] `retrieve(query, k)` → `list[Document]`
  - [ ] `as_tool()` → `BaseTool`
  - [ ] `rebuild(urls)` → `None`
- [ ] **2.1 Define `EmbeddingModel` Protocol** — `src/rag/embeddings.py`
  - [ ] `embed_documents(texts)` → `list[list[float]]`
  - [ ] `embed_query(text)` → `list[float]`
- [ ] **2.1 Implement `ChromaRetriever`** — wrapping `build_retriever()` logic → `src/rag/chroma_retriever.py`
  - [ ] Satisfy `isinstance(chroma_retriever, Retriever)` check
- [ ] **2.1 Implement embedding classes** — `DashScopeEmbeddings` → `src/rag/dashscope_embeddings.py`, `HuggingFaceEmbeddings` → `src/rag/hf_embeddings.py`
- [ ] **2.1 Extract document loader** — WebBaseLoader + text splitter orchestration → `src/rag/document_loader.py`
- [ ] **2.2 DI integration** — `ChromaRetriever(rebuilt=False)` receives config, not raw Settings

### Phase 3 — Optimizations

- [ ] **3.4 Incremental vectorstore rebuilds** — URL-chunk hash map to skip already-indexed URLs
  - [ ] Keep `_embedding_config_matches` check
  - [ ] Add progress bars for CLI indexing operations
- [ ] **3.4 Embedding cache** — LRU cache to avoid re-embedding same chunks on rebuild
- [ ] **3.4 Parallel document loading** — `asyncio.gather` for concurrent URL fetches
- [ ] **3.5 Deprecation shim** — `src/core/embeddings.py`, `src/core/retriever.py` → re-exports from `src/rag/`

## Dependencies

- `src/config/` — Settings/AppConfig with Chroma + Embedding namespaces
- `src/utils/retry.py` — unified retry decorators (Phase 1)
- External: `langchain-chroma`, `chromadb`, `dashscope`, `sentence-transformers`, `tiktoken`, `beautifulsoup4`
