---
name: project-overview
description: langgraph-rag project — functional scope architecture, two apps (QA + Chat), LangGraph pipeline, module map
metadata:
  type: project
---

# Project Overview — langgraph-rag

Project `langgraph-rag` ("only Subscribers") is a RAG (Retrieval-Augmented Generation) application built with LangGraph, FastAPI, and DashScope.

## Functional Scope Architecture

The project is organized into 5 functional scopes (not 3 architectural layers as originally designed):

| Scope | Current Code | Target Module | Responsibility |
|---|---|---|---|
| **RAG Pipeline** | `src/core/graph.py`, `nodes.py`, `state.py` | `src/graph/` | Graph topology, node factories, state management, question resolution |
| **Knowledge Retrieval** | `src/core/embeddings.py`, `retriever.py` | `src/rag/` | Embeddings, Chroma vectorstore, document loading, text splitting |
| **Web Search** | `src/core/web_search.py` | `src/web_search/` | URL discovery, Baidu/DDG providers, result normalization |
| **API & CLI** | `src/qa/`, `src/chat/api.py`, `src/chat/main.py` | `src/api/` | FastAPI endpoints, request/response models, DI, SSE, CLI |
| **Conversation Engine** | `src/chat/sessions.py`, `graph.py`, `nodes.py` (condense) | `src/sessions/` | Multi-turn sessions, condense, MemorySaver, persistence |

## Current Source Tree

```
src/
├── core/           # Shared RAG engine (~1700 LOC) — being split into graph/rag/web_search
├── qa/             # Single-shot Q&A (~700 LOC) — API migrating to src/api/
├── chat/           # Multi-turn chat (~1300 LOC) — API to api/, sessions to sessions/
├── graph/          # [NEW target dir] RAG pipeline SKILL.md created
├── rag/            # [NEW target dir] Knowledge retrieval SKILL.md created
├── web_search/     # [NEW target dir] Web search SKILL.md created
├── api/            # [NEW target dir] API & CLI SKILL.md created
└── sessions/       # [NEW target dir] Conversation engine SKILL.md created
```

## Tech Stack

- Python 3.11+, LangGraph 0.2-0.7, LangChain 0.3.x
- DashScope (Qwen `qwen-plus`) for LLM, `text-embedding-v4` (1024-dim) for embeddings
- Chroma (SQLite) for vector store, FastAPI + uvicorn for web
- Baidu HTML scraping + DuckDuckGo (`ddgs`) for web search
- `.env` config via python-dotenv, YAML planned for Phase 2

## Key Cross-Cutting Concerns

- **Node duplication**: `core/nodes.py` and `chat/nodes.py` share ~70% identical code — fix via `QuestionResolver` injection (Phase 1)
- **Global state**: `_graph`, `_settings`, `_rebuild_lock` as module-level globals in `qa/api.py` — fix via FastAPI DI (Phase 2)
- **No tests**: Zero test files exist — test harness planned for Phase 1
- **No streaming**: All graph execution is synchronous batch — SSE streaming planned for Phase 3
- **No typed errors**: All errors are bare `RuntimeError` — `RAGError` hierarchy planned for Phase 2

**Why:** The original `src/core/` module grew into a monolithic "shared RAG engine" mixing graph logic, embeddings, retriever lifecycle, and web search. The refactoring splits these into independent functional modules, each with its own SKILL.md that is self-contained and scoped to a single responsibility.

**How to apply:** When working on any code change, first identify which functional scope(s) it touches, then consult the corresponding SKILL.md in `src/<scope>/SKILL.md` for context, known issues, and to-do items.
