---
name: refactoring-plan
description: 3-phase roadmap complete, web search pipeline refactoring launched, 20 tech-debt items tracked, new lightweight path designed
metadata:
  type: project
  updated: 2026-06-09
---

# Refactoring Plan

The full historical plan is in `REFACTORING_PLAN.md` (1466 lines). The newly
launched web search pipeline refactoring is designed in `memory/refactor/SKILL.md`.

## Current Status (2026-06-09)

### Phases 1–3: Complete

The original three-phase refactoring plan is finished:

| Phase | Focus | Status |
|-------|-------|--------|
| Phase 1 | Extract components without behavioral change | Complete |
| Phase 2 | Interface abstractions, DI, typed events, config files | Complete |
| Phase 3 | SSE streaming, session persistence, metrics, optimization | Complete |

### Phase 1 Baseline (Complete)

- Test harness (`pytest`), shared utilities (`src/utils/`), config split (`src/config/`)
- Prompt extraction (`src/llm/prompts.py`), unified graph state (`src/graph/state.py`)
- Shared graph node implementations, compatibility facades

### Phase 2 Interfaces (Complete)

| Scope | Deliverable |
|---|---|
| `src/web_search/` | Provider protocol, Baidu/Bing/DDG providers, factory, discovery, noise filtering |
| `src/rag/` | Retriever/embedding protocols, `ChromaRetriever`, embedding modules, document loader |
| `src/sessions/` | Session models/registry, TTL cleanup, isolated Chroma, SQLite metadata/checkpoint storage |
| `src/graph/` | Unified builder, edge helpers, typed events, executor wrapper, metrics |
| `src/api/` | App-state DI, shared models, typed error handlers, SSE streaming, auth/CORS |

### Phase 3 Company-Readiness (Complete)

- YAML config with environment overlays (development/staging/production)
- Typed `RAGError` hierarchy, FastAPI error handlers
- Typed graph events (`GraphExecutor`), in-process metrics, SSE endpoints
- SQLite session metadata + checkpoint persistence (`SQLiteMemorySaver`)
- Deprecation warnings on legacy import paths
- CI, pre-commit, Docker, docker-compose, API-key auth, CORS, `/ready`
- Team docs: ARCHITECTURE, CHANGELOG, CONTRIBUTING, SECURITY, COMPANY_READINESS_GAPS

### Active Initiative: Web Search Pipeline Refactoring

**Why now:** The Phase 1–3 work reorganized modules and added abstractions, but
the web search path still forces one-shot discovered URLs through the full heavy
pipeline (fetch → split → embed → Chroma → retrieve → grade → generate). This
embeds content that is queried once then discarded.

**Accuracy note (2026-06-09):** URL fetching is **already parallel** —
`src/rag/document_loader.py` uses a `ThreadPoolExecutor`
(`_load_url_documents_batch`, `max_concurrent_loads=4`) with TTL caching and
input-order preservation. The remaining cost is **embedding + Chroma build**,
not fetching. The lightweight path's win comes from skipping embed/Chroma for
one-shot URLs, not from "adding" parallel fetch.

**Design document:** `memory/refactor/SKILL.md` (565 lines)

**Core insight:** The vector store pipeline was designed for persistent
documentation URLs queried many times. Web search URLs are used once — the
embedding and Chroma steps add cost but no value.

**Proposed architecture:** Two paths diverging at the entry point:

| Path | Trigger | Flow | Latency |
|---|---|---|---|
| Heavy (existing) | Explicit URLs, static docs, `web_search_lightweight=False` | Fetch → Split → Embed → Chroma → Retrieve → Generate | dominated by embed + Chroma build |
| Lightweight (new) | Web search discovered URLs | Parallel fetch → Extract text → Direct LLM prompt | fetch + single LLM call |

**What changes:**

| Module | New/Modified | Purpose |
|---|---|---|
| `src/web_search/content_fetcher.py` | New | Page text extraction; **reuses** `document_loader.load_source_documents` for parallel fetch + caching rather than reimplementing it |
| `src/web_search/prompt_builder.py` | New | Assemble direct-answer prompt from fetched pages |
| `src/graph/nodes/web_answer.py` | New | LLM node that reads pages directly (no retriever) |
| `src/graph/builder.py` | Modified | Add `build_lightweight_graph()` |
| `src/qa/main.py`, `src/qa/api.py` | Modified | Branch: web search → lightweight; explicit URLs → heavy |
| `src/config/settings.py` | Modified | Add `web_search_lightweight`, `web_search_max_page_tokens` |

**What stays:** URL discovery providers (Bing, Baidu, DuckDuckGo), noise
filtering, parallel fetch + caching in `document_loader.py`, document quality
filtering (`src/rag/document_quality.py`), all of `src/rag/` (heavy path
preserved for static docs), all API contracts.

**Trade-off (not a pure win):** The lightweight path drops the
`grade_documents` and `rewrite` self-correction loop. For 3–6 clean pages this
is fine; for noisy extraction or JS-rendered pages (empty text) it removes the
retry safety net. Keep `web_search_lightweight=False` as a permanent escape
hatch rather than deprecating it.

**Migration:** Three phases — A (build lightweight path, opt-in), B (validate
latency + quality), C (optimize: streaming, cross-query cache, JS rendering).

## Verification Status

- Full test suite: **113 tests — 113 passed** (previously 85 with 2 failures; both fixed on 2026-06-09)
- `python -m compileall src tests` — passed
- `git diff --check` — passed (line-ending warnings only)
- Branch: `web-developed` (ahead of `main`)

## Backward Compatibility Guarantees

- All CLI commands continue working
- All API endpoints maintain same request/response schemas
- `.env.example` settings remain valid
- Existing Chroma databases remain compatible
- Heavy path preserved as default; lightweight path is opt-in via `web_search_lightweight`
- Old import paths emit `DeprecationWarning`, remain functional

## Critical Tech Debt Items

**Resolved (Phase 1–3):**
1. ~~Node factories duplicated ~70% between `core/nodes.py` and `chat/nodes.py`~~ — unified via `QuestionResolver`
2. ~~`print()` statements in graph nodes~~ — replaced with `logging`
3. ~~`load_settings()` mutates `os.environ` as side effect~~ — split into pure loader + env setter
4. ~~Module-level globals `_graph`, `_settings` in API modules~~ — moved to FastAPI `app.state`

**Open (web search refactoring):**
5. Web search pipeline latency for one-shot URLs (embed + Chroma build dominate; fetch is already parallel) — addressed by lightweight path
6. No content quality scoring before indexing — **partially already solved** by `src/rag/document_quality.filter_quality_documents`; lightweight path skips indexing entirely
7. Search engine rank is the sole quality gate — lightweight path sends full page text to LLM
8. `web_search_top_k` config drift resolved; canonical default is 6

## Related Files

| File | Purpose |
|---|---|
| `REFACTORING_PLAN.md` | Full historical architectural plan (1466 lines) |
| `memory/refactor/SKILL.md` | Web search pipeline refactoring design (565 lines) |
| `PROBLEMS_DETECTED.md` | 12 root causes of web search slowness/poor quality |
| `docs/code-review-web-search.md` | Finding: web search not exposed as LangChain tool (now fixed) |
| `memory/current-process.md` | Live project state snapshot |
| `memory/refactor-daily-forms.md` | Dated implementation checklist |
| `COMPANY_READINESS_GAPS.md` | Remaining company-readiness work |
