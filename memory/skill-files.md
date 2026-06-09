---
name: skill-files
description: 7 functional-scope SKILL.md files in a cross-referenced hierarchy — 6 original scopes + new web search refactoring design
metadata:
  type: reference
  updated: 2026-06-09
---

# Skill Files — Functional Scope Architecture

## Reorganization (2026-06-05)

The SKILL.md files were reorganized from a **layer-based split** (system/core/qa/chat) to a **function-based split** (system/graph/rag/web_search/api/sessions). This matches the refactoring target in `REFACTORING_PLAN.md`.

### Old Structure (deleted)
- `SKILL.md` (root) — system architect, oversaw everything
- `src/core/SKILL.md` (340+ lines) — monolithic, mixed graph + embeddings + web search + config
- `src/qa/SKILL.md` — FastAPI layer for QA only
- `src/chat/SKILL.md` — chat app layer, mixed API + session concerns

### New Structure (current)

| File | Skill Name | Domain | Lines |
|---|---|---|---|
| `SKILL.md` | `system-architect` | Overall architecture, config, phase strategy, cross-cutting concerns | ~263 |
| `src/graph/SKILL.md` | `rag-pipeline-architect` | Graph topology, node factories, state management, question resolution | ~238 |
| `src/rag/SKILL.md` | `knowledge-retrieval-architect` | Embeddings, Chroma lifecycle, document loading, text splitting, retriever | ~174 |
| `src/web_search/SKILL.md` | `web-search-architect` | URL discovery, Baidu/Bing/DDG providers, normalization, noise filtering | ~164 |
| `src/api/SKILL.md` | `api-interface-architect` | FastAPI endpoints, request/response models, DI, SSE transport, CLI | ~305 |
| `src/sessions/SKILL.md` | `session-engine-architect` | Multi-turn sessions, condense, MemorySaver, TTL, SQLite persistence | ~284 |
| `memory/refactor/SKILL.md` | `web-search-refactor-architect` | Web search pipeline redesign — lightweight direct-LLM path vs heavy vector-store path, migration plan | ~565 |

### Addition: Web Search Refactor Architect (2026-06-09)

The 7th SKILL.md was added to design the web search pipeline refactoring:

- **Why needed:** The Phase 1–3 refactoring reorganized modules and added
  abstractions, but the web search path still forces one-shot discovered URLs
  through the full heavy pipeline (fetch → split → embed → Chroma → retrieve →
  grade → generate), taking 60–180 seconds for an index queried once and
  discarded.
- **Scope:** `memory/refactor/SKILL.md` documents the current pipeline end-to-end
  with latency breakdowns, proposes a lightweight "parallel fetch → extract text
  → direct LLM prompt" path, maps every file that must change, and provides a
  three-phase migration strategy.
- **Relationship to existing skills:** It references the three most relevant
  scopes — [[web-search-architect]] (URL discovery stays), [[rag-pipeline-architect]]
  (graph topology gains a lightweight variant), and [[knowledge-retrieval-architect]]
  (heavy path preserved for static docs).

### Wiki-Link Cross-References

All files use `[[wiki-link]]` notation for cross-referencing:
- Root → all 6 sub-skills: `[[rag-pipeline-architect]]`, `[[knowledge-retrieval-architect]]`, `[[web-search-architect]]`, `[[api-interface-architect]]`, `[[session-engine-architect]]`, `[[web-search-refactor-architect]]`
- Each sub-skill → `Parent: [[system-architect]]` + lists all siblings
- `memory/refactor/SKILL.md` → `Parent: [[system-architect]]`, Related: `[[web-search-architect]]`, `[[rag-pipeline-architect]]`, `[[knowledge-retrieval-architect]]`

### Content per File

Each functional-scope SKILL.md contains:
1. **Quick Reference** — key facts table
2. **File Map** — current code files with LOC, key symbols
3. **Flow/Design Patterns** — data flow, graph topology, design decisions
4. **Known Issues** — prioritized by severity, with file locations and fix phases
5. **Refactoring Target** — planned directory structure for that scope
6. **Refactoring To-Do List** — checkbox items sourced from `REFACTORING_PLAN.md`, organized by phase

The `memory/refactor/SKILL.md` additionally contains:
7. **Architecture Decision Record** — two-path design with decision tree
8. **Latency Breakdown** — per-phase timing for current vs target pipeline
9. **Content Extraction Design** — `FetchedPage` dataclass, `fetch_pages()` API
10. **Migration Strategy** — three phases (A: build, B: validate, C: optimize) with risk mitigations

**Why:** A layer-based split (core/engine, web/api, chat/session) forces cross-referencing for most real tasks because they span layers. A functional split means each SKILL.md is self-contained for its domain. The to-do lists naturally cluster per scope, and each scope maps 1:1 to a target module in the refactoring plan.

**How to apply:** When working on a task, match it to the functional scope, then read only that scope's SKILL.md. Cross-references are available via wiki-links if needed. The root `SKILL.md` should be consulted for cross-cutting concerns (config, error handling, logging, phase strategy). For web search pipeline changes, start with `memory/refactor/SKILL.md` to understand the two-path architecture, then consult [[web-search-architect]], [[rag-pipeline-architect]], and [[knowledge-retrieval-architect]] for the modules involved.
