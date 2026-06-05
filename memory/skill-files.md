---
name: skill-files
description: 6 functional-scope SKILL.md files in a cross-referenced hierarchy, reorganized from 3 layer-based files on 2026-06-05
metadata:
  type: reference
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
| `SKILL.md` | `system-architect` | Overall architecture, config, phase strategy, cross-cutting concerns | 263 |
| `src/graph/SKILL.md` | `rag-pipeline-architect` | Graph topology, node factories, state management, question resolution | 238 |
| `src/rag/SKILL.md` | `knowledge-retrieval-architect` | Embeddings, Chroma lifecycle, document loading, text splitting, retriever | 174 |
| `src/web_search/SKILL.md` | `web-search-architect` | URL discovery, Baidu/DDG providers, normalization, noise filtering | 164 |
| `src/api/SKILL.md` | `api-interface-architect` | FastAPI endpoints, request/response models, DI, SSE transport, CLI | 305 |
| `src/sessions/SKILL.md` | `session-engine-architect` | Multi-turn sessions, condense, MemorySaver, TTL, SQLite persistence | 284 |

### Wiki-Link Cross-References

All files use `[[wiki-link]]` notation for cross-referencing:
- Root → all 5: `[[rag-pipeline-architect]]`, `[[knowledge-retrieval-architect]]`, `[[web-search-architect]]`, `[[api-interface-architect]]`, `[[session-engine-architect]]`
- Each sub-skill → `Parent: [[system-architect]]` + lists all 4 siblings

### Content per File

Each functional-scope SKILL.md contains:
1. **Quick Reference** — key facts table
2. **File Map** — current code files with LOC, key symbols
3. **Flow/Design Patterns** — data flow, graph topology, design decisions
4. **Known Issues** — prioritized by severity, with file locations and fix phases
5. **Refactoring Target** — planned directory structure for that scope
6. **Refactoring To-Do List** — checkbox items sourced from `REFACTORING_PLAN.md`, organized by phase

**Why:** A layer-based split (core/engine, web/api, chat/session) forces cross-referencing for most real tasks because they span layers. A functional split means each SKILL.md is self-contained for its domain. The to-do lists naturally cluster per scope, and each scope maps 1:1 to a target module in the refactoring plan.

**How to apply:** When working on a task, match it to the functional scope, then read only that scope's SKILL.md. Cross-references are available via wiki-links if needed. The root `SKILL.md` should be consulted for cross-cutting concerns (config, error handling, logging, phase strategy).
