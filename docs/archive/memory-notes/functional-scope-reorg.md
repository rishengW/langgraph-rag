---
name: functional-scope-reorg
description: SKILL.md files reorganized from 3 architectural layers to 5 functional scopes on 2026-06-05, with to-do lists from REFACTORING_PLAN.md embedded
metadata:
  type: project
---

# Functional Scope Reorganization (2026-06-05)

## Why the change

The original layer-based split (system-architect / core-engine-architect / web-api-architect / chat-session-architect) created a false separation — most real tasks span layers, forcing constant cross-referencing. `src/core/SKILL.md` became a 340+ line dumping ground mixing graph topology, embeddings, Chroma lifecycle, and web search.

The functional split maps each SKILL.md 1:1 to a target module in the refactoring plan.

## What changed

### Deleted
- `src/core/SKILL.md` → split into `src/graph/` + `src/rag/` + `src/web_search/`
- `src/qa/SKILL.md` → merged into `src/api/`
- `src/chat/SKILL.md` → split into `src/api/` + `src/sessions/`

### Created
- `src/graph/SKILL.md` — RAG Pipeline (238 lines)
- `src/rag/SKILL.md` — Knowledge Retrieval (174 lines)
- `src/web_search/SKILL.md` — Web Search (164 lines)
- `src/api/SKILL.md` — API & CLI (305 lines)
- `src/sessions/SKILL.md` — Conversation Engine (284 lines)

### New directories
- `src/graph/`, `src/rag/`, `src/web_search/`, `src/api/`, `src/sessions/`

### Modified
- Root `SKILL.md` — updated architecture map, module responsibilities table, wiki-link references, added Refactoring To-Do List section
- `MEMORY.md` — updated with new entries
- All 4 memory files in `memory/` — updated to reflect functional scope architecture

## To-Do Lists Added

Each functional-scope SKILL.md now has a `## Refactoring To-Do List` section with checkbox items organized by phase, sourced from `REFACTORING_PLAN.md`:

| SKILL.md | Items | Key work |
|---|---|---|
| Root | 19 | pyproject.toml, test harness, logging, config split, requirements |
| `src/graph/` | 15 | Node unification (QuestionResolver), RAGState, prompt extraction |
| `src/rag/` | 13 | Retry consolidation, Protocol definitions, incremental indexing |
| `src/web_search/` | 7 | Provider Protocol, factory, fallback |
| `src/api/` | 17 | URL parsing dedup, global state elimination, DI, SSE |
| `src/sessions/` | 14 | condense tests, TTL cleanup, SQLite persistence, MemorySaver swap |

**Why:** The functional scope split means each SKILL.md is self-contained for its domain. When working on retrieval, you get embeddings + Chroma lifecycle + document loading in one file, without wading through graph topology or session management.

**How to apply:** Match a task to its functional scope, then read only that SKILL.md. Use wiki-link cross-references (`[[knowledge-retrieval-architect]]`, etc.) when a task truly spans scopes.
