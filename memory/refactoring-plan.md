---
name: refactoring-plan
description: 3-phase roadmap (15-20 days), to-do lists embedded in each functional-scope SKILL.md, 20 tech-debt items tracked
metadata:
  type: project
---

# Refactoring Plan

The full architectural plan is in `REFACTORING_PLAN.md` (1466 lines). Key facts:

## Current Snapshot (2026-06-07)

The working tree now contains the Phase 1 extraction baseline and a Phase 2
interfaces/abstractions implementation pass.

Implemented Phase 1 baseline:

- Test harness, shared utilities, config split, prompt extraction, unified graph
  state, shared graph node implementations, and compatibility facades.

Implemented Phase 2 scope pass:

- `src/web_search/`: provider protocol, provider classes, factory, DI-friendly
  discovery, and core compatibility.
- `src/rag/`: retriever/embedding protocols, Chroma retriever wrapper,
  embedding modules, document loader split, and core shims.
- `src/sessions/`: session models/registry, TTL cleanup, isolated Chroma cleanup,
  settings isolation helper, and chat compatibility.
- `src/graph/`: unified builder, edge helpers, typed events, executor wrapper,
  and legacy graph wrappers.
- `src/api/`: FastAPI app-state dependencies, shared models, and QA/chat state
  refactor away from module-level globals.

Verification status:

- Passed `git diff --check`.
- Passed bundled Python compile: `python -m compileall src tests`.
- Full pytest is blocked until the project Python environment is repaired.

## Phase Summary

| Phase | Focus | Effort | Risk | Branch |
|-------|-------|--------|------|--------|
| Phase 1 | Extract components without behavioral change | 3-5 days | Low | `refactor/phase-1-extract` |
| Phase 2 | Interface abstractions, DI, typed events, config files | 5-8 days | Medium | `refactor/phase-2-interfaces` |
| Phase 3 | SSE streaming, session persistence, metrics, optimization | 5-7 days | Medium-High | `refactor/phase-3-streaming` |

## To-Do Lists in SKILL.md Files

On 2026-06-05, actionable to-do lists from `REFACTORING_PLAN.md` were embedded into each functional-scope SKILL.md:

| SKILL.md | # To-Do Items | Key Phase 1 Items |
|---|---|---|
| Root `SKILL.md` | 19 | pyproject.toml, test harness, print→logging, config split, requirements regroup |
| `src/graph/SKILL.md` | 15 | Unify node factories (QuestionResolver), merged RAGState, extract prompts |
| `src/rag/SKILL.md` | 13 | Consolidate retry logic, Protocol definitions, incremental indexing |
| `src/web_search/SKILL.md` | 7 | Provider Protocol, factory, retry with fallback |
| `src/api/SKILL.md` | 17 | URL parsing dedup, global state elimination, DI, SSE endpoints |
| `src/sessions/SKILL.md` | 14 | condense tests, TTL cleanup, SQLite persistence, MemorySaver swap |

## Critical Tech Debt Items (from analysis)

**Priority 1 — Fix Immediately:**
1. Node factories duplicated ~70% between `core/nodes.py` and `chat/nodes.py`
2. `print()` statements in graph nodes (not structured logging)
3. `load_settings()` mutates `os.environ` as side effect
4. Module-level globals `_graph`, `_settings` in API modules

## Backward Compatibility Guarantees

- All CLI commands continue working through Phase 3
- All API endpoints maintain same request/response schemas through Phase 2
- `.env.example` settings remain valid
- Existing Chroma databases remain compatible
- Old import paths emit `DeprecationWarning` in Phase 3, remain functional

**Why:** The refactoring must be incremental and revertible. Each phase is a separate branch independently mergeable. The compatibility shims mean old code paths are preserved.

**How to apply:** Start with Phase 1 (lowest risk). Each to-do item references its step number in `REFACTORING_PLAN.md`. Verify against exit criteria before merging each phase.
