---
name: refactor-daily-forms
description: Dated refactoring process forms with complete, to-do, blocked, and note lines
metadata:
  type: project
  updated: 2026-06-07
---

# Refactor Daily Forms

This file is the single process checklist for completed work and next actions.
Each date gets one form. Every form line must start with one of these labels:

- `complete` - finished and verified as far as the current environment allows
- `to do` - planned next work
- `blocked` - cannot proceed until an external condition changes
- `note` - context that future agents should preserve

## Form Template

| Label | Scope | Line Item | Evidence / Next Action |
|---|---|---|---|
| complete |  |  |  |
| to do |  |  |  |
| blocked |  |  |  |
| note |  |  |  |

## 2026-06-07 Form

| Label | Scope | Line Item | Evidence / Next Action |
|---|---|---|---|
| complete | Phase 1 | Extraction baseline implemented. | Shared utilities, config split, prompt extraction, unified graph state, shared graph nodes, compatibility facades, and focused tests are present in the working tree. |
| complete | Phase 2 / web_search | Provider interface pass implemented. | `src/web_search/` contains protocol, Baidu/DDG providers, factory, discovery helpers, and `src/core/web_search.py` compatibility. |
| complete | Phase 2 / rag | Retrieval interface pass implemented. | `src/rag/` contains retriever/embedding protocols, `ChromaRetriever`, embedding modules, document loader, and core shims. |
| complete | Phase 2 / sessions | Session interface pass implemented. | `src/sessions/` contains models, registry, TTL cleanup, isolated Chroma cleanup, settings isolation, and chat compatibility. |
| complete | Phase 2 / graph | Unified graph interface pass implemented. | `src/graph/` contains unified builder, edges, events, executor, state, and shared nodes. Legacy QA/chat graph wrappers delegate to it. |
| complete | Phase 2 / api | App-state API dependency pass implemented. | `src/api/` contains dependency helpers and shared models. QA/chat APIs use app state instead of old module-level graph/session globals. |
| complete | Verification | Lightweight checks passed. | `git diff --check` passed. Bundled Python `compileall src tests` passed. Targeted scans found no old QA/chat API globals or runtime `print()` calls in refactored paths. |
| blocked | Verification | Full pytest cannot run in the current local Python setup. | Repair or recreate `.venv`; current launcher points to missing Python 3.11, bundled Python lacks `pytest`, and venv compiled `pydantic_core` is incompatible with bundled Python. |
| to do | Environment | Recreate the project Python environment. | Install a working Python matching the venv or rebuild `.venv`, then install requirements including pytest. |
| to do | Verification | Run the full test suite. | Run `python -m pytest -q` after environment repair; record failures and fixes in the next dated form. |
| to do | Phase 2 cleanup | Review compatibility shims and endpoint contracts after pytest passes. | Confirm old imports and QA/chat HTTP schemas still behave as promised through Phase 2. |
| to do | Phase 2 remaining | Decide whether to continue Phase 2 config/YAML/error-code work before Phase 3. | Compare against `REFACTORING_PLAN.md` Phase 2 exit criteria. |
| to do | Phase 3 planning | Prepare SSE/session persistence/observability work only after Phase 2 verification is green. | Add the next dated form before starting Phase 3 work. |
| note | Agent workflow | Future RAGRefactorDeveloper workers must update this file when they finish. | Add or update the current date form, mark completed lines, refresh to-do lines, and record blocked checks. |
