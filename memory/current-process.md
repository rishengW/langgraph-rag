---
name: current-process
description: Current refactoring process snapshot after Phase 1 extraction and Phase 2 interface implementation passes
metadata:
  type: project
  updated: 2026-06-08
---

# Current Process

Last updated: 2026-06-08

## Refactoring Status

Phase 1 extraction work is implemented in the working tree, and the current
advanced pass has implemented Phase 2 interfaces and abstractions across the
functional scopes.

On 2026-06-08, the active to-do list was completed through verification,
compatibility review, and the Phase 2/Phase 3 readiness decision. A later
Phase 2/3 implementation pass completed YAML config, typed errors, typed graph
events, LLM provider DI, SSE endpoints, metrics, deprecation warnings, and
SQLite session metadata storage.

The active process marker is:

- `advanced_phase2_interfaces_workbench/README.md`
- `memory/refactor-daily-forms.md`

## Completed Implementation Passes

### Phase 1 Baseline

- Added pytest project harness and focused offline tests.
- Added shared utilities for retry, networking, and URL parsing.
- Split pure settings/config loading into `src/config/`.
- Added unified graph state and shared graph node implementations.
- Extracted LLM prompt templates into `src/llm/prompts.py`.
- Kept old `src/core/*` and `src/chat/*` import paths as compatibility facades.

### Phase 2 Interface Pass

Five RAGRefactorDeveloper workers implemented one `SKILL.md` scope each:

- `src/web_search/`: provider protocol, Baidu/DDG providers, provider factory,
  DI-friendly discovery, and `src/core/web_search.py` compatibility.
- `src/rag/`: retriever/embedding protocols, `ChromaRetriever`, embedding
  modules, document loading orchestration, and core retriever/embedding shims.
- `src/sessions/`: session models/registry, TTL cleanup, isolated Chroma cleanup,
  session settings isolation, and `src/chat/sessions.py` compatibility.
- `src/graph/`: unified graph builder, edge helpers, typed graph events, executor
  wrapper, and legacy QA/chat graph delegation.
- `src/api/`: app-state dependency providers, shared API models, QA/chat app
  state refactor away from module globals, and preserved endpoint contracts.

## Verification State

Passed:

- `git diff --check`
- `python -m compileall src tests`
- Full pytest execution: `python -m pytest -q` passed with 66 tests green
- Targeted scans for removed QA/chat API globals and runtime `print()` calls
- Compatibility import check for legacy QA/chat model and graph/session exports
- Scoped compatibility tests:
  `python -m pytest -q tests\test_api_dependencies.py tests\test_graph_state_and_nodes.py tests\test_sessions.py`
- Config precedence, typed errors, graph event/metrics, SSE API, session
  storage, and deprecation compatibility tests

Environment note:

- `python` is the working verification runner in this environment and reports
  Python 3.11.7. `.venv\Scripts\python.exe` also reports Python 3.11.7 but
  lacks `pytest`, so prefer `python -m pytest -q` unless the venv is repaired.

## Current Implementation State

Completed in the latest pass:

- YAML config support through `config/default.yaml`, `load_settings(config_file=...)`,
  and CLI `--config` plumbing for QA/chat entry points.
- Typed `RAGError` hierarchy and shared FastAPI handlers with stable codes.
- Typed graph event emission through `GraphExecutor`, including node lifecycle,
  error, done, retriever summary, and grader summary events.
- LLM provider seam in `src/llm/provider.py`.
- SSE endpoints: `POST /query/stream` and
  `POST /chat/{thread_id}/message/stream`.
- Event-driven in-process metrics and `GET /metrics` on QA/chat apps.
- Optional stdlib SQLite session metadata storage and registry hooks.
- Deprecation warnings for legacy `src/core/*` and `src/chat/*` compatibility
  import paths.

Blocked:

- Full chat history recovery across process restart. The installed LangGraph
  package does not provide `langgraph.checkpoint.sqlite`, and requirements do
  not include another persistent checkpointer package. Session metadata is
  persisted, but checkpoint/state persistence needs a supported dependency or a
  larger custom checkpointer implementation.

## Next Process Step

Decide whether to add a supported LangGraph checkpoint persistence dependency
or implement a custom SQLite checkpointer. After that, wire recovered session
metadata plus persisted checkpoints into chat app startup and verify restart
recovery end to end.

Track the next concrete actions in `memory/refactor-daily-forms.md`. Future
workers should update that file at the end of each implementation pass.
