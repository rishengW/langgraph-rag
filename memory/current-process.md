---
name: current-process
description: Current project process snapshot after refactoring, persistence, and company-readiness passes
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
compatibility review, and the Phase 2/Phase 3 readiness decision. Later passes
completed YAML config, typed errors, typed graph events, LLM provider DI, SSE
endpoints, metrics, deprecation warnings, SQLite session metadata/checkpoint
storage, and major company-readiness foundations.

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
- Full pytest execution: `python -m pytest -q` passed with 74 tests green
- Targeted scans for removed QA/chat API globals and runtime `print()` calls
- Compatibility import check for legacy QA/chat model and graph/session exports
- Scoped compatibility tests:
  `python -m pytest -q tests\test_api_dependencies.py tests\test_graph_state_and_nodes.py tests\test_sessions.py`
- Config precedence, typed errors, graph event/metrics, SSE API, API security,
  readiness, session storage/checkpoint, and deprecation compatibility tests

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
- Optional stdlib SQLite session metadata storage, schema versioning,
  checkpoint storage, registry hooks, and chat startup recovery.
- Deprecation warnings for legacy `src/core/*` and `src/chat/*` compatibility
  import paths.
- CI workflow, dev quality configs, pre-commit, Dockerfile, docker compose,
  API-key auth, CORS config, `/ready`, environment overlays, and team docs.
- README refreshed to match the current company-readiness state, including
  config overlays, auth/CORS, SSE, metrics, Docker, verification, and
  SQLite-backed chat/session persistence.
- Repaired and verified the project `.venv` dependency set after a stale
  LangChain 1.x environment caused install/compatibility failures; explicit
  PyPI install now yields a clean pinned LangChain 0.3/LangGraph 0.6 stack.
- Updated `requirements.txt` to pin the direct runtime/test dependencies to the
  verified `.venv` versions and added direct `PyYAML`/`requests` entries for
  code imports that should not rely on transitive dependencies.

## Remaining Company-Readiness Work

- Install dev dependencies in CI/local dev to run `ruff`, `mypy`, and
  coverage-enforced tests when doing full quality-gate verification.
- Add dependency scanning (`pip-audit` or Dependabot).
- Add structured JSON logging and request/session correlation IDs.
- Add API versioning under `/api/v1/` while keeping compatibility shims.
- Add deployment-specific infrastructure (AWS/GCP/on-prem/PaaS).
- Consider rate limiting and a session export endpoint.

Track the next concrete actions in `memory/refactor-daily-forms.md`. Future
workers should update that file at the end of each implementation pass.
