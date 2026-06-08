---
name: refactor-daily-forms
description: Dated refactoring process forms with complete, to-do, blocked, and note lines
metadata:
  type: project
  updated: 2026-06-08
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

## 2026-06-08 Form

| Label | Scope | Line Item | Evidence / Next Action |
|---|---|---|---|
| note | Coordination | Started to-do completion pass with RAGRefactorDeveloper subagents. | Spawned one compatibility/endpoint review worker and one Phase 2/Phase 3 readiness worker; parent owns environment repair, test execution, and shared process checkpoints. |
| complete | Environment | Verified a working project Python runner. | `python --version` reports Python 3.11.7 and can import/run the installed test dependencies; `.venv\\Scripts\\python.exe` also reports Python 3.11.7 but lacks `pytest`, so use `python` for verification in this environment. |
| complete | Verification | Ran the full test suite. | `python -m pytest -q` passed: 44 tests green after compatibility regression coverage was added. |
| complete | Phase 2 cleanup | Locally checked compatibility shims and endpoint contracts. | `python -m compileall src tests` passed; direct import check confirmed legacy QA/chat model and graph/session exports still resolve. |
| complete | Phase 2 cleanup / compatibility | Reviewed scoped core/chat compatibility modules and QA/chat HTTP schemas. | Restored legacy `src.core.nodes` helper aliases `_question_tokens` and `_split_context_sentences`; added focused regression coverage in `tests/test_graph_state_and_nodes.py`; scoped tests and full suite passed. |
| complete | Phase 2 remaining | Decided whether Phase 2 config/YAML/error-code work should continue before Phase 3. | Continue a short Phase 2 completion pass before Phase 3 because YAML config, typed error codes, fuller typed event emission, and LLM provider DI are not complete. |
| complete | Phase 3 planning | Prepared SSE/session persistence/observability readiness plan. | Start Phase 3 only after the Phase 2 completion pass and `python -m pytest -q` are green; build SSE on typed events, persistence on config/checkpointer settings, and metrics on events/error codes. |
| complete | Phase 2 config | Completed YAML config support before Phase 3. | Added `config/default.yaml`, flat YAML loading, CLI `--config` support where applicable, and CLI > env > YAML > defaults tests. |
| complete | Phase 2 errors | Added typed RAG error hierarchy and FastAPI handlers. | See later Phase 2 errors checkpoint in this form for changed files and verification. |
| complete | Phase 2 events | Completed non-SSE typed graph event emission groundwork. | `src/graph/executor.py` now emits node_start/node_end/error/done plus retriever and grader summaries when inferable from chunks; SSE/API integration remains deferred. |
| complete | Phase 2 LLM DI | Added an LLM provider seam. | Added `src/llm/provider.py`; graph nodes now build chat models through `build_chat_model()` instead of constructing `ChatTongyi` directly. |
| note | Phase 3 readiness | Phase 3 remains gated on Phase 2 completion and green tests. | SSE, SQLite session persistence, metrics, incremental Chroma rebuilds, and deprecation warnings should come after the Phase 2 completion pass. |
| note | Coordination | Started Phase 2 completion and Phase 3 implementation pass. | Parent owns YAML config and final integration; RAGRefactorDeveloper subagents are assigned disjoint scopes for errors, events/metrics, and session persistence. |
| complete | Phase 2 config | Implemented YAML config support first. | Added `config/default.yaml`, flat YAML loading with CLI > env > YAML > built-in precedence, `--config` plumbing for QA/chat CLIs and app factories, and config precedence tests; `python -m pytest -q tests\test_config.py` passed. |
| complete | Phase 2 errors | Implemented typed error hierarchy and FastAPI handlers. | Added `src/errors.py`, `src/api/errors.py`, QA/chat handler registration, typed chat 404s, QA generic exception wrapping, and `tests/test_api_errors.py`; `python -m pytest -q tests\\test_api_errors.py tests\\test_api_dependencies.py`, `python -m compileall src tests`, `git diff --check`, and full `python -m pytest -q` passed. |
| complete | Phase 2 events / Phase 3 metrics | Completed typed event emission and metrics groundwork. | Added `src/graph/metrics.py`, event-driven metrics collection, and executor tests for event order, error events, retriever/grader summaries, and metrics; `python -m pytest -q` passed with 62 tests. |
| complete | Phase 3 SSE/API | Integrated typed graph events with SSE endpoints. | Added `POST /query/stream`, `POST /chat/{thread_id}/message/stream`, shared SSE serialization, and API streaming tests. |
| complete | Phase 3 sessions | Added optional session persistence storage groundwork. | Added `StorageBackend`, `SessionMetadata`, `InMemoryStorage`, and stdlib `SQLiteStorage` for metadata only; optional registry hooks preserve default in-memory behavior; `python -m pytest -q tests\\test_sessions.py` and full `python -m pytest -q` passed. |
| complete | Phase 3 sessions | Resolved full LangGraph checkpoint persistence without a new dependency. | Added `SQLiteMemorySaver`, which persists LangGraph `MemorySaver` checkpoint maps to SQLite; chat startup now restores metadata and checkpoint state. |
| complete | Phase 3 metrics | Added in-process metrics endpoint. | `GraphExecutor` records event-driven metrics and QA/chat apps expose `GET /metrics`; API stream tests verify metrics increment after streamed requests. |
| complete | Phase 3 deprecation | Added deprecation warnings for old compatibility import paths. | Added `src/_compat.py`, warnings in `src/core/*` and `src/chat/*` facades, and `tests/test_compat.py` coverage. |
| complete | Verification | Ran final integrated verification. | `python -m pytest -q` passed with 66 tests; `python -m compileall src tests` passed; `git diff --check` passed. |
| note | Company readiness | Started combined planning pass from `memory/current-process.md`, `memory/refactor-daily-forms.md`, and `COMPANY_READINESS_GAPS.md`. | Parent owns checkpoint persistence and integration; subagents may own independent CI/quality/container/docs/security slices. |
| complete | Current blocker | Implemented restart-safe chat checkpoint persistence without adding an unverified external dependency. | Added `SQLiteMemorySaver`, wired chat startup to SQLite metadata/checkpoint stores, restored persisted session metadata into runtime registry, and added restart-state tests. |
| complete | Company readiness / infra | Added CI, quality gates, dev requirements, Docker, and compose. | Added `.github/workflows/ci.yml`, Ruff/mypy/pytest-cov config in `pyproject.toml`, `requirements-dev.txt`, `.pre-commit-config.yaml`, `Dockerfile`, `docker-compose.yml`, and `.dockerignore`; marked `COMPANY_READINESS_GAPS.md` sections 1-3 complete. |
| complete | Company readiness / local tests | Kept local base pytest runnable while CI enforces coverage. | Coverage settings remain configured, but `--cov` enforcement moved to CI where `requirements-dev.txt` installs `pytest-cov`; local `python -m pytest -q` works without dev extras. |
| complete | Company readiness / verification | Resolved integration regressions from concurrent readiness work. | Fixed chat graph wrapper delegation when no explicit checkpointer is supplied; full local test suite is green. |
| complete | Company readiness / API security | Added API-key auth, CORS configuration, and readiness endpoint. | `API_KEY` unset keeps local mutation endpoints open; when set, QA/chat mutation endpoints require `Authorization: Bearer <key>`; QA/chat expose `/ready` using local app-state checks only; CORS is driven by `cors_allow_origins`/`CORS_ALLOW_ORIGINS`. |
| complete | Company readiness / docs | Added team documentation and updated readiness gap tracking. | Added `CONTRIBUTING.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, and `SECURITY.md`; updated `COMPANY_READINESS_GAPS.md` sections 8-12 without editing source or tests. |
| complete | Company readiness / persistence docs | Documented Chroma/session backup and restore boundaries. | `ARCHITECTURE.md` covers `.chroma/`, `.chroma/chat/<thread_id>/`, optional SQLite session metadata backup, restore steps, and the metadata-only limitation. |
| complete | Company readiness / database migrations | Added SQLite session schema versioning. | `SQLiteStorage` now creates a `schema_version` table and test coverage verifies the current schema version. |
| to do | Company readiness / security | Finish remaining security implementation tasks. | Dependency scanning remains open; scoped `sk-` hardcoded-key scan found no matches; CORS configuration and API-key auth are implemented. |
| complete | Company readiness / environments | Added multi-environment config support. | Added `config/development.yaml`, `config/staging.yaml`, and `config/production.yaml`; default config loading now overlays `config/{RAG_ENV}.yaml`, defaulting to `development`. |
| complete | Company readiness / API verification | Ran targeted API/config hardening tests. | `python -m pytest -q tests\test_api_security_readiness.py tests\test_api_dependencies.py tests\test_api_streaming.py tests\test_config.py` passed with 16 tests. |
| complete | Company readiness / full verification | Full local verification is green. | `python -m pytest -q` passed with 74 tests; `python -m compileall src tests` passed; `git diff --check` passed with line-ending warnings only. |
| complete | Company readiness / README | Updated README for the current implementation state. | Replaced stale notebook-era notes with setup, config precedence, QA/chat endpoints, SSE, auth/CORS, metrics/readiness, Docker, verification, persistence, and remaining readiness gaps. |
| complete | Verification | Ran documentation pass verification. | `git diff --check` completed without whitespace errors; Git reported line-ending warnings only. |

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
