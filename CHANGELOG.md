# Changelog

This project follows a lightweight changelog until formal releases begin.

## Unreleased

### Added

- Added contributor onboarding, architecture, changelog, and security
  documentation.
- Documented Chroma backup and restore procedures.
- Added CI, dev quality gate configuration, pre-commit hooks, Docker, compose,
  API-key auth, CORS config, `/ready`, environment overlays, and SQLite-backed
  chat checkpoint persistence.

### Known Gaps

- CI installs dev dependencies before running coverage, Ruff, and mypy; the
  local base environment may need `requirements-dev.txt` installed for those
  gates.
- Deployment-specific infrastructure, API versioning, dependency scanning,
  structured JSON logging, rate limiting, and session transcript export remain
  tracked readiness work.

## 2026-06-08

### Added

- Phase 2/3 refactoring work added YAML config support, typed RAG errors,
  typed graph events, LLM provider dependency injection, SSE endpoints,
  in-process metrics, optional SQLite session metadata storage, and deprecation
  warnings for compatibility import paths.

### Verification

- Full test suite passed in the recorded process checkpoint with 66 tests.
- `python -m compileall src tests` passed in the recorded process checkpoint.
- `git diff --check` passed in the recorded process checkpoint.
