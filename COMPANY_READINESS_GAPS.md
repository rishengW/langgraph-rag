# Company Readiness Gaps — langgraph-rag

Generated from the project audit on 2026-06-08. Each section lists what's missing,
why it matters, and a concrete deliverable. Ordered roughly by priority / effort.

---

## 1. CI/CD Pipeline

**Current state:** No CI at all. Tests run manually via `pytest`.

**Deliverables:**
- [x] `.github/workflows/ci.yml` — GitHub Actions workflow that runs on push/PR:
  - Set up Python 3.11
  - Install dependencies from `requirements.txt`
  - Run `pytest` with `--tb=short`
  - Run `ruff check .` (lint)
  - Run `mypy src/` (type check, see §2)
  - Fail the build on any issue
- [ ] Optional: `.github/workflows/release.yml` — build + publish to PyPI if this becomes a package

**Why:** Automated quality gates. No regressions slip through. Standard for any team project.

---

## 2. Static Analysis & Code Quality Gates

**Current state:** Type hints exist but aren't enforced. No linter/formatter configured.

**Deliverables:**
- [x] `pyproject.toml` entries for:
  - `[tool.ruff]` — lint + format rules, target Python 3.11
  - `[tool.mypy]` — strict mode (or `--strict` gradual rollout), ignore missing stubs for deps
  - `[tool.pytest.ini_options]` — add `--cov=src --cov-report=term --cov-fail-under=70`
- [x] `.pre-commit-config.yaml` — pre-commit hooks:
  - `ruff` (lint + format)
  - `mypy`
  - `check-yaml`, `check-toml`, `end-of-file-fixer`, `trailing-whitespace`
- [x] `requirements-dev.txt` (or a `[dev]` extra) listing: `pytest`, `pytest-cov`, `ruff`, `mypy`, `pre-commit`, `httpx`

**Why:** Enforces consistency. Catches type errors before they're bugs. Table stakes for team development.

---

## 3. Containerization

**Current state:** No Dockerfile. App runs directly on the host Python.

**Deliverables:**
- [x] `Dockerfile` — multi-stage build:
  - Stage 1: install deps
  - Stage 2: slim runtime image
  - Copy `src/`, `config/`, `requirements.txt`
  - `EXPOSE 8000 8001`
  - `CMD ["python", "-m", "src.qa.main", "serve", "--host", "0.0.0.0"]`
- [x] `docker-compose.yml` — orchestrate both QA (port 8000) and Chat (port 8001) services, with a shared volume for `.chroma/` and optional SQLite persistence
- [x] `.dockerignore` — exclude `.venv/`, `.git/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.chroma/`

**Why:** Reproducible environment. Anyone can run the app with `docker compose up`. Required for any cloud deployment.

---

## 4. Authentication & Authorization

**Current state:** Mutation endpoints use optional API-key auth. When `API_KEY`
is unset, local development remains open; when set, mutation requests must send
`Authorization: Bearer <key>`.

**Deliverables:**
- [x] `src/api/auth.py` — simple API-key middleware:
  - Read `API_KEY` from env/Settings
  - FastAPI dependency `require_api_key(request: Request)` that checks `Authorization: Bearer <key>` header
  - Return 401 if missing/invalid
  - Apply to all mutation endpoints (`POST /query`, `POST /chat`, `DELETE /chat/{id}`)
- [ ] Optional: rate limiting middleware (e.g., `slowapi` or manual token-bucket)

**2026-06-08 API security update:** Added `src/api/auth.py`, wired it to QA
`POST /query` and `POST /query/stream`, and to chat `POST /chat`,
`POST /chat/{thread_id}/message`, `POST /chat/{thread_id}/message/stream`, and
`DELETE /chat/{thread_id}`. Existing read-only, health, readiness, metrics, and
static UI endpoints remain public.

**Why:** Any API exposed beyond localhost needs auth. Even internal tools need basic protection.

---

## 5. Observability & Monitoring

**Current state:** `MetricsCollector` is exported through `/metrics`; liveness
and readiness are separate endpoints.

**Deliverables:**
- [x] `GET /metrics` endpoint (JSON or Prometheus text format) that exposes `MetricsCollector.snapshot()`
- [x] `GET /health` — already exists. Add `GET /ready` (readiness) with local state checks that avoid live external network calls
- [ ] Structured JSON logging format (configurable via `LOG_FORMAT=json` env var):
  - Include `timestamp`, `level`, `logger`, `message`, `request_id`, `session_id`
- [ ] Optional: OpenTelemetry tracing for graph node spans

**2026-06-08 readiness update:** Added `GET /ready` to QA and chat. The checks
use local app state only (`settings`, QA graph readiness, chat registry, and
configured Chroma path) and intentionally avoid live DashScope or network probes
so readiness cannot block on external services.

**Why:** You can't debug what you can't see. Metrics + structured logs are the minimum for production.

---

## 6. Configuration Hardening

**Current state:** `.env` + YAML loading works. Some rough edges remain.

**Deliverables:**
- [x] Validate `Settings` at startup — fail fast with a clear message if `DASHSCOPE_API_KEY` is missing or empty
- [x] `config/default.yaml` — already exists, verify it's complete and matches every Settings field
- [x] Configuration documentation comments in `default.yaml`
- [ ] Mask secrets in logs (already partially done via `secret_fingerprint` — extend to all log emissions)

**2026-06-08 config hardening update:** `load_settings()` already fails fast
when `DASHSCOPE_API_KEY` is empty. `config/default.yaml` now includes the new
non-secret API/CORS settings with comments that keep actual `API_KEY` and
DashScope secrets in environment variables.

**Why:** Misconfiguration is the #1 cause of production incidents. Fail fast, fail clearly.

---

## 7. API Versioning & Documentation

**Current state:** Flat endpoints (`/query`, `/chat`). No versioning. No OpenAPI customization.

**Deliverables:**
- [ ] Prefix all routes with `/api/v1/` (keep old routes as redirects or deprecation shims for one release)
- [ ] Custom OpenAPI title/description/version in FastAPI app config
- [ ] Pydantic model `Field(description=...)` on all request/response models so the auto-generated docs are useful
- [ ] Optional: `openapi.json` export script for sharing with API consumers

**Why:** Versioned APIs let you evolve without breaking clients. Good OpenAPI docs reduce support burden.

---

## 8. Database & Persistence

**Current state:** Chroma is local SQLite. Chat sessions have optional SQLite
metadata plus a stdlib SQLite-backed LangGraph memory checkpoint saver. Chat
startup restores persisted session metadata and checkpoint state.

**Deliverables:**
- [x] Session SQLite schema versioning — a `schema_version` table, migration on startup if needed
- [x] Document the Chroma directory structure and what a backup/restore looks like
- [ ] Optional: `POST /chat/{id}/export` — export a session transcript as JSON/Markdown

**2026-06-08 persistence update:** `ARCHITECTURE.md` documents the `.chroma/`
store, isolated chat stores under `.chroma/chat/<thread_id>/`, and
stop-the-service backup/restore steps. `src/sessions/sqlite.py` now creates a
`schema_version` table, and `src/sessions/checkpoint.py` persists LangGraph
`MemorySaver` checkpoint maps to SQLite without adding a new runtime dependency.

**Why:** Schema changes WILL happen. Versioned migrations prevent data loss.

---

## 9. Security Basics

**Current state:** SSL verification issues documented in `SSL_FIX.md`. Some Windows workarounds.

**Deliverables:**
- [x] `SECURITY.md` — how to report vulnerabilities, supported versions
- [ ] Dependency scanning — `pip-audit` or Dependabot on GitHub to flag known CVEs
- [x] Remove any remaining hardcoded defaults that look like keys (verify none exist)
- [x] CORS middleware configuration — currently likely wide open; restrict to configured origins

**2026-06-08 documentation update:** `SECURITY.md` now covers vulnerability
reporting, supported-version expectations, secret handling, Chroma/session data
sensitivity, backup handling, SSL troubleshooting boundaries, and open auth/CORS
and dependency-scanning gaps. Dependency scanning still needs implementation.

**2026-06-08 API security update:** Added CORS middleware driven by
`cors_allow_origins`/`CORS_ALLOW_ORIGINS`. Empty origins keep CORS closed by
default, while development config allows common localhost origins. A scoped
scan for `sk-` style key literals found no hardcoded secrets.

**Why:** Even internal tools need basic security hygiene. Dependency vulns are the easiest attack vector.

---

## 10. Documentation for Teams

**Current state:** README is good for solo devs. Missing team onboarding docs.

**Deliverables:**
- [x] `CONTRIBUTING.md` — how to set up, run tests, lint, commit conventions
- [x] `ARCHITECTURE.md` — extract the key diagrams and module descriptions from `REFACTORING_PLAN.md` into a shorter reference doc (the plan is 1465 lines — too long for newcomers)
- [ ] Docstrings on all public functions (many already have them — audit coverage)
- [x] `CHANGELOG.md` — start tracking changes by version

**2026-06-08 documentation update:** Added `CONTRIBUTING.md`,
`ARCHITECTURE.md`, and `CHANGELOG.md`. Public docstring coverage is not audited
in this pass because the task explicitly excludes source-code edits.

**Why:** The bus factor. If someone else needs to work on this, they need onboarding docs.

---

## 11. Production Deployment Config

**Current state:** No cloud/vendor deployment configuration.

**Deliverables (pick your platform):**
- [ ] **If AWS:** `infra/` with CDK or Terraform for ECS Fargate + EFS (for Chroma persistence)
- [ ] **If GCP:** Cloud Run + Cloud Storage FUSE for Chroma
- [ ] **If on-prem:** systemd unit files for the QA and Chat services
- [ ] **Minimal:** `Procfile` or `fly.toml` for quick PaaS deployment

**2026-06-08 documentation update:** No deployment config was added in this
docs-only pass. `ARCHITECTURE.md` records the current single-process deployment
shape and Chroma persistence expectations to support later AWS/GCP/on-prem/PaaS
work.

**Why:** The gap between "runs on my machine" and "runs in production" is mostly deployment config.

---

## 12. Multi-Environment Support

**Current state:** Single `.env` file. No dev/staging/prod distinction.

**Deliverables:**
- [x] `config/development.yaml`, `config/staging.yaml`, `config/production.yaml`
- [x] `RAG_ENV` env var to select which config file to load (defaults to `development`)
- [ ] Production config disables debug endpoints, enables stricter CORS, sets higher rate limits

**2026-06-08 documentation update:** `CONTRIBUTING.md` documents `--config`
usage for custom YAML files. A later config/source pass added `RAG_ENV`
selection and dev/staging/prod defaults.

**2026-06-08 config update:** Added development, staging, and production YAML
overlays. `load_settings()` now loads `config/default.yaml` plus
`config/{RAG_ENV}.yaml` when the default config path is used, with `RAG_ENV`
defaulting to `development`.

**Why:** You should never run the same config in dev and prod.

---

## Summary — Effort & Impact Matrix

| # | Gap | Effort | Impact |
|---|-----|--------|--------|
| 1 | CI/CD (GitHub Actions) | Low | **High** |
| 2 | Linting + Type Checking | Low | **High** |
| 3 | Docker + Compose | Low | **High** |
| 4 | Auth (API key) | Low | Medium |
| 5 | `/metrics` + `/ready` | Low | Medium |
| 6 | Config validation | Low | Medium |
| 7 | API versioning | Medium | Medium |
| 8 | DB export/migrations beyond v1 | Medium | Low |
| 9 | Security (audit + CORS + deps) | Low | Medium |
| 10 | Team docs | Medium | Medium |
| 11 | Deployment config | Medium | Medium |
| 12 | Multi-env config | Low | Low |

**Suggested order of attack for Codex:**
1 → 2 → 3 → 4 → 5 → 10 → 7 → 8 → 6 → 9 → 12 → 11

The first five items (CI, linting, Docker, auth, metrics) get you ~80% of the way to
"this looks like a company project" with the least effort.
