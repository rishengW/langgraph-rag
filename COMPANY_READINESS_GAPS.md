# Company Readiness Gaps — langgraph-rag

**Last audit:** 2026-06-15 — Quality-gate remediation complete after the
2026-06-14 readiness audits.

Generated from a full project audit evaluating eight weighted dimensions:
Architecture & Code Structure (17/20), Code Quality & Engineering Practices (14/15),
Testing (12/15), Documentation (9/10), DevOps & Infrastructure (8/10),
Security (7/8), Feature Completeness (8/10), and Operational Readiness (7/12).

Each section below lists what's missing, why it matters, and a concrete deliverable.
Ordered by priority × effort. Completed items are kept for audit trail.

---

## 1. CI/CD Pipeline — ✅ COMPLETE

**Deliverables:**
- [x] `.github/workflows/ci.yml` — GitHub Actions on push/PR:
  - Python 3.11 setup with pip cache
  - `pytest --tb=short --cov=src --cov-report=term --cov-fail-under=70`
  - `ruff check .`
  - `mypy src/`
- [ ] Optional: `.github/workflows/release.yml` — build + publish to PyPI if this becomes a package

**Why:** Automated quality gates. No regressions slip through.

---

## 2. Static Analysis & Code Quality Gates — ✅ COMPLETE

**Deliverables:**
- [x] `pyproject.toml` with Ruff (B/C4/E/F/I/SIM/UP), mypy `--strict`, pytest coverage gate at 70%
- [x] `.pre-commit-config.yaml` — ruff, ruff-format, mypy, check-yaml, check-toml, end-of-file-fixer, trailing-whitespace
- [x] `requirements-dev.txt` — pytest, pytest-cov, ruff, mypy, pre-commit, httpx

**Minor housekeeping:**
- [x] `CONTRIBUTING.md` reflects the configured Ruff, MyPy, pre-commit, and coverage checks
- [x] `.env.example` merge-conflict markers removed

**Why:** Enforces consistency. Catches type errors before they're bugs.

**2026-06-15 remediation update:** Ruff and strict MyPy now pass across the
full source tree, and the coverage gate passes at 70.77%. Verified with
`.\.venv\Scripts\ruff.exe check .`, `.\.venv\Scripts\mypy.exe src/`,
`.\.venv\Scripts\python.exe -m pytest -q`, and
`.\.venv\Scripts\python.exe -m pytest --tb=short --cov=src --cov-report=term --cov-fail-under=70`.

---

## 3. Containerization — ✅ COMPLETE

**Deliverables:**
- [x] `Dockerfile` — multi-stage build, python:3.11-slim, non-root user
- [x] `docker-compose.yml` — QA (:8000) + Chat (:8001) with named volumes for Chroma and session data
- [x] `.dockerignore` — excludes venv, git, pycache, pytest_cache, Chroma

**Why:** Reproducible environment. `docker compose up` for anyone.

---

## 4. Authentication & Authorization — 85% COMPLETE

**Deliverables:**
- [x] `src/api/auth.py` — Bearer token auth with `hmac.compare_digest`, 401 on mismatch
- [x] Wired to all mutation endpoints (QA query, Chat CRUD)
- [x] Open when `API_KEY` is unset (local development)
- [ ] **Rate limiting** — token-bucket or `slowapi` middleware on mutation endpoints
- [ ] **Per-session API keys** for chat threads (currently one global key)

**Why:** Any API exposed beyond localhost needs auth + abuse protection.

---

## 5. Observability & Monitoring — 60% COMPLETE

**Deliverables:**
- [x] `GET /metrics` (JSON snapshot from `MetricsCollector`) on both QA and Chat
- [x] `GET /health` (liveness) on both apps
- [x] `GET /ready` (readiness — local state checks, no external probes) on both apps
- [ ] **Structured JSON logging** — configurable via `LOG_FORMAT=json`, including:
  - `timestamp`, `level`, `logger`, `message`, `request_id`, `session_id`, `thread_id`
- [ ] **Request correlation IDs** — injected at middleware, threaded through graph execution
- [ ] **OpenTelemetry tracing** — spans for graph node execution, LLM calls, retrieval

**Why:** You can't debug what you can't see. Structured logs + tracing are the minimum for production observability.

---

## 6. Configuration Hardening — 75% COMPLETE

**Deliverables:**
- [x] Fail-fast on missing `DASHSCOPE_API_KEY`
- [x] `config/default.yaml` fully documented with inline comments
- [x] Multi-environment overlays (dev/staging/prod) via `RAG_ENV`
- [x] Secrets excluded from YAML (env/`.env` only)
- [ ] **Secret masking in all log emissions** — partial coverage today (`secret_fingerprint` exists but isn't universal)
- [ ] **Production config hardening** — disable debug endpoints, stricter CORS, lock down tool defaults
- [ ] **Config validation at startup** — validate all Settings fields, not just API key presence (e.g., Chroma path writability, model name validity)

**Why:** Misconfiguration is the #1 cause of production incidents. Fail fast, fail clearly.

---

## 7. API Versioning & Documentation — 0% COMPLETE

**Deliverables:**
- [ ] **API versioning** — prefix all routes with `/api/v1/`, keep old routes as deprecation shims for one release
- [ ] **OpenAPI customization** — title, description, version in FastAPI app config
- [ ] **Pydantic `Field(description=...)`** on all request/response models for useful auto-generated docs
- [ ] **`openapi.json` export script** for sharing with API consumers

**Why:** Versioned APIs let you evolve without breaking clients. Good OpenAPI docs reduce support burden.

---

## 8. Database & Persistence — 80% COMPLETE

**Deliverables:**
- [x] Session SQLite schema versioning via `schema_version` table
- [x] SQLite-backed LangGraph checkpoint persistence (`src/sessions/checkpoint.py`)
- [x] Chroma backup/restore documented in `ARCHITECTURE.md`
- [ ] **Online backup capability** — current procedure requires stopping servers; need live snapshot or export
- [ ] **`POST /chat/{id}/export`** — export a session transcript as JSON/Markdown
- [ ] **Formal multi-version migration runner** — current migrations are manual; need an automated runner for schema v1 → vN

**Why:** Schema changes WILL happen. Versioned, automated migrations prevent data loss.

---

## 9. Security — 75% COMPLETE

**Deliverables:**
- [x] `SECURITY.md` — vulnerability reporting, supported versions
- [x] CORS middleware driven by `cors_allow_origins`/`CORS_ALLOW_ORIGINS`
- [x] Hardcoded secrets audit — zero found
- [x] API-key auth on all mutation endpoints
- [x] Security policy reflects implemented API-key auth and configurable CORS
- [ ] **Dependency scanning** — Dependabot or `pip-audit` CI workflow to flag known CVEs in the LangChain/FastAPI/Chroma/DashScope chain
- [ ] **Secrets scanning in CI** — `detect-secrets` or `trufflehog` to prevent accidental key commits
- [ ] **Content Security Policy** — CSP headers on static UI endpoints

**Why:** Dependency vulns are the easiest attack vector. Even internal tools need security hygiene.

---

## 10. Documentation for Teams — 90% COMPLETE

**Deliverables:**
- [x] `CONTRIBUTING.md` — setup, testing, linting, commit style
- [x] `ARCHITECTURE.md` — system design, request flows, persistence, backup procedure
- [x] `CHANGELOG.md` — versioned change tracking
- [x] `SECURITY.md` — vulnerability reporting and secret handling
- [x] `COMPANY_READINESS_GAPS.md` — this file
- [x] `.env.example`, `SECURITY.md`, and `CONTRIBUTING.md` cleaned up after the 2026-06-14 audit
- [ ] **`CLAUDE.md`** — AI assistant onboarding (project conventions, architecture summary, verification commands)
- [ ] **Docstring coverage audit** — many public functions have docstrings but coverage hasn't been systematically verified
- [ ] **Inline `# REFACTOR:` cleanup** — ~dozen+ REFACTOR annotations throughout codebase should be triaged (convert to issues or resolve)

**Why:** The bus factor. If someone else needs to work on this, they need complete onboarding docs.

---

## 11. Testing — 70% COMPLETE

**Deliverables:**
- [x] 18 test files covering graph builder, sessions, API, streaming, config, tools, web search, URLs, retry
- [x] DI-friendly architecture (`GraphProviders`, `GraphNodeOverrides`) enables offline testing
- [x] CI coverage gate at 70%
- [ ] **Integration tests** — all tests are offline/mocked today; add a small smoke suite against real providers (or recorded fixtures via `vcr.py`)
- [ ] **Snapshot/regression tests** for LangGraph execution traces — capture graph event streams and assert on structure
- [ ] **Coverage floor raise** — target 80%+ (currently at 70%)
- [ ] **Property-based tests** for document quality filtering, URL scoring, and reranking (good candidates for Hypothesis)

**Why:** 70% is a decent floor, but the jump to 80%+ catches real regressions. Integration tests validate that the provider seams actually work.

---

## 12. Production Deployment Config — 0% COMPLETE

**Deliverables (pick your platform):**
- [ ] **AWS:** `infra/` with CDK or Terraform for ECS Fargate + EFS (Chroma persistence)
- [ ] **GCP:** Cloud Run + Cloud Storage FUSE for Chroma
- [ ] **On-prem:** systemd unit files for QA and Chat services
- [ ] **Minimal:** `Procfile` or `fly.toml` for quick PaaS deployment
- [ ] **Health check integration** — wire `/health` and `/ready` into the orchestrator's probes

**Why:** The gap between "runs on my machine" and "runs in production" is mostly deployment config.

---

## 13. Multi-Environment Support — 85% COMPLETE

**Deliverables:**
- [x] `config/development.yaml`, `config/staging.yaml`, `config/production.yaml`
- [x] `RAG_ENV` env var for environment overlay selection
- [ ] **Production config** — tighter CORS, disabled debug endpoints, higher rate limits, locked-down tool defaults
- [ ] **Staging config** — realistic production-like settings (not currently populated meaningfully)

**Why:** You should never run the same config in dev and prod.

---

## 14. Newly Identified Gaps (2026-06-14 Audit)

These were discovered during the scoring audit and were not in the original gaps document:

- [ ] **No CLAUDE.md** — AI coding assistants have no project-level instructions. Should document conventions, architecture summary, and verification commands.
- [x] **`.env.example` has merge conflicts** — resolved on 2026-06-15.
- [x] **`CONTRIBUTING.md` is stale** — updated on 2026-06-15 to document Ruff/Mypy/pre-commit/coverage checks.
- [ ] **No online backup** — current procedure requires stopping servers to copy Chroma SQLite files safely.
- [ ] **`src/core/` backward-compat layer** — all modules emit `DeprecationWarning`; should plan removal timeline and cut over remaining consumers.
- [ ] **`httpx` in `requirements-dev.txt` is unused** — listed but no test imports it; either adopt it for async API testing or remove.
- [ ] **Legacy workbenches in tree** — `advanced_phase1_workbench/` and `advanced_phase2_interfaces_workbench/` are excluded from linting but still occupy disk; should be archived or removed.

---

## Summary — Effort & Impact Matrix (Updated 2026-06-14)

Completed items are shown with a checkmark for audit trail.

| # | Gap | Effort | Impact | Status |
|---|-----|--------|--------|--------|
| 1 | CI/CD (GitHub Actions) | Low | **High** | ✅ |
| 2 | Linting + Type Checking | Low | **High** | ✅ |
| 3 | Docker + Compose | Low | **High** | ✅ |
| — | **Quality gates green under Ruff/MyPy/coverage** | Medium | **High** | ✅ |
| — | **Structured JSON logging + correlation IDs** | Medium | **High** | ⬜ |
| 4 | Auth (API key) | Low | Medium | ✅ |
| — | **Rate limiting** | Low | Medium | ⬜ |
| 5 | `/metrics` + `/health` + `/ready` | Low | Medium | ✅ |
| 6 | Config validation + secret masking | Low | Medium | 🔶 |
| 7 | API versioning (`/api/v1/`) | Medium | Medium | ⬜ |
| — | **Integration tests + coverage to 80%** | Medium | **High** | ⬜ |
| 8 | DB migrations beyond v1 + online backup | Medium | Medium | 🔶 |
| 9 | Security (CORS + dep scanning + secrets CI) | Low | Medium | 🔶 |
| 10 | Team docs + CLAUDE.md | Medium | Medium | 🔶 |
| — | **CLAUDE.md + stale doc cleanup** | Low | Medium | 🔶 |
| 11 | Deployment config | Medium | Medium | ⬜ |
| 12 | Multi-env config hardening | Low | Low | 🔶 |
| 13 | Production config hardening | Low | Low | ⬜ |
| — | Deprecate `src/core/` + archive workbenches | Low | Low | ⬜ |

**Legend:** ✅ Complete &nbsp; 🔶 Partial &nbsp; ⬜ Not started

---

## Suggested Order of Attack for Next Sprint

After the 2026-06-15 quality-gate remediation, the next readiness lift should
focus on production operations rather than CI cleanup:

1. **Structured JSON logging + correlation IDs** (Medium effort, High impact) — Single biggest observability gap
2. **Rate limiting** (Low effort, Medium impact) — Protect mutation endpoints from abuse
3. **Dependency scanning in CI** (Low effort, Medium impact) — Block known CVEs at the PR gate
4. **CLAUDE.md + remaining docs polish** (Low effort, Medium impact) — Add AI assistant onboarding and triage lingering `# REFACTOR:` annotations
5. **Integration test smoke suite** (Medium effort, High impact) — Even 3–5 tests against recorded fixtures catch provider-seam regressions

The jump from **90 → 95+** requires deployment config, API versioning, and production hardening — these are higher-effort and depend on where the project will actually run.


---

# Second Audit — Claude Opus 4.7 (Kiro)

**Audit date:** 2026-06-14
**Auditor:** Claude Opus 4.7 via Kiro
**Method:** Full repo walk + verification commands run locally (`pytest --cov`, `ruff check`, `mypy src/`, LOC count, source sampling, doc/CI/Docker review).

This is a second, independent audit added below the original (Claude Code / DeepSeek v4 Pro) audit so the two evaluations can be compared side-by-side. The two scores (Claude Code: **82**, this audit: **78**) differ by 4 points; the largest delta comes from how each auditor weighed lint/type-check debt and CI green-state.

**2026-06-15 update:** The CI green-state blockers identified in this historical
audit have been remediated. `ruff check .`, `mypy src/`, `pytest -q`, and the
70% coverage gate now pass locally. The stale `.env.example`, `SECURITY.md`,
and `CONTRIBUTING.md` findings were also fixed.

## Overall: **78 / 100**

Solid mid/senior-level engineering, production-capable for an internal beta. Not yet hardened for an external SaaS launch.

The project's own `COMPANY_READINESS_GAPS.md` self-scores 82. After verifying the codebase end-to-end against an independent metric set, this audit lands 4 points lower because the self-score doesn't fully account for:

- Strict-mypy debt and Ruff drift as of 2026-06-14 (remediated on 2026-06-15)
- Duplicated `core/qa/chat` vs `graph/api/sessions` layering — the migration is half-finished
- The `.env.example` merge conflict found on 2026-06-14 (remediated on 2026-06-15)
- "CI exists" vs "CI is green" — both Ruff and MyPy CI gates failed on 2026-06-14, but pass after the 2026-06-15 remediation

## Verification Snapshot (2026-06-14)

| Signal | Value |
|---|---|
| Source files | 88 (`src/`) |
| Source LOC | 10,718 |
| Test files | 18 |
| Test count | 171 passing |
| Coverage | 70.54% (gate 70%, branch on) |
| Ruff errors | 97 (65 auto-fixable) |
| Mypy strict errors | 124 across 28 files |
| `print()` in non-CLI source | 0 (CLI entrypoints only) |
| `TODO`/`FIXME`/`HACK` markers | 0 |
| `# REFACTOR:` annotations | ~25 (change rationale, not active TODOs) |
| Merge-conflict markers | 1 (`.env.example`) |

## Verification Snapshot (2026-06-15)

| Signal | Value |
|---|---|
| Ruff | `ruff check .` passes |
| MyPy strict | `mypy src/` passes, 88 source files checked |
| Tests | 171 passing |
| Coverage | 70.77% (gate 70%, branch on) |
| Merge-conflict markers | 0 in `.env.example`, `SECURITY.md`, `CONTRIBUTING.md`, `README.md` |
| Whitespace | `git diff --check` passes |

## Score Breakdown (10 weighted dimensions)

| # | Dimension | Weight | Score | Weighted |
|---|---|---|---|---|
| 1 | Architecture & module design | 15 | 12 | 12 |
| 2 | Code quality & idiomatic Python | 15 | 10 | 10 |
| 3 | Testing | 12 | 9 | 9 |
| 4 | Documentation | 8 | 7 | 7 |
| 5 | DevOps / CI / containerization | 10 | 8 | 8 |
| 6 | Security & secret handling | 10 | 7 | 7 |
| 7 | Observability & operations | 10 | 5 | 5 |
| 8 | Configuration management | 8 | 6 | 6 |
| 9 | Performance & resource handling | 7 | 6 | 6 |
| 10 | Maintainability & dev experience | 5 | 4 | 4 |
| | **Total** | **100** | | **78** |

### 1. Architecture & module design — 12 / 15

**Strengths**
- Clean functional decomposition: `graph/`, `rag/`, `web_search/`, `sessions/`, `api/`, `llm/`, `tools/`, `utils/`. Each ships a `SKILL.md` describing scope and intent.
- Excellent DI seams: `GraphProviders` / `GraphNodeOverrides` make the graph fully mockable. `JsonRequester` for tools, `WebSearchProvider` protocol, and the `LLMProvider` seam are textbook.
- Two-graph design (full Chroma path + lightweight fetch→prompt path) is well-motivated and now correctly routes per tool.
- Typed event system + `MetricsCollector` are real foundations for streaming and observability, not stubs.
- Typed error hierarchy in `src/errors.py` mapped to HTTP status in `api/errors.py`.

**Costs**
- `src/qa/api.py` and `src/chat/api.py` are the active FastAPI apps but `src/api/` exists as a partial unification (auth, dependencies, errors, streaming, models). The migration to `src/api/routers/` is unfinished. The two apps duplicate ~70% of source-refresh and graph-rebuild logic.
- `src/core/` deprecation shim has no removal timeline; every import emits a `DeprecationWarning` at startup.
- `qa/main.py` and `chat/main.py` mix CLI parsing, graph building, and REPL with `print()` for status — fine for a CLI, but mode-dispatch is duplicated across both.

### 2. Code quality & idiomatic Python — 10 / 15

**Strengths**
- Type hints on virtually every public symbol; `from __future__ import annotations` consistent.
- Zero `TODO`/`FIXME`/`HACK` markers in `src/` (rare and impressive).
- Logging uses `%s` lazy formatting consistently — no f-strings in log calls.
- Frozen dataclasses for `Settings` and `FetchPolicy`. Immutable by default.
- Retry helpers cleanly separate connection-error retry from generic retry.

**Historical costs from the 2026-06-14 audit, updated with remediation status**
- **Ruff drift as of 2026-06-14: 97 errors.** Top categories:
  - 37 × `I001` unsorted imports (auto-fixable)
  - 16 × `B008` (FastAPI `Depends()` in defaults — debatable, but the rule catches it)
  - 13 × `UP045` (`Optional[X]` instead of `X | None`)
  - 5 × `E402` module-import-not-at-top
  - 5 × `UP035` deprecated-import
  - Smaller categories below
  - 65 of 97 were auto-fixable. This made the lint gate fail on 2026-06-14; remediated on 2026-06-15 and `ruff check .` now passes.
- **Mypy strict drift as of 2026-06-14: 124 errors across 28 files.** Mostly `graph/` (44), `chat/` (27), `web_search/` (23). Real issues included missing return types on FastAPI handlers, untyped function calls in typed contexts, and `Missing named argument "messages" for "QueryResponse"`. Remediated on 2026-06-15 and `mypy src/` now passes.
- `.env.example` had unresolved merge conflict markers (lines 7–11). Fixed on 2026-06-15; no conflict markers remain in the readiness docs/env files checked.
- ~25 `# REFACTOR:` annotations across the source — change rationale, not active TODOs, but noise in a "finished" codebase.
- Handful of `print()` calls in non-CLI modules; acceptable as user-facing CLI output but inconsistent with the structured-logging direction.

### 3. Testing — 9 / 12

**Strengths**
- 171 passing tests in 18 files, clean offline runs (~5s).
- 70.5% coverage with a CI gate at 70%, branch coverage on.
- Smart use of DI: graph routing tests use `GraphProviders` to inject mock agents and tools rather than spinning up real LLMs.
- Test naming is descriptive (`test_web_answer_returns_grounded_refusal_when_no_readable_pages`, `test_build_lightweight_graph_routes_non_web_search_tools_back_to_agent`).

**Costs**
- **No integration tests.** Every test mocks providers. Real DashScope, real DeepSeek, real Bing, real Chroma never hit CI. A `vcr.py`-backed smoke suite or recorded-fixture suite would catch provider-seam regressions.
- No load/perf tests. For something with embedding + vector retrieval + LLM calls, baseline latency budgets per node would matter for production.
- Coverage at the floor (70%). The gate is enforceable but tight; several paths in `qa/api.py` and `chat/api.py` are below module-level threshold.
- No property-based tests. Document quality scoring, URL filtering, and reranking are good Hypothesis candidates.

### 4. Documentation — 7 / 8

**Strengths**
- `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, `COMPANY_READINESS_GAPS.md`, `PROBLEMS_DETECTED.md` all present and substantive.
- Per-module `SKILL.md` files describe scope and design intent. Rare, and great for onboarding.
- Endpoint table, env-var table, persistence section, Docker section in README.
- Tracked technical debt in two places (`COMPANY_READINESS_GAPS.md`, `PROBLEMS_DETECTED.md`) with status legends.

**Costs**
- `CONTRIBUTING.md` was stale in the 2026-06-14 audit. It was updated on 2026-06-15 to document the configured Ruff, MyPy, pre-commit, and coverage checks.
- No API reference doc (relies on FastAPI's auto-generated `/docs`).
- Mermaid in `ARCHITECTURE.md` is text-only — fine, but a rendered diagram would help non-technical stakeholders.
- No runbook / on-call doc.

### 5. DevOps / CI / containerization — 8 / 10

**Strengths**
- Multi-stage `Dockerfile` with non-root user, separate venv builder.
- `docker-compose.yml` runs both apps on different ports with named volumes.
- `.dockerignore` excludes the right things.
- GitHub Actions workflow runs pytest with coverage, ruff, and mypy on push/PR.
- `.pre-commit-config.yaml` wired with the standard set.

**Costs**
- **CI was red on 2026-06-14.** The workflow correctly failed on the 97 ruff and 124 mypy errors. This was the "CI exists but isn't enforced" pattern; the 2026-06-15 remediation restored green local gates.
- No release workflow, no tagging, no version bumping automation.
- No staging/prod deploy IaC.
- Docker image not built/published anywhere.
- No pip-audit / dependency-vulnerability scan in CI.

### 6. Security & secret handling — 7 / 10

**Strengths**
- `SECURITY.md` with reporting flow.
- API-key auth via `hmac.compare_digest` (timing-safe). Gated only on mutation endpoints; health/ready/metrics open by design.
- No hard-coded secrets in source.
- `.gitignore` excludes `.env` and `.chroma`.
- `secret_fingerprint()` for safe logging of key identity.
- CORS configurable via `cors_allow_origins`.

**Costs**
- **No rate limiting.** `/chat`, `/chat/{id}/message`, `/query` are all unthrottled. Trivial to DoS or run up a DashScope bill.
- No dependency scanning. Dependabot/pip-audit not configured.
- No secrets scanning (detect-secrets / trufflehog) on commits.
- Single global API key. No per-tenant or per-session credentialing.
- `.env.example` had merge conflict markers in the 2026-06-14 audit. This was resolved on 2026-06-15.
- `web_search_verify_ssl` and `SSL_FIX.md` document a "set false to bypass" knob — fine for local dev, but the existence of that path is a foot-gun if it leaks to production config.

### 7. Observability & operations — 5 / 10

**Strengths**
- `/health`, `/ready`, `/metrics` on both apps.
- `MetricsCollector` records typed graph events (`NodeStart`, `NodeEnd`, `Error`, `RetrieverResult`, `GraderDecision`).
- SSE streaming endpoints (`/query/stream`, `/chat/{id}/message/stream`) emit those events to clients.
- Per-page extraction logging (chars/tokens/error) lets you distinguish "discovery miss" from "extraction miss".
- Logging uses module-level loggers consistently.

**Costs**
- **No structured (JSON) logging.** Plain text only. Hard to ship to ELK/Loki/Datadog.
- **No request correlation IDs.** A request fanning out into agent → tool → web fetch → LLM produces logs that can't be tied back to one trace.
- No OpenTelemetry / distributed tracing. No spans for graph-node execution or external calls.
- No alerting hooks. The metrics endpoint is JSON pull-based; nothing pushes to a real metrics backend.
- No log levels per environment. Always INFO.

### 8. Configuration management — 6 / 8

**Strengths**
- Layered config with documented precedence: CLI > env > YAML > defaults.
- `config/default.yaml` plus per-env overlays (`development.yaml`, `staging.yaml`, `production.yaml`) selected by `RAG_ENV`.
- `Settings` is a frozen dataclass — immutable at runtime.
- Secrets stay out of YAML by design.
- Fail-fast on missing `DASHSCOPE_API_KEY`.

**Costs**
- `Settings` is a 40-field flat dataclass. Works but doesn't compose. The plan to split into namespaced sub-configs (LLMConfig, RetrievalConfig, etc.) is documented but unimplemented.
- Staging/production YAMLs aren't meaningfully different from `default.yaml` — not validated to produce production-shaped behavior.
- Limited config validation beyond API-key presence. Chroma path writability, model name validity, etc. aren't checked at startup.

### 9. Performance & resource handling — 6 / 7

**Strengths** (most of the heavy lifting in `PROBLEMS_DETECTED.md` is real work):
- Parallel URL loading via `ThreadPoolExecutor` (default 4 workers).
- In-process `SourceDocumentCache` (opt-in via `page_load_cache_ttl_seconds`).
- Per-page token budget on web fetches (`web_search_max_page_tokens=8000`).
- Lightweight graph bypasses Chroma rebuild for one-shot web-search queries.
- Embedding-config fingerprinting prevents unnecessary Chroma rebuilds.
- Retry with exponential backoff + jitter on DashScope calls.
- Async FastAPI with `asyncio.to_thread` wrapping sync graph invocations.

**Costs**
- LangGraph is invoked synchronously and offloaded to threads — no native async path through the graph.
- No request timeout / global cancel propagation. A slow LLM call can tie up a worker thread for the full `dashscope_request_timeout` (120s default).
- No connection pool sharing between providers.

### 10. Maintainability & dev experience — 4 / 5

**Strengths**
- `pyproject.toml` is clean and modern.
- `requirements.txt` pinned to exact versions; `requirements-dev.txt` separate.
- Pre-commit hooks installed.
- `SKILL.md` per module is excellent for AI-assisted maintenance.
- `MEMORY.md` and `memory/` directory capture process state across sessions.

**Costs**
- Before the 2026-06-15 remediation, lint/type-check debt (see #2) made the bar for "is my change ready to merge" unclear.
- `advanced_phase1_workbench/` and `advanced_phase2_interfaces_workbench/` directories sit in tree (now empty stubs but still noise).
- `install_log.txt` and `graph.png` are committed artifacts that probably shouldn't be.

## Comparison With The First Audit

| Dimension | Claude Code self-score | This audit | Delta | Reason |
|---|---|---|---|---|
| Overall | 82 | 78 | **−4** | Historical lint/type gate failure weighed heavier here |
| Architecture | 17/20 | 12/15 (≈16/20) | −1 | Duplicate qa/chat API layering not yet collapsed |
| Code quality | 14/15 | 10/15 | **−4** | 97 ruff errors + 124 mypy errors, env.example merge conflict |
| Testing | 12/15 | 9/12 (≈11/15) | −1 | Same observation: integration test gap |
| Docs | 9/10 | 7/8 (≈9/10) | 0 | Same |
| DevOps | 8/10 | 8/10 | 0 | Same |
| Security | 7/8 | 7/10 (≈6/8) | −1 | Heavier penalty for missing rate limiting + dep scanning |
| Operational readiness | 7/12 | 5/10 (≈6/12) | −1 | Heavier penalty for no JSON logs + no correlation IDs |
| Feature completeness | 8/10 | not separately scored | n/a | Folded into Architecture in this rubric |
| Configuration management | not separately scored | 6/8 | n/a | Split out from Architecture |
| Performance | not separately scored | 6/7 | n/a | New dimension |
| Dev experience | not separately scored | 4/5 | n/a | New dimension |

The two audits agree on the broad shape: solid engineering, real DI seams, real testing infrastructure, real gaps in observability + production hardening. They disagree on how heavily to penalize the 2026-06-14 CI failure state: the first audit credits items as "complete ✅" when configured (e.g., "Static Analysis & Code Quality Gates — ✅ COMPLETE"), this audit credits them only when **green**. On 2026-06-14, 124 mypy errors and 97 ruff errors blocked the configured gates, so "complete" overstated that day's state; the 2026-06-15 remediation corrected this.

## Historical Top 5 Items From 2026-06-14 (78 → 88)

These were different from the first audit's top-5 because they prioritized unblocking already-built infrastructure over building new infrastructure. Items marked completed were remediated on 2026-06-15.

1. **CI green-state remediation — completed 2026-06-15.** Ruff and strict MyPy blockers were fixed; `ruff check .` and `mypy src/` now pass. Keep these gates enforced as the merge bar.
2. **`.env.example` conflict cleanup — completed 2026-06-15.** Conflict markers were removed and the checked docs/env files are clean.
3. **Structured JSON logging + request correlation IDs.** Single biggest production-readiness gap. **Effort: 1 day. Impact: high.**
4. **Rate limiting on mutation endpoints.** `slowapi` middleware. **Effort: half a day. Impact: medium-high (cost protection + DoS protection).**
5. **Contributor docs refresh — completed 2026-06-15.** `CONTRIBUTING.md` now documents the Ruff/MyPy/pre-commit/coverage bar.

## Top Items For 88 → 95

- Finish `qa/api.py` + `chat/api.py` → `src/api/routers/` consolidation. The duplication is real and growing.
- OpenTelemetry tracing.
- Real integration tests against recorded provider fixtures (`vcr.py` or similar).
- Production deployment IaC (ECS / Cloud Run / k8s).
- API versioning under `/api/v1/`.
- Per-tenant API keys.
- Dependency scanning + secrets scanning in CI.

## What I Specifically Push Back On From The First Audit

- **Self-score of 82 was generous on 2026-06-14 because CI was red on both lint and types.** The first audit's matrix showed ✅ COMPLETE for "Static Analysis & Code Quality Gates" — the *configuration* was complete, but the codebase did not pass that configuration until the 2026-06-15 remediation.
- **Several "Fixed ✅" items in `PROBLEMS_DETECTED.md` are actually opt-in.** URL caching is opt-in with default disabled, JS fallback is opt-in, semantic-quality fallback for search is still out of scope. The status legend is honest about this, which is good — but the headline summary undersells the residual risk.
- **The "Suggested Order of Attack" jumped straight to JSON logging + rate limiting + dep scanning before the CI gate cleanup.** That cleanup is now complete, so the next highest-impact work is production operations: structured logging, rate limiting, dependency scanning, integration smoke tests, and deployment hardening.

## Verdict

For an internal LangGraph RAG service backing a small team, this is shippable today.

For a product with paying external users, it needs the top-5 items above before going public. The infrastructure is mostly in place — the gap is enforcement and the last 20% of plumbing (correlation IDs, rate limits, integration tests).

The codebase is genuinely well-architected. With the quality gates cleared on 2026-06-15, the next focused sprint can move the score toward ~88 by adding JSON logging, request correlation IDs, rate limiting, dependency scanning, and integration smoke tests without any large-effort items.
