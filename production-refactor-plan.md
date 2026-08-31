# Production Refactor Plan

**Project:** `langgraph-rag`  
**Status:** Proposed  
**Scope:** Production hardening, inbound MCP server, optional outbound MCP clients  
**Reference:** Comparison with `D:\deer-flow-main` and the existing `.kiro/specs/mcp-server/requirements.md`

## 1. Executive Summary

Refactor `langgraph-rag` into a transport-neutral RAG/agent runtime with thin HTTP, CLI, and MCP adapters. Preserve the project's strong RAG pipeline, typed streaming events, session rollback, and provider seams. Adopt DeerFlow's useful separation between reusable agent logic and product-facing applications, but do not copy its unsafe MCP configuration exposure, partial initialization, arbitrary command execution, or development runtime deployment patterns.

Deliver production readiness in stages:

1. Restore green quality gates and correct the MCP/security requirements.
2. Extract application services from FastAPI handlers.
3. Implement a stateless inbound MCP server with `stdio` and authenticated streamable HTTP.
4. Introduce an immutable, policy-aware tool catalog.
5. Optionally add lifecycle-managed outbound MCP clients.
6. Replace single-process persistence and authorization where multi-replica or multi-tenant operation is required.
7. Add production telemetry, deployment hardening, security scanning, and operational runbooks.

## 2. Goals

- Expose RAG capabilities safely to Kiro, Claude, Cursor, and other MCP clients.
- Optionally let the agent consume approved external MCP tools.
- Keep graph, retrieval, and session logic independent of transport frameworks.
- Make tool discovery, authorization, execution, and observability consistent.
- Support a defensible single-instance internal production topology first.
- Define a clear path to horizontally scalable, multi-tenant deployment.
- Preserve existing HTTP and CLI behavior during migration.

## 3. Non-Goals for the First Release

- Exposing every internal/local tool over MCP.
- Stateful MCP chat sessions before identity and ownership are implemented.
- Arbitrary user-configured `stdio` commands in hosted production.
- Horizontal scaling while SQLite, local Chroma, local uploads, and process-local locks remain.
- Replacing the existing RAG graph or web-search pipeline.
- Building an MCP administration UI before backend policy and redaction are complete.
## 4. Architectural Decisions

### 4.1 Separate MCP Directions

Treat these as separate bounded contexts:

- **Inbound MCP server:** external MCP clients invoke `langgraph-rag` RAG capabilities.
- **Outbound MCP client:** `langgraph-rag` discovers and invokes tools hosted by external MCP servers.

They may share tool metadata, policy, telemetry, and secret-provider abstractions, but they must not share transport lifecycle or startup entry points.

### 4.2 Target Runtime Shape

```text
Clients
  |-- Browser / REST client
  |-- CLI
  |-- MCP client (Kiro, Claude, Cursor)
  v
Transport adapters
  |-- HTTP API adapter
  |-- CLI adapter
  |-- Inbound MCP adapter
  v
Application services
  |-- RagApplicationService
  |-- ChatApplicationService
  |-- TurnExecutionService
  |-- SessionLifecycleService
  v
Domain/runtime
  |-- LangGraph workflows
  |-- ToolCatalog + ToolPolicy
  |-- RAG and web search
  |-- Sessions and checkpoints
  |-- Typed events and artifacts
  v
Infrastructure
  |-- LLM/search providers
  |-- Vector store
  |-- SQL/checkpoint store
  |-- Object storage
  |-- Optional outbound MCP servers
```

### 4.3 Proposed Package Layout

```text
src/
  application/
    models.py
    errors.py
    rag_service.py
    chat_service.py
    turn_execution.py
    session_lifecycle.py
  adapters/
    http/
    cli/
    mcp_server/
      main.py
      config.py
      auth.py
      tools.py
      transport.py
  mcp/
    catalog.py
    policy.py
    telemetry.py
    models.py
    client/
      config.py
      manager.py
      provider.py
      oauth.py
      adapter.py
  graph/
  rag/
  tools/
  sessions/
```

Initially these remain one repository and Python distribution. Split packages only after boundaries are stable and independently reusable.

## 5. Design Principles

1. FastAPI, CLI, and MCP handlers call application services; application services never import transport frameworks.
2. Tool schemas bound to the model and tools available to the dispatcher always come from the same immutable catalog snapshot.
3. Every tool call passes through authorization, validation, timeout, output bounding, redaction, audit, and metrics.
4. Production network transports fail closed when authentication or required dependencies are unavailable.
5. External MCP metadata and results are untrusted data, not instructions.
6. Configuration updates are versioned and published atomically; active runs keep their original generation.
7. Secrets are referenced, not returned, logged, or persisted in browser-readable configuration.
8. Local development defaults must never silently become production defaults.
## 6. Phase 0 — Requirements and Quality Baseline

### Tasks

- Fix all current `mypy src/` errors and restore a green merge gate.
- Update `.kiro/specs/mcp-server/requirements.md` to distinguish inbound server and outbound client behavior.
- Change production HTTP MCP behavior to fail startup when authentication is absent.
- Keep anonymous access only behind an explicit development setting.
- Add requirements for identity, ownership, rate limits, deadlines, schema limits, output limits, SSRF prevention, audit events, and secret redaction.
- Define required versus optional MCP dependencies and atomic/degraded startup behavior.
- Decide the first supported topology: single-instance internal production or multi-replica service.
- Create MCP `design.md` and `tasks.md` after requirements approval.

### Exit Criteria

- Tests, Ruff, formatting, and strict mypy pass from a clean environment.
- Security and transport behavior are unambiguous in the spec.
- The supported deployment topology is documented and enforced.
- No implementation begins with unresolved authentication or persistence assumptions.

## 7. Phase 1 — Extract Transport-Neutral Application Services

### Tasks

- Move graph invocation and result shaping out of `src/chat/api.py` and `src/qa/api.py`.
- Extract pre-turn checkpoint snapshot, cancellation, rollback, and completion hooks into `TurnExecutionService`.
- Extract graph/session creation and source refresh into `SessionLifecycleService`.
- Define stable request/result models such as `RagRequest`, `RagAnswer`, `SourceReference`, `TurnRequest`, and `TurnResult`.
- Define public errors separately from internal exceptions; never return raw provider exceptions.
- Keep current API routes as compatibility adapters over the new services.
- Preserve SSE event ordering, tool events, artifacts, cancellation, and checkpoint rollback behavior.

### Primary Existing Files

- `src/chat/api.py`
- `src/qa/api.py`
- `src/graph/executor.py`
- `src/graph/builder.py`
- `src/sessions/registry.py`
- `src/sessions/checkpoint.py`

### Exit Criteria

- HTTP and CLI behavior remains backward-compatible.
- Application services can be invoked in tests without constructing FastAPI requests.
- Cancellation and failed turns restore checkpoint state.
- MCP adapters can call the RAG engine without importing an HTTP route.

## 8. Phase 2 — Stateless Inbound MCP Server

### Initial Tools

1. `rag_ask`: answer a question against configured or explicitly supplied sources.
2. `rag_web_search_answer`: discover sources and return a grounded answer.

### Input Bounds

- Question: 1–16,000 characters.
- Source URLs: maximum 10, HTTPS by default.
- Maximum source/result count: configurable and bounded.
- Request deadline: mandatory server-side maximum.
- Reject unknown or oversized fields before graph invocation.

### Stable Result Contract

```json
{
  "answer": "...",
  "sources": [
    {
      "url": "https://example.com",
      "title": "Example",
      "citation_id": "source-1"
    }
  ],
  "request_id": "...",
  "grounded": true,
  "warnings": []
}
```

### Transport and Lifecycle

- Add a separate entry point: `python -m src.adapters.mcp_server.main`.
- Support `stdio` for local clients; protocol output goes only to stdout and logs only to stderr.
- Support authenticated streamable HTTP as a separately selected transport.
- Do not start FastAPI when MCP starts, and do not start MCP from FastAPI lifespan.
- Register all tools atomically; startup fails if a required tool cannot initialize.
- Handle SIGINT/SIGTERM, stop accepting calls, cancel or drain active calls by policy, and close resources within a bounded grace period.
- Pin the MCP SDK to an exact tested version in every dependency manifest.

### Exit Criteria

- Kiro or another MCP client can list and invoke both tools over `stdio`.
- Authenticated HTTP discovery and invocation pass contract tests.
- No local file, editor, memory mutation, or internal graph-node tool is advertised.
- Tool errors are structured and sanitized.
- Logs contain request/tool/outcome metadata but no arguments, source content, or secrets.
## 9. Phase 3 — Immutable Tool Catalog and Policy Boundary

### Core Interfaces

```python
@dataclass(frozen=True)
class ToolDescriptor:
    qualified_name: str
    display_name: str
    source: Literal["builtin", "local", "mcp"]
    server_name: str | None
    description: str
    input_schema: dict[str, object]
    risk_level: Literal["read", "write", "execute", "admin"]

@dataclass(frozen=True)
class ToolCatalogSnapshot:
    generation: int
    tools: tuple[BaseTool, ...]
    descriptors: tuple[ToolDescriptor, ...]

class ToolProvider(Protocol):
    async def start(self) -> None: ...
    async def snapshot(self) -> ToolCatalogSnapshot: ...
    async def health(self) -> ProviderHealth: ...
    async def close(self) -> None: ...
```

### Tasks

- Replace duplicated conditional registration in `_resolve_tools` and `_resolve_lightweight_tools` with catalog providers.
- Implement providers for built-in tools, document tools, memory tools, and optional MCP tools.
- Namespace remote tools as `mcp__<server>__<tool>`.
- Validate name length, description length, schema size/depth, property counts, enum counts, and unsupported constructs.
- Detect collisions before publishing a generation.
- Add risk metadata and principal/tenant allowlists.
- Add a single tool execution pipeline:
  `authorize -> validate -> limit -> invoke -> bound -> redact -> audit -> record outcome`.
- Extend typed tool events with source server, catalog generation, duration, and sanitized outcome.

### Graph Consistency

Because `agent_factory()` binds schemas at graph creation and `ToolNode` receives a static tool list, dynamic changes must rebuild both together.

Use a graph cache key such as:

```text
(settings fingerprint, catalog generation, session policy fingerprint)
```

At the start of a turn, acquire one catalog snapshot and use it for both model binding and dispatch. Never mutate the tool list in an active graph.

### Exit Criteria

- Full and lightweight graphs consume the same catalog abstraction.
- A model cannot call a tool omitted from the dispatcher or dispatch a tool omitted from model schemas.
- Catalog updates are atomic and observable by generation.
- Existing local tools retain current behavior and tests.

## 10. Phase 4 — Optional Outbound MCP Clients

Implement only if the agent must consume GitHub, database, filesystem, browser, or other external MCP tools.

### Tasks

- Add typed MCP server configuration for approved `stdio` and remote HTTP transports.
- Resolve secret references through a secret-provider interface.
- Use a lifecycle-managed multi-server client with explicit startup and shutdown.
- Retain transport/client sessions for the lifetime of a catalog generation.
- Implement connection, discovery, invocation, reconnect, timeout, cancellation, and health behavior.
- Support OAuth token caching and refresh with per-server concurrency locks.
- Publish a new catalog only after all required servers validate successfully.
- Quarantine failed optional servers and expose degraded readiness explicitly.
- Consider deferred tool search when remote schema count exceeds the direct-binding budget.

### Production Restrictions

- Disable arbitrary `stdio` configuration in hosted production.
- Permit only allowlisted executable/image identities and fixed argument templates.
- Prefer sidecars or isolated workloads for approved local MCP servers.
- Require HTTPS and endpoint allowlists for remote servers.
- Block loopback, link-local, cloud metadata, and private-network destinations unless explicitly approved.
- Validate redirects and OAuth token endpoints independently.

### Exit Criteria

- External tools are namespaced, policy-checked, bounded, and fully auditable.
- Configuration reload cannot create a partial active catalog.
- All clients and subprocesses close cleanly during generation replacement and shutdown.
- Remote failure cannot corrupt session checkpoints or crash unrelated requests.
## 11. Phase 5 — Production Security, Persistence, and Scale

### Authentication and Authorization

- Replace the optional global API key with principal-based authentication for production.
- Prefer OIDC/JWT or workload identity; optionally require mTLS for service-to-service MCP.
- Require ownership checks for every session read, write, stream, upload, download, and delete operation.
- Protect MCP configuration and tool-policy administration with separate admin permissions.
- Fail production startup if network transports lack required authentication.
- Use constant-time credential comparison where shared keys remain temporarily supported.

### Abuse and Cost Controls

- Add per-principal and per-tenant rate, concurrency, token, search, and cost quotas.
- Bound message length, URL count, upload count/size, graph duration, tool calls per turn, retries, and response size.
- Propagate cancellation and one deadline through API/MCP, graph, LLM, search, retrieval, and tool layers.
- Return sanitized public errors with internal correlation IDs.

### Single-Instance Internal Production Option

Enforce and document:

- Exactly one chat worker/process.
- One encrypted persistent volume.
- No horizontal scaling.
- Restricted network access.
- Tested restart recovery and backup/restore.
- Explicit maintenance windows for SQLite/Chroma backups.

### Multi-Replica Option

Before horizontal scaling, move to:

- PostgreSQL for session metadata and LangGraph checkpoints.
- Alembic or an equivalent migration runner.
- Redis or database advisory locks for per-thread serialization.
- Object storage for uploads and generated artifacts.
- A server-mode vector database or PostgreSQL vector extension.
- Shared rate-limit/catalog metadata storage.
- Queue/worker execution for long ingestion and extraction jobs.
- Tenant IDs on every durable object.

Avoid loading pickle payloads from stores that another tenant or compromised component can modify.

### Exit Criteria

- The declared topology survives restart and dependency-failure tests.
- Unauthorized principals cannot access known thread IDs or artifacts.
- Limits prevent unbounded LLM/search/tool spend.
- Multi-replica deployment is prohibited until shared persistence and distributed locking are active.

## 12. Phase 6 — Observability and Operations

### Structured Logs

Emit JSON records containing bounded fields:

```text
service, environment, request_id, trace_id, principal_id, tenant_id,
thread_id, catalog_generation, mcp_server, tool_name, tool_call_id,
outcome, duration_ms
```

Never log authorization headers, OAuth tokens, secret values, full documents, or sensitive tool arguments.

### Metrics

- `mcp_server_connections`
- `mcp_server_health`
- `mcp_tool_calls_total`
- `mcp_tool_call_duration_seconds`
- `mcp_tool_failures_total`
- `mcp_tool_timeouts_total`
- `mcp_catalog_generation`
- `mcp_catalog_reload_failures_total`
- `graph_runs_total`
- `graph_run_duration_seconds`
- `llm_tokens_total`
- `provider_cost_total`
- `active_streams`

Do not use thread IDs, URLs, or other unbounded values as metric labels.

### Tracing

Create one OpenTelemetry trace across:

```text
HTTP/MCP request -> application service -> graph -> model -> tool/MCP -> retrieval/search -> response
```

### Health Model

- `/health/live`: process is alive.
- `/health/ready`: required database, checkpoint, catalog, model, and transport dependencies are ready.
- `/health/dependencies`: admin-only detailed dependency status.

### Operational Deliverables

- Deployment, rollback, backup, restore, key rotation, MCP outage, and incident runbooks.
- SLOs for availability, latency, tool failure rate, and grounded-answer success.
- Alerts based on user impact rather than log volume.
## 13. Deployment Target

Recommended production services:

```text
reverse-proxy / ingress
frontend
api-gateway
agent-runtime
mcp-server-http
mcp-connector-worker (optional)
postgres
redis
object-storage
vector-database
telemetry-collector
```

Local MCP clients continue to launch the `stdio` server as a separate subprocess.

### Container Controls

- Pin base images by digest.
- Run as non-root with a read-only root filesystem.
- Drop Linux capabilities and prohibit privileged containers.
- Do not mount the host Docker socket into application containers.
- Set CPU, memory, process, and filesystem limits.
- Add liveness/readiness probes and graceful termination periods.
- Apply network policies and egress restrictions.
- Generate SBOMs, scan images, and sign release artifacts.
- Never copy `.env`, MCP credentials, or local configuration secrets into images.

## 14. Administration UI — Last, Not First

After policy and redaction are complete, an admin-only UI may expose:

- Server name and enabled state.
- Redacted endpoint/transport configuration.
- Credential status, never credential values.
- Connection health and last sanitized error.
- Advertised tool inventory and risk class.
- Per-tool allow/deny policy.
- Active catalog generation.
- Test-connection and controlled reload actions.
- Audit history.

The browser must never receive command environments, authorization headers, OAuth tokens, client secrets, refresh tokens, or secret-bearing URLs.

## 15. Test and CI Plan

### MCP Contract Tests

1. Tool discovery and invocation over `stdio`.
2. Tool discovery and invocation over authenticated streamable HTTP.
3. Invalid required configuration fails atomically.
4. Optional server failure produces explicit degraded readiness.
5. Duplicate names are namespaced or rejected.
6. Oversized/deep schemas are rejected.
7. Timeouts and cancellation close upstream work.
8. Client disconnect follows configured cancel/continue policy.
9. Server restart and reconnect behavior.
10. OAuth refresh is concurrency-safe.
11. Secrets never appear in APIs, logs, errors, or snapshots.
12. SSRF and private-network restrictions.
13. Tool output truncation and redaction.
14. Checkpoint rollback after tool/MCP failure.
15. Catalog generation consistency during reload.
16. Authorization, ownership, and tenant isolation.
17. Load tests for concurrent streams and tool calls.

### Merge Gates

```text
pytest with branch coverage
ruff check
ruff format --check
strict mypy
MCP contract/integration tests
dependency vulnerability scan
secret scan
SAST
container scan
frontend lint/typecheck/build
migration upgrade/downgrade test
```

Pin direct production dependencies exactly or through a committed lock file, and test upgrades deliberately.

## 16. Migration and Rollback Strategy

- Keep current HTTP routes as compatibility adapters during service extraction.
- Add feature flags for inbound MCP, outbound MCP, catalog generations, and new persistence backends.
- Run old and new execution paths against recorded fixtures before switching defaults.
- Migrate sessions with explicit schema versions and idempotent migrations.
- Publish catalog generation changes atomically; retain the prior generation until active calls drain.
- Use blue/green or canary deployment for network MCP and persistence changes.
- Roll back application code independently from irreversible data migrations.
- Back up durable state before every migration and rehearse restoration.

## 17. Prioritized Backlog

### P0 — Release Blockers

- Restore strict mypy and all merge gates.
- Correct MCP production authentication requirements.
- Extract transport-neutral application services.
- Implement stateless inbound MCP over `stdio`.
- Add authenticated streamable HTTP MCP.
- Close session/history ownership gaps.
- Declare and enforce the supported deployment topology.

### P1 — Production Controls

- Immutable `ToolCatalog` and centralized `ToolPolicy`.
- Request, cost, concurrency, timeout, and cancellation limits.
- Structured logs, tracing, exported metrics, and audit events.
- Dependency/secret/container scanning.
- PostgreSQL, distributed locking, and object storage if replicas are required.
- MCP schema validation, output bounding, and hostile-metadata tests.

### P2 — Product and Scale

- Optional outbound MCP client subsystem.
- Deferred tool discovery/search.
- Administration API and UI.
- API versioning under `/api/v1/`.
- Release automation, infrastructure as code, SLOs, and disaster recovery exercises.
- Removal timeline for deprecated compatibility modules.

## 18. Definition of Done

The production refactor is complete only when:

- HTTP, CLI, and MCP use the same application services without duplicated graph logic.
- Every tool is cataloged, namespaced, authorized, bounded, observable, and auditable.
- Production network transports fail closed.
- Secrets cannot be retrieved through management APIs or logs.
- Cancellation and failures preserve checkpoint consistency.
- The supported deployment topology is enforced and tested.
- CI includes quality, security, MCP contract, migration, and deployment checks.
- Backup, restore, rollback, key rotation, and MCP outage procedures have been rehearsed.
- Multi-tenant or horizontally scalable claims are made only after identity, shared persistence, distributed locking, and tenant isolation are verified.
