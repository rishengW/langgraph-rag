# Design Document

## Overview

The production refactor introduces a transport-neutral application boundary before adding MCP transports. The inbound MCP server and optional outbound MCP client are separate bounded contexts. Phase 1 extracts the existing non-streaming QA orchestration into `RagApplicationService`; Phase 2 adds the stateless inbound MCP adapter. Outbound MCP remains prohibited in first-release deployment configuration.

The first supported production topology is one internal instance and exactly one chat worker using one protected persistent volume for SQLite, Chroma, uploads, and artifacts. Network access is restricted and authenticated. Horizontal scaling is rejected until shared persistence, distributed locks, object storage, a server vector store, and shared policy/quota state are implemented.

## Architecture

```text
HTTP / CLI / inbound MCP adapters
                 |
                 v
        application services
      RagApplicationService
                 |
                 v
 graph + RAG + web search runtime
                 |
                 v
 local providers and persistence

optional outbound MCP client (later phase)
  separate config + lifecycle + health
  -> policy/tool catalog boundary -> approved remote tools
```

### Boundary rules

- `src/application/` imports domain/runtime modules but never FastAPI or MCP SDK modules.
- HTTP and MCP adapters translate transport contracts to application models.
- Inbound and outbound MCP have separate entry points and active sessions.
- One immutable tool/configuration generation is validated before publication.
- Raw provider exceptions remain internal and are correlated by `request_id`.

## Components and Interfaces

### RagApplicationService

`RagApplicationService.ask()` owns URL normalization, optional discovery, source metadata, graph selection, rebuild serialization, global graph promotion, rollback, invocation, and result shaping. Infrastructure is supplied through `RagServiceDependencies`; process-global QA graph state is supplied through `RagGraphState` callbacks. This keeps direct tests independent of FastAPI while preserving current behavior.

## Data Models

```python
@dataclass(frozen=True, slots=True)
class RagRequest:
    question: str
    urls: str | list[str] | None
    rebuild: bool
    web_search: bool
    debug: bool
    request_id: str

@dataclass(frozen=True, slots=True)
class SourceReference:
    url: str
    title: str | None
    citation_id: str | None

@dataclass(frozen=True, slots=True)
class RagAnswer:
    answer: str | None
    error: str | None
    success: bool
    messages: list[str] | None
    source_urls: list[str] | None
    source_mode: str | None
    source_note: str | None
    request_id: str
    sources: tuple[SourceReference, ...]
```

`RagAnswer` retains legacy HTTP fields while adding stable source objects and a correlation ID for future adapters. The HTTP compatibility adapter returns only the existing `QueryResponse` fields.

### Inbound MCP adapter (Phase 2)

A later `src/adapters/mcp_server/` package will contain configuration, auth, transport, lifecycle, and canonical tool adapters. It will expose only `rag_ask` and `rag_web_search_answer`, translate their closed schemas to `RagRequest`, enforce limits/deadlines before invoking the service, and translate `RagAnswer` to the stable MCP result contract.

### Optional outbound MCP client (Phase 4)

The outbound client will use its own lifecycle manager and configuration namespace behind immutable tool catalog and policy interfaces. The first-release configuration schema rejects outbound MCP configuration entirely. No arbitrary hosted stdio command is accepted.

## Data and State

The Phase 1 QA service is stateless per call except for deliberate replacement of the process-global QA graph after a successful explicit-source rebuild. A rebuild snapshots the prior graph/settings, clears the global graph to release local file handles, and restores the snapshot on failure. Web-discovered graphs remain per-request and are not promoted.

Future network requests carry trusted `principal_id`, optional `tenant_id`, `request_id`, an immutable configuration generation, and a mandatory deadline. Durable resources record owner identity. Active requests retain their starting generation.

### Immutable tool catalog and policy boundary (Phase 3)

`src/mcp/` defines frozen `ToolDescriptor` and `ToolCatalogSnapshot` values plus provider lifecycle, catalog publication, policy execution, and bounded telemetry seams. Full and lightweight graph builders compose the same provider set; only the heavy retriever versus required lightweight web-search contribution differs. One validated snapshot supplies model binding and dispatch, and its generation/descriptors are attached to the compiled graph for turn execution.

Local providers cover injected tools, retrieval, memory, builtin read/compute tools, local document readers, and session-confined editors. Outbound MCP remains disabled behind an explicit empty provider seam. Future remote descriptors must use `mcp__<server>__<tool>` and pass bounded schema validation before atomic publication.

Every cataloged invocation passes through one sync/async BaseTool-preserving pipeline: authorize, validate, deadline/concurrency limit, invoke, output bound, redact, audit, and outcome record. Stable audit/event outcomes carry bounded metadata only; arguments, provider exception text, and result content are excluded.

## Error Handling

- Existing typed `RAGError` instances preserve compatibility.
- Unexpected exceptions become `RagApplicationError` with a stable public message and `request_id`; the original exception is retained only as `internal_cause` and chained for internal diagnostics.
- The HTTP adapter maps the sanitized application error to the existing typed HTTP error boundary.
- Future MCP adapters return structured stable codes and never raw provider text.
- Web-search failure keeps current fallback behavior but logs only exception type plus request ID in the new service.

## Security and Lifecycle

Network startup validates environment, auth, transport, limits, secret references, required dependencies, and the complete tool generation before bind. Anonymous HTTP requires both development environment and an explicit disabled-by-default flag. URL policy validates initial and redirected destinations, disallows prohibited address ranges by default, and bounds redirects. Shutdown marks readiness false, rejects new work, drains/cancels within a bounded grace period, and closes resources in dependency order.

## Correctness Properties

### Property 1: Bounded-context isolation

For every supported process entry point, starting one bounded context activates only its selected transport and lifecycle resources; it never activates FastAPI or the opposite MCP direction.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 10.5**

### Property 2: Canonical closed tool surface

For every valid configuration generation, the advertised inbound tool names are an atomically published subset of `{rag_ask, rag_web_search_answer}` and every accepted input satisfies the closed schema and all hard bounds before downstream invocation.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 8.1, 8.2, 8.3, 8.4, 8.5**

### Property 3: Stable bounded results

For every tool outcome, each source contains a URL; successful results contain all stable contract fields; and sources, warnings, debug data, and serialized bytes never exceed configured hard ceilings.

**Validates: Requirements 2.6, 2.8, 5.5**

### Property 4: Production network transports fail closed

For every production Network_Transport configuration lacking valid trusted authentication, startup completes with failure before socket bind or tool publication. Anonymous network access succeeds only for explicit development plus the anonymous-development flag.

**Validates: Requirements 3.3, 3.4, 3.5, 3.6**

### Property 5: Trusted identity and ownership confinement

For every protected resource and action, only an authorized trusted Principal/Tenant owner can act; caller-supplied identity cannot override trusted identity; unknown and unauthorized identifiers produce non-disclosing equivalent public responses.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6**

### Property 6: Limits and deadline propagation

For every request, effective limits do not exceed hard ceilings, the effective deadline does not exceed the server maximum, and remaining time passed to each downstream seam is no greater than its caller's remaining time. Expiry/cancellation prevents subsequent side effects.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4**

### Property 7: URL destination safety

For every initial URL, redirect, resolution, and connection target, fetch occurs only when scheme, host, address, port, redirect count, and policy all permit it; default policy rejects credentials, non-HTTPS, special/private ranges, and metadata destinations.

**Validates: Requirements 6.3, 6.4, 6.5, 6.6, 6.7**

### Property 8: Error and telemetry non-disclosure

For arbitrary provider exception text, credentials, arguments, questions, and source content, no public error, audit event, metric label, health response, or ordinary log contains prohibited values; correlated results contain a request ID and stable outcome metadata.

**Validates: Requirements 5.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6**

### Property 9: Atomic dependency and configuration publication

For every required-dependency failure, no transport or partial tool set becomes active. For every optional-dependency failure, either startup fails or one explicit degraded generation omits all dependent tools. Failed reload retains the prior generation and closes unpublished resources.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 10.4**

### Property 10: Secret-reference confinement

For every configured secret, persisted and serialized configuration contains only its reference, and runtime values never appear in management responses, tool metadata, errors, logs, or audit events.

**Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**

### Property 11: Bounded graceful shutdown

For every shutdown signal and in-flight workload, readiness becomes false before new work is rejected; resources close in dependency order; and the process stops waiting after the configured hard grace ceiling.

**Validates: Requirements 10.1, 10.2, 10.3**

### Property 12: Single-instance topology enforcement

For every production deployment configuration using local SQLite, Chroma, uploads, checkpoints, or process-local locks, worker and replica count must equal one. Counts above one are accepted only after every shared-state prerequisite is enabled and validated.

**Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6**

### Property 13: QA application-service parity

For every legacy POST `/query` request fixture, direct service invocation and the HTTP adapter produce equivalent legacy fields, source selection, debug messages, graph promotion, and rollback transitions. Unexpected exceptions expose only the sanitized message and request ID.

**Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5, 12.6**

## Testing Strategy

### Phase 0 and Phase 1

- Strict mypy over `src/`, Ruff lint, scoped Ruff format checks, and compileall.
- Direct service tests for explicit/default/discovered sources, debug shaping, provider failure sanitization, graph promotion, and rebuild rollback.
- HTTP parity tests using the existing FastAPI `TestClient` and monkeypatched infrastructure seams.
- Existing QA, chat, graph, web-search, and tool tests as regression coverage.

### Phase 2 and later

- Contract tests over stdio and authenticated streamable HTTP.
- Generated boundary tests for question, URL, schema, result, output, deadline, and quota limits.
- Startup fault injection for every required/optional dependency.
- SSRF cases for redirect chains, special address ranges, DNS changes, and metadata endpoints.
- Concurrent reload, cancellation, ownership, and shutdown tests.
- Secret marker injection across responses, logs, audit events, and configuration snapshots.

## Deployment and Operations

Production starts one chat worker/process and uses one protected persistent volume. Local development may use the existing reload command, but production instructions prohibit reload and additional workers. Backup and restore run in maintenance windows with the writer stopped or quiesced. Restricted network policy permits only required provider/search egress. See `docs/production-topology.md`.
