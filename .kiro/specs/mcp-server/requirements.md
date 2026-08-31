# Requirements Document

**Status:** Approved
**Approval basis:** The user explicitly approved implementation of the production refactor plan and requested this requirements, design, and task slice.
**Supported first production topology:** Single-instance internal production only.

## Introduction

This feature makes the RAG runtime transport-neutral and defines two separate MCP bounded contexts. The **inbound MCP server** lets external MCP clients invoke this project's RAG capabilities. The optional **outbound MCP client** lets this project invoke tools hosted by approved external MCP servers. They may share policy, telemetry, and secret-reference abstractions, but they do not share transport lifecycle, configuration ownership, or startup entry points.

The first MCP release is stateless and inbound. Stateful MCP chat and outbound clients remain deferred until principal identity, ownership, policy, and lifecycle controls are implemented. Existing HTTP and CLI behavior remains compatible while application logic moves behind transport-neutral services.

## Glossary

- **Inbound_MCP_Server**: A separately started adapter that exposes this project's RAG capabilities to MCP clients.
- **Outbound_MCP_Client**: An optional lifecycle-managed subsystem that invokes allowlisted external MCP servers.
- **Principal**: The authenticated workload or user identity assigned by a trusted authentication boundary.
- **Tenant**: An optional isolation scope associated with a Principal.
- **Resource_Owner**: The Principal and Tenant recorded when a session, upload, artifact, or durable resource is created.
- **Network_Transport**: Streamable HTTP or any other remotely reachable transport.
- **Stdio_Transport**: A local subprocess transport whose protocol output uses standard output and diagnostics use standard error.
- **Required_Dependency**: A dependency without which the selected capability cannot safely serve.
- **Optional_Dependency**: A dependency whose failure may produce an explicitly reported degraded state without activating partial configuration.
- **Server_Deadline**: A mandatory bounded execution deadline controlled by the server.
- **Secret_Reference**: A non-secret identifier resolved by an approved secret provider at runtime.

## Requirements

### Requirement 1: Separate MCP bounded contexts

**User Story:** As an operator, I want inbound and outbound MCP isolated, so that one direction cannot silently activate or control the other.

#### Acceptance Criteria

1. THE system SHALL implement the Inbound_MCP_Server and Outbound_MCP_Client as distinct bounded contexts with separate configuration, lifecycle managers, startup entry points, and health state.
2. WHEN the Inbound_MCP_Server starts, THE system SHALL NOT start an Outbound_MCP_Client or the FastAPI application.
3. WHEN the FastAPI application or an Outbound_MCP_Client starts, THE system SHALL NOT implicitly start the Inbound_MCP_Server.
4. THE first MCP release SHALL require only the Inbound_MCP_Server; deployment configuration SHALL reject any attempt to enable or configure an Outbound_MCP_Client until a later approved phase.
5. THE contexts MAY share immutable tool metadata, policy interfaces, telemetry interfaces, and secret-provider interfaces, but SHALL NOT share active transport sessions.

### Requirement 2: Canonical inbound tool contracts

**User Story:** As an MCP client, I want a small stable tool surface, so that calls remain predictable and bounded.

#### Acceptance Criteria

1. WHEN inbound MCP startup succeeds, THE server SHALL atomically advertise exactly the enabled subset of the canonical tools `rag_ask` and `rag_web_search_answer`, and SHALL advertise no internal graph-node, memory-mutation, editor, file, or administration tool.
2. THE `rag_ask` tool SHALL accept a question and zero to the configured maximum source URLs and SHALL use configured default sources when URLs are absent.
3. THE `rag_web_search_answer` tool SHALL discover sources through the existing web-search boundary and return a grounded answer when web search is enabled.
4. THE question field SHALL contain 1 to 16,000 characters after transport decoding and SHALL contain at least one non-whitespace character.
5. THE default maximum source URL count SHALL be 10; an operator MAY configure a lower value or a bounded higher value that does not exceed the implementation hard ceiling.
6. THE maximum search result and returned source counts SHALL be configurable, positive, and bounded by implementation hard ceilings.
7. THE server SHALL accept HTTPS source URLs by default and SHALL reject unknown fields, malformed fields, and fields exceeding declared limits before graph, search, or provider invocation.
8. EVERY source object in any tool result SHALL contain `url` and MAY contain `title` and `citation_id`; WHEN a tool succeeds, THE result SHALL contain `answer`, bounded `sources`, `request_id`, `grounded`, and bounded `warnings`.

### Requirement 3: Transport and production authentication

**User Story:** As an operator, I want network MCP to fail closed, so that production cannot start anonymously.

#### Acceptance Criteria

1. THE Inbound_MCP_Server SHALL support Stdio_Transport for local clients and authenticated streamable HTTP as a separately selected Network_Transport.
2. WHEN Stdio_Transport is selected, THE server SHALL require no bearer credential, SHALL emit protocol messages only to standard output, and SHALL emit logs only to standard error.
3. WHEN a Network_Transport is selected in production and trusted authentication configuration is absent or invalid, THE server SHALL fail startup before binding a socket or advertising tools.
4. Anonymous Network_Transport SHALL be permitted only when both the environment is explicitly `development` and an explicit development-only anonymous-HTTP setting is enabled.
5. THE development-only anonymous setting SHALL be rejected in production and SHALL default to disabled.
6. WHEN bearer credentials remain supported, THE server SHALL compare them in constant time, reject missing or invalid credentials before tool execution, and return a bearer authentication challenge without logging the credential.
7. THE server SHALL validate the configured transport, non-empty bind host, and port range 1 through 65535 before starting.

### Requirement 4: Principal identity and resource ownership

**User Story:** As a security administrator, I want every network action attributed and owned, so that resources cannot be accessed by unrelated callers.

#### Acceptance Criteria

1. WHEN a Network_Transport request is accepted, THE authentication boundary SHALL provide a stable Principal identifier and, where applicable, a Tenant identifier; caller-supplied identity fields SHALL NOT override trusted identity.
2. WHEN Stdio_Transport is used, THE server SHALL assign an explicit local-process Principal rather than treating identity as absent.
3. WHEN a session, upload, artifact, checkpoint, or durable resource is created, THE system SHALL record its Resource_Owner.
4. BEFORE any read, write, stream, download, delete, or continuation of an owned resource, THE system SHALL verify that the Principal and Tenant are authorized for that resource.
5. IF ownership cannot be established or verified, THEN the operation SHALL fail without revealing whether a resource identifier exists.
6. Stateful MCP tools SHALL remain prohibited until ownership checks and per-session serialization are implemented and tested.

### Requirement 5: Rate, concurrency, cost, deadline, and cancellation controls

**User Story:** As an operator, I want bounded work per principal and request, so that abuse or provider stalls cannot exhaust the service.

#### Acceptance Criteria

1. THE system SHALL enforce configurable bounded per-Principal and, when present, per-Tenant request-rate, concurrent-call, search, token, tool-call, retry, and cost limits.
2. EVERY network or MCP tool request SHALL receive a Server_Deadline no greater than a configured mandatory maximum; clients MAY request a shorter deadline but SHALL NOT extend the server maximum.
3. THE remaining Server_Deadline SHALL propagate through application services, graph execution, LLM calls, web search, retrieval, URL fetching, and outbound tool calls.
4. WHEN a deadline expires or a client disconnect follows the configured cancellation policy, THE system SHALL cancel or stop upstream work, reject further side effects, and return a sanitized timeout or cancellation result.
5. THE server SHALL bound response bytes, source count, warning count, debug output, and tool/provider output before serialization.
6. Limit counters SHALL use bounded-cardinality identifiers and SHALL NOT use URLs, full thread IDs, questions, or source content as metric labels.

### Requirement 6: Schema, URL, and SSRF safety

**User Story:** As a security administrator, I want untrusted inputs constrained before I/O, so that schemas and URLs cannot reach prohibited resources.

#### Acceptance Criteria

1. THE server SHALL publish closed input schemas that reject unknown properties and declare required fields, scalar lengths, collection counts, and supported formats.
2. THE server SHALL enforce hard limits for schema byte size, nesting depth, property count, enum count, and output size before accepting external tool metadata or results.
3. BEFORE fetching a caller-supplied, discovered, redirected, or outbound-MCP URL, THE system SHALL validate its scheme, normalized host, resolved addresses, port, and endpoint allow/deny policy.
4. BY DEFAULT the system SHALL reject non-HTTPS URLs, embedded credentials, loopback, link-local, multicast, unspecified, private-network, and cloud-metadata destinations.
5. THE system SHALL independently validate every redirect target and SHALL stop after a bounded redirect count.
6. DNS resolution and connection SHALL be protected against rebinding by validating resolved addresses at the connection boundary where the HTTP client permits.
7. Exceptions for private destinations SHALL require explicit production policy approval and SHALL be narrower than a general private-network allow switch.

### Requirement 7: Sanitized errors, audit events, and redaction

**User Story:** As an operator, I want useful diagnostics without secret or provider leakage, so that incidents can be investigated safely.

#### Acceptance Criteria

1. EVERY accepted request SHALL receive an internal correlation/request ID that is returned in structured success and sanitized error results.
2. PUBLIC errors SHALL use stable codes and sanitized messages and SHALL NOT include raw provider exceptions, stack traces, credentials, source content, authorization headers, or secret-bearing URLs.
3. Internal exception details MAY be logged only through approved sanitization and redaction and SHALL be associated with the correlation ID.
4. THE system SHALL emit bounded audit events for authentication outcome, authorization outcome, tool invocation, policy denial, limit rejection, timeout/cancellation, configuration publication, startup, degraded state, and shutdown.
5. Audit events SHALL include bounded identity, tool, outcome, duration, and configuration-generation metadata but SHALL NOT include questions, full arguments, documents, tokens, secrets, or unbounded URLs.
6. A logging, metrics, tracing, or audit sink failure SHALL NOT expose secrets or convert a successful tool operation into a client-visible failure; required audit durability MAY instead fail the protected operation according to explicit policy.

### Requirement 8: Required and optional dependency startup

**User Story:** As an operator, I want atomic startup and explicit degradation, so that partial unsafe capability sets are never served.

#### Acceptance Criteria

1. BEFORE accepting requests, THE server SHALL validate configuration, resolve required Secret_References, initialize required dependencies, validate all required tool schemas, and construct one immutable advertised tool generation.
2. IF any Required_Dependency or required tool fails initialization, THEN startup SHALL fail atomically without binding the selected transport or publishing a partial tool set.
3. IF an Optional_Dependency fails, THEN the server MAY start only when all tools depending on it are omitted atomically and readiness explicitly reports a sanitized degraded state.
4. A configuration reload SHALL publish either one fully validated generation or retain the previous generation; it SHALL NOT mutate an active generation in place.
5. Active requests SHALL retain the generation with which they started until completion or cancellation.
6. Detailed dependency diagnostics SHALL be restricted to an administrative interface; public readiness SHALL disclose only bounded status.

### Requirement 9: Secret references and configuration safety

**User Story:** As a security administrator, I want secrets referenced rather than exposed, so that configuration can be inspected safely.

#### Acceptance Criteria

1. Production configuration SHALL store Secret_References rather than secret values in browser-readable, API-readable, logged, or persisted tool configuration.
2. Secret values SHALL be resolved only at runtime through an approved secret-provider interface and SHALL NOT be returned by configuration, health, audit, or tool-list endpoints.
3. THE system SHALL redact authorization headers, access tokens, refresh tokens, client secrets, command environments, and secret-bearing URL components from errors and logs.
4. Outbound stdio command execution SHALL remain disabled in hosted production until executable/image allowlists and fixed argument templates are implemented.
5. Remote outbound MCP endpoints SHALL require HTTPS and explicit endpoint policy before activation.

### Requirement 10: Lifecycle and graceful shutdown

**User Story:** As an operator, I want deterministic lifecycle behavior, so that shutdown and replacement do not corrupt state or leak processes.

#### Acceptance Criteria

1. WHEN SIGINT or SIGTERM is received, THE server SHALL stop accepting new work and mark readiness false before draining or cancelling active work according to policy.
2. THE shutdown grace period SHALL be configurable and bounded; after it expires, remaining work SHALL be cancelled and process exit SHALL continue.
3. Shutdown SHALL close network listeners, provider clients, outbound MCP sessions/subprocesses, graph resources, checkpoint stores, audit exporters, and background workers in dependency order.
4. Startup failure and configuration-generation replacement SHALL close every resource initialized for the unpublished generation.
5. Stdio and Network_Transport lifecycle failures SHALL be isolated from the FastAPI and optional outbound-client processes.

### Requirement 11: First supported production topology

**User Story:** As an operator, I want an honest supported topology, so that local persistence is not deployed with unsafe scaling assumptions.

#### Acceptance Criteria

1. THE first supported deployment SHALL be single-instance internal production with exactly one chat worker/process.
2. SQLite databases, Chroma data, uploads, and generated artifacts SHALL reside on one protected persistent volume accessible only to that instance.
3. THE deployment SHALL use restricted ingress and egress, authenticated network transports, restart recovery, tested backup/restore, and explicit maintenance windows for SQLite/Chroma backups.
4. Horizontal scaling and multiple chat workers SHALL be explicitly prohibited while process-local locks or local SQLite, Chroma, uploads, checkpoints, or artifacts remain authoritative.
5. Multi-replica support SHALL require shared transactional persistence, distributed per-resource locks, object storage, a server-mode vector store, shared quota/catalog state, migrations, and tenant identifiers on durable objects before the prohibition is removed.
6. Production launch commands and operations documentation SHALL set or require one worker while local development commands MAY retain reload behavior.

### Requirement 12: Transport-neutral QA compatibility slice

**User Story:** As a developer, I want the existing QA route backed by an application service, so that future HTTP and MCP adapters reuse the same behavior.

#### Acceptance Criteria

1. THE non-streaming stateless QA orchestration SHALL reside under `src/application/` and SHALL NOT import FastAPI or another transport framework.
2. THE application layer SHALL define stable `RagRequest`, `RagAnswer`, and `SourceReference` models and separate sanitized public errors from retained internal causes.
3. POST `/query` SHALL be a thin compatibility adapter over `RagApplicationService` and SHALL preserve its existing response fields, source mode/note behavior, debug-message shaping, lightweight web-search selection, graph promotion, and rebuild rollback.
4. Unexpected provider or infrastructure exceptions SHALL produce a sanitized public message with a request/correlation ID and SHALL NOT add raw exception text to the public response.
5. This slice SHALL NOT refactor chat streaming or alter typed tool-event behavior except for focused type corrections required to restore strict checking.
6. Focused tests SHALL invoke the service without FastAPI and SHALL verify HTTP response parity.
