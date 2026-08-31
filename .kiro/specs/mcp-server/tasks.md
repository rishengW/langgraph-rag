# Implementation Plan

## Overview

Tasks are ordered by production-refactor phase. Checked items are implemented in this approved slice; unchecked items remain future work and retain their phase prerequisites.

## Tasks

- [x] 1. Approve production requirements and topology (Phase 0)
  - [x] 1.1 Separate inbound MCP server and optional outbound MCP client bounded contexts
    - Define separate configuration, entry points, lifecycle, health, and first-release outbound prohibition.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  - [x] 1.2 Define canonical inbound tools and contracts
    - Specify `rag_ask`, `rag_web_search_answer`, question/URL/result limits, closed schemas, stable results, and server deadlines.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_
  - [x] 1.3 Define fail-closed production transport and identity policy
    - Specify stdio local identity, authenticated network startup, development-only anonymous access, principal identity, and ownership.
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_
  - [x] 1.4 Define security, dependency, secret, and lifecycle controls
    - Specify limits, cancellation, SSRF prevention, sanitization, auditing, atomic/degraded startup, secret references, and graceful shutdown.
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 9.1, 9.2, 9.3, 9.4, 9.5, 10.1, 10.2, 10.3, 10.4, 10.5_
  - [x] 1.5 Document and enforce the single-instance internal production topology
    - Require one production chat worker, one protected persistent volume, restricted network, backup/restore, and no horizontal scaling.
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_
  - [x] 1.6 Restore and validate baseline quality gates
    - Fix strict mypy and Ruff import-order errors with focused diffs.
    - Scope Ruff formatting to refactor-owned paths because repository-wide formatting would rewrite legacy files.
    - Run Ruff, strict mypy, compileall, and available pytest coverage.
    - _Requirements: 12.5, 12.6_

- [ ] 2. Extract transport-neutral application services (Phase 1)
  - [x] 2.1 Define stable QA application contracts and sanitized errors
    - Add `RagRequest`, `RagAnswer`, `SourceReference`, and public/internal error separation under `src/application/`.
    - _Requirements: 7.1, 7.2, 7.3, 12.1, 12.2, 12.4_
  - [x] 2.2 Extract non-streaming stateless QA orchestration
    - Preserve URL resolution, source mode/note, lightweight web search, debug messages, global graph promotion, and rebuild rollback.
    - Inject graph, settings, search, and invocation seams without FastAPI imports.
    - _Requirements: 12.1, 12.3, 12.4_
  - [x] 2.3 Convert POST `/query` to a compatibility adapter
    - Translate `QueryRequest` to `RagRequest` and `RagAnswer` to the unchanged `QueryResponse` fields.
    - Leave chat streaming and typed tool events unchanged except focused type corrections.
    - _Requirements: 12.3, 12.5_
  - [x] 2.4 Add direct-service and HTTP parity tests
    - Cover direct invocation, source/debug behavior, sanitized unexpected failures, and legacy HTTP response shape.
    - _Requirements: 12.4, 12.6_
  - [x] 2.5 Extract remaining chat/turn/session application services
    - Move checkpoint snapshot, cancellation, rollback, completion hooks, and session lifecycle behind transport-neutral services.
    - Preserve SSE event ordering and artifact behavior.
    - _Requirements: 5.3, 5.4, 12.5_

- [x] 3. Implement stateless inbound MCP server (Phase 2)
  - [x] 3.1 Add closed configuration and canonical tool schemas
    - Reject outbound configuration, unknown fields, oversized fields, invalid transport settings, and unsafe production anonymous settings.
    - _Requirements: 1.4, 2.1, 2.4, 2.5, 2.6, 2.7, 3.7, 6.1, 6.2_
  - [x] 3.2 Implement `rag_ask` and `rag_web_search_answer` adapters
    - Call `RagApplicationService`, return the stable bounded contract, and advertise no internal tools.
    - _Requirements: 2.1, 2.2, 2.3, 2.8, 12.1_
  - [x] 3.3 Implement stdio transport and lifecycle
    - Keep stdout protocol-only and stderr diagnostic-only; assign a local principal.
    - _Requirements: 3.1, 3.2, 4.2, 10.1, 10.2, 10.3_
  - [x] 3.4 Implement authenticated streamable HTTP
    - Fail production startup before bind when auth is absent; permit anonymous access only behind the explicit development gate.
    - _Requirements: 3.3, 3.4, 3.5, 3.6, 4.1_
  - [x] 3.5 Add mandatory deadlines, cancellation, limits, audit, and redaction
    - Apply one bounded execution pipeline before and after application invocation.
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_
  - [x] 3.6 Add MCP contract and startup fault tests
    - Test both transports, auth combinations, schema bounds, atomic startup, timeout/cancellation, shutdown, and secret non-disclosure.
    - _Requirements: 2.1, 2.7, 3.1, 3.3, 8.1, 8.2, 10.1, 10.2, 10.3_

- [ ] 4. Add immutable tool catalog and policy boundary (Phase 3)
  - [ ] 4.1 Publish immutable validated catalog generations
    - Validate names, collisions, schema limits, risk metadata, and model/dispatcher consistency before atomic publication.
    - _Requirements: 6.1, 6.2, 8.1, 8.4, 8.5_
  - [ ] 4.2 Centralize authorization and execution policy
    - Implement authorize, validate, limit, invoke, bound, redact, audit, and outcome recording.
    - _Requirements: 4.1, 4.3, 4.4, 4.5, 5.1, 5.5, 7.4_

- [ ] 5. Add optional outbound MCP clients (Phase 4)
  - [ ] 5.1 Introduce outbound configuration only after separate approval
    - Keep lifecycle/configuration separate and resolve only Secret_References.
    - _Requirements: 1.1, 1.4, 9.1, 9.2_
  - [ ] 5.2 Implement required/optional provider publication semantics
    - Fail required dependencies atomically, quarantine optional failures, and expose degraded readiness.
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_
  - [ ] 5.3 Enforce outbound endpoint and process restrictions
    - Apply HTTPS endpoint policy, SSRF/redirect controls, executable allowlists, bounded invocation, reconnect, and clean shutdown.
    - _Requirements: 6.3, 6.4, 6.5, 6.6, 6.7, 9.4, 9.5, 10.3, 10.4_

- [ ] 6. Add production identity, persistence, and scale controls (Phase 5)
  - [ ] 6.1 Implement principal/tenant ownership and quotas
    - Cover every session, upload, artifact, checkpoint, and stream operation.
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1_
  - [ ] 6.2 Enforce single-instance deployment capabilities
    - Reject replicas/workers above one while local authoritative stores or locks remain.
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.6_
  - [ ] 6.3 Gate multi-replica support on shared-state prerequisites
    - Add shared transactional persistence, migrations, distributed locks, object storage, server vector store, shared quotas/catalog state, and tenant IDs.
    - _Requirements: 11.4, 11.5_

- [ ] 7. Complete observability and operations (Phase 6)
  - [ ] 7.1 Export bounded structured logs, metrics, traces, and audit events
    - Preserve redaction and bounded cardinality across all transports and tools.
    - _Requirements: 5.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 9.3_
  - [ ] 7.2 Complete lifecycle and dependency health operations
    - Add public liveness/readiness, restricted dependency health, bounded draining, and replacement cleanup.
    - _Requirements: 8.6, 10.1, 10.2, 10.3, 10.4, 10.5_
  - [ ] 7.3 Rehearse production runbooks and release gates
    - Test deployment, rollback, backup, restore, key rotation, MCP outage, restart recovery, and incident procedures.
    - _Requirements: 9.1, 9.2, 11.2, 11.3, 11.4, 11.5, 11.6_

## Task Dependency Graph

```json
{
  "waves": [
    {
      "wave": 1,
      "description": "Approved requirements, topology, and quality baseline.",
      "tasks": ["1"]
    },
    {
      "wave": 2,
      "description": "Transport-neutral application services.",
      "tasks": ["2"],
      "dependsOn": ["1"]
    },
    {
      "wave": 3,
      "description": "Stateless inbound MCP transports and contracts.",
      "tasks": ["3"],
      "dependsOn": ["2"]
    },
    {
      "wave": 4,
      "description": "Immutable tool catalog and centralized policy.",
      "tasks": ["4"],
      "dependsOn": ["3"]
    },
    {
      "wave": 5,
      "description": "Separately approved outbound clients and production identity/persistence controls.",
      "tasks": ["5", "6"],
      "dependsOn": ["4"]
    },
    {
      "wave": 6,
      "description": "Production observability and operational rehearsal.",
      "tasks": ["7"],
      "dependsOn": ["5", "6"]
    }
  ],
  "criticalPath": ["1", "2", "3", "4", "6", "7"]
}
```

Task 5 cannot begin until a separate approval permits outbound MCP configuration. Task 6.3 cannot enable multiple replicas until all listed shared-state prerequisites pass. Phase 2 transport work depends on the Phase 1 application service boundary.

## Notes

- Do not refactor chat streaming during the approved Phase 1 first slice.
- Preserve uncommitted user changes and do not commit or push as part of task execution.
- Required validation is Ruff lint, scoped Ruff formatting, strict mypy, compileall, and as much pytest as installed local dependencies permit.
