# Single-Instance Production Topology

## Supported topology

The first supported production deployment is an internal, single-instance service:

- exactly one chat worker/process; no reload mode in production;
- one protected persistent volume containing SQLite databases, Chroma data, uploads, and generated artifacts;
- authenticated ingress and restricted provider/search egress;
- no horizontal scaling, shared-writer mount, or active/active failover;
- one separately launched inbound MCP process when MCP is enabled; stdio is local, and streamable HTTP must fail startup without production authentication.

The current `python -m src.frontend.chat.main serve` and container command use Uvicorn's default single worker. Production manifests must set `RAG_WORKER_COUNT=1` and `RAG_REPLICA_COUNT=1`; startup also checks `WEB_CONCURRENCY` and `UVICORN_WORKERS` as worker-count aliases. Malformed, conflicting, or above-one declarations fail startup before graphs, local persistence, locks, quotas, or MCP tools initialize. Production launch automation must not add `--workers`, multiple chat replicas, or reload. `docker-compose.yml` is a local-development convenience, not a production manifest; its QA and chat services and named volumes do not establish a supported multi-service production topology.

## Persistent state

Mount one encrypted/protected volume at the deployment's state root and configure all local state beneath it. Limit filesystem access to the service identity. Stop or quiesce writers for consistent SQLite/Chroma backups, use explicit maintenance windows, retain backup checksums, and rehearse restoration into an isolated instance before declaring a backup valid.

A restart must recover session metadata/checkpoints and existing Chroma collections from the same volume. A failed restore, inaccessible volume, or required local database failure makes readiness false; production must not silently start with empty replacement state.

## Network and authentication

Expose only the required internal ingress through a reverse proxy or service mesh. Production HTTP APIs and streamable HTTP MCP must have trusted authentication; anonymous MCP HTTP is development-only and requires both `MCP_ENVIRONMENT=development` and `MCP_ALLOW_ANONYMOUS_HTTP=true`. Launch MCP separately with `python -m src.frontend.adapters.mcp_server.main`; use `MCP_TRANSPORT=stdio` for local subprocess clients or `MCP_TRANSPORT=http` for the SDK Streamable HTTP ASGI app. Production HTTP requires an HTTPS `MCP_PUBLIC_BASE_URL`, an exact `MCP_ALLOWED_HOSTS` allowlist, and a bearer key resolved from the server-side environment variable named by `MCP_AUTH_SECRET_ENV` (default `API_KEY`). Wildcard hosts and origins are rejected outside development.

The inbound HTTP transport enables the SDK Host/Origin checks against exact configured values. Source URLs are HTTPS-only and every pre-invocation DNS result must be public; however, the current legacy loaders cannot pin those validated addresses to later connections or independently enforce this policy on every redirect. Do not claim outbound-fetch DNS-rebinding resistance. Keep restricted egress or an allowlisting proxy in front of source fetches until connection-boundary address pinning and redirect validation replace that gap.

Permit egress only to configured LLM, embedding, search, and approved source endpoints. Never expose SQLite, Chroma files, uploads, secrets, or dependency diagnostics without the administrative authentication boundary.

## Health and shutdown operations

QA, chat, and inbound MCP HTTP expose unauthenticated `GET /health` liveness and `GET /ready` readiness with status-only, no-store responses. Detailed fixed-name dependency states are available at `GET /admin/health/dependencies`; FastAPI requires the configured `API_KEY`, authenticated MCP HTTP requires its trusted bearer scope, and anonymous-development MCP hides the route. Do not route the administrative endpoint through public ingress.

On shutdown, readiness becomes false before listener stop/drain. Inbound MCP rejects new calls, waits only within `MCP_SHUTDOWN_GRACE_SECONDS`, cancels remaining work, and closes the RAG service and exporters in dependency order. Catalog publication retains leased generations, then closes retired, quarantined, or unpublished providers with a hard per-provider bound; one cleanup failure cannot prevent later resources from receiving a close attempt.

## Scaling prohibition

Do not increase workers or replicas while any authoritative state uses local SQLite, local Chroma, local uploads/artifacts, or process-local locks. Multi-replica support requires, at minimum:

1. shared transactional session/checkpoint persistence with migrations;
2. distributed per-session/resource locks;
3. object storage for uploads and artifacts;
4. a server-mode vector store;
5. shared rate-limit, cost, and catalog-generation state;
6. tenant IDs and ownership checks on every durable object.

Deployment review and startup must reject replica/worker counts above one until every prerequisite is implemented and tested. The code-level gate requires one canonical registration for each item above (quota and catalog state are separate registrations). Each registration must identify a real implementation, be explicitly marked configured by the composition root, and pass a runtime validator. Missing, duplicate, unconfigured, or failed registrations reject the complete scaled topology; validator exception details are not exposed.

The current QA, chat, and inbound MCP composition roots intentionally provide no shared-state registrations, so their production default remains exactly one worker and one replica. There is deliberately no environment-only capability switch: environment claims cannot attest that a backend is implemented or validated. A future shared-state composition root must inject the complete validated generation and use those same implementations for runtime state before production scaling is supported.

## Operations checklist

- Verify worker and replica count is exactly one.
- Verify the persistent volume is mounted, writable by only the service identity, and included in backup/restore tests.
- Verify production auth, exact allowed hosts/origins, mandatory deadlines, and request/output limits before binding any network MCP transport.
- Verify the MCP process is separate from FastAPI and advertises only `rag_ask` and `rag_web_search_answer`.
- Account for the documented legacy-fetcher DNS/redirect pinning gap with restricted egress.
- Verify liveness, readiness, graceful termination, restart recovery, and disk-capacity alerts.
- Record backup, restore, rollback, key-rotation, and maintenance procedures.

## Quality-gate migration

At the Phase 0 baseline, `ruff format --check .` would rewrite 108 legacy files. CI therefore checks formatting only for the refactor-owned Python paths (`src/backend/application`, `src/frontend/adapters/mcp_server`, `src/qa/api.py`, `tests/test_rag_application_service.py`, and `tests/test_mcp_server.py`) while `ruff check .` and strict `mypy src/` remain repository-wide. Expand the format scope incrementally when touched legacy files are deliberately formatted; do not use a bulk formatting commit as part of this production slice.
