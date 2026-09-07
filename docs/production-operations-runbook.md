# Production Operations Rehearsal and Release Gate

This runbook is the release-blocking procedure for the supported internal production topology. It does not turn a local test into production evidence: first run the automated controls, then perform every drill against a production-equivalent isolated environment, retain restricted evidence, and validate the resulting evidence document. Any failed or missing step blocks release.

## Scope and safety invariants

- Run exactly one chat worker and one replica while SQLite, Chroma, uploads, artifacts, checkpoints, or process-local locks remain authoritative. Never attach two writers to the persistent volume.
- Outbound MCP clients are separate dependencies and must not be activated implicitly.
- Keep all authoritative local state under one protected, encrypted persistent-volume root. Restrict it to the service identity.
- Store secret references in configuration and evidence, never secret values. Resolve values only through the approved runtime provider.
- Use immutable image digests, a restricted maintenance window, restricted ingress/egress, and an authenticated administrative boundary.
- Store full command output, approvals, checksums, timestamps, and observations in the restricted change/incident system. Put only bounded references in the release evidence JSON.

## Automated rehearsal gate

Run this focused gate from a clean checkout with the release dependencies installed. It exercises topology rejection, launch declarations, application rollback, required/optional MCP outage behavior, SQLite/session recovery, graceful shutdown, evidence validation, and reference-only key rotation:

```bash
python -m pytest --tb=short \
  tests/test_production_release_gate.py \
  tests/test_deployment_topology.py::test_production_accepts_one_worker_and_replica_with_matching_aliases \
  tests/test_deployment_topology.py::test_replica_count_rejects_scaled_production_and_names_local_blockers \
  tests/test_deployment_topology.py::test_production_launch_configuration_declares_one_instance \
  tests/test_rag_application_service.py::test_failed_rebuild_restores_previous_global_graph \
  tests/test_provider_publication.py::test_required_failure_is_atomic_and_retains_previous_generation \
  tests/test_provider_publication.py::test_optional_failure_omits_all_dependent_tools_and_degrades_readiness \
  tests/test_sessions.py::test_sqlite_storage_round_trips_metadata \
  tests/test_sessions.py::test_registry_restores_session_from_metadata \
  tests/test_sessions.py::test_sqlite_memory_saver_restores_graph_state_after_reopen \
ruff check src/deployment tests/test_production_release_gate.py
ruff format --check src/deployment tests/test_production_release_gate.py
mypy src/
python -m compileall -q src tests
```

The repository-wide CI suite remains mandatory. Preserve its logs by immutable commit and workflow-run reference. A focused pass is not permission to ignore another CI failure.

## Evidence workflow

1. Copy `config/production-rehearsal.example.json` into a restricted per-release workspace. Never edit the checked-in template into a real environment inventory.
2. Replace every placeholder. Use immutable artifact, volume, ticket, backup, and secret-reference identifiers only. Do not include credentials, authorization headers, endpoint query strings, documents, or source content.
3. Complete all drills below. Each `evidence_ref` must resolve in the restricted change system to operator, UTC start/end, commands or platform actions, observations, approval, and rollback decision.
4. Keep every drill within the chosen maximum age (seven days by default, 30 days hard maximum).
5. Run the fail-closed gate:

```bash
python -m src.deployment.release_gate /restricted/release-evidence.json --max-age-hours 168
```

A valid result is a bounded JSON summary with `status` equal to `passed`, the release ID, and eight scenarios. Exit code `2`, stale evidence, a failed drill, unknown fields, unsupported scaling, unpinned images, raw secret fields, non-quiesced state, or mismatched checksums blocks release.

## `deployment` drill

**Preconditions**

- Record the candidate image as `sha256:<digest>` and a distinct known-good rollback digest. Verify signature/SBOM/scans in the release system.
- Confirm `RAG_ENV=production`, `RAG_WORKER_COUNT=1`, `RAG_REPLICA_COUNT=1`, `WEB_CONCURRENCY=1`, no reload flag, authenticated ingress, exact hosts/origins, and restricted egress.
- Confirm the protected persistent volume is mounted at the expected root and only the service identity can read/write it.
- Complete and verify the backup drill before changing code or schema.

**Procedure**

1. Enter the approved maintenance window and mark readiness false or remove the instance from ingress.
2. Drain requests within the configured grace period, stop the prior chat process, and confirm no writer retains the volume. Do not use blue/green writers against local state.
3. Start the candidate digest as one chat process.
4. Confirm public `/health` is live and `/ready` is ready. Through the restricted authenticated route, confirm required dependency health without copying details into public evidence.
5. Invoke one authenticated QA request and one controlled chat/session request.
6. Verify logs/audit events are correlated and contain no credential, question, source-content, or secret marker.

**Pass criteria:** one candidate instance serves; readiness and smoke checks pass; state uses the intended volume; auth and tool surface fail closed. Otherwise execute rollback.

## `rollback` drill

1. Trigger rollback using a harmless rehearsal criterion (for example, an intentionally unmet synthetic readiness condition), not a real customer-impacting fault.
2. Remove the candidate from ingress, mark it unready, drain, and stop it before starting another writer.
3. If no incompatible state change occurred, start the recorded rollback image digest against the same verified volume. If a migration or incompatible write occurred, keep writers stopped and restore the verified pre-change backup first.
4. Repeat health, readiness, authenticated QA/chat, MCP tool-list, and state-recovery checks.
5. Confirm the failed candidate cannot still receive traffic or write state.

**Pass criteria:** the pinned previous artifact returns to ready, durable canaries remain accessible, and only one writer exists. Record rollback duration and decision authority.

## `backup` drill

Use a platform volume snapshot when available. The portable file-copy rehearsal below is valid only with every writer stopped or quiesced:

```bash
# STATE_ROOT and EVIDENCE_DIR are operator-controlled, protected paths.
test -n "$STATE_ROOT" && test -n "$EVIDENCE_DIR"
# Stop/drain the chat process and every maintenance process that can mutate state.
( cd "$STATE_ROOT" && find . -type f -print0 | sort -z | xargs -0 sha256sum ) \
  > "$EVIDENCE_DIR/state-files.sha256"
tar --xattrs --acls -C "$STATE_ROOT" -czf "$EVIDENCE_DIR/state.tgz" .
sha256sum "$EVIDENCE_DIR/state.tgz" > "$EVIDENCE_DIR/state.tgz.sha256"
sha256sum "$EVIDENCE_DIR/state-files.sha256" \
  > "$EVIDENCE_DIR/state-files.manifest.sha256"
```

Confirm the captured root includes every configured SQLite/session/checkpoint database, Chroma collection, upload, and generated artifact path. Protect and retain the archive, file manifest, platform snapshot ID, encryption/key reference, and checksums according to policy. Never back up `.env` files or resolved secrets as application state.

**Pass criteria:** writers were quiesced; one complete encrypted artifact/snapshot and checksum manifest exist; retention and restore owner are recorded.

## `restore` drill

1. Provision a clean isolated volume and instance with no production ingress, provider side effects, scheduled jobs, or shared-writer access.
2. Verify the archive checksum before extraction, then restore to the clean root:

```bash
sha256sum -c "$EVIDENCE_DIR/state.tgz.sha256"
mkdir -p "$ISOLATED_STATE_ROOT"
tar --xattrs --acls -C "$ISOLATED_STATE_ROOT" -xzf "$EVIDENCE_DIR/state.tgz"
( cd "$ISOLATED_STATE_ROOT" && sha256sum -c "$EVIDENCE_DIR/state-files.sha256" )
sha256sum "$EVIDENCE_DIR/state-files.sha256"
```

3. Start exactly one isolated worker using the candidate configuration and restored volume. Confirm readiness.
4. Verify representative pre-backup session metadata, checkpoint history, Chroma retrieval, upload access, and generated artifact access. Use identifiers recorded before backup; do not synthesize replacement state after a failed lookup.
5. Stop the isolated instance and dispose of the restored copy according to data-handling policy.

Put the SHA-256 of `state-files.sha256` in both `backup_sha256` and `restore_sha256` only after all file checks and application-level recovery checks pass. Set `isolated_restore` true only for a genuinely isolated restore.

## `key_rotation` drill

1. Inventory applicable references: `API_KEY` injection, the environment variable named by `MCP_AUTH_SECRET_ENV`, and each outbound `authorization_secret_ref`. Record provider/identifier pairs only.
2. Create a new secret-provider version without changing the persisted reference object. Do not place either value in configuration, evidence, shell arguments, tickets, logs, or health output.
3. In the maintenance window, inject the new runtime version and restart/reload only the owning bounded context.
4. Verify the new credential succeeds, the prior credential receives the same sanitized authentication failure as any invalid credential, and unauthenticated production HTTP remains rejected.
5. Search bounded logs/audit output for a non-secret rotation correlation ID, not for the credential itself. Revoke the old provider version after validation.
6. Revert to the prior provider version and stop the rollout if new authentication or dependency readiness fails; record only version references.

**Pass criteria:** configuration and evidence still contain references only, the new value works, the old value is rejected/revoked, and neither value appears in serialized or diagnostic output.

## `mcp_outage` drill

Perform the outage with network policy or a controlled unavailable test endpoint; never redirect traffic to an unapproved destination.

1. Deny one optional outbound MCP provider. Publish/reload the catalog and confirm the provider's complete tool set is omitted atomically, public readiness is bounded and degraded, and unaffected required tools still work.
2. Restore the optional provider and confirm a new complete catalog generation becomes ready; active calls retain their starting generation until completion.
3. Deny one required provider in an isolated startup/reload rehearsal. Confirm startup/publication fails atomically or retains the previous generation, with no partial tool set.
4. Confirm the FastAPI lifecycle does not implicitly start or restart the outbound client. A separate outbound outage must not change the chat tool surface.
5. Verify timeout, reconnect, cancellation, and shutdown remain bounded and raw provider errors or credentials do not reach public responses/logs.

**Pass criteria:** optional failure is explicit degradation with dependent tools omitted; required failure never publishes partial capability; recovery publishes one validated generation.

## `restart_recovery` drill

1. Before the backup/restart, record restricted references to a controlled session, checkpointed conversation, Chroma-backed retrieval canary, upload, and generated artifact on the authoritative volume.
2. Gracefully stop the one instance. Confirm readiness becomes false before drain and the process exits within the configured grace period.
3. Restart the same pinned image and configuration with the same volume. Do not create an empty replacement volume if mount or database access fails.
4. Confirm readiness, then retrieve each pre-restart canary through its normal authenticated ownership boundary. Continue the checkpointed session once and confirm history remains intact.
5. Perform an ungraceful process-termination rehearsal only in the isolated environment, then repeat database and canary checks after restart.

**Pass criteria:** existing state is recovered without identifier substitution or cross-owner access, no duplicate writer exists, and inaccessible/corrupt required state makes readiness false.

## `incident_response` drill

1. Open a synthetic incident and assign commander, operations lead, security lead, communications owner, and scribe references.
2. Classify impact from readiness, latency, timeout, tool-failure, grounded-answer, and audit signals. Do not paste questions, source content, credentials, or unbounded URLs into the incident record.
3. Contain by removing ingress, disabling the affected optional provider/tool generation, or stopping the single writer. Preserve the volume and bounded audit/trace references before mutation.
4. Choose and execute one recovery branch: restart, required-provider recovery, key rotation, application rollback, or verified state restore. Apply the matching section above without concurrent local-state writers.
5. Validate health, readiness, ownership, canonical tools, state canaries, and secret non-disclosure before restoring traffic.
6. Record a UTC timeline, decisions, evidence references, customer-impact statement, follow-up owner, and expiry/rotation of temporary access. Close only after an independent operator verifies recovery.

**Pass criteria:** the team detects, contains, recovers, and verifies within the rehearsal objective; evidence stays bounded and secret-free; every follow-up has an owner.

## Release decision

Release only when automated CI is green, all eight scenario records are fresh and passed, backup/restore checksums match, writers were quiesced, restore was isolated, images are distinct pinned digests, references contain no secret values, and the release gate exits successfully. Any exception requires a new rehearsal; there is no warning-only override in the verifier.
