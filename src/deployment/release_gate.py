"""Fail-closed validation for production operations rehearsal evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from ..errors import ConfigurationError
from ..mcp.secrets import SecretReference
from .topology import DeploymentTopology, DeploymentTopologyError, validate_single_instance_deployment

REHEARSAL_SCHEMA_VERSION = 1
DEFAULT_MAX_REHEARSAL_AGE = timedelta(days=7)
MAX_REHEARSAL_AGE = timedelta(days=30)
MAX_EVIDENCE_BYTES = 64 * 1024
MAX_SECRET_REFERENCES = 32
MAX_REFERENCE_CHARS = 256
_FUTURE_CLOCK_SKEW = timedelta(minutes=5)
_SAFE_REFERENCE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class RehearsalScenario(StrEnum):
    """Closed production procedures that must be rehearsed before release."""

    DEPLOYMENT = "deployment"
    ROLLBACK = "rollback"
    BACKUP = "backup"
    RESTORE = "restore"
    KEY_ROTATION = "key_rotation"
    MCP_OUTAGE = "mcp_outage"
    RESTART_RECOVERY = "restart_recovery"
    INCIDENT_RESPONSE = "incident_response"


REQUIRED_REHEARSAL_SCENARIOS = tuple(RehearsalScenario)


class ReleaseGateError(ValueError):
    """Raised with a sanitized message when release evidence is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class DrillEvidence:
    """One successful, bounded reference to externally retained drill evidence."""

    scenario: RehearsalScenario
    evidence_ref: str
    completed_at: datetime


@dataclass(frozen=True, slots=True)
class BackupRestoreEvidence:
    """Evidence that a quiesced backup was restored and checksum-verified in isolation."""

    artifact_ref: str
    backup_sha256: str
    restore_sha256: str
    writers_quiesced: bool
    isolated_restore: bool


@dataclass(frozen=True, slots=True)
class ProductionReleaseEvidence:
    """Validated release evidence safe to retain without runtime secret values."""

    release_id: str
    operator_ref: str
    image_digest: str
    rollback_image_digest: str
    topology: DeploymentTopology
    state_volume_ref: str
    secret_references: tuple[SecretReference, ...]
    backup_restore: BackupRestoreEvidence
    drills: tuple[DrillEvidence, ...]

    def public_summary(self) -> dict[str, int | str]:
        """Return a bounded summary that excludes infrastructure and evidence details."""

        return {
            "status": "passed",
            "release_id": self.release_id,
            "scenario_count": len(self.drills),
        }


def _closed_object(
    value: object,
    *,
    expected_fields: frozenset[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ReleaseGateError(f"{label} must be an object")

    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ReleaseGateError(f"{label} contains invalid fields")
        result[key] = item
    if set(result) != expected_fields:
        raise ReleaseGateError(f"{label} fields do not match the closed schema")
    return result


def _array(value: object, *, label: str, minimum: int, maximum: int) -> list[object]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ReleaseGateError(f"{label} has an invalid item count")
    return list(value)


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReleaseGateError(f"{label} must be an integer")
    return value


def _safe_reference(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ReleaseGateError(f"{label} must be a string")
    if (
        not 1 <= len(value) <= MAX_REFERENCE_CHARS
        or value != value.strip()
        or "://" in value
        or _SAFE_REFERENCE_PATTERN.fullmatch(value) is None
    ):
        raise ReleaseGateError(f"{label} is invalid")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ReleaseGateError(f"{label} must be a lowercase SHA-256 checksum")
    return value


def _image_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _IMAGE_DIGEST_PATTERN.fullmatch(value) is None:
        raise ReleaseGateError(f"{label} must pin an image by SHA-256 digest")
    return value


def _timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise ReleaseGateError(f"{label} must be an ISO-8601 timestamp")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise ReleaseGateError(f"{label} must be an ISO-8601 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReleaseGateError(f"{label} must include a timezone")
    return parsed.astimezone(UTC)


def _validate_rehearsal_window(
    completed_at: datetime,
    *,
    now: datetime,
    max_age: timedelta,
) -> None:
    if completed_at > now + _FUTURE_CLOCK_SKEW:
        raise ReleaseGateError("Rehearsal evidence cannot be future-dated")
    if completed_at < now - max_age:
        raise ReleaseGateError("Rehearsal evidence is stale")


def _validate_topology(value: object) -> DeploymentTopology:
    topology_document = _closed_object(
        value,
        expected_fields=frozenset({"workers", "replicas"}),
        label="topology",
    )
    workers = _integer(topology_document["workers"], label="topology workers")
    replicas = _integer(topology_document["replicas"], label="topology replicas")
    try:
        return validate_single_instance_deployment(
            environment="production",
            environ={
                "RAG_WORKER_COUNT": str(workers),
                "RAG_REPLICA_COUNT": str(replicas),
            },
        )
    except DeploymentTopologyError:
        raise ReleaseGateError("Release evidence declares an unsupported topology") from None


def _validate_secret_references(value: object) -> tuple[SecretReference, ...]:
    raw_references = _array(
        value,
        label="secret_references",
        minimum=1,
        maximum=MAX_SECRET_REFERENCES,
    )
    references: list[SecretReference] = []
    seen: set[tuple[str, str]] = set()
    for raw_reference in raw_references:
        try:
            reference = SecretReference.from_config(raw_reference)
        except (ConfigurationError, TypeError, ValueError):
            raise ReleaseGateError(
                "secret_references must contain only closed Secret_Reference objects"
            ) from None
        identity = (reference.provider, reference.identifier)
        if identity in seen:
            raise ReleaseGateError("secret_references must not contain duplicates")
        seen.add(identity)
        references.append(reference)
    return tuple(references)


def _validate_backup_restore(value: object) -> BackupRestoreEvidence:
    document = _closed_object(
        value,
        expected_fields=frozenset(
            {
                "artifact_ref",
                "backup_sha256",
                "restore_sha256",
                "writers_quiesced",
                "isolated_restore",
            }
        ),
        label="backup_restore",
    )
    backup_sha256 = _sha256(document["backup_sha256"], label="backup checksum")
    restore_sha256 = _sha256(document["restore_sha256"], label="restore checksum")
    if document["writers_quiesced"] is not True:
        raise ReleaseGateError("Backup evidence must confirm quiesced writers")
    if document["isolated_restore"] is not True:
        raise ReleaseGateError("Restore evidence must confirm isolated restoration")
    if backup_sha256 != restore_sha256:
        raise ReleaseGateError("Restored state checksum does not match the backup")
    return BackupRestoreEvidence(
        artifact_ref=_safe_reference(document["artifact_ref"], label="backup artifact reference"),
        backup_sha256=backup_sha256,
        restore_sha256=restore_sha256,
        writers_quiesced=True,
        isolated_restore=True,
    )


def _validate_drills(
    value: object,
    *,
    now: datetime,
    max_age: timedelta,
) -> tuple[DrillEvidence, ...]:
    raw_drills = _array(
        value,
        label="drills",
        minimum=len(REQUIRED_REHEARSAL_SCENARIOS),
        maximum=len(REQUIRED_REHEARSAL_SCENARIOS),
    )
    drills_by_scenario: dict[RehearsalScenario, DrillEvidence] = {}
    for raw_drill in raw_drills:
        document = _closed_object(
            raw_drill,
            expected_fields=frozenset(
                {"scenario", "passed", "evidence_ref", "completed_at"}
            ),
            label="drill evidence",
        )
        raw_scenario = document["scenario"]
        if not isinstance(raw_scenario, str):
            raise ReleaseGateError("Drill scenario is invalid")
        try:
            scenario = RehearsalScenario(raw_scenario)
        except ValueError:
            raise ReleaseGateError("Drill scenario is invalid") from None
        if scenario in drills_by_scenario:
            raise ReleaseGateError("Drill scenarios must be unique")
        if document["passed"] is not True:
            raise ReleaseGateError("Every required production drill must pass")
        completed_at = _timestamp(document["completed_at"], label="drill completion")
        _validate_rehearsal_window(completed_at, now=now, max_age=max_age)
        drills_by_scenario[scenario] = DrillEvidence(
            scenario=scenario,
            evidence_ref=_safe_reference(document["evidence_ref"], label="drill evidence reference"),
            completed_at=completed_at,
        )

    if set(drills_by_scenario) != set(REQUIRED_REHEARSAL_SCENARIOS):
        raise ReleaseGateError("Evidence must contain every required production drill")
    return tuple(drills_by_scenario[scenario] for scenario in REQUIRED_REHEARSAL_SCENARIOS)


def validate_release_evidence(
    value: object,
    *,
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_REHEARSAL_AGE,
) -> ProductionReleaseEvidence:
    """Validate one closed production rehearsal document without exposing raw values."""

    if max_age <= timedelta(0) or max_age > MAX_REHEARSAL_AGE:
        raise ReleaseGateError("Maximum rehearsal age is outside the supported range")
    validation_time = datetime.now(UTC) if now is None else now
    if validation_time.tzinfo is None or validation_time.utcoffset() is None:
        raise ReleaseGateError("Release gate validation time must include a timezone")
    validation_time = validation_time.astimezone(UTC)

    document = _closed_object(
        value,
        expected_fields=frozenset(
            {
                "schema_version",
                "release_id",
                "environment",
                "operator_ref",
                "image_digest",
                "rollback_image_digest",
                "topology",
                "state_volume_ref",
                "secret_references",
                "backup_restore",
                "drills",
            }
        ),
        label="release evidence",
    )
    if _integer(document["schema_version"], label="schema_version") != REHEARSAL_SCHEMA_VERSION:
        raise ReleaseGateError("Unsupported release evidence schema version")
    environment = document["environment"]
    if not isinstance(environment, str) or environment.casefold() != "production":
        raise ReleaseGateError("Release evidence environment must be production")

    image_digest = _image_digest(document["image_digest"], label="release image")
    rollback_image_digest = _image_digest(
        document["rollback_image_digest"],
        label="rollback image",
    )
    if image_digest == rollback_image_digest:
        raise ReleaseGateError("Release and rollback images must be distinct pinned artifacts")

    return ProductionReleaseEvidence(
        release_id=_safe_reference(document["release_id"], label="release_id"),
        operator_ref=_safe_reference(document["operator_ref"], label="operator_ref"),
        image_digest=image_digest,
        rollback_image_digest=rollback_image_digest,
        topology=_validate_topology(document["topology"]),
        state_volume_ref=_safe_reference(
            document["state_volume_ref"],
            label="state_volume_ref",
        ),
        secret_references=_validate_secret_references(document["secret_references"]),
        backup_restore=_validate_backup_restore(document["backup_restore"]),
        drills=_validate_drills(
            document["drills"],
            now=validation_time,
            max_age=max_age,
        ),
    )


def load_release_evidence(
    path: Path,
    *,
    now: datetime | None = None,
    max_age: timedelta = DEFAULT_MAX_REHEARSAL_AGE,
) -> ProductionReleaseEvidence:
    """Load a bounded UTF-8 JSON document and apply the production release gate."""

    try:
        payload = path.read_bytes()
    except OSError:
        raise ReleaseGateError("Release evidence could not be read") from None
    if not payload or len(payload) > MAX_EVIDENCE_BYTES:
        raise ReleaseGateError("Release evidence size is invalid")
    try:
        decoded = payload.decode("utf-8")
        value: object = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ReleaseGateError("Release evidence is not valid UTF-8 JSON") from None
    return validate_release_evidence(value, now=now, max_age=max_age)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate production runbook rehearsal evidence before release.",
    )
    parser.add_argument("evidence", type=Path, help="Path to the restricted evidence JSON file")
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=int(DEFAULT_MAX_REHEARSAL_AGE.total_seconds() // 3600),
        help="Maximum age of every drill (default: 168, hard maximum: 720)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the release gate and emit only a bounded pass/fail summary."""

    arguments = _build_argument_parser().parse_args(argv)
    try:
        evidence = load_release_evidence(
            arguments.evidence,
            max_age=timedelta(hours=arguments.max_age_hours),
        )
    except (OverflowError, ReleaseGateError):
        print(
            json.dumps(
                {
                    "status": "failed",
                    "code": "production_release_gate_failed",
                    "message": "Production rehearsal evidence did not pass validation.",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    print(json.dumps(evidence.public_summary(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MAX_REHEARSAL_AGE",
    "MAX_EVIDENCE_BYTES",
    "MAX_REHEARSAL_AGE",
    "REHEARSAL_SCHEMA_VERSION",
    "REQUIRED_REHEARSAL_SCENARIOS",
    "BackupRestoreEvidence",
    "DrillEvidence",
    "ProductionReleaseEvidence",
    "RehearsalScenario",
    "ReleaseGateError",
    "load_release_evidence",
    "main",
    "validate_release_evidence",
]
