from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from src.backend.mcp import EnvironmentSecretProvider, SecretReference, SecretResolver
from src.deployment.release_gate import (
    REQUIRED_REHEARSAL_SCENARIOS,
    ReleaseGateError,
    load_release_evidence,
    main,
    validate_release_evidence,
)


def _valid_document(now: datetime) -> dict[str, object]:
    completed_at = (now - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    checksum = "c" * 64
    return {
        "schema_version": 1,
        "release_id": "release-2026-07",
        "environment": "production",
        "operator_ref": "on-call-primary",
        "image_digest": f"sha256:{'a' * 64}",
        "rollback_image_digest": f"sha256:{'b' * 64}",
        "topology": {"workers": 1, "replicas": 1},
        "state_volume_ref": "volume/rag-production-01",
        "secret_references": [
            {"provider": "env", "identifier": "API_KEY"},
            {"provider": "env", "identifier": "MCP_API_KEY"},
        ],
        "backup_restore": {
            "artifact_ref": "backup/release-2026-07",
            "backup_sha256": checksum,
            "restore_sha256": checksum,
            "writers_quiesced": True,
            "isolated_restore": True,
        },
        "drills": [
            {
                "scenario": scenario.value,
                "passed": True,
                "evidence_ref": f"change/release-2026-07/{scenario.value}",
                "completed_at": completed_at,
            }
            for scenario in REQUIRED_REHEARSAL_SCENARIOS
        ],
    }


def _drills(document: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], document["drills"])


def test_valid_release_evidence_closes_every_required_operations_gate() -> None:
    now = datetime(2026, 7, 20, 12, tzinfo=UTC)

    evidence = validate_release_evidence(_valid_document(now), now=now)

    assert evidence.release_id == "release-2026-07"
    assert evidence.topology.environment == "production"
    assert evidence.topology.worker_count == 1
    assert evidence.topology.replica_count == 1
    assert tuple(item.scenario for item in evidence.drills) == REQUIRED_REHEARSAL_SCENARIOS
    assert evidence.backup_restore.backup_sha256 == evidence.backup_restore.restore_sha256
    assert evidence.public_summary() == {
        "status": "passed",
        "release_id": "release-2026-07",
        "scenario_count": 8,
    }


@pytest.mark.parametrize("case", ["missing", "duplicate", "failed", "stale", "future"])
def test_release_gate_rejects_incomplete_or_invalid_drill_evidence(case: str) -> None:
    now = datetime(2026, 7, 20, 12, tzinfo=UTC)
    document = _valid_document(now)
    drills = _drills(document)

    if case == "missing":
        drills.pop()
    elif case == "duplicate":
        drills[-1] = deepcopy(drills[0])
    elif case == "failed":
        drills[0]["passed"] = False
    elif case == "stale":
        drills[0]["completed_at"] = (now - timedelta(days=8)).isoformat()
    else:
        drills[0]["completed_at"] = (now + timedelta(hours=1)).isoformat()

    with pytest.raises(ReleaseGateError):
        validate_release_evidence(document, now=now)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("writers_quiesced", False),
        ("isolated_restore", False),
        ("restore_sha256", "d" * 64),
    ],
)
def test_release_gate_requires_recoverable_quiesced_state(field: str, value: object) -> None:
    now = datetime(2026, 7, 20, 12, tzinfo=UTC)
    document = _valid_document(now)
    backup_restore = cast(dict[str, object], document["backup_restore"])
    backup_restore[field] = value

    with pytest.raises(ReleaseGateError):
        validate_release_evidence(document, now=now)


@pytest.mark.parametrize(("dimension", "count"), [("workers", 2), ("replicas", 2)])
def test_release_gate_rejects_scaled_local_state_topology(dimension: str, count: int) -> None:
    now = datetime(2026, 7, 20, 12, tzinfo=UTC)
    document = _valid_document(now)
    topology = cast(dict[str, object], document["topology"])
    topology[dimension] = count

    with pytest.raises(ReleaseGateError, match="unsupported topology"):
        validate_release_evidence(document, now=now)


def test_release_gate_accepts_only_secret_references_and_sanitizes_rejection() -> None:
    now = datetime(2026, 7, 20, 12, tzinfo=UTC)
    document = _valid_document(now)
    marker = "raw-production-secret-marker"
    references = cast(list[dict[str, object]], document["secret_references"])
    references[0]["value"] = marker

    with pytest.raises(ReleaseGateError) as captured:
        validate_release_evidence(document, now=now)

    assert marker not in str(captured.value)


def test_key_rotation_reuses_the_reference_without_persisting_secret_values() -> None:
    reference = SecretReference(provider="env", identifier="ROTATING_MCP_TOKEN")
    runtime_environment = {reference.identifier: "old-runtime-secret"}
    resolver = SecretResolver(
        {
            "env": EnvironmentSecretProvider(
                {reference.identifier},
                environment=runtime_environment,
            )
        }
    )

    old_value = asyncio.run(resolver.resolve(reference))
    runtime_environment[reference.identifier] = "new-runtime-secret"
    new_value = asyncio.run(resolver.resolve(reference))
    persisted = json.dumps(reference.to_config(), sort_keys=True)

    assert old_value.reveal() == "old-runtime-secret"
    assert new_value.reveal() == "new-runtime-secret"
    assert "old-runtime-secret" not in persisted
    assert "new-runtime-secret" not in persisted
    assert "old-runtime-secret" not in repr(old_value)
    assert "new-runtime-secret" not in repr(new_value)


def test_file_gate_and_cli_emit_only_a_bounded_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = datetime.now(UTC)
    document = _valid_document(now)
    path = tmp_path / "restricted-evidence.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_release_evidence(path, now=now)
    exit_code = main([str(path), "--max-age-hours", "24"])
    output = capsys.readouterr()

    assert loaded.release_id == "release-2026-07"
    assert exit_code == 0
    assert output.err == ""
    assert json.loads(output.out) == {
        "release_id": "release-2026-07",
        "scenario_count": 8,
        "status": "passed",
    }
    assert cast(str, document["image_digest"]) not in output.out
    assert "volume/rag-production-01" not in output.out


def test_cli_failure_is_sanitized(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    marker = "raw-production-secret-marker"
    path = tmp_path / "invalid-evidence.json"
    path.write_text(json.dumps({"secret_value": marker}), encoding="utf-8")

    exit_code = main([str(path)])
    output = capsys.readouterr()

    assert exit_code == 2
    assert output.out == ""
    assert marker not in output.err
    assert json.loads(output.err) == {
        "code": "production_release_gate_failed",
        "message": "Production rehearsal evidence did not pass validation.",
        "status": "failed",
    }


def test_operations_runbook_and_template_cover_the_closed_gate() -> None:
    runbook = Path("docs/production-operations-runbook.md").read_text(encoding="utf-8")
    template = json.loads(
        Path("config/production-rehearsal.example.json").read_text(encoding="utf-8")
    )

    assert "python -m src.deployment.release_gate" in runbook
    for scenario in REQUIRED_REHEARSAL_SCENARIOS:
        assert f"`{scenario.value}`" in runbook
    assert {drill["scenario"] for drill in template["drills"]} == {
        scenario.value for scenario in REQUIRED_REHEARSAL_SCENARIOS
    }
    assert set(template) == {
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
