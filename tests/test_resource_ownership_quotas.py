from __future__ import annotations

import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.messages import AIMessage

from src.backend.security import (
    Principal,
    QuotaBudget,
    QuotaCharge,
    QuotaLimits,
    QuotaManager,
    ResourceOwner,
)
from src.backend.sessions import ChatSessionRegistry, SQLiteMemorySaver, SQLiteStorage
from src.config.loader import load_settings
from src.errors import QuotaExceededError, ResourceNotFoundError
from src.frontend.chat import api as chat_api


def _budget(**overrides: int) -> QuotaBudget:
    values = {
        "requests_per_minute": 100,
        "concurrent_calls": 100,
        "searches_per_minute": 100,
        "tokens_per_minute": 100,
        "tool_calls_per_minute": 100,
        "retries_per_minute": 100,
        "cost_units_per_minute": 100,
    }
    values.update(overrides)
    return QuotaBudget(**values)


def _limits(
    *,
    principal: QuotaBudget | None = None,
    tenant: QuotaBudget | None = None,
) -> QuotaLimits:
    return QuotaLimits(
        principal=principal or _budget(),
        tenant=tenant or _budget(),
        max_tracked_identities=100,
    )


def test_registry_authorizes_complete_owner_and_does_not_touch_foreign_probe(
    mock_settings,
) -> None:
    now = 10.0
    registry = ChatSessionRegistry(
        cleanup=lambda _session: None,
        time_func=lambda: now,
    )
    owner_principal = Principal("principal-a", "tenant-a")
    session = registry.create(
        graph=object(),
        settings=mock_settings,
        source_urls=[],
        source_mode="defaults",
        thread_id="owned-thread",
        owner=ResourceOwner.from_principal(owner_principal),
    )

    now = 20.0
    assert registry.get_owned("owned-thread", Principal("principal-b", "tenant-a")) is None
    assert registry.get_owned("owned-thread", Principal("principal-a", "tenant-b")) is None
    assert session.last_accessed_at == 10.0
    assert registry.get_owned("missing-thread", owner_principal) is None
    assert registry.get_owned("owned-thread", owner_principal) is session
    assert session.last_accessed_at == 20.0

    ownerless = registry.create(
        graph=object(),
        settings=mock_settings,
        source_urls=[],
        source_mode="defaults",
        thread_id="legacy-ownerless",
    )
    assert ownerless.owner is None
    assert registry.get_owned("legacy-ownerless", owner_principal) is None
    assert registry.delete_owned("owned-thread", Principal("principal-b", "tenant-a")) is False
    assert registry.delete_owned("owned-thread", owner_principal) is True


def test_session_owner_round_trips_and_v1_rows_migrate_ownerless(tmp_path) -> None:
    database = tmp_path / "sessions.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE schema_version (id INTEGER PRIMARY KEY, version INTEGER NOT NULL)"
        )
        connection.execute("INSERT INTO schema_version VALUES (1, 1)")
        connection.execute(
            """
            CREATE TABLE sessions (
                thread_id TEXT PRIMARY KEY,
                source_urls TEXT NOT NULL,
                source_mode TEXT NOT NULL,
                config TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_accessed_at REAL NOT NULL,
                chroma_dir TEXT,
                isolated_chroma INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("legacy", "[]", "defaults", "{}", 1.0, 2.0, ".chroma", 0),
        )

    storage = SQLiteStorage(database)
    legacy = storage.load("legacy")
    assert legacy is not None
    assert legacy.owner is None

    owned = replace(
        legacy,
        thread_id="owned",
        owner=ResourceOwner("principal-a", "tenant-a"),
    )
    storage.save(owned)
    assert storage.load("owned") == owned
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}
        version = connection.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()[
            0
        ]
    assert {"owner_principal_id", "owner_tenant_id"}.issubset(columns)
    assert version == SQLiteStorage.SCHEMA_VERSION == 2


def test_checkpoint_operations_require_recorded_owner(tmp_path) -> None:
    saver = SQLiteMemorySaver(tmp_path / "checkpoints.sqlite3", enforce_ownership=True)
    owner = ResourceOwner("principal-a", "tenant-a")
    foreign = ResourceOwner("principal-a", "tenant-b")
    saver.register_owner("thread-a", owner)

    snapshot = saver.snapshot_thread("thread-a", owner=owner)
    saver.restore_thread("thread-a", snapshot, owner=owner)
    assert saver.owner_for_thread("thread-a") == owner

    failures: list[ResourceNotFoundError] = []
    for candidate in (None, foreign):
        with pytest.raises(ResourceNotFoundError) as exc_info:
            saver.snapshot_thread("thread-a", owner=candidate)
        failures.append(exc_info.value)
    with pytest.raises(ResourceNotFoundError) as unknown:
        saver.snapshot_thread("unknown", owner=owner)
    assert {str(error) for error in [*failures, unknown.value]} == {"Resource not found."}

    saver.delete_thread("thread-a", owner=owner)
    assert saver.owner_for_thread("thread-a") is None


@pytest.mark.parametrize(
    ("charge_field", "budget_field"),
    [
        ("requests", "requests_per_minute"),
        ("searches", "searches_per_minute"),
        ("tokens", "tokens_per_minute"),
        ("tool_calls", "tool_calls_per_minute"),
        ("retries", "retries_per_minute"),
        ("cost_units", "cost_units_per_minute"),
    ],
)
def test_each_principal_quota_dimension_is_enforced(
    charge_field: str,
    budget_field: str,
) -> None:
    principal_budget = replace(_budget(), **{budget_field: 1})
    quotas = QuotaManager(_limits(principal=principal_budget))
    principal = Principal("principal-a")
    values = dict.fromkeys(QuotaCharge.__dataclass_fields__, 0)
    values[charge_field] = 1
    charge = QuotaCharge(**values)

    quotas.acquire(principal, charge).release()
    with pytest.raises(QuotaExceededError, match="Request quota exceeded"):
        quotas.acquire(principal, charge)


def test_quota_concurrency_is_separate_per_principal_and_tenant() -> None:
    quotas = QuotaManager(
        _limits(
            principal=_budget(concurrent_calls=1),
            tenant=_budget(concurrent_calls=1),
        )
    )
    first = quotas.acquire(Principal("principal-a", "tenant-a"), QuotaCharge(requests=0))
    with pytest.raises(QuotaExceededError):
        quotas.acquire(Principal("principal-a", "tenant-a"), QuotaCharge(requests=0))
    with pytest.raises(QuotaExceededError):
        quotas.acquire(Principal("principal-b", "tenant-a"), QuotaCharge(requests=0))

    other_tenant = quotas.acquire(
        Principal("principal-a", "tenant-b"),
        QuotaCharge(requests=0),
    )
    other_tenant.release()
    first.release()
    quotas.acquire(Principal("principal-b", "tenant-a"), QuotaCharge(requests=0)).release()


def test_quota_window_expires_without_unbounded_identity_labels() -> None:
    now = 0.0
    quotas = QuotaManager(
        QuotaLimits(
            principal=_budget(requests_per_minute=1),
            tenant=_budget(),
            max_tracked_identities=1,
        ),
        time_func=lambda: now,
    )
    quotas.acquire(Principal("principal-a")).release()
    with pytest.raises(QuotaExceededError):
        quotas.acquire(Principal("principal-a"))
    with pytest.raises(QuotaExceededError):
        quotas.acquire(Principal("principal-b"))

    now = 60.0
    quotas.acquire(Principal("principal-b")).release()


class _HTTPGraph:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def get_state(self, _config: object) -> object:
        return SimpleNamespace(values={"messages": list(self.messages)})

    def invoke(self, inputs: dict[str, object], _config: object) -> dict[str, object]:
        self.messages.extend(inputs["messages"])  # type: ignore[arg-type]
        self.messages.append(AIMessage(content="owned answer"))
        return {"messages": list(self.messages)}

    def stream(self, inputs: dict[str, object], config: object = None, **_kwargs: object):
        del config
        self.messages.extend(inputs["messages"])  # type: ignore[arg-type]
        answer = AIMessage(content="owned stream")
        self.messages.append(answer)
        yield {"generate": {"messages": [answer]}}


def test_http_owner_boundary_covers_session_stream_upload_artifact_and_delete(
    monkeypatch,
    isolated_settings,
    tmp_path,
) -> None:
    settings_a = isolated_settings(
        api_key="trusted-token",
        api_principal_id="principal-a",
        api_tenant_id="tenant-a",
        source_urls=["https://default.test"],
        web_search_enabled=False,
        file_read_enabled=True,
        file_read_root=str(tmp_path / "files"),
    )
    monkeypatch.setattr(chat_api, "load_settings", lambda: settings_a)
    monkeypatch.setattr(chat_api, "build_chat_graph", lambda *_args, **_kwargs: _HTTPGraph())
    headers = {
        "Authorization": "Bearer trusted-token",
        "X-Principal-Id": "caller-spoof",
        "X-Tenant-Id": "caller-spoof",
    }

    with TestClient(chat_api.create_app()) as client:
        started = client.post("/chat", json={"web_search": False}, headers=headers)
        assert started.status_code == 200
        thread_id = started.json()["thread_id"]
        session = client.app.state.session_registry.get(thread_id, touch=False)
        assert session is not None
        assert session.owner == ResourceOwner("principal-a", "tenant-a")
        assert client.app.state.chat_checkpointer.owner_for_thread(thread_id) == session.owner

        upload = client.post(
            f"/chat/{thread_id}/upload",
            files={"files": ("notes.txt", b"owned content", "text/plain")},
            headers=headers,
        )
        assert upload.status_code == 200
        filename = upload.json()["files"][0]["filename"]
        assert (
            client.get(f"/chat/{thread_id}/files/{filename}", headers=headers).content
            == b"owned content"
        )

        settings_b = replace(settings_a, api_principal_id="principal-b")
        client.app.state.settings = settings_b
        client.app.state.config = settings_b
        protected_requests = [
            client.post(
                f"/chat/{thread_id}/message",
                json={"message": "foreign write"},
                headers=headers,
            ),
            client.post(
                f"/chat/{thread_id}/message/stream",
                json={"message": "foreign stream"},
                headers=headers,
            ),
            client.get(f"/chat/{thread_id}/history", headers=headers),
            client.post(
                f"/chat/{thread_id}/upload",
                files={"files": ("foreign.txt", b"foreign", "text/plain")},
                headers=headers,
            ),
            client.get(f"/chat/{thread_id}/files/{filename}", headers=headers),
            client.delete(f"/chat/{thread_id}/files/{filename}", headers=headers),
            client.delete(f"/chat/{thread_id}", headers=headers),
        ]
        unknown = client.get("/chat/unknown/history", headers=headers)
        assert all(response.status_code == 404 for response in protected_requests)
        assert all(response.json() == unknown.json() for response in protected_requests)

        client.app.state.settings = settings_a
        client.app.state.config = settings_a
        stream = client.post(
            f"/chat/{thread_id}/message/stream",
            json={"message": "owned stream"},
            headers=headers,
        )
        assert stream.status_code == 200
        assert "event: done" in stream.text
        assert client.delete(f"/chat/{thread_id}", headers=headers).status_code == 200
        assert client.app.state.chat_checkpointer.owner_for_thread(thread_id) is None


def test_http_rejects_caller_identity_fields(monkeypatch, isolated_settings) -> None:
    configured = isolated_settings(
        api_principal_id="trusted-principal",
        api_tenant_id="trusted-tenant",
        web_search_enabled=False,
    )
    monkeypatch.setattr(chat_api, "load_settings", lambda: configured)
    monkeypatch.setattr(chat_api, "build_chat_graph", lambda *_args, **_kwargs: _HTTPGraph())

    with TestClient(chat_api.create_app()) as client:
        response = client.post(
            "/chat",
            json={
                "web_search": False,
                "principal_id": "caller-principal",
                "tenant_id": "caller-tenant",
            },
        )
    assert response.status_code == 422


def test_http_tenant_request_quota_is_shared_across_principals(
    monkeypatch,
    isolated_settings,
) -> None:
    configured = isolated_settings(
        api_principal_id="principal-a",
        api_tenant_id="tenant-a",
        web_search_enabled=False,
        quota_principal_requests_per_minute=10,
        quota_tenant_requests_per_minute=1,
    )
    monkeypatch.setattr(chat_api, "load_settings", lambda: configured)
    monkeypatch.setattr(chat_api, "build_chat_graph", lambda *_args, **_kwargs: _HTTPGraph())

    with TestClient(chat_api.create_app()) as client:
        assert client.post("/chat", json={"web_search": False}).status_code == 200
        other = replace(configured, api_principal_id="principal-b")
        client.app.state.settings = other
        client.app.state.config = other
        limited = client.post("/chat", json={"web_search": False})

    assert limited.status_code == 429
    assert limited.json() == {
        "detail": {
            "code": "QUOTA_EXCEEDED",
            "message": "Request quota exceeded.",
        }
    }


def test_quota_configuration_rejects_out_of_range_values(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("QUOTA_PRINCIPAL_CONCURRENT_CALLS", "0")
    with pytest.raises(ValueError, match="quota_principal_concurrent_calls"):
        load_settings(env_file=tmp_path / "missing.env", config_file=None)


_SAFE_ID = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="._-",
    ),
    min_size=1,
    max_size=32,
)


@settings(max_examples=50, deadline=None)
@given(
    owner_principal=_SAFE_ID,
    owner_tenant=st.one_of(st.none(), _SAFE_ID),
    caller_principal=_SAFE_ID,
    caller_tenant=st.one_of(st.none(), _SAFE_ID),
)
def test_trusted_identity_and_ownership_confinement_property(
    owner_principal: str,
    owner_tenant: str | None,
    caller_principal: str,
    caller_tenant: str | None,
) -> None:
    """Property 5: Trusted identity and ownership confinement.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6**
    """

    owner = ResourceOwner(owner_principal, owner_tenant)
    caller = Principal(caller_principal, caller_tenant)
    assert owner.authorizes(caller) is (
        owner_principal.strip() == caller_principal.strip()
        and (owner_tenant.strip() if owner_tenant is not None else None)
        == (caller_tenant.strip() if caller_tenant is not None else None)
    )
