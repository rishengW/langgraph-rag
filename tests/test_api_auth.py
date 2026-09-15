"""Tests for API-key authentication boundaries and proxy rate limiting."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.frontend.api.amap_proxy import ClientRateLimiter
from src.frontend.api.auth import require_principal


@pytest.fixture
def mini_app() -> FastAPI:
    app = FastAPI()

    @app.get("/protected")
    async def protected(principal: object = Depends(require_principal)) -> dict[str, str]:
        return {"principal": principal.principal_id}

    return app


@pytest.mark.parametrize("environment", ["production", "staging"])
def test_require_principal_fails_closed_without_key_in_protected_environments(
    mini_app: FastAPI, monkeypatch: pytest.MonkeyPatch, environment: str
) -> None:
    monkeypatch.setenv("RAG_ENV", environment)
    monkeypatch.delenv("API_KEY", raising=False)

    with TestClient(mini_app, raise_server_exceptions=False) as client:
        response = client.get("/protected")

    assert response.status_code == 503
    assert "API key" in response.json()["detail"]


@pytest.mark.parametrize("environment", ["development", "testing", ""])
def test_require_principal_stays_open_without_key_outside_protected_environments(
    mini_app: FastAPI, monkeypatch: pytest.MonkeyPatch, environment: str
) -> None:
    monkeypatch.setenv("RAG_ENV", environment)
    monkeypatch.delenv("API_KEY", raising=False)

    with TestClient(mini_app) as client:
        response = client.get("/protected")

    assert response.status_code == 200
    assert response.json()["principal"] == "api-key-client"


def test_require_principal_accepts_valid_bearer_key(
    mini_app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_ENV", "production")
    monkeypatch.setenv("API_KEY", "secret-key")

    with TestClient(mini_app) as client:
        rejected = client.get("/protected")
        accepted = client.get("/protected", headers={"Authorization": "Bearer secret-key"})

    assert rejected.status_code == 401
    assert accepted.status_code == 200


def test_rate_limiter_bounds_requests_within_window() -> None:
    limiter = ClientRateLimiter(max_requests=3, window_seconds=60.0)

    assert [limiter.allow("ip-a", now=float(i)) for i in range(5)] == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert limiter.allow("ip-b", now=1.0) is True


def test_rate_limiter_window_slides_and_evicts_stale_clients() -> None:
    limiter = ClientRateLimiter(max_requests=2, window_seconds=10.0)

    assert limiter.allow("ip-a", now=1.0) is True
    assert limiter.allow("ip-a", now=2.0) is True
    assert limiter.allow("ip-a", now=3.0) is False
    assert limiter.allow("ip-a", now=11.5) is True

    for index in range(100):
        limiter.allow(f"bulk-{index}", now=12.0)

    assert "ip-a" not in limiter._hits or limiter._hits["ip-a"]
