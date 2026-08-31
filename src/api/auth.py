"""Trusted API-key authentication and principal assignment dependencies."""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request, status

from ..security import Principal


def _configured_api_key(request: Request) -> str:
    settings = getattr(request.app.state, "config", None)
    settings_key = getattr(settings, "api_key", "") if settings is not None else ""
    return (settings_key or os.getenv("API_KEY", "")).strip()


def _configured_principal(request: Request) -> Principal:
    settings = getattr(request.app.state, "config", None)
    principal_id = getattr(settings, "api_principal_id", "api-key-client")
    tenant_value = getattr(settings, "api_tenant_id", "")
    tenant_id = str(tenant_value or "").strip() or None
    return Principal(principal_id=str(principal_id or "api-key-client"), tenant_id=tenant_id)


def require_principal(request: Request) -> Principal:
    """Authenticate the request and return only server-configured identity.

    Caller-provided request fields and arbitrary identity headers are never
    consulted, so they cannot override the trusted authentication boundary.
    """

    expected_key = _configured_api_key(request)
    if expected_key:
        header = request.headers.get("Authorization", "")
        scheme, _, supplied_key = header.partition(" ")
        is_valid = (
            scheme.lower() == "bearer"
            and bool(supplied_key)
            and hmac.compare_digest(supplied_key.strip(), expected_key)
        )
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return _configured_principal(request)


def require_api_key(request: Request) -> None:
    """Backward-compatible guard for endpoints that do not consume identity."""

    require_principal(request)


def require_admin_api_key(request: Request) -> Principal:
    """Fail closed when no authenticated administrative boundary is configured."""

    if not _configured_api_key(request):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    return require_principal(request)


__all__ = ["require_admin_api_key", "require_api_key", "require_principal"]
