"""API-key authentication dependency for mutation endpoints."""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request, status


def _configured_api_key(request: Request) -> str:
    settings = getattr(request.app.state, "config", None)
    settings_key = getattr(settings, "api_key", "") if settings is not None else ""
    return (settings_key or os.getenv("API_KEY", "")).strip()


def require_api_key(request: Request) -> None:
    """Require ``Authorization: Bearer <API_KEY>`` when an API key is configured.

    Args:
        request: Incoming FastAPI request.

    Raises:
        HTTPException: If API-key authentication is enabled and the request is
            missing a matching bearer token.
    """

    expected_key = _configured_api_key(request)
    if not expected_key:
        return

    header = request.headers.get("Authorization", "")
    scheme, _, supplied_key = header.partition(" ")
    is_valid = (
        scheme.lower() == "bearer"
        and bool(supplied_key)
        and hmac.compare_digest(supplied_key.strip(), expected_key)
    )
    if is_valid:
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key.",
        headers={"WWW-Authenticate": "Bearer"},
    )


__all__ = ["require_api_key"]
