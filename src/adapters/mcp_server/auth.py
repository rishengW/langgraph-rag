"""Transitional shared-bearer authentication for inbound MCP HTTP."""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass, field
from uuid import uuid4

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl, TypeAdapter

from ...mcp.observability import (
    MCPObservability,
    ObservationOutcome,
    default_observability,
)
from .audit import AuditEvent, emit_audit
from .config import MCPSettings

_REQUIRED_SCOPE = "rag:invoke"


@dataclass(frozen=True, slots=True)
class ResolvedHTTPAuth:
    """Runtime-only authentication material; the token is excluded from repr."""

    principal_id: str
    token: str = field(repr=False)


class SharedBearerTokenVerifier:
    """Verify one transition-period shared key with constant-time comparison."""

    def __init__(
        self,
        auth: ResolvedHTTPAuth,
        observability: MCPObservability | None = None,
    ) -> None:
        self._auth = auth
        self._observability = observability or default_observability()

    async def verify_token(self, token: str) -> AccessToken | None:
        matched = hmac.compare_digest(token.encode("utf-8"), self._auth.token.encode("utf-8"))
        outcome: ObservationOutcome = "authenticated" if matched else "denied"
        emit_audit(
            AuditEvent(
                request_id=uuid4().hex,
                principal_id=self._auth.principal_id if matched else "unauthenticated",
                tool="authentication",
                outcome=outcome,
                duration_ms=0,
                signal="authentication",
                transport="http",
            ),
            self._observability,
        )
        if not matched:
            return None
        return AccessToken(
            token="redacted",
            client_id=self._auth.principal_id,
            subject=self._auth.principal_id,
            scopes=[_REQUIRED_SCOPE],
        )


def resolve_http_auth(settings: MCPSettings) -> ResolvedHTTPAuth | None:
    if settings.transport != "http" or settings.allow_anonymous_http:
        return None
    token = os.getenv(settings.auth_secret_env, "").strip()
    if len(token) < 16:
        raise ValueError(
            "Authenticated MCP HTTP requires a secret of at least 16 characters "
            f"from the environment reference {settings.auth_secret_env}"
        )
    return ResolvedHTTPAuth(principal_id=settings.http_principal_id, token=token)


def sdk_auth_settings(settings: MCPSettings) -> AuthSettings:
    base_url = settings.effective_public_base_url
    url_adapter = TypeAdapter(AnyHttpUrl)
    return AuthSettings(
        issuer_url=url_adapter.validate_python(base_url),
        resource_server_url=url_adapter.validate_python(f"{base_url}{settings.path}"),
        required_scopes=[_REQUIRED_SCOPE],
    )


def current_principal_id(settings: MCPSettings) -> str:
    if settings.transport == "stdio":
        return "local-process"
    access_token = get_access_token()
    if access_token is not None:
        return access_token.subject or access_token.client_id
    return "anonymous-development"


__all__ = [
    "ResolvedHTTPAuth",
    "SharedBearerTokenVerifier",
    "current_principal_id",
    "resolve_http_auth",
    "sdk_auth_settings",
]
