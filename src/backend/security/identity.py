"""Trusted request identity and durable resource ownership values."""

from __future__ import annotations

import hmac
from dataclasses import dataclass

_MAX_IDENTITY_CHARS = 128


@dataclass(frozen=True, slots=True)
class Principal:
    """Identity supplied by a trusted authentication boundary."""

    principal_id: str
    tenant_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "principal_id",
            _validated_identity(self.principal_id, "principal"),
        )
        if self.tenant_id is not None:
            object.__setattr__(
                self,
                "tenant_id",
                _validated_identity(self.tenant_id, "tenant"),
            )

    @classmethod
    def local_process(cls) -> Principal:
        """Return the explicit identity used by local non-network callers."""

        return cls(principal_id="local-process")


@dataclass(frozen=True, slots=True)
class ResourceOwner:
    """Principal and optional tenant recorded for one durable resource."""

    principal_id: str
    tenant_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "principal_id",
            _validated_identity(self.principal_id, "owner principal"),
        )
        if self.tenant_id is not None:
            object.__setattr__(
                self,
                "tenant_id",
                _validated_identity(self.tenant_id, "owner tenant"),
            )

    @classmethod
    def from_principal(cls, principal: Principal) -> ResourceOwner:
        """Copy trusted identity into immutable ownership metadata."""

        return cls(principal_id=principal.principal_id, tenant_id=principal.tenant_id)

    def authorizes(self, principal: Principal) -> bool:
        """Return whether both trusted principal and tenant match this owner."""

        principal_matches = _constant_time_equal(self.principal_id, principal.principal_id)
        if self.tenant_id is None or principal.tenant_id is None:
            tenant_matches = self.tenant_id is None and principal.tenant_id is None
        else:
            tenant_matches = _constant_time_equal(self.tenant_id, principal.tenant_id)
        return principal_matches and tenant_matches


def _constant_time_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _validated_identity(value: str, kind: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{kind.capitalize()} identity must be a string")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > _MAX_IDENTITY_CHARS
        or any(ord(character) < 32 or ord(character) == 127 for character in normalized)
    ):
        raise ValueError(f"{kind.capitalize()} identity is invalid")
    return normalized


__all__ = ["Principal", "ResourceOwner"]
