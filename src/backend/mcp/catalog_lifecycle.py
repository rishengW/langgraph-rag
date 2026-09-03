"""Bounded lifecycle operations layered over immutable catalog validation."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from uuid import uuid4

from . import catalog as _catalog
from .catalog import ToolCatalog as _BaseToolCatalog
from .models import (
    ProviderHealth,
    ToolCatalogReadiness,
    ToolCatalogSnapshot,
    ToolProvider,
    ToolProviderRegistration,
)
from .observability import MCPObservability

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER_CLEANUP_TIMEOUT_SECONDS = 5.0
MAX_PROVIDER_CLEANUP_TIMEOUT_SECONDS = 60.0
_CLEANUP_TIMEOUT: ContextVar[float] = ContextVar(
    "mcp_catalog_cleanup_timeout",
    default=DEFAULT_PROVIDER_CLEANUP_TIMEOUT_SECONDS,
)


async def _bounded_close_provider(
    provider: ToolProvider,
    *,
    phase: str,
) -> None:
    """Attempt one provider close without stalling replacement or shutdown."""

    try:
        async with asyncio.timeout(_CLEANUP_TIMEOUT.get()):
            await provider.close()
    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
        raise
    except BaseException as exc:
        logger.warning(
            "Tool provider cleanup failed provider_type=%s phase=%s error_type=%s",
            type(provider).__name__[:128],
            phase[:128],
            type(exc).__name__[:128],
        )


# Base publication and lease release deliberately share one private cleanup
# seam. Replacing that seam preserves the existing atomic publication logic
# while making unpublished, quarantined, and retired cleanup bounded.
setattr(  # noqa: B010 - deliberate lifecycle seam replacement
    _catalog,
    "_close_provider_quietly",
    _bounded_close_provider,
)


class ToolCatalog(_BaseToolCatalog):
    """Immutable catalog with bounded replacement and administrative health."""

    def __init__(
        self,
        observability: MCPObservability | None = None,
        *,
        cleanup_timeout_seconds: float = DEFAULT_PROVIDER_CLEANUP_TIMEOUT_SECONDS,
    ) -> None:
        if (
            isinstance(cleanup_timeout_seconds, bool)
            or not isinstance(cleanup_timeout_seconds, int | float)
            or not 0 < cleanup_timeout_seconds <= MAX_PROVIDER_CLEANUP_TIMEOUT_SECONDS
        ):
            raise ValueError("Provider cleanup timeout is outside the supported bound")
        super().__init__(observability)
        self._cleanup_timeout_seconds = float(cleanup_timeout_seconds)

    def restricted_dependency_health(self) -> dict[str, object]:
        """Return bounded provider diagnostics for an administrative surface."""

        return {
            "status": self._readiness.status,
            "generation": self._readiness.generation,
            "providers": [
                {
                    "name": health.provider,
                    "status": health.status,
                    "generation": health.generation,
                    "detail": health.detail,
                }
                for health in self._provider_health
            ],
        }

    @asynccontextmanager
    async def acquire_snapshot(self) -> AsyncIterator[ToolCatalogSnapshot]:
        """Lease a generation while retaining this catalog's cleanup bound."""

        token = _CLEANUP_TIMEOUT.set(self._cleanup_timeout_seconds)
        try:
            async with super().acquire_snapshot() as snapshot:
                yield snapshot
        finally:
            _CLEANUP_TIMEOUT.reset(token)

    async def publish(
        self,
        providers: Sequence[ToolProvider | ToolProviderRegistration],
    ) -> ToolCatalogSnapshot:
        """Publish atomically with bounded cleanup on every replacement path."""

        token = _CLEANUP_TIMEOUT.set(self._cleanup_timeout_seconds)
        try:
            return await super().publish(providers)
        finally:
            _CLEANUP_TIMEOUT.reset(token)

    async def close(self) -> None:
        """Mark non-ready first, then close every provider without failure coupling."""

        request_id = uuid4().hex
        started = time.monotonic()
        token = _CLEANUP_TIMEOUT.set(self._cleanup_timeout_seconds)
        try:
            async with self._publication_lock:
                deferred = tuple(
                    provider
                    for generation_providers in self._retired_providers.values()
                    for provider in generation_providers
                )
                providers_to_close = tuple(
                    {id(provider): provider for provider in (*self._providers, *deferred)}.values()
                )
                registrations = self._registrations
                current_generation = self._snapshot.generation if self._snapshot is not None else 0

                self._providers = ()
                self._registrations = ()
                self._retired_providers = {}
                self._active_leases = {}
                self._readiness = ToolCatalogReadiness(
                    status="closed",
                    generation=current_generation,
                )
                self._provider_health = tuple(
                    ProviderHealth(
                        provider=registration.name,
                        status="closed",
                        generation=current_generation,
                    )
                    for registration in registrations
                )

                for provider in reversed(providers_to_close):
                    await _bounded_close_provider(provider, phase="shutdown")

                self._observe(
                    signal="lifecycle",
                    outcome="closed",
                    request_id=request_id,
                    generation=current_generation,
                    started=started,
                )
        finally:
            _CLEANUP_TIMEOUT.reset(token)


# Keep direct imports from ``src.backend.mcp.catalog`` on the same bounded public type.
setattr(  # noqa: B010 - keep direct submodule imports on the bounded type
    _catalog,
    "ToolCatalog",
    ToolCatalog,
)


__all__ = [
    "DEFAULT_PROVIDER_CLEANUP_TIMEOUT_SECONDS",
    "MAX_PROVIDER_CLEANUP_TIMEOUT_SECONDS",
    "ToolCatalog",
]
