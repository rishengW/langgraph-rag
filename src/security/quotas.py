"""Bounded process-local quotas keyed by trusted principal and tenant identity."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from ..errors import QuotaExceededError
from .identity import Principal

_HARD_MAX_RATE = 1_000_000_000
_HARD_MAX_CONCURRENCY = 1_024
_HARD_MAX_TRACKED_IDENTITIES = 100_000
_WINDOW_SECONDS = 60.0
_USAGE_FIELDS = (
    "requests",
    "searches",
    "tokens",
    "tool_calls",
    "retries",
    "cost_units",
)
_KeyT = TypeVar("_KeyT")


@dataclass(frozen=True, slots=True)
class QuotaCharge:
    """Usage charged atomically for one accepted operation."""

    requests: int = 1
    searches: int = 0
    tokens: int = 0
    tool_calls: int = 0
    retries: int = 0
    cost_units: int = 1

    def __post_init__(self) -> None:
        for name in _USAGE_FIELDS:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= _HARD_MAX_RATE
            ):
                raise ValueError(f"Quota charge {name} must be a bounded non-negative integer")


@dataclass(frozen=True, slots=True)
class QuotaBudget:
    """Rolling-minute usage and active-call bounds for one identity scope."""

    requests_per_minute: int
    concurrent_calls: int
    searches_per_minute: int
    tokens_per_minute: int
    tool_calls_per_minute: int
    retries_per_minute: int
    cost_units_per_minute: int

    def __post_init__(self) -> None:
        _bounded_positive(self.requests_per_minute, _HARD_MAX_RATE, "request rate")
        _bounded_positive(self.concurrent_calls, _HARD_MAX_CONCURRENCY, "concurrency")
        _bounded_positive(self.searches_per_minute, _HARD_MAX_RATE, "search rate")
        _bounded_positive(self.tokens_per_minute, _HARD_MAX_RATE, "token rate")
        _bounded_positive(self.tool_calls_per_minute, _HARD_MAX_RATE, "tool-call rate")
        _bounded_positive(self.retries_per_minute, _HARD_MAX_RATE, "retry rate")
        _bounded_positive(self.cost_units_per_minute, _HARD_MAX_RATE, "cost rate")

    def maximum_for(self, field_name: str) -> int:
        return {
            "requests": self.requests_per_minute,
            "searches": self.searches_per_minute,
            "tokens": self.tokens_per_minute,
            "tool_calls": self.tool_calls_per_minute,
            "retries": self.retries_per_minute,
            "cost_units": self.cost_units_per_minute,
        }[field_name]


@dataclass(frozen=True, slots=True)
class QuotaLimits:
    """Separate immutable budgets for principals and optional tenants."""

    principal: QuotaBudget = field(
        default_factory=lambda: QuotaBudget(
            requests_per_minute=120,
            concurrent_calls=4,
            searches_per_minute=30,
            tokens_per_minute=250_000,
            tool_calls_per_minute=240,
            retries_per_minute=600,
            cost_units_per_minute=500_000,
        )
    )
    tenant: QuotaBudget = field(
        default_factory=lambda: QuotaBudget(
            requests_per_minute=600,
            concurrent_calls=16,
            searches_per_minute=120,
            tokens_per_minute=1_000_000,
            tool_calls_per_minute=1_000,
            retries_per_minute=3_000,
            cost_units_per_minute=2_000_000,
        )
    )
    max_tracked_identities: int = 4_096

    def __post_init__(self) -> None:
        _bounded_positive(
            self.max_tracked_identities,
            _HARD_MAX_TRACKED_IDENTITIES,
            "tracked identity count",
        )


@dataclass(slots=True)
class _UsageWindow:
    events: deque[tuple[float, QuotaCharge]] = field(default_factory=deque)
    totals: dict[str, int] = field(default_factory=lambda: dict.fromkeys(_USAGE_FIELDS, 0))

    def prune(self, now: float) -> None:
        while self.events and now - self.events[0][0] >= _WINDOW_SECONDS:
            _timestamp, charge = self.events.popleft()
            for name in _USAGE_FIELDS:
                self.totals[name] -= getattr(charge, name)

    def accepts(self, charge: QuotaCharge, budget: QuotaBudget) -> bool:
        return all(
            self.totals[name] + getattr(charge, name) <= budget.maximum_for(name)
            for name in _USAGE_FIELDS
        )

    def append(self, now: float, charge: QuotaCharge) -> None:
        self.events.append((now, charge))
        for name in _USAGE_FIELDS:
            self.totals[name] += getattr(charge, name)


class QuotaLease:
    """Idempotent active-call lease returned by :class:`QuotaManager`."""

    def __init__(
        self,
        manager: QuotaManager,
        principal_key: tuple[str | None, str],
        tenant_key: str | None,
    ) -> None:
        self._manager = manager
        self._principal_key = principal_key
        self._tenant_key = tenant_key
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._manager._release(self._principal_key, self._tenant_key)

    def __enter__(self) -> QuotaLease:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.release()


class QuotaManager:
    """Thread-safe rolling limiter for the supported single-process topology."""

    def __init__(
        self,
        limits: QuotaLimits | None = None,
        *,
        time_func: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limits = limits or QuotaLimits()
        self._time = time_func
        self._lock = threading.Lock()
        self._principal_windows: dict[tuple[str | None, str], _UsageWindow] = {}
        self._tenant_windows: dict[str, _UsageWindow] = {}
        self._active_principals: dict[tuple[str | None, str], int] = {}
        self._active_tenants: dict[str, int] = {}

    @property
    def limits(self) -> QuotaLimits:
        return self._limits

    def acquire(
        self,
        principal: Principal,
        charge: QuotaCharge | None = None,
    ) -> QuotaLease:
        """Atomically enforce rolling usage and active-call bounds."""

        usage = charge or QuotaCharge()
        now = self._time()
        principal_key = (principal.tenant_id, principal.principal_id)
        tenant_key = principal.tenant_id
        with self._lock:
            principal_window = self._window(self._principal_windows, principal_key, now)
            tenant_window = (
                None if tenant_key is None else self._window(self._tenant_windows, tenant_key, now)
            )
            if not principal_window.accepts(usage, self._limits.principal):
                raise QuotaExceededError()
            if tenant_window is not None and not tenant_window.accepts(usage, self._limits.tenant):
                raise QuotaExceededError()
            if (
                self._active_principals.get(principal_key, 0)
                >= self._limits.principal.concurrent_calls
            ):
                raise QuotaExceededError()
            if (
                tenant_key is not None
                and self._active_tenants.get(tenant_key, 0) >= self._limits.tenant.concurrent_calls
            ):
                raise QuotaExceededError()

            principal_window.append(now, usage)
            if tenant_window is not None:
                tenant_window.append(now, usage)
            self._active_principals[principal_key] = (
                self._active_principals.get(principal_key, 0) + 1
            )
            if tenant_key is not None:
                self._active_tenants[tenant_key] = self._active_tenants.get(tenant_key, 0) + 1
        return QuotaLease(self, principal_key, tenant_key)

    def _window(
        self,
        windows: dict[_KeyT, _UsageWindow],
        key: _KeyT,
        now: float,
    ) -> _UsageWindow:
        window = windows.get(key)
        if window is not None:
            window.prune(now)
            return window
        if len(windows) >= self._limits.max_tracked_identities:
            for candidate, existing in tuple(windows.items()):
                existing.prune(now)
                if not existing.events:
                    del windows[candidate]
        if len(windows) >= self._limits.max_tracked_identities:
            raise QuotaExceededError()
        window = _UsageWindow()
        windows[key] = window
        return window

    def _release(
        self,
        principal_key: tuple[str | None, str],
        tenant_key: str | None,
    ) -> None:
        with self._lock:
            _decrement(self._active_principals, principal_key)
            if tenant_key is not None:
                _decrement(self._active_tenants, tenant_key)


def estimate_token_units(text: str | None) -> int:
    """Return a deterministic conservative token estimate for quota charging."""

    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def estimated_cost_units(charge: QuotaCharge) -> int:
    """Convert bounded work dimensions into provider-independent cost units."""

    return max(
        1,
        charge.cost_units,
        charge.tokens + charge.searches * 100 + charge.tool_calls * 10 + charge.retries * 5,
    )


def _bounded_positive(value: int, maximum: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"Quota {label} must be between 1 and {maximum}")


def _decrement(values: dict[_KeyT, int], key: _KeyT) -> None:
    remaining = values.get(key, 0) - 1
    if remaining > 0:
        values[key] = remaining
    else:
        values.pop(key, None)


__all__ = [
    "QuotaBudget",
    "QuotaCharge",
    "QuotaLease",
    "QuotaLimits",
    "QuotaManager",
    "estimate_token_units",
    "estimated_cost_units",
]
