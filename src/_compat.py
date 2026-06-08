"""Compatibility helpers for legacy import paths."""

from __future__ import annotations

import warnings


def warn_deprecated_import(old: str, new: str) -> None:
    """Emit a standard deprecation warning for legacy module imports."""

    warnings.warn(
        f"Import from {old} is deprecated; import from {new} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


__all__ = ["warn_deprecated_import"]
