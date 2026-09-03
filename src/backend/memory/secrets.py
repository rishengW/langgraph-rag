"""Credential screening for memory writes.

A plain-text JSON file must not become a secret store, so content and tags are
screened before anything is persisted. ``find_secret_match`` returns the *name*
of the matching pattern, never the matched text, which keeps the "must not echo
the secret" guarantee at the type level: callers have nothing sensitive to leak.
"""

from __future__ import annotations

import re
from typing import Final

SECRET_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "pem_private_key",
        re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE),
    ),
    (
        "sk_token",
        re.compile(r"sk-[A-Za-z0-9_-]{16,}", re.IGNORECASE),
    ),
    (
        "aws_access_key_id",
        re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}", re.IGNORECASE),
    ),
    (
        "bearer_token",
        re.compile(r"\bbearer\s+\S{20,}", re.IGNORECASE),
    ),
    (
        "assigned_secret",
        re.compile(
            r"[\w.-]*(?:password|passwd|secret|api[_-]?key|token)[\w.-]*\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
)


def find_secret_match(text: str) -> str | None:
    """Return the name of the first matching Secret_Pattern, else ``None``.

    Never returns any part of ``text``.
    """

    if not text:
        return None
    for name, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return name
    return None


def find_secret_in_any(texts: object) -> str | None:
    """Return the first Secret_Pattern name matched by any string in ``texts``."""

    if isinstance(texts, str):
        return find_secret_match(texts)
    if texts is None:
        return None
    try:
        items = list(texts)  # type: ignore[call-overload]
    except TypeError:
        return None
    for item in items:
        if isinstance(item, str):
            name = find_secret_match(item)
            if name is not None:
                return name
    return None


__all__ = [
    "SECRET_PATTERNS",
    "find_secret_in_any",
    "find_secret_match",
]
