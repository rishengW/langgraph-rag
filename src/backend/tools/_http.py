from __future__ import annotations

from collections.abc import Callable
from typing import Any

import requests

JsonRequester = Callable[..., Any]


def request_json(
    url: str,
    *,
    params: dict[str, Any],
    requester: JsonRequester | None = None,
    timeout: int = 10,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Call an HTTP JSON endpoint with a tiny injectable surface for tests."""

    get = requester or requests.get
    kwargs: dict[str, Any] = {"params": params, "timeout": timeout}
    if headers:
        kwargs["headers"] = headers

    response = get(url, **kwargs)
    if isinstance(response, dict):
        return response
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


__all__ = ["JsonRequester", "request_json"]
