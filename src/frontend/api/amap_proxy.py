"""Hardened same-origin proxy for AMap JavaScript API service calls."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus, unquote

import requests

AMAP_REST_API_BASE_URL = "https://restapi.amap.com"
AMAP_WEB_API_BASE_URL = "https://webapi.amap.com"
AMAP_SERVICE_HOST_PATH = "/_AMapService"
AMAP_PROXY_USER_AGENT = "langgraph-rag/1.0 (AMap JS API proxy)"
MAX_AMAP_PROXY_PATH_CHARS = 512
MAX_AMAP_PROXY_QUERY_CHARS = 4096
MAX_AMAP_PROXY_RESPONSE_BYTES = 5_000_000

_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._~-]+$")
_SAFE_QUERY_NAME_RE = re.compile(r"^[A-Za-z0-9_.\[\]-]{1,64}$")
_ALLOWED_REST_PATHS = frozenset(
    {
        # The basic JS SDK map used by the chat UI authenticates and resolves
        # administrative metadata through these two endpoints.
        "v3/iasdkauth",
        "v3/config/district",
    }
)
_ALLOWED_WEB_PATH_PREFIXES = ("v4/map/styles",)
_SAFE_EXACT_CONTENT_TYPES = {
    "application/json",
    "application/javascript",
    "application/octet-stream",
    "text/javascript",
    "text/plain",
}

ProxyRequester = Callable[..., Any]


class AMapProxyError(RuntimeError):
    """A client-safe proxy error carrying an HTTP status code."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class AMapProxyResponse:
    """Bounded upstream response safe to return through FastAPI."""

    content: bytes
    media_type: str
    status_code: int = 200


def build_amap_client_config(settings: Any) -> dict[str, object]:
    """Return the strict browser-visible subset of AMap configuration."""

    enabled = bool(
        (getattr(settings, "map_enabled", False) or getattr(settings, "directions_enabled", False))
        and str(getattr(settings, "amap_js_api_key", "") or "").strip()
        and str(getattr(settings, "amap_js_security_code", "") or "").strip()
    )
    if not enabled:
        return {"amap": {"enabled": False}}

    return {
        "amap": {
            "enabled": True,
            "js_api_key": str(settings.amap_js_api_key).strip(),
            "service_host": AMAP_SERVICE_HOST_PATH,
            "api_version": "2.0",
            "coordinate_system": "gcj02",
        }
    }


def validate_amap_proxy_path(path: str) -> str:
    """Validate an AMap-relative API path without normalizing attacker input."""

    value = path or ""
    if (
        not value
        or value != value.strip()
        or len(value) > MAX_AMAP_PROXY_PATH_CHARS
    ):
        raise AMapProxyError("Invalid AMap service path.", status_code=400)
    decoded = unquote(value)
    if decoded != value:
        raise AMapProxyError("Invalid AMap service path.", status_code=400)
    if any(marker in value for marker in ("\\", ":", "//")):
        raise AMapProxyError("Invalid AMap service path.", status_code=400)
    segments = value.split("/")
    if any(
        not segment
        or segment in {".", ".."}
        or _SAFE_PATH_SEGMENT_RE.fullmatch(segment) is None
        for segment in segments
    ):
        raise AMapProxyError("Invalid AMap service path.", status_code=400)
    if value not in _ALLOWED_REST_PATHS and not any(
        value == prefix or value.startswith(f"{prefix}/")
        for prefix in _ALLOWED_WEB_PATH_PREFIXES
    ):
        raise AMapProxyError("Unsupported AMap service path.", status_code=404)
    return value


def fetch_amap_proxy_response(
    path: str,
    query_items: Iterable[tuple[str, str]],
    *,
    security_code: str,
    timeout_seconds: int = 10,
    requester: ProxyRequester | None = None,
) -> AMapProxyResponse:
    """Fetch one fixed-host AMap service response with the security code injected."""

    safe_path = validate_amap_proxy_path(path)
    secret = (security_code or "").strip()
    if not secret:
        raise AMapProxyError("AMap JavaScript service is not configured.", status_code=503)

    params: list[tuple[str, str]] = []
    query_size = 0
    for raw_name, raw_value in query_items:
        name = str(raw_name)
        value = str(raw_value)
        if _SAFE_QUERY_NAME_RE.fullmatch(name) is None:
            raise AMapProxyError("Invalid AMap service query.", status_code=400)
        try:
            query_size += len(quote_plus(name)) + len(quote_plus(value)) + 2
        except UnicodeError as exc:
            raise AMapProxyError("Invalid AMap service query.", status_code=400) from exc
        if query_size > MAX_AMAP_PROXY_QUERY_CHARS:
            raise AMapProxyError("AMap service query is too large.", status_code=400)
        if name.casefold() == "jscode":
            continue
        params.append((name, value))
    params.append(("jscode", secret))

    base_url = (
        AMAP_WEB_API_BASE_URL
        if any(
            safe_path == prefix or safe_path.startswith(f"{prefix}/")
            for prefix in _ALLOWED_WEB_PATH_PREFIXES
        )
        else AMAP_REST_API_BASE_URL
    )
    get = requester or requests.get
    try:
        response = get(
            f"{base_url}/{safe_path}",
            params=params,
            timeout=max(1, int(timeout_seconds)),
            headers={
                "Accept": "application/json, text/plain, */*",
                "User-Agent": AMAP_PROXY_USER_AGENT,
            },
            allow_redirects=False,
            stream=True,
        )
    except (requests.Timeout, TimeoutError) as exc:
        raise AMapProxyError("AMap service request timed out.", status_code=504) from exc
    except Exception as exc:
        raise AMapProxyError("AMap service request failed.", status_code=502) from exc

    if isinstance(response, dict):
        try:
            content = json.dumps(
                response,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise AMapProxyError(
                "AMap service returned an invalid response.", status_code=502
            ) from exc
        if len(content) > MAX_AMAP_PROXY_RESPONSE_BYTES:
            raise AMapProxyError("AMap service response is too large.", status_code=502)
        return AMapProxyResponse(content=content, media_type="application/json")

    try:
        status_code = int(getattr(response, "status_code", 502) or 502)
        if not 200 <= status_code < 300:
            raise AMapProxyError("AMap service returned an upstream error.", status_code=502)

        content = _bounded_response_content(response)
        content_type = ""
        headers = getattr(response, "headers", None)
        if headers is not None:
            content_type = str(headers.get("content-type", "") or "")
        return AMapProxyResponse(
            content=content,
            media_type=_safe_media_type(content_type),
            status_code=status_code,
        )
    except AMapProxyError:
        raise
    except Exception as exc:
        raise AMapProxyError("AMap service response failed.", status_code=502) from exc
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            with suppress(Exception):
                close()


def _bounded_response_content(response: Any) -> bytes:
    chunks: list[bytes] = []
    total = 0
    iterator = getattr(response, "iter_content", None)
    if callable(iterator):
        for chunk in iterator(chunk_size=64 * 1024):
            if not chunk:
                continue
            data = chunk.encode("utf-8") if isinstance(chunk, str) else bytes(chunk)
            total += len(data)
            if total > MAX_AMAP_PROXY_RESPONSE_BYTES:
                raise AMapProxyError("AMap service response is too large.", status_code=502)
            chunks.append(data)
        return b"".join(chunks)

    raw_content = getattr(response, "content", b"")
    content = raw_content.encode("utf-8") if isinstance(raw_content, str) else bytes(raw_content)
    if len(content) > MAX_AMAP_PROXY_RESPONSE_BYTES:
        raise AMapProxyError("AMap service response is too large.", status_code=502)
    return content


def _safe_media_type(content_type: str) -> str:
    media_type = content_type.partition(";")[0].strip().lower()
    if media_type in _SAFE_EXACT_CONTENT_TYPES:
        return media_type
    return "application/octet-stream"


__all__ = [
    "AMAP_SERVICE_HOST_PATH",
    "AMapProxyError",
    "AMapProxyResponse",
    "build_amap_client_config",
    "fetch_amap_proxy_response",
    "validate_amap_proxy_path",
]
