from __future__ import annotations

import logging
import os
import ssl

logger = logging.getLogger(__name__)

DEFAULT_DASHSCOPE_HTTP_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_USER_AGENT = "only-subcribers/1.0"


def configure_ssl_from_env() -> None:
    """Apply the legacy SSL-debug environment override if requested."""

    if os.getenv("DISABLE_SSL_VERIFY", "").lower() != "true":
        return

    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["CURL_CA_BUNDLE"] = ""
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception as exc:
        logger.warning("Could not disable SSL verification: %s", exc)


def parse_dashscope_base_url(raw_value: str | None) -> str:
    value = (raw_value or "").strip().rstrip("/")
    if not value:
        return ""
    if not value.startswith(("http://", "https://")):
        raise RuntimeError(
            "DASHSCOPE_HTTP_BASE_URL must start with http:// or https://. "
            f"Got: {value!r}"
        )
    return value


def configure_dashscope_base_url(base_url: str) -> None:
    """Apply DashScope base URL to environment and SDK global settings."""

    if base_url:
        os.environ["DASHSCOPE_HTTP_BASE_URL"] = base_url
        effective_url = base_url
    else:
        os.environ.pop("DASHSCOPE_HTTP_BASE_URL", None)
        effective_url = DEFAULT_DASHSCOPE_HTTP_BASE_URL

    try:
        import dashscope

        dashscope.base_http_api_url = effective_url
    except Exception:
        pass


def ensure_user_agent() -> None:
    os.environ.setdefault("USER_AGENT", DEFAULT_USER_AGENT)

