from __future__ import annotations

import logging
import gc
import os
import shutil
import ssl
import stat
import time
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any, TypeVar

from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_HTTP_STATUSES = {
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
}


def is_retryable_connection_error(error: Exception) -> bool:
    """Return whether an exception looks like a transient network failure."""

    error_msg = str(error).upper()
    retry_markers = (
        "SSL",
        "CERTIFICATE",
        "EOF",
        "CONNECTION",
        "MAX RETRIES",
        "TIMEOUT",
        "REMOTE END",
        "TEMPORARILY UNAVAILABLE",
    )
    return any(marker in error_msg for marker in retry_markers)


def invoke_with_retry(
    chain: Any,
    input_data: Any,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Any:
    """Invoke a LangChain runnable with retry logic for transient failures."""

    last_error: Exception | None = None
    max_retries = max(1, max_retries)

    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        except (OSError, ConnectionError, TimeoutError, RequestException, ssl.SSLError) as exc:
            last_error = exc
            if not is_retryable_connection_error(exc):
                raise
            logger.warning(
                "SSL/Connection error on attempt %d/%d: %s",
                attempt + 1,
                max_retries,
                exc,
            )
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt) + (os.urandom(1)[0] / 256)
                logger.info("Retrying in %.1f seconds...", delay)
                time.sleep(delay)
            else:
                logger.error("Failed after %d attempts: %s", max_retries, exc)
        except Exception:
            logger.exception("Non-retryable error")
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("Invocation failed without an error response.")


def call_with_retry(
    operation: Callable[[], T],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable: Callable[[Exception], bool] | None = None,
    log_label: str = "operation",
) -> T:
    """Run an arbitrary operation with exponential backoff."""

    last_error: Exception | None = None
    max_retries = max(1, max_retries)
    retryable = retryable or is_retryable_connection_error

    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if not retryable(exc):
                raise
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "%s attempt %d/%d failed; retrying in %.1fs: %s",
                    log_label,
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
            else:
                logger.error("%s failed after %d attempts: %s", log_label, max_retries, exc)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{log_label} failed without an error response.")


def dashscope_call_with_retry(
    client: Any,
    kwargs: dict[str, Any],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Any:
    """Call a DashScope client and retry retryable HTTP/network failures."""

    last_error: Exception | None = None
    max_retries = max(1, max_retries)

    for attempt in range(max_retries):
        try:
            response = client.call(**kwargs)
            if response.status_code == HTTPStatus.OK:
                return response

            error = RuntimeError(
                "DashScope embedding error "
                f"({response.status_code}, {response.code}): {response.message}"
            )
            if response.status_code not in RETRYABLE_HTTP_STATUSES:
                raise error
            last_error = error
        except (OSError, ConnectionError, TimeoutError, RequestException, ssl.SSLError) as exc:
            last_error = exc

        if attempt < max_retries - 1:
            delay = base_delay * (2**attempt)
            logger.warning(
                "DashScope embedding attempt %d/%d failed; retrying in %.1fs: %s",
                attempt + 1,
                max_retries,
                delay,
                last_error,
            )
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("DashScope embedding call failed without an error response.")


def remove_tree_with_retry(
    path: Path,
    *,
    max_retries: int = 10,
    delay: float = 1.0,
    operation_logger: logging.Logger | None = None,
) -> None:
    """Remove a directory tree with retries for Windows file locking."""

    log = operation_logger or logger
    if not path.exists():
        return

    def handle_remove_error(func, fpath, exc_info):
        try:
            if os.path.exists(fpath):
                os.chmod(fpath, stat.S_IWUSR | stat.S_IRUSR)
                os.unlink(fpath)
                log.debug("Force-removed locked file: %s", fpath)
        except Exception as exc:
            log.warning("Could not force-remove %s: %s", fpath, exc)

    def remove_tree_manually(dirpath: str) -> bool:
        try:
            for root, dirs, files in os.walk(dirpath, topdown=False):
                for name in files:
                    filepath = os.path.join(root, name)
                    try:
                        os.chmod(filepath, stat.S_IWUSR | stat.S_IRUSR)
                        os.unlink(filepath)
                    except Exception as exc:
                        log.warning("Failed to remove file %s: %s", filepath, exc)
                for name in dirs:
                    inner_dir = os.path.join(root, name)
                    try:
                        os.rmdir(inner_dir)
                    except Exception as exc:
                        log.warning("Failed to remove dir %s: %s", inner_dir, exc)
            if os.path.exists(dirpath):
                os.rmdir(dirpath)
                return True
        except Exception as exc:
            log.warning("Manual tree removal failed: %s", exc)
        return False

    for attempt in range(max_retries):
        try:
            gc.collect()
            time.sleep(0.1)
            shutil.rmtree(str(path), onerror=handle_remove_error)
            if not path.exists():
                log.info("Successfully removed .chroma directory after %d attempt(s)", attempt + 1)
                return
        except (PermissionError, OSError) as exc:
            if attempt < max_retries - 1:
                log.warning(
                    "Failed to remove %s (attempt %d/%d), retrying in %.1f seconds: %s",
                    path,
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
            else:
                log.warning("Standard removal failed, attempting manual file-by-file removal")
                if remove_tree_manually(str(path)):
                    log.info("Manual removal succeeded")
                    return
                log.error("Failed to remove %s after %d attempts and manual removal", path, max_retries)
                raise

    if path.exists():
        log.warning("Standard removal did not delete %s; attempting manual file-by-file removal", path)
        if remove_tree_manually(str(path)):
            log.info("Manual removal succeeded")
            return
        raise OSError(f"Failed to remove {path} after {max_retries} attempts")
