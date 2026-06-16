from __future__ import annotations

from .._compat import warn_deprecated_import
from ..config import DEFAULT_URLS, Settings
from ..config.loader import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_DASHSCOPE_HTTP_BASE_URL,
    apply_runtime_environment,
    load_settings,
    load_yaml_config,
    parse_bool,
    secret_fingerprint,
)
from ..config.loader import (
    parse_optional_int as _parse_optional_int,
)
from ..config.loader import (
    parse_urls as _parse_urls,
)
from ..utils.networking import (
    configure_dashscope_base_url as _configure_dashscope_base_url,
)
from ..utils.networking import (
    parse_dashscope_base_url as _parse_dashscope_base_url,
)

warn_deprecated_import("src.core.config", "src.config")


__all__ = [
    "DEFAULT_CONFIG_FILE",
    "DEFAULT_DASHSCOPE_HTTP_BASE_URL",
    "DEFAULT_URLS",
    "Settings",
    "_configure_dashscope_base_url",
    "_parse_dashscope_base_url",
    "_parse_optional_int",
    "_parse_urls",
    "apply_runtime_environment",
    "load_settings",
    "load_yaml_config",
    "parse_bool",
    "secret_fingerprint",
]
