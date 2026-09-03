from __future__ import annotations

import json

import pytest

from src.frontend.api.amap_proxy import (
    MAX_AMAP_PROXY_RESPONSE_BYTES,
    AMapProxyError,
    build_amap_client_config,
    fetch_amap_proxy_response,
    validate_amap_proxy_path,
)
from src.config import Settings


def test_amap_client_config_exposes_only_browser_safe_values():
    settings = Settings(
        dashscope_api_key="test-key",
        map_enabled=True,
        amap_web_service_key="server-secret",
        amap_js_api_key="browser-key",
        amap_js_security_code="security-secret",
    )

    config = build_amap_client_config(settings)
    serialized = json.dumps(config)

    assert config == {
        "amap": {
            "enabled": True,
            "js_api_key": "browser-key",
            "service_host": "/_AMapService",
            "api_version": "2.0",
            "coordinate_system": "gcj02",
        }
    }
    assert "server-secret" not in serialized
    assert "security-secret" not in serialized


def test_amap_client_config_is_disabled_without_each_required_setting():
    base = {
        "dashscope_api_key": "test-key",
        "map_enabled": True,
        "amap_js_api_key": "browser-key",
        "amap_js_security_code": "security-secret",
    }

    assert build_amap_client_config(Settings(**{**base, "map_enabled": False})) == {
        "amap": {"enabled": False}
    }
    assert build_amap_client_config(Settings(**{**base, "amap_js_api_key": ""})) == {
        "amap": {"enabled": False}
    }
    assert build_amap_client_config(Settings(**{**base, "amap_js_security_code": ""})) == {
        "amap": {"enabled": False}
    }


@pytest.mark.parametrize(
    "path",
    [
        "https://evil.test/v3/config/district",
        "v3/../secrets",
        "v3//config/district",
        "v3/%2e%2e/secrets",
        "%76%33/iasdkauth",
        " v3/iasdkauth",
        "v2/unsupported",
        "v3/place/text",
        "v5/direction/driving",
        "v3\\config\\district",
    ],
)
def test_amap_proxy_rejects_unsafe_or_unsupported_paths(path):
    with pytest.raises(AMapProxyError):
        validate_amap_proxy_path(path)


def test_amap_proxy_uses_fixed_rest_host_and_overrides_caller_jscode():
    captured = {}

    def requester(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return {"status": "1", "info": "OK"}

    result = fetch_amap_proxy_response(
        "v3/config/district",
        [("keywords", "Shanghai"), ("jscode", "attacker-value")],
        security_code="real-security-code",
        requester=requester,
    )

    assert captured["url"] == "https://restapi.amap.com/v3/config/district"
    assert ("jscode", "attacker-value") not in captured["kwargs"]["params"]
    assert ("jscode", "real-security-code") in captured["kwargs"]["params"]
    assert captured["kwargs"]["allow_redirects"] is False
    assert json.loads(result.content) == {"status": "1", "info": "OK"}


def test_amap_proxy_uses_web_host_for_map_styles():
    captured = {}

    def requester(url, **kwargs):
        captured["url"] = url
        return {"style": "ok"}

    fetch_amap_proxy_response(
        "v4/map/styles",
        [],
        security_code="security-code",
        requester=requester,
    )

    assert captured["url"] == "https://webapi.amap.com/v4/map/styles"


def test_amap_proxy_requires_server_side_security_code():
    with pytest.raises(AMapProxyError) as exc_info:
        fetch_amap_proxy_response(
            "v3/config/district",
            [],
            security_code="",
            requester=lambda *args, **kwargs: {},
        )

    assert exc_info.value.status_code == 503
    assert "security-code" not in str(exc_info.value)


@pytest.mark.parametrize("name", ["", "bad name", "host:override", "x" * 65])
def test_amap_proxy_rejects_invalid_query_names(name):
    with pytest.raises(AMapProxyError) as exc_info:
        fetch_amap_proxy_response(
            "v3/iasdkauth",
            [(name, "value")],
            security_code="security-code",
            requester=lambda *args, **kwargs: {},
        )

    assert exc_info.value.status_code == 400


def test_amap_proxy_bounds_encoded_query_size():
    with pytest.raises(AMapProxyError) as exc_info:
        fetch_amap_proxy_response(
            "v3/iasdkauth",
            [("key", "\u4e0a" * 1000)],
            security_code="security-code",
            requester=lambda *args, **kwargs: {},
        )

    assert exc_info.value.status_code == 400


def test_amap_proxy_bounds_discarded_jscode_values_too():
    with pytest.raises(AMapProxyError) as exc_info:
        fetch_amap_proxy_response(
            "v3/iasdkauth",
            [("jscode", "x" * 5000)],
            security_code="security-code",
            requester=lambda *args, **kwargs: {},
        )

    assert exc_info.value.status_code == 400


def test_amap_proxy_bounds_streamed_response_and_always_closes_it():
    class Response:
        status_code = 200
        headers = {"content-type": "image/svg+xml"}
        closed = False

        def iter_content(self, *, chunk_size):
            assert chunk_size > 0
            yield b"x" * (MAX_AMAP_PROXY_RESPONSE_BYTES // 2 + 1)
            yield b"y" * (MAX_AMAP_PROXY_RESPONSE_BYTES // 2 + 1)

        def close(self):
            self.closed = True

    response = Response()
    with pytest.raises(AMapProxyError) as exc_info:
        fetch_amap_proxy_response(
            "v4/map/styles",
            [],
            security_code="security-code",
            requester=lambda *args, **kwargs: response,
        )

    assert exc_info.value.status_code == 502
    assert response.closed is True


def test_amap_proxy_does_not_reflect_active_image_content_types():
    class Response:
        status_code = 200
        headers = {"content-type": "image/svg+xml"}
        content = b"<svg/>"
        closed = False

        def close(self):
            self.closed = True

    response = Response()
    result = fetch_amap_proxy_response(
        "v4/map/styles",
        [],
        security_code="security-code",
        requester=lambda *args, **kwargs: response,
    )

    assert result.media_type == "application/octet-stream"
    assert response.closed is True


def test_amap_proxy_wraps_stream_read_failures_and_ignores_close_failures():
    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}

        def iter_content(self, *, chunk_size):
            raise OSError("connection reset")
            yield b""  # pragma: no cover - keeps this method an iterator

        def close(self):
            raise OSError("close failed")

    with pytest.raises(AMapProxyError) as exc_info:
        fetch_amap_proxy_response(
            "v3/iasdkauth",
            [],
            security_code="security-code",
            requester=lambda *args, **kwargs: Response(),
        )

    assert exc_info.value.status_code == 502
    assert "connection reset" not in str(exc_info.value)
