from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx2
import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client

from src.backend.application import RagAnswer, SourceReference
from src.frontend.adapters.mcp_server.auth import resolve_http_auth
from src.frontend.adapters.mcp_server.config import MCPSettings, load_mcp_settings
from src.frontend.adapters.mcp_server.lifecycle import initialize_runtime
from src.frontend.adapters.mcp_server.tools import CANONICAL_TOOL_NAMES, create_mcp_server
from src.frontend.adapters.mcp_server.transport import build_http_app, transport_security
from src.frontend.adapters.mcp_server.url_policy import UnsafeSourceURLError, URLValidator

_PUBLIC_ADDRESSES = ("93.184.216.34",)


class FakeRagService:
    def __init__(
        self,
        *,
        answer: str = "grounded answer",
        delay: float = 0,
        source_mode: str = "defaults",
    ) -> None:
        self.answer = answer
        self.delay = delay
        self.source_mode = source_mode
        self.requests: list[Any] = []
        self.active = 0
        self.max_active = 0
        self.cancelled = False
        self.closed = False
        self.started = asyncio.Event()

    async def ask(self, request: Any) -> RagAnswer:
        self.requests.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            sources = tuple(
                SourceReference(url=url, citation_id=f"source-{index}")
                for index, url in enumerate(request.urls or (), start=1)
            )
            return RagAnswer(
                answer=self.answer,
                error=None,
                success=True,
                messages=None,
                source_urls=[source.url for source in sources],
                source_mode=self.source_mode,
                source_note=None,
                request_id=request.request_id,
                sources=sources,
            )
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        self.closed = True


def _validator(addresses: tuple[str, ...] = _PUBLIC_ADDRESSES) -> URLValidator:
    async def async_resolver(_host: str, _port: int) -> tuple[str, ...]:
        return addresses

    return URLValidator(
        async_resolver=async_resolver,
        sync_resolver=lambda _host, _port: addresses,
    )


def test_config_defaults_disabled_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MCP_ENABLED", raising=False)
    assert load_mcp_settings(env_file="missing.env", config_file=None).enabled is False

    with pytest.raises(ValueError, match="MCP_ALLOWED_HOSTS"):
        MCPSettings(enabled=True, transport="http", environment="production")
    with pytest.raises(ValueError, match="Anonymous"):
        MCPSettings(
            enabled=True,
            transport="http",
            environment="staging",
            allow_anonymous_http=True,
        )
    with pytest.raises(ValueError, match="wildcards"):
        MCPSettings(
            enabled=True,
            transport="http",
            environment="production",
            allowed_hosts=("example.com:*",),
            public_base_url="https://example.com",
        )

    with pytest.raises(ValueError, match="reserved operational endpoint"):
        MCPSettings(enabled=True, path="/health")

    monkeypatch.setenv("MCP_OUTBOUND_ENABLED", "true")
    with pytest.raises(ValueError, match="Outbound"):
        load_mcp_settings(env_file="missing.env", config_file=None)


def test_http_auth_is_required_and_secret_repr_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = MCPSettings(
        enabled=True,
        transport="http",
        auth_secret_env="TEST_MCP_SECRET",
        public_base_url="http://testserver",
        allowed_hosts=("testserver",),
    )
    monkeypatch.delenv("TEST_MCP_SECRET", raising=False)
    with pytest.raises(ValueError, match="requires a secret"):
        resolve_http_auth(settings)

    secret = "correct-horse-battery-staple"
    monkeypatch.setenv("TEST_MCP_SECRET", secret)
    resolved = resolve_http_auth(settings)
    assert resolved is not None
    assert secret not in repr(resolved)
    assert settings.auth_secret_env in repr(settings)


def test_url_validation_rejects_unsafe_destinations_and_checks_every_address() -> None:
    async def run() -> None:
        safe = _validator()
        assert await safe.validate(["https://example.com/path"], maximum=10) == [
            "https://example.com/path"
        ]
        for url in (
            "http://example.com",
            "https://user:password@example.com",
            "https://127.0.0.1",
            "https://metadata.google.internal",
            "https://example.com:8443",
        ):
            with pytest.raises(UnsafeSourceURLError):
                await safe.validate([url], maximum=10)

        mixed = _validator(("93.184.216.34", "10.0.0.5"))
        with pytest.raises(UnsafeSourceURLError, match="prohibited"):
            await mixed.validate(["https://example.com"], maximum=10)

    asyncio.run(run())


def test_in_memory_client_lists_exactly_two_closed_tools_and_invokes_both() -> None:
    async def run() -> None:
        service = FakeRagService(source_mode="web_search")
        server, _ = create_mcp_server(
            service=service,
            settings=MCPSettings(enabled=True),
            url_validator=_validator(),
        )
        async with Client(server) as client:
            listed = await client.list_tools()
            assert tuple(tool.name for tool in listed.tools) == CANONICAL_TOOL_NAMES
            assert all(tool.input_schema["additionalProperties"] is False for tool in listed.tools)
            assert await server.list_resources() == []
            assert await server.list_prompts() == []

            ask = await client.call_tool(
                "rag_ask",
                {"question": "What is documented?", "sources": ["https://example.com/a"]},
            )
            assert ask.is_error is False
            assert ask.structured_content is not None
            assert set(ask.structured_content) == {
                "answer",
                "sources",
                "request_id",
                "grounded",
                "warnings",
            }
            web = await client.call_tool("rag_web_search_answer", {"question": "What is new?"})
            assert web.is_error is False

        assert service.requests[0].urls == ["https://example.com/a"]
        assert service.requests[0].web_search is False
        assert service.requests[1].urls is None
        assert service.requests[1].web_search is True

    asyncio.run(run())


def test_unknown_and_oversized_inputs_are_sanitized_without_invocation() -> None:
    async def run() -> None:
        service = FakeRagService()
        server, _ = create_mcp_server(service=service, settings=MCPSettings(enabled=True))
        marker = "super-secret-marker"
        async with Client(server) as client:
            unknown = await client.call_tool(
                "rag_web_search_answer", {"question": "hello", "unknown": marker}
            )
            oversized = await client.call_tool(
                "rag_web_search_answer", {"question": marker + ("x" * 16_001)}
            )
        assert unknown.is_error and oversized.is_error
        texts = " ".join(block.text for result in (unknown, oversized) for block in result.content)
        assert marker not in texts
        for result in (unknown, oversized):
            payload = json.loads(result.content[0].text)
            assert payload["code"] == "INVALID_REQUEST"
            assert payload["request_id"]
        assert service.requests == []

    asyncio.run(run())


def test_timeout_concurrency_output_bounds_and_redaction(caplog: pytest.LogCaptureFixture) -> None:
    async def run() -> None:
        timeout_service = FakeRagService(delay=0.1)
        timeout_server, _ = create_mcp_server(
            service=timeout_service,
            settings=MCPSettings(enabled=True, deadline_seconds=0.01),
        )
        async with Client(timeout_server) as client:
            timed_out = await client.call_tool("rag_web_search_answer", {"question": "hello"})
        assert timed_out.is_error
        assert json.loads(timed_out.content[0].text)["code"] == "DEADLINE_EXCEEDED"
        assert timeout_service.cancelled

        concurrent_service = FakeRagService(delay=0.02)
        concurrent_server, _ = create_mcp_server(
            service=concurrent_service,
            settings=MCPSettings(enabled=True, max_concurrency=1),
        )
        await asyncio.gather(
            concurrent_server.call_tool("rag_web_search_answer", {"question": "one"}),
            concurrent_server.call_tool("rag_web_search_answer", {"question": "two"}),
        )
        assert concurrent_service.max_active == 1

        secret = "secret-output-marker"
        output_service = FakeRagService(answer=f"Bearer {secret} " + ("z" * 10_000))
        output_server, _ = create_mcp_server(
            service=output_service,
            settings=MCPSettings(
                enabled=True,
                max_answer_chars=10_000,
                max_output_bytes=1024,
            ),
            redaction_values=(secret,),
        )
        caplog.set_level(logging.INFO, logger="mcp.audit")
        async with Client(output_server) as client:
            bounded = await client.call_tool("rag_ask", {"question": "hello"})
        assert bounded.structured_content is not None
        encoded = json.dumps(bounded.structured_content).encode("utf-8")
        assert len(encoded) <= 1024
        assert secret not in encoded.decode()
        assert "output_truncated" in bounded.structured_content["warnings"]
        assert secret not in caplog.text

    asyncio.run(run())


def test_atomic_tool_initialization_failure_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.frontend.adapters.mcp_server import tools as tools_module

    original = tools_module.Tool.from_function
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("required tool unavailable")
        return original(*args, **kwargs)

    monkeypatch.setattr(tools_module.Tool, "from_function", fail_second)
    with pytest.raises(RuntimeError, match="required tool unavailable"):
        create_mcp_server(service=FakeRagService(), settings=MCPSettings(enabled=True))


def test_authenticated_stateless_http_discovery_invocation_and_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        secret = "0123456789abcdef"
        monkeypatch.setenv("TEST_MCP_HTTP_KEY", secret)
        settings = MCPSettings(
            enabled=True,
            transport="http",
            auth_secret_env="TEST_MCP_HTTP_KEY",
            public_base_url="http://testserver",
            allowed_hosts=("testserver",),
            allowed_origins=("http://trusted.example",),
        )
        runtime = await initialize_runtime(
            settings, service=FakeRagService(source_mode="web_search")
        )
        app = build_http_app(runtime)
        assert transport_security(settings).enable_dns_rebinding_protection
        transport = httpx2.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with (
                httpx2.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                    headers={"Authorization": f"Bearer {secret}"},
                ) as http_client,
                Client(
                    streamable_http_client("http://testserver/mcp", http_client=http_client)
                ) as client,
            ):
                assert (
                    tuple(tool.name for tool in (await client.list_tools()).tools)
                    == CANONICAL_TOOL_NAMES
                )
                result = await client.call_tool("rag_web_search_answer", {"question": "latest"})
                assert result.is_error is False

                health = await http_client.get("/health")
                readiness = await http_client.get("/ready")
                dependencies = await http_client.get("/admin/health/dependencies")
                assert health.json() == {"status": "ok"}
                assert readiness.json() == {"status": "ready"}
                assert "dependencies" not in readiness.json()
                assert dependencies.status_code == 200
                assert dependencies.json()["status"] == "ready"
                assert secret not in dependencies.text

            async with httpx2.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as anonymous:
                assert (await anonymous.get("/health")).json() == {"status": "ok"}
                assert (await anonymous.get("/ready")).json() == {"status": "ready"}
                admin = await anonymous.get("/admin/health/dependencies")
                assert admin.status_code == 401
                assert "dependencies" not in admin.text
                response = await anonymous.post(
                    "/mcp",
                    json={"jsonrpc": "2.0", "id": 1, "method": "server/discover", "params": {}},
                    headers={
                        "Accept": "application/json, text/event-stream",
                        "Content-Type": "application/json",
                    },
                )
                assert response.status_code == 401
                assert response.headers["www-authenticate"].startswith("Bearer ")
        await runtime.shutdown()

    asyncio.run(run())


def test_graceful_shutdown_stops_readiness_and_is_bounded() -> None:
    async def run() -> None:
        service = FakeRagService(delay=1)
        settings = MCPSettings(enabled=True, shutdown_grace_seconds=0.01)
        runtime = await initialize_runtime(settings, service=service)
        call = asyncio.create_task(
            runtime.server.call_tool("rag_web_search_answer", {"question": "hello"})
        )
        await service.started.wait()
        started = time.monotonic()
        drained = await runtime.shutdown()
        elapsed = time.monotonic() - started
        await asyncio.gather(call, return_exceptions=True)

        assert drained is False
        assert elapsed < 0.25
        assert runtime.ready is False
        assert runtime.lifecycle_status == "closed"
        assert runtime.gate.accepting is False
        assert runtime.gate.active_count == 0
        assert runtime.public_readiness() == {"status": "not_ready"}
        assert runtime.public_liveness() == {"status": "ok"}
        assert service.cancelled is True
        assert service.closed is True
        assert call.done()

    asyncio.run(run())


def test_runtime_startup_failure_closes_initialized_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.frontend.adapters.mcp_server import lifecycle as lifecycle_module

    async def run() -> None:
        service = FakeRagService()

        def fail_server(**_kwargs: Any) -> None:
            raise RuntimeError("server publication failed")

        monkeypatch.setattr(lifecycle_module, "create_mcp_server", fail_server)
        with pytest.raises(RuntimeError, match="server publication failed"):
            await initialize_runtime(MCPSettings(enabled=True), service=service)
        assert service.closed is True

    asyncio.run(run())


def test_anonymous_development_http_hides_dependency_diagnostics() -> None:
    async def run() -> None:
        settings = MCPSettings(
            enabled=True,
            transport="http",
            allow_anonymous_http=True,
            allowed_hosts=("testserver",),
        )
        runtime = await initialize_runtime(settings, service=FakeRagService())
        app = build_http_app(runtime)
        transport = httpx2.ASGITransport(app=app)
        async with (
            app.router.lifespan_context(app),
            httpx2.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client,
        ):
            response = await client.get("/admin/health/dependencies")
            assert response.status_code == 404
            assert "dependencies" not in response.text
        await runtime.shutdown()

    asyncio.run(run())


@hypothesis_settings(max_examples=8, deadline=None)
@given(active_calls=st.integers(min_value=0, max_value=6))
def test_bounded_shutdown_property(active_calls: int) -> None:
    """Property 11: bounded graceful shutdown.

    **Validates: Requirements 10.1, 10.2, 10.3**
    """

    async def run() -> None:
        service = FakeRagService(delay=1)
        runtime = await initialize_runtime(
            MCPSettings(
                enabled=True,
                shutdown_grace_seconds=0.01,
                max_concurrency=2,
            ),
            service=service,
        )
        calls = [
            asyncio.create_task(
                runtime.server.call_tool(
                    "rag_web_search_answer",
                    {"question": f"question-{index}"},
                )
            )
            for index in range(active_calls)
        ]
        if calls:
            await service.started.wait()
            await asyncio.sleep(0)

        started = time.monotonic()
        drained = await runtime.shutdown()
        elapsed = time.monotonic() - started
        await asyncio.gather(*calls, return_exceptions=True)

        assert elapsed < 0.25
        assert drained is (active_calls == 0)
        assert runtime.lifecycle_status == "closed"
        assert runtime.gate.active_count == 0
        assert all(call.done() for call in calls)

    asyncio.run(run())
