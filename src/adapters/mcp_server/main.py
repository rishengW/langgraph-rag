"""Separate entry point for the stateless inbound MCP server."""

from __future__ import annotations

import asyncio
import logging
import sys
from uuid import uuid4

from .audit import AuditEvent, emit_audit
from .config import load_mcp_settings
from .lifecycle import initialize_runtime
from .transport import run_selected_transport


def _configure_diagnostics() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


async def _run() -> None:
    settings = load_mcp_settings()
    runtime = await initialize_runtime(settings)
    await run_selected_transport(runtime)


def main() -> int:
    """Run MCP without importing or starting either FastAPI application."""

    _configure_diagnostics()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        return 130
    except Exception:
        emit_audit(
            AuditEvent(
                request_id=uuid4().hex,
                principal_id="server",
                tool="startup",
                outcome="failed",
                duration_ms=0,
                signal="lifecycle",
                transport="internal",
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
