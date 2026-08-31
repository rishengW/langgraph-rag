"""Separate entry point for the stateless inbound MCP server."""

from __future__ import annotations

import asyncio
import logging
import sys

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
    except Exception as exc:
        logging.getLogger(__name__).error(
            "Inbound MCP startup failed error_type=%s", type(exc).__name__
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
