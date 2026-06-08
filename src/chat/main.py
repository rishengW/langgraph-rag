"""CLI entry point for the chat app.

Two subcommands:

* ``serve`` — run the FastAPI web app (port 8001 by default).
* ``chat``  — interactive REPL in the terminal. Useful for quick
  testing and for environments without a browser.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

from langchain_core.messages import HumanMessage

from ..config import load_settings, secret_fingerprint
from ..core.web_search import (
    discover_urls_from_web,
    settings_for_discovered_urls,
)


def _print_urls(label: str, urls: Iterable[str]) -> None:
    print(label)
    for index, url in enumerate(urls, start=1):
        print(f"  {index}. {url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the only Subcribers chat agent.")
    sub = parser.add_subparsers(dest="mode", required=True, help="Operation mode")

    # serve
    serve = sub.add_parser("serve", help="Start the chat web server")
    serve.add_argument(
        "--host",
        default=os.getenv("CHAT_API_HOST", "127.0.0.1"),
        help="Host to bind to (default: 127.0.0.1)",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("CHAT_API_PORT", "8001")),
        help="Port to bind to (default: 8001)",
    )
    serve.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload (development mode)",
    )
    serve.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file. Env vars and CLI flags override it.",
    )

    # chat (REPL)
    chat = sub.add_parser("chat", help="Interactive terminal chat")
    chat.add_argument(
        "--urls",
        type=str,
        default="",
        help="Comma-separated source URLs (skips web search)",
    )
    chat.add_argument(
        "--web-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deprecated compatibility flag; chat searches when URLs are not explicit",
    )
    chat.add_argument(
        "--seed-question",
        default="",
        help=(
            "Question used to seed web search when --urls is empty. "
            "If omitted, the configured default URLs are used."
        ),
    )
    chat.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file. Env vars and CLI flags override it.",
    )

    return parser.parse_args()


def _serve(args: argparse.Namespace) -> None:
    from .api import create_app
    import uvicorn

    app = create_app(api_host=args.host, api_port=args.port, config_file=args.config)
    print(f"💬 Starting only Subcribers chat server at http://{args.host}:{args.port}")
    print(f"📖 Open http://{args.host}:{args.port} in a browser to chat")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level="info")


def _repl(args: argparse.Namespace) -> None:
    from .graph import _build_memory_saver, build_chat_graph

    urls = [u.strip() for u in args.urls.split(",") if u.strip()] or None

    settings = (
        load_settings(urls=urls)
        if args.config is None
        else load_settings(urls=urls, config_file=args.config)
    )
    base_settings = settings
    print(f"DashScope API key loaded: {secret_fingerprint(settings.dashscope_api_key)}")

    rebuild = False
    if urls is None and settings.web_search_enabled and args.seed_question.strip():
        print(f"---WEB SEARCH ({settings.web_search_provider})---")
        try:
            found = discover_urls_from_web(args.seed_question.strip(), settings)
        except Exception as exc:
            print(f"Web search failed; using configured source URLs instead: {exc}")
            found = []
        if found:
            settings = settings_for_discovered_urls(base_settings, found)
            rebuild = True
            _print_urls("Discovered source URLs:", found)
        else:
            _print_urls("No web search results; using configured source URLs:", settings.source_urls)
    elif urls:
        _print_urls("Using explicit source URLs:", urls)
    else:
        _print_urls("Using configured source URLs:", settings.source_urls)

    print("\nBuilding chat graph (this may index sources on the first run)...")
    checkpointer = _build_memory_saver()
    graph = build_chat_graph(
        settings,
        rebuild_vectorstore=rebuild,
        checkpointer=checkpointer,
    )

    thread_id = "cli"
    config = {"configurable": {"thread_id": thread_id}}

    print("\nReady. Type your question, or 'exit' / Ctrl-D to quit.\n")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", ":q"}:
            break

        if urls is None and base_settings.web_search_enabled:
            try:
                found = discover_urls_from_web(prompt, base_settings)
            except Exception as exc:
                print(f"Web search failed; keeping current sources: {exc}")
                found = []

            if found and found != settings.source_urls:
                settings = settings_for_discovered_urls(base_settings, found)
                _print_urls("Refreshed source URLs from web search:", found)
                graph = build_chat_graph(
                    settings,
                    rebuild_vectorstore=True,
                    checkpointer=checkpointer,
                )

        try:
            result = graph.invoke({"messages": [HumanMessage(content=prompt)]}, config)
        except Exception as exc:
            print(f"⚠️  Error: {exc}")
            continue

        messages = result.get("messages", []) if isinstance(result, dict) else []
        answer = ""
        for msg in reversed(messages):
            kind = getattr(msg, "type", None) or msg.__class__.__name__.lower()
            content = getattr(msg, "content", "")
            if (kind.startswith("ai") or kind == "assistant") and (content or "").strip():
                answer = content if isinstance(content, str) else str(content)
                break

        print(f"Assistant: {answer or '(no reply produced)'}\n")

    print("Bye.")


def main() -> None:
    args = parse_args()
    if args.mode == "serve":
        _serve(args)
    elif args.mode == "chat":
        _repl(args)
    else:
        print("Unknown mode. Use 'serve' or 'chat'.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
