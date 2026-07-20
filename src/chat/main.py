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
from collections.abc import Iterable
from dataclasses import replace

from langchain_core.messages import HumanMessage

from ..config import load_settings, secret_fingerprint
from ..core.web_search import (
    discover_urls_from_web,
    settings_for_discovered_urls,
)
from ..graph.events import DoneEvent, ErrorEvent, TokenEvent
from ..graph.executor import GraphExecutor


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
            "Compatibility seed for heavyweight web search. Lightweight chat "
            "searches inside the graph when a message requires it."
        ),
    )
    chat.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file. Env vars and CLI flags override it.",
    )

    return parser.parse_args()


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .api import create_app

    app = create_app(api_host=args.host, api_port=args.port, config_file=args.config)
    print(f"Starting Subscribers chat server at http://{args.host}:{args.port}")
    print(f"Open http://{args.host}:{args.port} in a browser to chat")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level="info")


def _repl(args: argparse.Namespace) -> None:
    from ..graph.builder import build_lightweight_graph
    from .graph import _build_memory_saver, build_chat_graph

    urls = [u.strip() for u in args.urls.split(",") if u.strip()] or None

    settings = (
        load_settings(urls=urls)
        if args.config is None
        else load_settings(urls=urls, config_file=args.config)
    )
    base_settings = settings
    graph_owned_web = (
        urls is None
        and settings.web_search_enabled
        and settings.web_search_lightweight
    )
    print(f"DashScope API key loaded: {secret_fingerprint(settings.dashscope_api_key)}")

    rebuild = False
    if graph_owned_web:
        settings = replace(settings, source_urls=[])
        print("Web search will run inside the chat graph when needed.")
    elif urls is None and settings.web_search_enabled and args.seed_question.strip():
        print(f"---WEB SEARCH ({settings.web_search_provider})---")
        try:
            found = discover_urls_from_web(args.seed_question.strip(), settings)
        except Exception as exc:
            print(f"Web search failed; using configured source URLs instead: {exc}")
            found = []
        if found:
            settings = settings_for_discovered_urls(
                replace(base_settings, web_search_enabled=False), found
            )
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
    if graph_owned_web:
        graph = build_lightweight_graph(
            settings=settings,
            mode="chat",
            checkpointer=checkpointer,
        )
    else:
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

        if (
            not graph_owned_web
            and urls is None
            and base_settings.web_search_enabled
        ):
            try:
                found = discover_urls_from_web(prompt, base_settings)
            except Exception as exc:
                print(f"Web search failed; keeping current sources: {exc}")
                found = []

            if found and found != settings.source_urls:
                settings = settings_for_discovered_urls(
                    replace(base_settings, web_search_enabled=False), found
                )
                _print_urls("Refreshed source URLs from web search:", found)
                graph = build_chat_graph(
                    settings,
                    rebuild_vectorstore=True,
                    checkpointer=checkpointer,
                )

        try:
            executor = GraphExecutor(graph)
            answer = ""
            streamed_any = False
            printed_prefix = False
            inputs: dict[str, object] = {"messages": [HumanMessage(content=prompt)]}
            if graph_owned_web:
                inputs.update(
                    {
                        "current_question": prompt,
                        "source_urls": [],
                        "source_mode": "web_search",
                        "sub_questions": [],
                        "expanded_queries": [],
                        "search_queries": [],
                        "web_search_results": [],
                        "web_search_result_metadata": [],
                        "web_answer_attempts": 0,
                        "web_answer_no_readable_content": False,
                        "expansion_attempted": False,
                    }
                )
            for event in executor.stream(
                inputs,
                config,
                stream_tokens=True,
            ):
                if isinstance(event, TokenEvent):
                    if not printed_prefix:
                        print("Assistant: ", end="", flush=True)
                        printed_prefix = True
                    print(event.token, end="", flush=True)
                    answer += event.token
                    streamed_any = True
                elif isinstance(event, DoneEvent):
                    # Fall back to the final answer if no tokens were streamed
                    # (e.g. the model/provider did not emit token chunks).
                    if not answer and event.answer:
                        answer = event.answer
                elif isinstance(event, ErrorEvent):
                    print(f"\nError: {event.message}")
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        if streamed_any:
            # Terminate the streamed line.
            print("\n")
        else:
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
