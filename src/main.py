from __future__ import annotations

import argparse
import os
import pprint
from typing import Iterable

from .config import load_settings, secret_fingerprint
from .graph import build_graph


def _print_urls(label: str, urls: Iterable[str]) -> None:
    print(label)
    for index, url in enumerate(urls, start=1):
        print(f"  {index}. {url}")


def parse_args() -> argparse.Namespace:
    default_rebuild = os.getenv("RAG_REBUILD_DB", "false").lower() in ("true", "1", "yes")
    
    parser = argparse.ArgumentParser(description="Run the local LangGraph RAG agent.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=default_rebuild,
        help="With no subcommand, rebuild the local Chroma vector database and exit.",
    )
    
    # Create subparsers for CLI vs server mode
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # CLI mode (default, for backward compatibility)
    cli_parser = subparsers.add_parser("query", help="Run a query from command line")
    cli_parser.add_argument(
        "question",
        help="Question to ask the RAG agent.",
    )
    cli_parser.add_argument(
        "--urls",
        type=str,
        default="",
        help="Comma-separated URLs to use for RAG (overrides SOURCE_URLS env var).",
    )
    cli_parser.add_argument(
        "--rebuild",
        action="store_true",
        default=default_rebuild,
        help="Delete and rebuild the local Chroma vector database.",
    )
    cli_parser.add_argument(
        "--web-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search the web for source URLs when --urls is omitted.",
    )
    
    # Server mode
    server_parser = subparsers.add_parser("serve", help="Start the web server")
    server_parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("API_HOST", "127.0.0.1"),
        help="Host to bind to (default: 127.0.0.1)",
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", "8000")),
        help="Port to bind to (default: 8000)",
    )
    server_parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload on file changes (development mode)",
    )
    server_parser.add_argument(
        "--rebuild",
        action="store_true",
        default=default_rebuild,
        help="Rebuild vector database on startup",
    )
    
    # For backward compatibility, allow positional argument without subcommand
    args, unknown = parser.parse_known_args()
    
    # If no subcommand and there are positional args, assume it's a query
    if args.mode is None and args.rebuild and not unknown:
        args.mode = "rebuild"

    if args.mode is None and unknown:
        # Reconstruct as query mode
        question = unknown[0]
        urls = ""
        rebuild = default_rebuild
        
        # Look for --urls and --rebuild flags in unknown
        for i, arg in enumerate(unknown):
            if arg == "--urls" and i + 1 < len(unknown):
                urls = unknown[i + 1]
            elif arg == "--rebuild":
                rebuild = True
        
        # Create a namespace to mimic old behavior
        args.mode = "query"
        args.question = question
        args.urls = urls
        args.rebuild = rebuild
    
    return args


def main() -> None:
    args = parse_args()
    
    if args.mode == "serve":
        # Start the FastAPI server
        from .api import create_app
        import uvicorn
        
        app = create_app(
            rebuild_db=args.rebuild,
            api_host=args.host,
            api_port=args.port,
        )
        
        print(f"🚀 Starting RAG LangGraph web server at http://{args.host}:{args.port}")
        print(f"📖 Open your browser and navigate to http://{args.host}:{args.port}")
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    elif args.mode == "rebuild":
        settings = load_settings()
        build_graph(settings, rebuild_vectorstore=True)
        print("Rebuilt Chroma vector database.")
    else:
        # CLI mode (default, for backward compatibility)
        from .graph_executor import run_rag_query
        
        if not hasattr(args, 'question'):
            print("Error: No mode specified. Use 'query' or 'serve'.")
            print("For backward compatibility, you can also just provide a question.")
            return
        
        # Parse URLs from command line if provided
        urls = None
        if args.urls.strip():
            urls = [url.strip() for url in args.urls.split(",") if url.strip()]
        
        settings = load_settings(urls=urls)
        print(f"DashScope API key loaded: {secret_fingerprint(settings.dashscope_api_key)}")
        discovered_from_search = False
        if urls is None and args.web_search and settings.web_search_enabled:
            from .web_search import discover_urls_from_web, settings_for_discovered_urls

            print(f"---WEB SEARCH ({settings.web_search_provider})---")
            try:
                discovered_urls = discover_urls_from_web(args.question, settings)
            except Exception as exc:
                discovered_urls = []
                print(f"Web search failed; using configured source URLs instead: {exc}")

            if discovered_urls:
                urls = discovered_urls
                settings = settings_for_discovered_urls(settings, urls)
                discovered_from_search = True
                _print_urls("Discovered source URLs:", urls)
            else:
                _print_urls("No web search results; using configured source URLs:", settings.source_urls)
        elif urls is not None:
            _print_urls("Using explicit source URLs:", urls)
        else:
            _print_urls("Using configured source URLs:", settings.source_urls)

        graph = build_graph(settings, rebuild_vectorstore=args.rebuild or discovered_from_search)
        
        result = run_rag_query(
            question=args.question,
            urls=urls,
            settings=settings,
            rebuild_vectorstore=args.rebuild or discovered_from_search,
            graph=graph,
            verbose=True,
        )
        
        if result["error"]:
            print(f"\n❌ Error: {result['error']}")
        else:
            print("\n✅ FINAL ANSWER")
            print("=" * 50)
            print(result["answer"])


if __name__ == "__main__":
    main()
