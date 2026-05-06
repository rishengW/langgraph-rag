\
from __future__ import annotations

import argparse
import os
import pprint

from .config import load_settings
from .graph import build_graph


def parse_args() -> argparse.Namespace:
    default_rebuild = os.getenv("RAG_REBUILD_DB", "false").lower() in ("true", "1", "yes")
    
    parser = argparse.ArgumentParser(description="Run the local LangGraph RAG agent.")
    
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
        graph = build_graph(settings, rebuild_vectorstore=args.rebuild)
        
        result = run_rag_query(
            question=args.question,
            urls=urls,
            settings=settings,
            rebuild_vectorstore=args.rebuild,
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
