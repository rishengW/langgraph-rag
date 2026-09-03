"""Execute RAG queries and extract results."""

from __future__ import annotations

import pprint
from typing import Any

from .._compat import warn_deprecated_import
from ..graph.builder import build_graph
from .config import Settings

warn_deprecated_import("src.core.graph_executor", "src.graph.executor")


def run_rag_query(
    question: str,
    urls: list[str] | None = None,
    settings: Settings | None = None,
    rebuild_vectorstore: bool = False,
    graph: Any = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute a RAG query and return the final answer.
    
    Args:
        question: The question to ask.
        urls: Optional list of source URLs. If provided, overrides settings.source_urls.
        settings: Settings object. If None, loads from .env.
        rebuild_vectorstore: Whether to rebuild the Chroma vector database.
        graph: Compiled LangGraph. If None, builds from settings.
        verbose: Whether to print intermediate outputs.
    
    Returns:
        Dictionary with keys:
        - "answer": The final generated answer (string)
        - "error": Error message if something went wrong (string or None)
        - "messages": Full message history from the graph (list)
    """
    
    try:
        # Load settings if not provided
        if settings is None:
            from .config import load_settings
            settings = load_settings(urls=urls)
        
        # Override URLs if provided
        if urls is not None:
            # Create a new settings object with overridden URLs
            from dataclasses import replace
            settings = replace(settings, source_urls=urls)
        
        # Build graph if not provided
        if graph is None:
            graph = build_graph(settings=settings, rebuild_vectorstore=rebuild_vectorstore)
        
        # Prepare inputs for the graph
        inputs = {
            "messages": [
                ("user", question),
            ]
        }
        
        # Execute the graph and collect outputs
        final_output = None
        all_messages = []
        
        for output in graph.stream(inputs):
            final_output = output
            
            if verbose:
                for key, value in output.items():
                    pprint.pprint(f"Output from node '{key}':")
                    pprint.pprint("---")
                    pprint.pprint(value, indent=2, width=100, depth=None)
                pprint.pprint("\n---\n")
            
            # Collect all messages from the output
            for _key, value in output.items():
                if isinstance(value, dict) and "messages" in value:
                    all_messages.extend(value["messages"])
        
        # Extract the final answer
        answer = None
        if final_output and "generate" in final_output:
            final_messages = final_output["generate"].get("messages", [])
        elif final_output and "agent" in final_output:
            final_messages = final_output["agent"].get("messages", [])
        else:
            final_messages = []

        if final_messages:
            # Convert message object to string
            last_msg = final_messages[-1]
            answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        
        return {
            "answer": answer or "No answer generated",
            "error": None,
            "messages": all_messages,
        }
    
    except Exception as e:
        return {
            "answer": None,
            "error": str(e),
            "messages": [],
        }
