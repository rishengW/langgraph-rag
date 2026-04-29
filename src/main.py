\
from __future__ import annotations

import argparse
import pprint

from .config import load_settings
from .graph import build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local LangGraph RAG agent.")
    parser.add_argument(
        "question",
        nargs="?",
        default="What does Lilian Weng say about the types of agent memory?",
        help="Question to ask the RAG agent.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and rebuild the local Chroma vector database.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()
    graph = build_graph(settings, rebuild_vectorstore=args.rebuild)

    inputs = {
        "messages": [
            ("user", args.question),
        ]
    }

    final_output = None

    for output in graph.stream(inputs):
        final_output = output
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=100, depth=None)
        pprint.pprint("\n---\n")

    # Print the final generated answer in a cleaner way when available.
    if final_output and "generate" in final_output:
        final_messages = final_output["generate"].get("messages", [])
        if final_messages:
            print("\nFINAL ANSWER")
            print("============")
            print(final_messages[-1])


if __name__ == "__main__":
    main()
