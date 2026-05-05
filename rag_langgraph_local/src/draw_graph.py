\
from __future__ import annotations

from pathlib import Path

from .config import load_settings
from .graph import build_graph


def main() -> None:
    settings = load_settings()
    graph = build_graph(settings, rebuild_vectorstore=False)

    output_path = Path("graph.png")

    try:
        image_bytes = graph.get_graph(xray=True).draw_mermaid_png()
        output_path.write_bytes(image_bytes)
        print(f"Graph image written to {output_path.resolve()}")
    except Exception as exc:
        print(f"Could not draw graph: {exc}")


if __name__ == "__main__":
    main()
