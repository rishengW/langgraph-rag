# RAG LangGraph Local Project

This is a multi-file Python version of the original `rag-langgraph.ipynb` notebook.

It builds a LangGraph RAG agent that:

1. Loads Lilian Weng blog posts from the web.
2. Splits the pages into chunks.
3. Stores embeddings in a local Chroma vector database.
4. Uses a retriever tool inside a LangGraph agent.
5. Grades retrieved context for relevance.
6. Generates an answer, or rewrites the query and tries again.

## Important security note

The notebook contained hard-coded API keys. They were intentionally removed from this Python version. Put your own keys in `.env` instead, and rotate any keys that were previously exposed in notebooks, GitHub, or shared files.

## Project structure

```text
rag_langgraph_local/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── config.py
    ├── draw_graph.py
    ├── graph.py
    ├── main.py
    ├── nodes.py
    ├── retriever.py
    └── state.py
```

## Setup

From inside this folder:

```bash
python -m venv .venv
```

Activate the virtual environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

On Windows PowerShell, use:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and set:

```text
DASHSCOPE_API_KEY=your_real_key_here
```

## First run

The first run downloads web pages, downloads the embedding model, and builds the local Chroma database.

```bash
python -m src.main "What does Lilian Weng say about the types of agent memory?" --rebuild
```

## Later runs

Once `.chroma/` exists, you can run without rebuilding:

```bash
python -m src.main "What are common prompt engineering techniques?"
```

## Optional: draw the graph

```bash
python -m src.draw_graph
```

This attempts to write `graph.png`. Graph rendering may require internet access or optional rendering dependencies depending on your LangGraph installation.

## Notes

- This still calls DashScope/Tongyi through an API. The code runs locally, but the LLM is not local unless you replace `ChatTongyi` with a local chat model.
- The default model is `qwen-vl-max` to preserve the notebook behavior. For a cheaper text-only model, set `QWEN_MODEL=qwen-plus` or another DashScope chat model you have access to.
- The default embeddings model is `sentence-transformers/all-mpnet-base-v2`. It is downloaded the first time it is used.
"# langgraph-rag" 
