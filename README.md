# RAG LangGraph Local Project

This is a multi-file Python version of the original `rag-langgraph.ipynb` notebook.

It builds a LangGraph RAG agent that:

1. Loads configured URLs, custom URLs, or web-search results from the web.
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
    ├── api.py
    ├── config.py
    ├── draw_graph.py
    ├── graph.py
    ├── graph_executor.py
    ├── main.py
    ├── nodes.py
    ├── retriever.py
    ├── state.py
    └── static/
        ├── index.html
        └── script.js
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

Optional DashScope network settings:

```text
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSION=1024
EMBEDDING_BATCH_SIZE=10
DASHSCOPE_REQUEST_TIMEOUT=120
DASHSCOPE_MAX_RETRIES=3
DASHSCOPE_HTTP_BASE_URL=
```

`text-embedding-v4` uses the same `DASHSCOPE_API_KEY` as Tongyi/Qwen. Use
`DASHSCOPE_HTTP_BASE_URL` only if your DashScope account or network requires a
non-default endpoint.

## Running the application

### Command-line interface (CLI)

Ask a question directly from the terminal:

```bash
python -m src.main query "What does Lilian Weng say about the types of agent memory?" --rebuild
```

Options:
- `query`: Run a single query
- `--urls`: Comma-separated URLs (overrides web search and env defaults)
- `--rebuild`: Rebuild vector database from scratch

Example with custom URLs:

```bash
python -m src.main query "Your question here" --urls "https://example.com,https://another.com" --rebuild
```

### Web interface

Start the FastAPI web server:

```bash
python -m src.main serve
```

This starts the server at `http://127.0.0.1:8000` by default.

**Server options:**
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to bind to (default: 8000)
- `--rebuild`: Rebuild vector database on startup
- `--reload`: Enable auto-reload on file changes (development mode)

**Examples:**

Start server on localhost:

```bash
python -m src.main serve
```

Start on all interfaces on port 5000:

```bash
python -m src.main serve --host 0.0.0.0 --port 5000
```

Development mode with auto-reload:

```bash
python -m src.main serve --reload
```

Then open your browser to the printed URL and:

1. Enter your question in the text area
2. Leave Source URLs empty to search the web automatically, or provide comma-separated URLs for custom sources
3. (Optional) Disable web search to use the configured `SOURCE_URLS` defaults when Source URLs is empty
4. (Optional) Check "Rebuild vector database" to force rebuild
5. Click "Ask Question"
6. View the answer, sources used, and any errors

**API Endpoints:**

- `GET /` - Web interface
- `GET /health` - Health check
- `POST /query` - Submit a query (JSON):

```json
{
  "question": "Your question here",
  "urls": "https://example.com,https://another.com",
  "web_search": true,
  "rebuild": false
}
```

### Environment variables for API

You can configure the API server via environment variables:

```text
API_HOST=0.0.0.0
API_PORT=8000
WEB_SEARCH_ENABLED=true
WEB_SEARCH_PROVIDER=duckduckgo
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_REGION=wt-wt
WEB_SEARCH_TIMELIMIT=
WEB_SEARCH_VERIFY_SSL=true
```

When `urls` is empty and `web_search` is true, the API searches the web using the question, builds an isolated temporary Chroma index for the discovered URLs, and answers from those pages. If `web_search` is false, the app uses `SOURCE_URLS` from `.env` or the built-in default URLs.

Set `WEB_SEARCH_PROVIDER=baidu` to use Baidu instead of DuckDuckGo for automatic source discovery. This can be useful when DashScope/Qwen works best without a proxy but DuckDuckGo search does not.

## First run

The first run downloads web pages, calls DashScope `text-embedding-v4`, and builds the local Chroma database.

**CLI:**

```bash
python -m src.main query "What does Lilian Weng say about the types of agent memory?" --rebuild
```

**Or via web interface:**

```bash
python -m src.main serve --rebuild
```

Then open http://127.0.0.1:8000 and check the "Rebuild vector database" checkbox when asking your first question.

## Later runs

Once `.chroma/` exists, you can run without rebuilding:

**CLI:**

```bash
python -m src.main query "What are common prompt engineering techniques?"
```

**Web interface:**

```bash
python -m src.main serve
```

Then open http://127.0.0.1:8000 and ask questions directly through the web UI.

## Optional: draw the graph

```bash
python -m src.draw_graph
```

This attempts to write `graph.png`. Graph rendering may require internet access or optional rendering dependencies depending on your LangGraph installation.

## Notes

- This still calls DashScope/Tongyi through an API. The code runs locally, but the LLM and default embeddings are not local unless you replace `ChatTongyi` and `text-embedding-v4` with local models.
- The default model is `qwen-plus`, a text chat model suitable for RAG. Use a `qwen-vl-*` model only if your workflow needs multimodal input.
- The default embeddings model is `text-embedding-v4`. Existing Chroma stores without matching embedding metadata are rebuilt automatically so vector dimensions stay consistent.
