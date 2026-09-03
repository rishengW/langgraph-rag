# Architecture Reference

This is the short architecture reference for contributors. The full historical
plan remains in `REFACTORING_PLAN.md`.

## Runtime Shape

```mermaid
flowchart TD
    Client[CLI or Browser] --> API[FastAPI Chat App]
    API --> Config[Settings Loader]
    API --> Graph[LangGraph Workflow]
    Graph --> Nodes[Shared Graph Nodes]
    Nodes --> LLM[DashScope / Qwen Provider]
    Nodes --> Retriever[Retriever Tool]
    Retriever --> Chroma[(Local Chroma Store)]
    Retriever --> Sources[Configured URLs / Explicit URLs / Web Search]
    Graph --> Events[Typed Graph Events]
    Events --> SSE[SSE Stream Endpoints]
    Events --> Metrics[/metrics Snapshot]
```

The chat app builds one graph per thread and uses a SQLite-backed LangGraph
memory saver for restart-safe multi-turn state.

## Main Modules

| Path | Purpose |
|---|---|
| `src/config/` | Pure `Settings` dataclass and `.env`/YAML loading. |
| `src/graph/` | Shared LangGraph builder, state, edges, typed events, executor, and metrics. |
| `src/graph/nodes/` | Shared node factories for agent, retrieval grading, rewrite, generate, and chat condense. |
| `src/rag/` | Chroma retriever lifecycle, document loading, and embedding providers. |
| `src/web_search/` | Baidu and DuckDuckGo discovery providers behind a common protocol. |
| `src/llm/` | LLM provider seam and prompt templates. |
| `src/api/` | FastAPI dependency setup, shared models, typed error handlers, and SSE formatting. |
| `src/chat/` | Chat CLI, API factory, session integration, static UI, and compatibility wrappers. |
| `src/sessions/` | Chat session model, registry, Chroma isolation, SQLite metadata storage, and checkpoint persistence. |
| `src/core/` | Backward-compatible import facades for older code paths. |

## Request Flow

Chat request flow:

1. `POST /chat` creates a thread ID and chooses default, explicit, or
   web-search sources.
2. Explicit and web-search sessions use an isolated Chroma directory under
   `.chroma/chat/<thread_id>/`. Default-source sessions share `.chroma/`.
3. `POST /chat/{thread_id}/message` invokes the per-session graph with the
   thread ID in LangGraph config.
4. `GET /chat/{thread_id}/history` reads the current checkpoint, which is
   persisted through `SQLiteMemorySaver` when the chat app is running normally.

## Persistence

Chroma is the durable vector index. By default it lives in `.chroma/` and is
backed by local SQLite files managed by Chroma. The project also writes
`.chroma/embedding_config.json` so an index is rebuilt when the embedding model
or embedding dimension changes.

Session persistence has two SQLite pieces under the chat Chroma area:

- `src/sessions/sqlite.py` stores thread ID, source URLs, source mode, selected
  config, timestamps, Chroma path, isolation flag, and schema version.
- `src/sessions/checkpoint.py` stores LangGraph `MemorySaver` checkpoint maps
  in SQLite, preserving chat transcript state across process restart without an
  external checkpointer dependency.

The compiled graph itself is still runtime-only. On chat app startup, persisted
session metadata is loaded, graphs are rebuilt against the same thread IDs and
source settings, and the persisted checkpoint saver supplies the prior state.

## Backup And Restore

Stop the chat server before copying Chroma or session SQLite files.
Copying a live Chroma directory can capture SQLite files while Chroma still has
open handles.

PowerShell backup example:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
New-Item -ItemType Directory -Force -Path "backups\$stamp"
Copy-Item -Recurse -Force ".chroma" "backups\$stamp\.chroma"
```

If using chat sessions, copy the session metadata and checkpoint SQLite files
from `.chroma/chat/` into the same backup directory:

```powershell
Copy-Item -Force ".chroma\chat\sessions.sqlite3" "backups\$stamp\sessions.sqlite3"
Copy-Item -Force ".chroma\chat\checkpoints.sqlite3" "backups\$stamp\checkpoints.sqlite3"
```

Restore by stopping the services, moving the current `.chroma/` aside, copying
the backup `.chroma/` into the project root, restoring the session databases to
`.chroma/chat/`, then starting the services again.

If `.chroma/embedding_config.json` does not match the configured embedding
model or dimension, the retriever rebuilds the Chroma store on the next build.
That is expected and safer than querying vectors with mismatched dimensions.

## Schema Migration Note

The current SQLite session metadata store creates both `sessions` and
`schema_version` tables. There is not yet a multi-version migration runner, so
future schema changes should still be treated as manual migrations until a
formal migration path exists:

1. Stop the application.
2. Back up the SQLite database and `.chroma/`.
3. Apply the schema change or recreate metadata from known active sessions.
4. Start the application and verify session creation, lookup, deletion, and
   metadata listing.

This limitation does not affect Chroma's own internal SQLite schema, which is
managed by Chroma.

## Operational Endpoints

| App | Endpoint | Purpose |
|---|---|---|
| Chat | `GET /health` | Liveness and session count. |
| Chat | `POST /chat` | Start a thread. |
| Chat | `POST /chat/{thread_id}/message` | Batch chat turn. |
| Chat | `POST /chat/{thread_id}/message/stream` | SSE graph event stream. |
| Chat | `GET /chat/{thread_id}/history` | Current persisted checkpoint transcript. |
| Chat | `DELETE /chat/{thread_id}` | Delete a session. |
| Chat | `GET /metrics` | In-process graph metrics snapshot. |

## Known Readiness Gaps

The following remain open and are tracked in `COMPANY_READINESS_GAPS.md`:

- Automated multi-version SQLite migrations beyond schema version 1.
- Dependency scanning.
- Structured JSON logging and request/session correlation IDs.
- API versioning under `/api/v1/`.
- Deployment-specific infrastructure.
