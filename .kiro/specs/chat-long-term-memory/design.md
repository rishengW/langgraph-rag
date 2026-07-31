# Design Document

## Overview

Long-term memory is added as a new persistence package (`src/memory/`) plus one tool
module (`src/tools/memory_tool.py`) that exposes three `StructuredTool`s to the chat
agent. A single JSON document at `memory/long_term_memory.json` holds every
`MemoryRecord`. The chat turn builder gains a read-only recall injector that prepends a
labelled `SystemMessage`, mirroring the existing upload-context note.

The feature is off by default (`memory_enabled=False`), so no existing behaviour, test,
or deployment changes until the flag is set.

Three design decisions carry most of the weight:

1. **A dedicated `src/memory/` package, not a module inside `src/tools/`.** The store is
   consumed by three different layers: the tools (`src/tools/`), the chat turn builder
   (`src/chat/api.py`, `src/chat/main.py`), and session deletion (`src/chat/api.py`
   delete endpoint). Putting persistence under `src/tools/` would force `src/chat/` to
   import from `src/tools/` for a non-tool concern. `src/memory/` parallels
   `src/sessions/`, which already owns its own SQLite persistence and is imported by the
   chat layer the same way.

2. **`thread_id` arrives through `RunnableConfig`, never through the tool schema.**
   Requirement 10 criterion 7 forbids an LLM-supplied `thread_id`. Verified against the
   installed `langchain_core` 0.3.86: a tool function parameter annotated
   `RunnableConfig` is injected by LangChain and excluded from `tool.args`, and
   `langgraph.prebuilt.ToolNode` propagates the invoke-time config, so
   `config["configurable"]["thread_id"]` reaches the tool body. This also means graphs
   stay session-agnostic and can still be built once per source set, as today.

3. **The store is a serializer/parser pair around a whole-document rewrite.** Records are
   at most `memory_max_records` (default 500) short strings, so read-modify-write of the
   entire document per mutation is cheap and removes every partial-update failure mode.
   Requirement 5's round-trip property becomes a directly testable law over that pair.

## Architecture

```
src/memory/                      NEW package
├── __init__.py                  public surface: MemoryStore, get_memory_store, models
├── models.py                    MemoryRecord, MemoryScope, MemoryCategory, limits
├── serialization.py             serialize_document / parse_document  (the pair)
├── relevance.py                 normalize_content, derive_query_terms, rank_records
├── secrets.py                   SECRET_PATTERNS, find_secret_match
├── store.py                     MemoryStore: lock + atomic write + CRUD + pruning
├── recall.py                    format_records, build_memory_note, MemoryCallBudget
└── SKILL.md                     module skill doc (project convention)

src/tools/memory_tool.py         NEW: three StructuredTool factories
src/tools/__init__.py            + re-exports
src/graph/builder.py             + wiring in _resolve_tools AND _resolve_lightweight_tools
src/config/settings.py           + 8 memory_* fields
src/config/loader.py             + SETTING_ENV_NAMES entries, + _coerce_setting branches
src/llm/prompts.py               + 3 bullets in AGENT_SYSTEM_PROMPT
src/chat/api.py                  + injector call in _graph_inputs_for_turn
                                 + session-scoped purge in delete_chat
src/chat/main.py                 + injector call in the REPL turn loop
.env.example, config/default.yaml + documented defaults
.gitignore                       DONE: memory/long_term_memory.json{,.corrupt-*}
```

### Layering

```
              ┌──────────────────────────────┐
              │  LLM (agent node)            │
              └───────────┬──────────────────┘
                          │ tool_calls
              ┌───────────▼──────────────────┐
              │  ToolNode                    │ config={"configurable":{"thread_id"}}
              └───────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │  src/tools/memory_tool.py          │  save_memory / recall_memory /
        │  (3 StructuredTool factories)      │  forget_memory
        └─────────────────┬──────────────────┘
                          │
   ┌──────────────────────▼───────────────────────┐      ┌───────────────────────┐
   │  src/memory/store.py  MemoryStore            │◄─────┤ src/chat/api.py       │
   │  · process-wide RLock per resolved path      │      │ · recall injector     │
   │  · read()  parse_document                    │      │ · delete_chat purge   │
   │  · mutate() serialize_document + os.replace  │      │ src/chat/main.py REPL │
   └──────────────────────┬───────────────────────┘      └───────────────────────┘
                          │
              ┌───────────▼──────────────────┐
              │  memory/long_term_memory.json│
              └──────────────────────────────┘
```

## Components and Interfaces

### 1. `src/memory/models.py`

```python
MemoryScope    = Literal["global", "session"]
MemoryCategory = Literal["fact", "preference", "entity", "task"]

CATEGORIES: Final[frozenset[str]] = frozenset(("fact", "preference", "entity", "task"))
SCOPES:     Final[frozenset[str]] = frozenset(("global", "session"))

SCHEMA_VERSION: Final[int] = 1
MAX_TAGS: Final[int] = 10
MAX_TAG_CHARS: Final[int] = 40
MAX_QUERY_CHARS: Final[int] = 500
MAX_QUERY_TERMS: Final[int] = 50
MAX_FORGET_DELETES: Final[int] = 10
MAX_SCOPE_ID_CHARS: Final[int] = 200
MAX_TOOL_CALLS_PER_TURN: Final[int] = 10
ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"\A[0-9a-f]{32}\Z")

@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str                      # uuid4().hex, 32 lowercase hex
    scope: MemoryScope
    scope_id: str | None         # None when scope == "global"
    category: MemoryCategory
    content: str
    tags: tuple[str, ...]
    created_at: str              # ISO-8601 UTC, timespec="seconds"
    updated_at: str
    last_recalled_at: str | None

    @property
    def recency(self) -> str:    # effective recency for pruning (R8.2)
        return self.last_recalled_at or self.created_at

@dataclass(frozen=True, slots=True)
class MemoryDocument:
    version: int
    updated_at: str
    records: tuple[MemoryRecord, ...]
```

`MemoryRecord` is frozen so mutations go through `dataclasses.replace`, which keeps the
"leave every other record unchanged" criteria (R4.3, R3.5) structurally easy to satisfy
and easy to assert on in tests.

Timestamps use `datetime.now(UTC).isoformat(timespec="seconds")`. A `clock` callable is
injectable into `MemoryStore` so tests get deterministic timestamps.

### 2. `src/memory/serialization.py` — the pair (R5)

```python
def serialize_document(doc: MemoryDocument) -> str
def parse_document(raw: str, *, fallback_updated_at: str) -> tuple[MemoryDocument, list[str]]
```

`parse_document` returns the document plus a list of warning strings rather than logging
directly, so it stays pure and unit-testable; `MemoryStore` emits the warnings.

Parse rules, in order:

| Condition | Result | Requirement |
|---|---|---|
| `json.loads` raises, or top level is not a `dict` | raise `CorruptDocumentError` (store quarantines) | R5.6 |
| `version` absent / non-int / `> 1` | return empty document, warning naming both versions | R5.7 |
| record not a `dict` | skip, warning with index | R5.5 |
| `id` / `content` / `scope` absent, null, non-str, blank, or `scope` outside `SCOPES` | skip record, warning with index + field | R5.5 |
| `scope_id`, `category`, `tags`, `created_at`, `updated_at`, `last_recalled_at` wrong type | substitute the documented default, keep record, warning | R5.8 |
| unknown keys | dropped, never written back | R5.4 |

Field-level substitution defaults (R5.8): `category` → `"fact"`, `tags` → `()`,
`scope_id` → `None`, `last_recalled_at` → `None`, `created_at`/`updated_at` → the
document's `updated_at` (or `fallback_updated_at` when that is itself unusable).

`serialize_document` writes exactly the three top-level keys and the nine record fields
in the declared order, with `ensure_ascii=False`, `indent=2`, and a trailing newline.
Because both sides use a fixed field order and `MemoryRecord` carries no extra state,
`parse(serialize(doc)) == doc` holds by construction — the property test in
`tests/test_memory_store.py` locks it in.

### 3. `src/memory/relevance.py` (R3.1)

```python
def normalize_content(text: str) -> str:
    return " ".join(text.casefold().split())

def derive_query_terms(query: str) -> tuple[str, ...]:
    """Normalize, split on spaces, keep first 50 distinct terms of len >= 2."""

def relevance_score(record: MemoryRecord, terms: Sequence[str]) -> int:
    """Count distinct terms present in normalized content or any normalized tag."""

def rank_records(
    records: Iterable[MemoryRecord], terms: Sequence[str], *, top_k: int
) -> list[MemoryRecord]:
    """Score > 0, sorted by (-score, updated_at desc, id asc), truncated to top_k."""
```

Pure functions, no `Settings`, no I/O. `rank_records` is the single ordering authority
referenced by R3.1, R4.2, and R7.1, so the three call sites cannot drift.

Sort key: `(-score, _desc(updated_at), record.id)` where descending `updated_at` is
achieved by sorting on the negated ordinal via a two-pass `sorted` (stable sort:
first by `id` ascending, then by `(-score, updated_at reversed)`) — implemented as
`sorted(sorted(candidates, key=lambda r: r.id), key=lambda r: (-score[r.id], r.updated_at), reverse=...)`
is *not* used because `reverse=True` would invert the id tie-break. Instead the key is
`(-score, _neg_iso(r.updated_at), r.id)` with `_neg_iso` mapping an ISO string to a
sortable descending surrogate (its per-character complement is avoided; the
implementation converts to a `datetime` and negates the POSIX timestamp). This keeps one
`sorted` call and an unambiguous total order.

### 4. `src/memory/secrets.py` (R9.1, R9.5)

```python
SECRET_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("pem_private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.I)),
    ("sk_token",        re.compile(r"sk-[A-Za-z0-9_-]{16,}", re.I)),
    ("aws_access_key",  re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", re.I)),
    ("bearer_token",    re.compile(r"\bbearer\s+\S{20,}", re.I)),
    ("assigned_secret", re.compile(
        r"\b\w*(?:password|passwd|secret|api_key|token)\w*\s*[:=]\s*\S{8,}", re.I)),
)

def find_secret_match(text: str) -> str | None:
    """Return the pattern name of the first match, or None. Never returns the text."""
```

Returning the *pattern name* rather than the match keeps R9.1's "contains none of the
matched text" guarantee at the type level.

### 5. `src/memory/store.py` — `MemoryStore`

```python
class MemoryStoreError(RuntimeError): ...
class MemoryPathError(MemoryStoreError): ...      # R9.7 (".." segment)
class MemoryWriteError(MemoryStoreError): ...     # R6.6

class MemoryStore:
    def __init__(
        self,
        path: Path,
        *,
        max_records: int,
        max_record_chars: int,
        clock: Callable[[], str] = _utc_now_iso,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None: ...

    # reads
    def read(self) -> tuple[MemoryRecord, ...]                       # R5.9, R6.9, R6.10
    def in_scope(self, thread_id: str | None) -> tuple[MemoryRecord, ...]  # R10.2, R10.3

    # mutations (each is one locked read-modify-write + one atomic replace)
    def save(self, *, content, category, tags, scope, thread_id) -> SaveOutcome
    def recall(self, *, query, thread_id, top_k, max_chars) -> RecallOutcome
    def forget(self, *, memory_id, query, thread_id) -> ForgetOutcome
    def purge_session(self, thread_id: str) -> int                   # R10.5
```

Outcome objects are small frozen dataclasses (`ok: bool`, `message: str`,
`record_count: int`, plus operation-specific fields). The tool layer turns them into the
LLM-facing string; the store never formats a failure marker itself. This keeps store
tests asserting on structure rather than on prose.

#### Locking (R6.4)

One `threading.RLock` per resolved absolute path, held for the whole read-modify-write.
Stores are cached per path so the three tools, the injector, and the delete endpoint all
share the same lock:

```python
_STORES: dict[Path, MemoryStore] = {}
_STORES_LOCK = threading.Lock()

def get_memory_store(settings: Settings) -> MemoryStore:
    """Resolve the path from Settings only (R9.3) and return the cached store."""
```

`resolve_store_path(settings)` returns `Path(settings.memory_store_path or
"memory/long_term_memory.json")`, raising `MemoryPathError` when any part equals `".."`
(R9.7), and resolving relative values against the process working directory (R9.4,
R11.6). Cross-process serialization is explicitly out of scope (R6.4); the atomic
replace still guarantees no reader sees a partial document.

#### Caching (R6.10)

`read()` keeps `(mtime_ns, st_size, records)`. It re-`stat`s on every call and discards
the cache on any difference. `mutate()` always re-reads under the lock before applying,
so the cache can never serve a superseded set to a writer.

#### Atomic write (R6.1, R6.2, R6.5, R6.7, R6.8)

```
_write_document(doc):
    path.parent.mkdir(parents=True, exist_ok=True)          # R6.5
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        write text, flush, os.fsync(fd), close
        for attempt in range(4):                            # 1 + 3 retries, R6.8
            try:
                os.replace(tmp, path); return               # R6.2 single operation
            except PermissionError:                         # Windows sharing violation
                if attempt == 3: raise
                sleeper(0.05 * 2**attempt)                  # 50 + 100 + 200 = 350 ms
    except OSError as exc:
        raise MemoryWriteError(...) from exc
    finally:
        if tmp still exists: unlink, warn on failure        # R6.7
```

`os.replace` is atomic on both POSIX and Windows and overwrites an existing target, which
is what R6.2 requires. `PermissionError` is the concrete Windows manifestation of a
sharing violation, hence the bounded retry with exponential backoff capped at 350 ms.

#### Save flow (R2, R8, R9.6, R10)

```
validate in fixed order (R2.8):
  content presence → content length → category → scope → tag count → tag length
then find_secret_match(content) then each tag                       (R9.6)
then, under the lock:
  records = re-read
  duplicate = first r where (normalized(content), scope, scope_id, category) match
  if duplicate:
      replace(duplicate, tags=new_tags, updated_at=now)             (R2.4)
      prune nothing                                                 (R8.8)
  else:
      while len(records) >= max_records:                            (R8.2)
          evict min by (recency, id) across the WHOLE store         (R8.1, R8.5)
      append new record
  write
```

Pruning deliberately scans the whole store, not the in-scope subset: scoping the eviction
would make the global cap unreachable once one session filled it. This is the one place
the refined requirements corrected the original draft.

#### Recall flow (R3)

Recall is a *mutation* because it stamps `last_recalled_at` (R3.5, R3.7). Determinism
(R3.4) holds because `last_recalled_at` appears in neither the rendered output (R3.2
renders `id`, `category`, `content`, `updated_at` only) nor the sort key (R3.1 sorts on
`updated_at`). The recency field feeds pruning alone.

Output assembly (R3.2) appends whole record blocks while the running length stays within
`memory_context_max_chars`, so no partial block is ever emitted.

#### Forget flow (R4)

```
id supplied and well-formed → delete that in-scope record, ignore query   (R4.1)
id supplied but malformed    → error naming the parameter                 (R4.8)
id well-formed, no in-scope match (incl. out-of-scope hit) → not-found    (R4.6)
no id, query supplied        → rank_records(top_k=None), delete first 10  (R4.2)
no id, query matches nothing → no-match string, store untouched           (R4.7)
neither supplied (blank counts as absent) → error                         (R4.5)
```

### 6. `src/memory/recall.py`

```python
MEMORY_NOTE_LABEL: Final[str] = "LONG-TERM MEMORY (recalled):"   # 27 chars, unique (R7.2)

def format_records(records, *, max_chars, include_category=True) -> str
def build_memory_note(store, *, message, thread_id, top_k, max_chars) -> str | None

class MemoryCallBudget:
    """Per-thread tool-call budget for one turn (R1.9)."""
    def reset(self, thread_id: str) -> None
    def consume(self, thread_id: str) -> bool
```

`build_memory_note` catches every exception, logs one warning, and returns `None` so the
turn proceeds (R7.6). It calls `store.read()` and `rank_records` directly — never
`store.recall()` — which is what makes injection read-only (R7.7).

The 2000 ms bound in R7.6 is enforced by the store's own bounded work (a single file read
plus an O(n·terms) scan over at most 500 short records); no timer thread is introduced.
The design note records this as a bound satisfied by construction rather than by
enforcement, and the test asserts the scan completes well under the bound at
`memory_max_records`.

`MemoryCallBudget` is a process-level singleton reset once per turn from
`_graph_inputs_for_turn` (FastAPI) and from the REPL loop (CLI). Both call sites run on
every turn regardless of `memory_auto_recall_enabled`, so the budget resets even when
auto-recall is off.

### 7. `src/tools/memory_tool.py`

Follows the `build_*_tool` playbook in `src/tools/SKILL.md`. All three factories live in
one module because they share one `MemoryStore` handle and one feature flag; splitting
them across three modules would triplicate the store wiring for no gain. Precedent:
`file_read_enabled` gates four tools, and `_files.py` holds their shared access logic.

```python
FAILURE_MARKER: Final[str] = "MEMORY_ERROR:"          # R12.1, R12.5

class SaveMemoryInput(BaseModel):
    content:  str        = Field(..., min_length=1, max_length=10_000, description=...)
    category: str | None = Field(default=None, description="fact | preference | entity | task")
    tags:     list[str]  = Field(default_factory=list, max_length=10, description=...)
    scope:    str | None = Field(default=None, description="global | session")

class RecallMemoryInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description=...)

class ForgetMemoryInput(BaseModel):
    memory_id: str | None = Field(default=None, min_length=1, max_length=32, ...)
    query:     str | None = Field(default=None, min_length=1, max_length=500, ...)

def build_save_memory_tool(settings, *, store=None, budget=None) -> BaseTool
def build_recall_memory_tool(settings, *, store=None, budget=None) -> BaseTool
def build_forget_memory_tool(settings, *, store=None, budget=None) -> BaseTool
def build_memory_tools(settings, *, store=None, budget=None) -> list[BaseTool]
```

`store=` / `budget=` are the injectable test seams the playbook calls for, replacing the
`requester=` seam used by HTTP tools.

No schema field names a path, directory, or file (R9.3). No schema field names
`thread_id` (R10.7). `content` carries a static `max_length=10_000` because a Pydantic
field constraint cannot reference runtime `Settings`; the authoritative
`memory_max_record_chars` check lives in the store and returns an error string rather
than a validation exception.

Each tool body:

```python
def _run_save(
    content: str,
    category: str | None = None,
    tags: list[str] | None = None,
    scope: str | None = None,
    *,
    config: RunnableConfig = None,
) -> str:
    thread_id = _thread_id_from(config)             # config["configurable"]["thread_id"]
    return _guard("save_memory", thread_id, lambda: store.save(...))
```

`_guard` is the single error boundary implementing R12:

- catches `BaseException` except `KeyboardInterrupt`/`SystemExit`, returning
  `f"{FAILURE_MARKER} <operation> failed: <cause>"` truncated to 500 chars (R12.1);
- checks the call budget first and returns a marker-prefixed refusal when exhausted
  (R1.9);
- logs one record at INFO on success and WARNING on failure (R12.2, R12.3), wrapped in
  its own `try/except Exception: pass` so a broken logger cannot change the outcome
  (R12.4);
- truncates a successful payload to `memory_context_max_chars` with a trailing
  `… [truncated]` marker (R12.6);
- guarantees successful payloads never start with `FAILURE_MARKER`, so a no-match or
  not-found result is distinguishable from a failure (R12.5).

Tool names and descriptions (R1.1, R1.4):

| Name | Description (abridged) |
|---|---|
| `save_memory` | "Store a durable fact, preference, or task the user states about themselves so later sessions can use it. Use when the user says their name, a preference, an attribute, or an ongoing task." |
| `recall_memory` | "Look up previously stored facts about the user by keyword. Use before answering a question about the user that the current conversation does not already answer." |
| `forget_memory` | "Delete stored memories by id or by keyword. Use when the user asks you to forget something." |

### 8. Graph wiring (R1.1, R1.2, R1.3)

Identical three lines appended to both `_resolve_tools` and `_resolve_lightweight_tools`
in `src/graph/builder.py`, after the `file_read_enabled` block:

```python
if settings.memory_enabled:
    tools.extend(tool_module.build_memory_tools(settings))
```

`build_memory_tools` returns the three tools in a fixed order, which is what makes R1.3's
"same three names and same three schemas on both graphs" hold by construction rather than
by two hand-maintained blocks. This is a small step toward the tool-registry refactor
already listed in `src/tools/SKILL.md`.

### 9. Chat integration

**`src/chat/api.py::_graph_inputs_for_turn`** — the note is prepended *ahead of* the
upload-context note (R7.2):

```python
turn_messages: list[Any] = []
if settings is not None:
    MEMORY_BUDGET.reset(session.thread_id)                      # R1.9
    if settings.memory_enabled and settings.memory_auto_recall_enabled:
        note = build_memory_note(
            get_memory_store(settings),
            message=message,
            thread_id=session.thread_id,
            top_k=settings.memory_recall_top_k,
            max_chars=settings.memory_context_max_chars,
        )
        if note is not None:
            turn_messages.append(SystemMessage(content=note))
    upload_note = _new_upload_context(session, settings)
    if upload_note is not None:
        turn_messages.append(SystemMessage(content=upload_note))
turn_messages.append(HumanMessage(content=message))
```

The injected message is persisted in the checkpoint like any other turn message (R7.8).
`_serialize_messages` already drops every `role == "system"` message from the history
response, so R7.5 needs no code change — only a test asserting the memory note never
reaches the UI.

**`src/chat/main.py`** — the REPL builds `inputs` inline; the same guarded block is added
before the `HumanMessage`, giving the CLI identical selection, placement, and limits
(R7.9). A shared `build_turn_messages(settings, thread_id, message, upload_note=None)`
helper in `src/memory/recall.py` keeps the two call sites from drifting.

**`src/chat/api.py::delete_chat`** — session-scoped purge (R10.5, R10.9) reuses the
existing best-effort pattern already applied to uploads:

```python
try:
    if settings.memory_enabled:
        await asyncio.to_thread(get_memory_store(settings).purge_session, thread_id)
except Exception as exc:
    logger.warning("Failed to purge session memory for thread %s: %s", thread_id, exc)
```

The `except` keeps a purge failure from blocking the registry deletion, exactly as R10.9
requires.

### 10. Configuration

`src/config/settings.py` — eight fields appended near the other tool flags:

```python
memory_enabled: bool = False
memory_store_path: str = ""
memory_max_records: int = 500
memory_max_record_chars: int = 1000
memory_recall_top_k: int = 5
memory_context_max_chars: int = 2000
memory_default_scope: str = "global"
memory_auto_recall_enabled: bool = True
```

`src/config/loader.py`:

- eight `SETTING_ENV_NAMES` entries (`MEMORY_ENABLED`, `MEMORY_STORE_PATH`,
  `MEMORY_MAX_RECORDS`, `MEMORY_MAX_RECORD_CHARS`, `MEMORY_RECALL_TOP_K`,
  `MEMORY_CONTEXT_MAX_CHARS`, `MEMORY_DEFAULT_SCOPE`, `MEMORY_AUTO_RECALL_ENABLED`);
- `memory_enabled` and `memory_auto_recall_enabled` added to the `parse_bool` branch,
  which already implements R11.7's accepted-value set (`true`, `1`, `yes`, `on`);
- a new `_MEMORY_INT_RANGES` table and branch that **raises** `ValueError` naming the
  field, the offending value, and the accepted range (R11.4, R11.8);
- `memory_default_scope` trimmed and case-folded, raising `ValueError` outside
  `{global, session}` (R11.5).

> **Deviation worth flagging.** Every other integer setting in `_coerce_setting` *clamps*
> out-of-range input (`max(1, parsed)`). R11.4 and R11.8 require a hard failure instead.
> The design follows the requirement, using the `rerank_strategy` branch — which already
> raises `ValueError` on an invalid value — as the in-file precedent. A non-integer value
> raises `ValueError` from `int(value)` already; the new branch wraps it to add the field
> name.

R11.9 needs no code: `memory_recall_top_k > memory_max_records` loads unchanged and
recall simply cannot return more records than exist.

`.env.example` and `config/default.yaml` each get the eight documented defaults (R11.3).

### 11. Prompt (R1.4)

Three bullets appended to `TOOLS AVAILABLE WHEN ENABLED:` in `AGENT_SYSTEM_PROMPT`, plus
one line in the "DEFAULT to calling a tool for..." list covering the new
remember/recall category, since memory is a category the existing guidance does not
address.

## Data Models

The in-memory types are defined in `src/memory/models.py` above (`MemoryRecord`,
`MemoryDocument`, `MemoryScope`, `MemoryCategory`). The on-disk representation is the
`version: 1` JSON document below.

```json
{
  "version": 1,
  "updated_at": "2026-07-30T09:15:04+00:00",
  "records": [
    {
      "id": "6f1c2b9a4d7e4f0b8c3a1d5e9f206b74",
      "scope": "global",
      "scope_id": null,
      "category": "preference",
      "content": "Prefers concise answers with code examples.",
      "tags": ["style", "answers"],
      "created_at": "2026-07-30T09:15:04+00:00",
      "updated_at": "2026-07-30T09:15:04+00:00",
      "last_recalled_at": null
    }
  ]
}
```

## Flows

### Save

```
user: "remember I prefer metric units"
  agent → save_memory{content:"Prefers metric units", category:"preference"}
    ToolNode (config.thread_id="a1b2…")
      _guard: budget ok → INFO log on return
        validate order → secret scan → LOCK
          re-read (cache stat-checked)
          duplicate? no → capacity? ok → append
          serialize → tmp file → fsync → os.replace (retry ≤3)
        UNLOCK
      → "Saved memory 6f1c…b74 (preference)."
```

### Auto recall injection

```
POST /chat/{tid}/message
  _graph_inputs_for_turn
    MEMORY_BUDGET.reset(tid)
    memory_enabled && memory_auto_recall_enabled?
      build_memory_note → store.read() → rank_records(top_k=5) → format ≤2000 chars
        any failure → log warning, return None          (turn continues)
    messages = [memory SystemMessage?, upload SystemMessage?, HumanMessage]
  graph.invoke(inputs, {"configurable": {"thread_id": tid}})
  GET /chat/{tid}/history → _serialize_messages drops both system notes
```

### Session delete

```
DELETE /chat/{tid}
  sessions.delete(tid) → 404 if unknown
  best-effort: rmtree upload dir            (existing)
  best-effort: store.purge_session(tid)     (new; global records untouched)
  → {"status": "deleted"}
```

## Error Handling

| Layer | Strategy |
|---|---|
| `parse_document` | Returns warnings; raises only `CorruptDocumentError` for an unparseable document |
| `MemoryStore` | Raises typed `MemoryStoreError` subclasses; quarantines a corrupt file to `<name>.corrupt-<UTC>` and continues empty |
| `_guard` (tool layer) | Sole boundary: converts everything to a `MEMORY_ERROR:`-prefixed string, never raises (R12.1) |
| Recall injector | Swallows everything, logs one warning, returns `None` (R7.6) |
| Delete endpoint | Swallows purge failure, logs one warning, still reports deletion (R10.9) |
| Logging | `logging.getLogger(__name__)`, INFO on success / WARNING on failure; counts and ids only, never content, tags, or query text (R9.2, R12.7) |

## Correctness Properties

These are the invariants the implementation must hold. Each is stated so it can be
asserted directly, and each maps to a test in the section that follows.

### Property 1: Round-trip identity (R5.3)

**Validates: Requirements 5.3**

For every `MemoryDocument` `d` reachable through the public API,
`parse_document(serialize_document(d)).document == d`, comparing `version` and `records`
field by field and in order. This is the headline property; it holds by construction
because serialization uses a fixed field order and `MemoryRecord` carries no state outside
its nine fields.

### Property 2: Write atomicity (R6.2, R6.7)

**Validates: Requirements 6.2, 6.7**

After any `_write_document` call, whether it succeeded or failed, the target file contains
exactly one complete document — either the previous one or the new one — and no `*.tmp`
sibling remains.

### Property 3: Count invariant (R8.6)

**Validates: Requirements 8.6**

After any successful mutation, `len(records) == pre_count + added - deleted` and
`len(records) <= memory_max_records`.

### Property 4: Ordering totality (R3.1)

**Validates: Requirements 3.1**

`rank_records` induces a total order: for any two distinct records the key
`(-score, updated_at desc, id asc)` differs, because ids are unique. Two calls over an
unchanged record set therefore return identical sequences.

### Property 5: Recall determinism (R3.4)

**Validates: Requirements 3.4**

`store.recall(query, thread_id)` called twice with no intervening `save` or `forget`
returns byte-identical strings, even though the first call writes `last_recalled_at`.
Holds because `last_recalled_at` appears in neither the rendered output nor the sort key.

### Property 6: Injection is read-only (R7.7)

**Validates: Requirements 7.7**

`build_memory_note` leaves the store file's content and mtime unchanged, so auto-recall
never influences pruning recency.

### Property 7: No-raise boundary (R12.1)

**Validates: Requirements 12.1**

Every Memory_Tool invocation returns a `str` for every input, including a store that
raises an unexpected exception type and a logger that raises. Only `KeyboardInterrupt` and
`SystemExit` propagate.

### Property 8: Failure discriminability (R12.5)

**Validates: Requirements 12.5**

A returned string starts with `FAILURE_MARKER` if and only if the operation's outcome was
`failure`. No-match and not-found are successes and must not carry the marker.

### Property 9: Scope isolation (R10.3)

**Validates: Requirements 10.3**

A record with `scope == "session"` appears in `in_scope(t)` if and only if
`t == record.scope_id` and `t is not None`.

### Property 10: No agent-controlled I/O surface (R9.3, R10.7)

**Validates: Requirements 9.3, 10.7**

For each of the three tools, `tool.args` contains no key naming a filesystem path and no
key named `thread_id`; the store path is a pure function of `Settings`.

### Property 11: Disabled means inert (R1.2, R1.8)

**Validates: Requirements 1.2, 1.8**

With `memory_enabled=False`, no tool named `save_memory`/`recall_memory`/`forget_memory`
is bound to either graph, and no byte is read from or written to the configured path.

### Property 12: Secret exclusion (R9.1, R9.2)

**Validates: Requirements 9.1, 9.2**

No string returned to the agent and no log record contains text that matched a
Secret_Pattern, nor any Memory_Record content, tag value, or query text.

## Testing Strategy

New files, matching the existing `tests/test_*.py` layout. The property labels below refer
to the Correctness Properties section.

**`tests/test_memory_store.py`**
- **Round-trip property (R5.3)** — the headline test. Generates documents over the field
  domains (empty records, `scope_id=None`, `last_recalled_at=None`, non-ASCII content,
  full `memory_max_records`) and asserts `parse(serialize(doc)) == doc` field by field
  and in order.
- Parse resilience: missing required field skips one record and keeps the rest (R5.5);
  wrong-typed optional field keeps the record with the documented default (R5.8);
  unknown keys dropped and not written back (R5.4); future `version` yields empty and
  leaves the file untouched (R5.7); corrupt JSON quarantines and continues (R5.6);
  missing file yields empty with no file created (R5.9, R6.9).
- Atomicity: monkeypatch `os.replace` to raise `PermissionError` and assert 4 attempts,
  the documented backoff via a fake `sleeper`, no leftover `*.tmp`, and an unchanged
  target (R6.7, R6.8). A read-only directory asserts `MemoryWriteError` (R6.6).
- Concurrency: 8 threads × 20 saves against one store; assert the final count equals 160
  (or `memory_max_records`) and no record was lost (R6.4).
- Restart: build a store, save, drop the instance, rebuild from the same path, assert
  field-for-field equality (R6.3). Cache invalidation via an out-of-band file rewrite
  (R6.10).
- Capacity: eviction picks the oldest effective recency across scopes (R8.2), ties break
  on ascending id (R8.5), duplicate update evicts nothing at the cap (R8.8),
  over-capacity load reads all then reduces on next write (R8.7).

**`tests/test_memory_tools.py`**
- Pydantic rejects blank `content`, an 11-element `tags`, an over-length `query` (R1.5).
- `tool.args` contains no `thread_id`, no path-like field (R9.3, R10.7); `config`-injected
  `thread_id` reaches the store (asserted through a fake store).
- Every failure path returns a `MEMORY_ERROR:`-prefixed string and does not raise,
  including a store that raises an unexpected exception type (R12.1); a broken logger does
  not change the outcome (R12.4).
- Validation order with several simultaneous violations names only the first (R2.8).
- Secret refusal for content and for a tag, with the matched text absent from the message
  (R9.1); refusal ordering after limit checks (R9.6).
- Recall determinism across two identical calls (R3.4); `last_recalled_at` stamped but
  absent from output (R3.5).
- Forget precedence with both id and query (R4.1), the 10-record cap (R4.2), idempotency
  (R4.4), out-of-scope id reported not-found (R4.6).
- Scope: `session` save with no `thread_id` errors (R10.8); session records invisible to
  another thread (R10.3); `purge_session` keeps global records (R10.5).
- Budget: the 11th call in one turn is refused (R1.9).

**`tests/test_memory_integration.py`**
- `_resolve_tools` and `_resolve_lightweight_tools` both expose the same three names and
  schemas when enabled, neither when disabled, and no file is touched when disabled
  (R1.1–R1.3, R1.8).
- `_graph_inputs_for_turn` places the memory note ahead of the upload note (R7.2), omits
  it on zero matches (R7.3) and when auto-recall is off (R7.4), and survives a raising
  store (R7.6). Injection is read-only (R7.7).
- `_serialize_messages` excludes the memory note from history (R7.5).
- `AGENT_SYSTEM_PROMPT` names all three tools (R1.4).

**`tests/test_config.py`** (extended) — env mapping for all eight fields, boolean
accepted-value set (R11.7), out-of-range and non-integer failures naming field, value,
and range (R11.4, R11.8), invalid `memory_default_scope` failure (R11.5), the default
path when `memory_store_path` is blank (R11.6), `.env.example` and `config/default.yaml`
both documenting all eight (R11.3), and `..` in the path refused (R9.7).

Every test uses `tmp_path` for the store, so no test touches
`memory/long_term_memory.json`.

Verification commands, per the project playbook:

```
python -m pytest -p no:warnings -q
python -m compileall src tests
git diff --check
```

## Requirements Traceability

| Req | Design components |
|---|---|
| 1. Tools available | `build_memory_tools`, both `_resolve_*` wiring sites, `AGENT_SYSTEM_PROMPT`, Pydantic schemas, `MemoryCallBudget` |
| 2. Saving | `MemoryStore.save`, `normalize_content`, fixed validation order in `memory_tool` |
| 3. Recalling | `MemoryStore.recall`, `derive_query_terms`, `rank_records`, `format_records` |
| 4. Forgetting | `MemoryStore.forget`, `rank_records`, `ID_PATTERN` |
| 5. JSON format & round-trip | `serialization.py`, `MemoryDocument`, `SCHEMA_VERSION`, quarantine in `store.read` |
| 6. Durable & atomic | `MemoryStore._write_document`, per-path `RLock`, stat-based cache, `os.replace` + bounded retry |
| 7. Auto recall injection | `recall.build_memory_note`, `build_turn_messages`, `_graph_inputs_for_turn`, REPL loop, `_serialize_messages` |
| 8. Capacity & pruning | `MemoryRecord.recency`, eviction in `MemoryStore.save`, over-capacity read path |
| 9. Secrets & privacy | `secrets.py`, `resolve_store_path`, log-field discipline in `_guard` and `store` |
| 10. Scope | `MemoryScope`, `MemoryStore.in_scope`, `purge_session`, `RunnableConfig` thread_id resolution, delete endpoint |
| 11. Configuration | `Settings` fields, `SETTING_ENV_NAMES`, `_coerce_setting` range/scope branches, `.env.example`, `config/default.yaml` |
| 12. Errors & observability | `_guard`, `FAILURE_MARKER`, typed store exceptions, module `logging` at INFO/WARNING |

## Open Design Notes

1. **Loader fail-fast vs clamp.** R11.4/R11.8 require the loader to fail; the rest of
   `_coerce_setting` clamps. The design raises, following the `rerank_strategy`
   precedent. Flag now if you would rather clamp and relax the requirement.
2. **`content` max length is duplicated.** The Pydantic schema uses a static
   `max_length=10_000` (an upper sanity bound) while `memory_max_record_chars` is enforced
   in the store, because a field constraint cannot read runtime `Settings`. The store's
   check is authoritative and returns an error string rather than a validation exception.
3. **R7.6's 2000 ms bound** is met by construction (one file read plus a bounded scan) and
   is asserted by a timing test rather than enforced by a watchdog. Adding a real timeout
   would require a thread per turn for no practical gain.
