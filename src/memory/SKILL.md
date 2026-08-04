---
name: memory-architect
description: >
  Use this skill when working on the chat agent's long-term memory — the JSON
  store, the save/recall/forget tools, automatic recall injection, memory
  scoping, pruning, or credential screening. Covers the serializer/parser pair
  and its round-trip property, the atomic write path, the process-wide lock,
  and the MEMORY_ERROR failure protocol. Trigger on mentions of long_term_memory
  .json, MemoryStore, MemoryRecord, save_memory, recall_memory, forget_memory,
  build_memory_tools, build_memory_note, build_turn_messages, MemoryCallBudget,
  memory_enabled, MEMORY_STORE_PATH, or memory scope/pruning.
---

# Memory Architect — src/memory/

Domain: durable, cross-session memory for the chat agent, persisted as one JSON
document. Parent: `SKILL.md` (root). Siblings: `src/tools/SKILL.md`,
`src/sessions/SKILL.md`, `src/chat/SKILL.md`, `src/graph/SKILL.md`,
`src/config/SKILL.md`.

Spec: `.kiro/specs/chat-long-term-memory/` (requirements, design, tasks).

## Quick Reference

| Fact | Value |
|---|---|
| Feature flag | `MEMORY_ENABLED` (default `false`) — when off, nothing is read or created |
| Store file | `memory/long_term_memory.json`, override with `MEMORY_STORE_PATH` |
| Git status | The store file and its `.corrupt-*` siblings are git-ignored; the `memory/*.md` docs stay tracked |
| Schema version | `1` (`models.SCHEMA_VERSION`); a higher version on disk is refused, not downgraded |
| Default cap | 500 records total, 1000 chars each, top-5 recall, 2000-char injection budget |
| LLM-facing tools | `save_memory`, `recall_memory`, `forget_memory` (`src/tools/memory_tool.py`) |
| Wiring sites | **TWO** in `src/graph/builder.py`: `_resolve_tools` and `_resolve_lightweight_tools` |
| Failure marker | `MEMORY_ERROR:` — success payloads never start with it |
| Turn assembly | `recall.build_turn_messages`, called by `src/chat/api.py` and `src/chat/main.py` |
| Tests | `tests/test_memory_{primitives,store,mutations,recall_note,tools,integration}.py` |

## File Map

```
src/memory/
├── __init__.py         # public surface
├── models.py           # MemoryRecord, MemoryDocument, limits, ID_PATTERN, RECORD_FIELDS
├── serialization.py    # serialize_document / parse_document  (the pair)
├── relevance.py        # normalize_content, derive_query_terms, rank_records
├── secrets.py          # SECRET_PATTERNS, find_secret_match
├── store.py            # MemoryStore: lock, atomic write, save/recall/forget/purge
└── recall.py           # format_records, build_memory_note, budget, turn assembly
```

## Why a Package, Not a Tool Module

Three layers read the store: the tools (`src/tools/`), the chat turn builders
(`src/chat/api.py`, `src/chat/main.py`), and session deletion (the `DELETE
/chat/{tid}` endpoint). Putting persistence under `src/tools/` would force
`src/chat/` to import from `src/tools/` for a non-tool concern. This package
parallels `src/sessions/`, which already owns its own storage.

## The Store Contract

One JSON document, rewritten whole on every mutation. At 500 short records that
costs nothing and removes every partial-update failure mode.

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
      "content": "Prefers concise answers.",
      "tags": ["style"],
      "created_at": "2026-07-30T09:15:04+00:00",
      "updated_at": "2026-07-30T09:15:04+00:00",
      "last_recalled_at": null
    }
  ]
}
```

`RECORD_FIELDS` in `models.py` fixes the field order both sides of the pair use.
**Changing that tuple changes the on-disk contract** and will break the
round-trip test.

### Round-trip property

`parse_document(serialize_document(doc)).records == doc.records`, field by field
and in order. This is the headline test in `tests/test_memory_store.py`, checked
across 11 document shapes plus a document at the cap. Anything that makes the
parser normalize, truncate, or reorder a field breaks it — which is why
`scope_id` length is *documented* at 200 chars but not enforced by truncation.

### Resilience rules

| On disk | Result |
|---|---|
| File missing | Empty list, no file created, no warning |
| Unparseable JSON, or top level not an object | Renamed to `<name>.corrupt-<stamp>`, continue empty, one warning |
| `version` absent / non-int / outside 1..1 | Empty list, file untouched, one warning |
| Record missing `id`/`content`/`scope`, or `scope` outside the set | That record skipped, the rest kept, one warning |
| Optional field with the wrong JSON type | Documented default substituted, record kept, one warning |
| Unknown keys | Dropped, never written back |

Quarantine names strip colons: a raw ISO-8601 timestamp is not a legal Windows
filename and the rename would fail on the platform this project runs on.

### Atomic write

`mkdir(parents=True)` → `tempfile.mkstemp` in the target directory → write →
`flush` → `os.fsync` → `os.replace`. A reader sees the previous or the new
document, never half of one. `os.replace` is atomic on POSIX and Windows and
overwrites an existing target.

`PermissionError` from the replace is retried 3 further times at 50/100/200 ms
(350 ms total) through an injected `sleeper`. That is the Windows sharing
violation (WinError 32), which clears in milliseconds. A disk-full `OSError` is
**not** retried — it will not clear, so retrying only delays the failure.

Failures name exactly one of three causes: path is a directory, path not
writable, filesystem error. A failed write leaves no `*.tmp` debris and does not
disturb the document on disk. If cleanup also fails, that is logged and the
original write error still propagates — masking a disk-full error with "could
not unlink" sends a debugger down the wrong path.

### Locking and caching

One `threading.RLock` per resolved path in a module-level table, so every
`MemoryStore` for the same file shares it, and `get_memory_store` returns one
instance per path. Cross-process serialization is **out of scope**; the atomic
replace still guarantees no torn reads.

Reads cache on `(st_mtime_ns, st_size)` and re-`stat` every call. Mutations call
`_read_locked(force=True)`, so a read-modify-write always starts from the file.
Without that, a same-size out-of-band rewrite inside the mtime granularity could
let a write build on stale records.

## Scope Model

Two values, no `user_id` dimension yet.

- `global` — visible to every thread, and to invocations with no thread at all.
- `session` — visible only on an exact `thread_id` match. A session record whose
  `scope_id` is missing is therefore visible nowhere.

`thread_id` is **never** a tool parameter. It arrives through the injected
`RunnableConfig` (`config["configurable"]["thread_id"]`), which LangChain keeps
out of `args_schema` and `ToolNode` propagates. A `session`-scoped save with no
resolvable thread is refused rather than silently downgraded to `global`.

Session records are purged by the `DELETE /chat/{tid}` endpoint, best-effort:
a purge failure is logged and does not block the session deletion.

## Pruning

Eviction scans the **whole store**, not the caller's scope. Scoping it would
make the global cap unreachable once one session filled it. Order is ascending
`(effective recency, id)`, where effective recency is `last_recalled_at or
created_at`, so ties are deterministic.

A duplicate update never evicts, even at the cap. A document loaded above the
cap reads fully and is reduced on the next write, never during a read.

## Recall vs Injection

Both rank through `relevance.rank_records`, the single ordering authority, so
explicit recall, keyword forget, and auto-injection cannot drift.

| | `recall_memory` tool | auto-injection (`build_memory_note`) |
|---|---|---|
| Writes | Yes — stamps `last_recalled_at` | **No** |
| Feeds pruning recency | Yes | No |
| Output | Tool string | `SystemMessage` prefixed `LONG-TERM MEMORY (recalled):` |

Determinism holds because `last_recalled_at` appears in neither the rendered
output nor the sort key. **Do not "simplify" `build_memory_note` to call
`store.recall()`** — that would make every turn a write and make recency
meaningless.

Matching is substring-based against normalized content and tags, not
token-equality: CJK text has no whitespace boundaries, so token-equality would
score nearly every Chinese memory at zero.

## Automatic Extraction

Automatic extraction is opt-in through `memory_extraction_enabled` and uses the
same main chat model as the conversation. Two hooks can schedule work:

- `session_start` selects at most one eligible previous session when a new
  session is created.
- `round_complete` runs after a completed checkpointed turn whenever the user
  turn count reaches `extraction_watermark + memory_extraction_turn_interval`.

The authoritative turn count always comes from checkpointed `HumanMessage`
objects. The watermark is a whole number in `SessionMetadata.config` under the
single key `extraction_watermark`; legacy or unusable values behave as 0. A
registry-less CLI runtime uses an in-memory watermark, so its value does not
survive process restart.

| Extraction result | Watermark policy |
|---|---|
| One or more candidates persisted | Advance to observed turn count |
| Empty array or unusable response | Advance |
| Every candidate refused by `MemoryStore.save` | Advance |
| Empty transcript slice | Advance |
| Model raised or timed out | Hold for retry |
| Checkpoint read failed | Hold for retry |
| Store write or watermark write failed | Hold for retry |

`ExtractionScheduler` uses daemon `threading.Thread` workers and a non-blocking
bounded semaphore. It has no queue, rejects duplicate work for a thread, and
does not join workers during shutdown. `ThreadPoolExecutor` is intentionally not
used because its process-exit hook joins non-daemon workers and could delay
shutdown for the full extraction timeout.

Every candidate still goes through `MemoryStore.save`, including credential
screening, duplicate updates, length limits, and capacity eviction. Successful
automatic records carry an `auto:<trigger>` provenance tag. The extraction
prompt fences the transcript as untrusted conversation data and explicitly
states that text inside the fence is data rather than an instruction; it never
includes stored memory records, store paths, or configuration values.

## Failure Protocol

`_guarded` in `src/tools/memory_tool.py` is the only error boundary.

- Catches everything except `KeyboardInterrupt`/`SystemExit`; returns a string.
- Failures start with `MEMORY_ERROR:` and are capped at 500 chars.
- Successes **never** start with the marker, so "no matching memory" and "no
  memory with id X was found" are distinguishable from real failures.
- One log line per call: INFO on success, WARNING on failure, wrapped so a
  broken logger cannot change the outcome.
- Log records carry counts and ids only — never content, tags, or query text.

Credential screening refuses content and tags matching a `SECRET_PATTERN`
(PEM key header, `sk-`, `AKIA`/`ASIA`, `bearer`, `key=value` secrets). It runs
*after* the limit checks and *before* duplicate detection, and returns the
pattern *name* only, so no matched text can be echoed.

Save validation has a fixed order — content presence, content length, category,
scope, tag count, tag length — so a call breaking several rules always reports
the same one.

Per turn, at most `MAX_TOOL_CALLS_PER_TURN` (10) memory tool calls are allowed.
The budget is reset by `build_turn_messages`, which both entry points call on
every turn regardless of whether auto-recall is on.

## Adding a Setting

Follow `src/config/SKILL.md`, plus: memory integer bounds **raise** on
out-of-range input instead of clamping like their neighbours. That is deliberate
(the `rerank_strategy` branch is the in-file precedent) — a nonsensical cap
should fail at startup rather than silently become 1.

## Known Issues

| # | Issue | Severity | Location | Fix |
|---|---|---|---|---|
| 1 | Cross-process writers can supersede one another; only intra-process writes are serialized | Low | `store._lock` | A file lock (`portalocker`) if a multi-process deployment appears |
| 2 | Two graph wiring sites must be kept in sync by hand | Medium | `graph/builder.py` | `build_memory_tools` returns a fixed list, which limits the damage; the tool-registry refactor in `src/tools/SKILL.md` would remove it |
| 3 | Recall is lexical, so paraphrases miss ("likes metric" vs "prefers SI") | Medium | `relevance.py` | Embedding recall behind a new setting |
| 4 | `content` length is bounded twice (static 10k in the schema, `memory_max_record_chars` in the store) | Low | `memory_tool.py` | Unavoidable: a Pydantic constraint cannot read runtime Settings |
| 5 | Whole-document rewrite is O(n) per mutation | Low | `store._persist_locked` | Fine at 500 records; SQLite if the cap ever grows by orders of magnitude |
| 6 | No `user_id` scope, so `global` means "this deployment's single user" | Low | `models.MemoryScope` | Add a third scope dimension when multi-user arrives |
| 7 | A timed-out model invocation is abandoned, not cancelled, because synchronous `model.invoke` is not interruptible | Low | `extraction.MemoryExtractor._call_model` | Use a provider-native cancellable API if one becomes available |
| 8 | Existing checkpoint backlog has no batch drain; session creation considers only one previous session | Low | `chat.memory_hooks.on_session_start` | Add an explicit one-off backfill command if adoption requires it |

## Refactoring To-Do List

- [ ] **Embedding recall** behind `memory_semantic_recall_enabled`, reusing the
      `web_search_semantic_*` model plumbing.
- [ ] **Extraction backfill command** for pre-existing checkpoint history.
- [ ] **`list_memories` tool** for a full inventory dump.
- [ ] **File lock** for multi-process deployments (Known Issue #1).
- [ ] **Migration hook** for `SCHEMA_VERSION` 2 so a bump upgrades rather than
      discards.

## Testing Strategy

| Test | Approach |
|---|---|
| Round-trip property | Enumerated document shapes; `parse(serialize(d)).records == d.records` |
| Parse resilience | Hand-built JSON per rule in the resilience table |
| Atomicity | Monkeypatch `os.replace` to raise; assert attempts, backoff, no debris, unchanged target |
| Concurrency | 8 threads x 20 saves; assert nothing is lost and the cap holds |
| Restart | Rebuild a store from the same path; assert field-for-field equality |
| Determinism | Two identical recalls return byte-identical strings |
| Read-only injection | Assert content **and** mtime unchanged after `build_memory_note` |
| No-raise boundary | Fake stores raising `ZeroDivisionError`, `MemoryError`; a raising logger |
| Wiring | Both resolvers, parametrized; the Chroma retriever is stubbed so no vector store is needed |
| Isolation | Every test points the store at `tmp_path` |

Never let a test write the real `memory/long_term_memory.json`: it would corrupt
the developer's own memories, and the `.gitignore` rule would hide it from
`git status`.

## Dependencies

- `src/config/Settings` — the eight `memory_*` fields; the store path comes from
  here and nowhere else.
- `langchain_core` — `StructuredTool`, `RunnableConfig` injection, `SystemMessage`
  and `HumanMessage` in `recall.build_turn_messages`.
- Standard library only for persistence: `json`, `os`, `tempfile`, `threading`,
  `uuid`, `re`, `datetime`.
