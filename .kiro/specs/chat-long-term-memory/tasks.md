# Implementation Plan

## Overview

Tasks are ordered bottom-up so every step is independently runnable: configuration and
pure functions first, then the store, then the tool layer, then integration. Each task
ends in a state where `python -m pytest -p no:warnings -q` passes.

The feature stays inert until `MEMORY_ENABLED=true`, so tasks 1 through 17 cannot change
any existing behaviour. The first user-visible change lands in task 18.

## Task Dependency Graph

```json
{
  "waves": [
    {
      "wave": 1,
      "description": "Configuration and the pure data model. No dependencies.",
      "tasks": ["1", "4"]
    },
    {
      "wave": 2,
      "description": "Pure functions and config documentation. Parallelizable.",
      "tasks": ["2", "3", "5", "6", "7"],
      "dependsOn": ["1", "4"]
    },
    {
      "wave": 3,
      "description": "Round-trip property test and the store read path.",
      "tasks": ["5.1", "8"],
      "dependsOn": ["5", "6", "7"]
    },
    {
      "wave": 4,
      "description": "Atomic write path.",
      "tasks": ["9"],
      "dependsOn": ["8"]
    },
    {
      "wave": 5,
      "description": "Store mutations. Parallelizable once the write path exists.",
      "tasks": ["10", "11", "12"],
      "dependsOn": ["9"]
    },
    {
      "wave": 6,
      "description": "Store test suites, including durability and concurrency.",
      "tasks": ["10.1", "11.1", "12.1", "13"],
      "dependsOn": ["10", "11", "12"]
    },
    {
      "wave": 7,
      "description": "Recall note builder and per-turn call budget.",
      "tasks": ["14", "14.1"],
      "dependsOn": ["10", "11", "12"]
    },
    {
      "wave": 8,
      "description": "The three StructuredTools and their tests.",
      "tasks": ["15", "15.1"],
      "dependsOn": ["14"]
    },
    {
      "wave": 9,
      "description": "Exports and both graph wiring sites.",
      "tasks": ["16"],
      "dependsOn": ["15"]
    },
    {
      "wave": 10,
      "description": "Prompt and chat integration. First user-visible change.",
      "tasks": ["17", "18"],
      "dependsOn": ["16"]
    },
    {
      "wave": 11,
      "description": "CLI REPL integration.",
      "tasks": ["19"],
      "dependsOn": ["18"]
    },
    {
      "wave": 12,
      "description": "Cross-layer integration tests.",
      "tasks": ["20"],
      "dependsOn": ["16", "17", "18", "19"]
    },
    {
      "wave": 13,
      "description": "Module skill doc and full-suite verification.",
      "tasks": ["21"],
      "dependsOn": ["20"]
    }
  ],
  "criticalPath": ["4", "5", "8", "9", "10", "14", "15", "16", "18", "20", "21"]
}
```

```
1 (Settings + loader)
├── 2 (.env.example, default.yaml)
└── 3 (config tests)

4 (models.py)
├── 5 (serialization) ──── 5.1 (round-trip property test)
├── 6 (relevance)
└── 7 (secrets)

5, 6, 7 ──► 8 (store: path + read + cache)
             └── 9 (atomic write)
                  ├── 10 (save) ──── 10.1 (save tests)
                  ├── 11 (recall) ── 11.1 (determinism tests)
                  ├── 12 (forget + purge) ── 12.1 (forget tests)
                  └── 13 (durability + concurrency tests)

10, 11, 12 ──► 14 (recall note + call budget) ── 14.1 (note tests)
                └── 15 (three tools) ── 15.1 (tool tests)
                     └── 16 (exports + both graph wiring sites)
                          ├── 17 (prompt)
                          ├── 18 (chat API injector + session purge)
                          │    └── 19 (CLI REPL injector)
                          └── 20 (integration tests)  [needs 16, 17, 18, 19]

20 ──► 21 (SKILL.md + full verification)
```

Critical path: 4 → 5 → 8 → 9 → 10 → 14 → 15 → 16 → 18 → 20 → 21.

Tasks 2, 3, 6, and 7 are independent of the store and can be done in any order once their
prerequisite (1 or 4) is complete.

## Tasks

- [x] 1. Add the eight memory settings and their environment mapping
- Add `memory_enabled: bool = False`, `memory_store_path: str = ""`, `memory_max_records: int = 500`, `memory_max_record_chars: int = 1000`, `memory_recall_top_k: int = 5`, `memory_context_max_chars: int = 2000`, `memory_default_scope: str = "global"`, and `memory_auto_recall_enabled: bool = True` to `Settings` in `src/config/settings.py`, next to the existing tool flags
- Add the eight `SETTING_ENV_NAMES` entries in `src/config/loader.py` (`MEMORY_ENABLED`, `MEMORY_STORE_PATH`, `MEMORY_MAX_RECORDS`, `MEMORY_MAX_RECORD_CHARS`, `MEMORY_RECALL_TOP_K`, `MEMORY_CONTEXT_MAX_CHARS`, `MEMORY_DEFAULT_SCOPE`, `MEMORY_AUTO_RECALL_ENABLED`)
- Add `memory_enabled` and `memory_auto_recall_enabled` to the existing `parse_bool` branch of `_coerce_setting`, which already implements the accepted-value set
- Add a `_MEMORY_INT_RANGES` table and a `_coerce_setting` branch that raises `ValueError` naming the field, the offending value, and the accepted range for `memory_max_records` (1..10000), `memory_max_record_chars` (1..10000), `memory_recall_top_k` (1..50), and `memory_context_max_chars` (1..20000); wrap the `int(value)` failure to name the field
- Add a `memory_default_scope` branch that trims and case-folds the value and raises `ValueError` outside `{global, session}`
- _Requirements: 11.1, 11.2, 11.4, 11.5, 11.7, 11.8, 11.9_

- [x] 2. Document the memory settings
- Add the eight variables with their default values to `.env.example` next to the other tool flags
- Add the eight keys with their default values to `config/default.yaml`
- _Requirements: 11.3_

- [x] 3. Extend the config tests
- In `tests/test_config.py`, assert env mapping for all eight fields, the boolean accepted-value set, out-of-range and non-integer failures naming field/value/range, an invalid `memory_default_scope` failure, and that `memory_recall_top_k > memory_max_records` loads without failing
- Assert both `.env.example` and `config/default.yaml` mention all eight names
- _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.7, 11.8, 11.9_

- [x] 4. Create the memory package data model
- Create `src/memory/__init__.py` and `src/memory/models.py`
- Define `MemoryScope`, `MemoryCategory`, `CATEGORIES`, `SCOPES`, `SCHEMA_VERSION = 1`, `MAX_TAGS`, `MAX_TAG_CHARS`, `MAX_QUERY_CHARS`, `MAX_QUERY_TERMS`, `MAX_FORGET_DELETES`, `MAX_SCOPE_ID_CHARS`, `MAX_TOOL_CALLS_PER_TURN`, and `ID_PATTERN`
- Define frozen slotted `MemoryRecord` with the nine fields and a `recency` property returning `last_recalled_at or created_at`
- Define frozen slotted `MemoryDocument` with `version`, `updated_at`, `records`
- Add a `_utc_now_iso` helper using `datetime.now(UTC).isoformat(timespec="seconds")`
- _Requirements: 5.2, 8.2_

- [x] 5. Implement the serializer/parser pair
- Create `src/memory/serialization.py` with `serialize_document(doc) -> str` writing exactly three top-level keys and the nine record fields in declared order, `ensure_ascii=False`, `indent=2`, trailing newline
- Implement `parse_document(raw, *, fallback_updated_at) -> tuple[MemoryDocument, list[str]]` returning warnings rather than logging
- Raise `CorruptDocumentError` when `json.loads` fails or the top level is not a dict
- Return an empty document with a warning when `version` is absent, non-integer, or greater than 1
- Skip a record and warn with its zero-based index and field name when `id`, `content`, or `scope` is absent, null, non-string, blank, or `scope` is outside `SCOPES`
- Substitute documented defaults and keep the record when an optional field has the wrong JSON type: `category` to `"fact"`, `tags` to `()`, `scope_id` and `last_recalled_at` to `None`, `created_at`/`updated_at` to the document's `updated_at`
- Drop unknown record keys so they are never written back
- _Requirements: 5.1, 5.2, 5.4, 5.5, 5.6, 5.7, 5.8_

- [x] 5.1 Write the round-trip property test
- Create `tests/test_memory_store.py` asserting `parse_document(serialize_document(d)).records == d.records` field by field and in order
- Cover an empty `records` array, records with `scope_id=None`, records with `last_recalled_at=None`, non-ASCII content, and a document at `memory_max_records`
- Add parse-resilience cases for missing required field, wrong-typed optional field, unknown keys not written back, future `version`, and corrupt JSON
- _Requirements: 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

- [x] 6. Implement relevance scoring
- Create `src/memory/relevance.py` with `normalize_content`, `derive_query_terms` (normalize, split on spaces, first 50 distinct terms of length >= 2), `relevance_score` (count distinct terms present in normalized content or any normalized tag), and `rank_records`
- `rank_records` keeps score >= 1, sorts by `(-score, updated_at descending, id ascending)` in one `sorted` call using a negated POSIX timestamp for the descending component, and truncates to `top_k`
- Keep the module pure: no `Settings`, no I/O
- Add tests for term derivation bounds, tag matching, the total ordering, and stability across repeated calls
- _Requirements: 3.1, 3.3_

- [x] 7. Implement secret screening
- Create `src/memory/secrets.py` with `SECRET_PATTERNS` covering the PEM private-key header, `sk-` plus 16 or more `[A-Za-z0-9_-]`, `AKIA`/`ASIA` plus 16 uppercase alphanumerics, `bearer` plus whitespace plus 20 or more non-whitespace characters, and a key name containing `password`/`passwd`/`secret`/`api_key`/`token` followed by `=` or `:` and 8 or more non-whitespace characters, all case-insensitive
- Implement `find_secret_match(text) -> str | None` returning the pattern name, never the matched text
- Add tests for each pattern plus a negative case, asserting the return value never contains the input secret
- _Requirements: 9.1, 9.5_

- [x] 8. Implement store path resolution and the read path
- Create `src/memory/store.py` with `MemoryStoreError`, `MemoryPathError`, `MemoryWriteError`
- Implement `resolve_store_path(settings)` returning `Path(settings.memory_store_path or "memory/long_term_memory.json")`, raising `MemoryPathError` when any part equals `".."`, and resolving relative values against the working directory
- Implement `MemoryStore.__init__(path, *, max_records, max_record_chars, clock, sleeper)` and a per-path `threading.RLock`
- Implement `read()` returning an empty tuple when the file is missing without creating it, quarantining an unparseable file to `<name>.corrupt-<UTC timestamp>` and continuing empty, emitting parser warnings through `logging`, and caching on `(st_mtime_ns, st_size)` with a re-`stat` on every call
- Implement `in_scope(thread_id)` per the scope rules
- Implement `get_memory_store(settings)` with a module-level per-path cache guarded by its own lock, so tools, the injector, and session delete share one lock
- _Requirements: 5.6, 5.9, 6.9, 6.10, 9.3, 9.4, 9.7, 10.2, 10.3, 11.6, 12.7_

- [x] 9. Implement the atomic write path
- Add `_write_document(doc)` creating missing parent directories, writing to a unique `tempfile.mkstemp` file in the target directory, flushing and `os.fsync`, then `os.replace` onto the target
- Retry `os.replace` at most 3 further times on `PermissionError` with 50, 100, then 200 ms waits through the injected `sleeper`, then raise `MemoryWriteError`
- Always remove the temporary file on failure, logging one warning if that removal also fails, and leave the previous target content untouched
- Add tests: monkeypatch `os.replace` to raise `PermissionError` and assert 4 attempts, the backoff sequence, no leftover `*.tmp`, and an unchanged target; assert `MemoryWriteError` when the parent path is unwritable
- _Requirements: 6.1, 6.2, 6.5, 6.6, 6.7, 6.8_

- [x] 10. Implement save with validation, deduplication, and pruning
- Add `MemoryStore.save(*, content, category, tags, scope, thread_id) -> SaveOutcome`
- Validate in the fixed order content presence, content length after trimming, category, scope, tag count, tag length, returning an outcome naming only the first violation
- Run `find_secret_match` on the trimmed content and each surviving tag after validation and before duplicate detection
- Normalize tags: trim, drop blanks, keep the first of case-folded duplicates, preserve supplied order
- Detect a duplicate on matching normalized content, scope, scope identifier, and category; replace its tag list and `updated_at` only, leaving id, content, `created_at`, `last_recalled_at`, scope, scope identifier, and category unchanged, and prune nothing
- For a new record, generate `uuid4().hex`, set `created_at == updated_at`, `last_recalled_at = None`, and evict across the whole store by ascending `(recency, id)` until the count is `max_records - 1` before appending
- Reject a `session` scope when no `thread_id` is resolved; set `scope_id` from the resolved `thread_id` and never from a parameter
- Log one record naming the deleted count, deleted identifiers, and resulting count whenever eviction happens
- _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.8, 9.1, 9.2, 9.6, 10.1, 10.4, 10.6, 10.8_

- [x] 10.1 Test save behaviour
- Cover the happy path and returned identifier, the default category, tag normalization, duplicate update field preservation, new-record timestamps, blank and over-length content, over-count and over-length tags, invalid category, the fixed validation order with several simultaneous violations, secret refusal for content and for a tag with the matched text absent, and refusal ordering after limit checks
- Cover eviction picking the oldest effective recency across scopes, the ascending-id tie-break, a duplicate update evicting nothing at the cap, and the count invariant
- Cover a `session` save with no `thread_id` returning an error and an invalid scope value
- _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 8.2, 8.5, 8.6, 8.8, 9.1, 9.6, 10.6, 10.8_

- [x] 11. Implement recall
- Add `MemoryStore.recall(*, query, thread_id, top_k, max_chars) -> RecallOutcome`
- Reject a blank query and a query longer than 500 characters without touching any record
- Rank in-scope records with `rank_records`, render id, category, content, and `updated_at` only, and append whole record blocks while staying within `max_chars` so no partial block is emitted
- Return a no-match outcome leaving every field unchanged when zero terms are derived or zero records score
- Stamp `last_recalled_at` on every returned record, leaving all other fields unchanged, persist before returning, keep the count unchanged, and prune nothing
- Return a read-failure outcome leaving the file unchanged and stamping nothing when the read fails
- _Requirements: 3.1, 3.2, 3.3, 3.5, 3.6, 3.7, 3.8_

- [x] 11.1 Test recall determinism
- Assert two identical calls with no intervening write return byte-identical strings even though the first stamped `last_recalled_at`
- Assert `last_recalled_at` appears in neither the output nor the ordering, that `max_chars` truncation drops whole blocks, and that a blank, whitespace-only, or over-length query is rejected without a write
- _Requirements: 3.2, 3.4, 3.5, 3.6_

- [x] 12. Implement forget and session purge
- Add `MemoryStore.forget(*, memory_id, query, thread_id) -> ForgetOutcome` treating absent, empty, and whitespace-only values as not supplied
- A well-formed 32-character hex id deletes that in-scope record and ignores any query supplied in the same call
- A malformed non-empty id, or a query longer than 500 characters, returns an error naming the parameter
- A well-formed id with no in-scope match, including an out-of-scope hit, returns not-found and leaves the store unchanged
- A query with no id deletes the ranked in-scope matches, stopping after 10, and reports the deleted count; zero matches returns no-match and leaves the store unchanged
- Neither parameter supplied returns an error stating one is required
- Every deletion leaves all other records' fields untouched
- Add `purge_session(thread_id) -> int` deleting only `session`-scoped records whose scope identifier matches, retaining every `global` record and every other scope identifier
- _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 10.5_

- [x] 12.1 Test forget and purge
- Cover id-and-query precedence, the 10-record cap with the remainder retained, idempotency across two calls with the same id, an out-of-scope id reported not-found, neither-parameter error, malformed id, over-length query, and non-interference with other records
- Cover `purge_session` retaining global records and other sessions' records
- _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 10.5_

- [x] 13. Test durability and concurrency
- Assert 8 threads times 20 saves against one store lose no record and respect the count invariant
- Assert a store rebuilt from the same path after dropping the instance returns every record field-for-field
- Assert an out-of-band file rewrite invalidates the cache on the next read
- Assert a document loaded above `memory_max_records` reads fully without a write, warns, and is reduced on the next write
- _Requirements: 6.3, 6.4, 6.10, 8.7_

- [x] 14. Implement the recall note builder and call budget
- Create `src/memory/recall.py` with `MEMORY_NOTE_LABEL = "LONG-TERM MEMORY (recalled):"`
- Implement `format_records(records, *, max_chars)` and `build_memory_note(store, *, message, thread_id, top_k, max_chars) -> str | None` that reads through `store.read()` and `rank_records`, never `store.recall()`, scores against the first 500 characters of the message, catches every exception, logs one warning, and returns `None` on failure or zero matches
- Implement `MemoryCallBudget` with `reset(thread_id)` and `consume(thread_id)` enforcing `MAX_TOOL_CALLS_PER_TURN`, plus a module-level `MEMORY_BUDGET` singleton
- Implement `build_turn_messages(settings, thread_id, message, upload_note=None)` returning the ordered turn message list so the API and CLI cannot drift
- Export the public surface from `src/memory/__init__.py`
- _Requirements: 1.9, 7.1, 7.2, 7.3, 7.6, 7.7_

- [x] 14.1 Test the note builder
- Assert the note carries the label, contains each selected record's id and content, respects `max_chars`, and returns `None` on zero matches
- Assert `build_memory_note` leaves the store file's content and mtime unchanged
- Assert a store whose `read` raises yields `None` and one logged warning
- Assert the budget refuses the 11th call in one turn and resets per turn
- _Requirements: 1.9, 7.2, 7.3, 7.6, 7.7_

- [x] 15. Implement the three memory tools
- Create `src/tools/memory_tool.py` with `FAILURE_MARKER = "MEMORY_ERROR:"`
- Define `SaveMemoryInput` (`content` required with `min_length=1` and `max_length=10000`, optional `category`, `tags` list capped at 10, optional `scope`), `RecallMemoryInput` (`query` required, 1..500), and `ForgetMemoryInput` (optional `memory_id` 1..32, optional `query` 1..500), with no path-like and no `thread_id` field
- Implement `_thread_id_from(config)` reading `config["configurable"]["thread_id"]` from an injected `RunnableConfig` parameter so it stays out of the LLM-facing schema
- Implement `_guard(operation, thread_id, call)` as the single error boundary: check the call budget first, catch every exception except `KeyboardInterrupt` and `SystemExit`, return a `FAILURE_MARKER`-prefixed string truncated to 500 characters on failure, truncate a successful payload to `memory_context_max_chars` with a trailing truncation marker, guarantee successful payloads never start with the marker, and log INFO on success and WARNING on failure inside its own suppressed try/except
- Implement `build_save_memory_tool`, `build_recall_memory_tool`, `build_forget_memory_tool` with `store=` and `budget=` test seams, and `build_memory_tools(settings, *, store=None, budget=None)` returning the three tools in fixed order with names `save_memory`, `recall_memory`, `forget_memory` and descriptions of 1..300 characters
- _Requirements: 1.5, 9.2, 9.3, 10.7, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7_

- [x] 15.1 Test the tool layer
- Create `tests/test_memory_tools.py`
- Assert Pydantic rejects blank content, an 11-element tag list, and an over-length query
- Assert `tool.args` contains no `thread_id` and no path-like key for all three tools, and that a config-injected `thread_id` reaches a fake store
- Assert every failure path returns a `FAILURE_MARKER`-prefixed string and never raises, including a store raising an unexpected exception type and a logger that raises
- Assert no-match and not-found results are successes without the marker
- Assert the factories produce tools with the expected names and non-empty descriptions
- _Requirements: 1.5, 10.7, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

- [x] 16. Re-export the tools and wire both graph paths
- Add `SaveMemoryInput`, `RecallMemoryInput`, `ForgetMemoryInput`, and the four builder functions to the imports and `__all__` of `src/tools/__init__.py`, alphabetically
- Append `if settings.memory_enabled: tools.extend(tool_module.build_memory_tools(settings))` after the `file_read_enabled` block in both `_resolve_tools` and `_resolve_lightweight_tools` in `src/graph/builder.py`
- _Requirements: 1.1, 1.2, 1.3_

- [x] 17. Announce the tools to the model
- Add three bullets to the `TOOLS AVAILABLE WHEN ENABLED:` list in `AGENT_SYSTEM_PROMPT` in `src/llm/prompts.py`, one per tool, each stating the condition for calling it
- Add one line to the "DEFAULT to calling a tool for..." guidance covering the remember and recall category
- _Requirements: 1.4, 1.6, 1.7_

- [x] 18. Integrate the injector into the chat API
- In `src/chat/api.py::_graph_inputs_for_turn`, reset `MEMORY_BUDGET` for the session's `thread_id` on every turn, then when `memory_enabled` and `memory_auto_recall_enabled` are both set, prepend the memory `SystemMessage` ahead of the upload-context note and ahead of the `HumanMessage`
- In `delete_chat`, add a best-effort `purge_session` call through `asyncio.to_thread`, wrapped in `try/except Exception` with one logged warning, so a purge failure does not block the registry deletion
- _Requirements: 1.9, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.8, 10.5, 10.9_

- [x] 19. Integrate the injector into the CLI REPL
- In `src/chat/main.py`, build the turn message list through the shared `build_turn_messages` helper so the CLI applies the same selection, placement, and character limit, and reset the call budget each turn
- _Requirements: 7.9_

- [x] 20. Add the integration tests
- Create `tests/test_memory_integration.py`
- Assert both `_resolve_tools` and `_resolve_lightweight_tools` expose the same three names and schemas when enabled and neither when disabled, and that nothing reads or writes the configured path when disabled
- Assert `_graph_inputs_for_turn` places the memory note ahead of the upload note, omits it on zero matches and when auto-recall is off, and completes the turn when the store raises
- Assert `_serialize_messages` excludes the memory note from the history response
- Assert `AGENT_SYSTEM_PROMPT` names all three tools
- Use `tmp_path` for every store so no test touches `memory/long_term_memory.json`
- _Requirements: 1.1, 1.2, 1.3, 1.4, 1.8, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

- [x] 21. Document the module and run the full verification
- Create `src/memory/SKILL.md` following the sibling skill-file convention: quick reference, file map, the store contract, the scope model, the failure-marker protocol, and the testing strategy
- Run `python -m pytest -p no:warnings -q`, `python -m compileall src tests`, and `git diff --check`, fixing anything they surface
- _Requirements: 5.3, 6.4, 12.7_

## Notes

**Verification after every task.** The project playbook expects all three of these to
exit 0:

```
python -m pytest -p no:warnings -q
python -m compileall src tests
git diff --check
```

Line-ending warnings from `git diff --check` on Windows are expected and harmless.

**Two wiring sites, not one.** `src/graph/builder.py` has both `_resolve_tools` and
`_resolve_lightweight_tools`. Task 16 must edit both, or memory will work on the RAG graph
and silently vanish on the lightweight web-search graph. `build_memory_tools` returning a
fixed list is what keeps the two blocks identical.

**Never touch the real store in tests.** Every test uses `tmp_path`. A test that writes
`memory/long_term_memory.json` will pollute the developer's actual memory file, and the
`.gitignore` rule will hide the damage from `git status`.

**Three deviations carried over from the design, already agreed:**

1. Task 1 makes the loader *raise* on out-of-range integers, while every other integer in
   `_coerce_setting` clamps via `max(1, parsed)`. This follows Requirements 11.4 and 11.8,
   using the existing `rerank_strategy` branch as the in-file precedent.
2. Task 15 gives `content` a static `max_length=10000` in the Pydantic schema because a
   field constraint cannot read runtime `Settings`. The authoritative
   `memory_max_record_chars` check lives in the store (task 10) and returns an error string
   rather than raising a validation error.
3. Requirement 7.6's 2000 ms bound is met by construction — one file read plus a bounded
   scan over at most `memory_max_records` short records — and asserted by a timing test in
   task 14.1 rather than enforced by a watchdog thread.

**Recall is a write.** Task 11 persists `last_recalled_at`, so `recall_memory` mutates the
file. Task 14 deliberately does *not* go through `store.recall()`, which is the only reason
auto-injection stays read-only and recall output stays deterministic. Do not "simplify"
task 14 to call `store.recall()`.

**Pruning scans the whole store, not the in-scope subset.** Task 10 evicts across every
scope. Scoping the eviction would make the global cap unreachable once one session filled
it; this was a defect in the first requirements draft and is corrected in Requirement 8.2.
