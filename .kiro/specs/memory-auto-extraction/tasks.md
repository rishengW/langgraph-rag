# Implementation Plan: Automatic Memory Extraction

## Overview

Bottom-up, in Python, matching the repository's existing layout: configuration and pure
functions first, then the Extractor, then the scheduler, then the session-layer bookkeeping,
then the trigger hooks, then the wiring. Every task ends with
`python -m pytest -p no:warnings -q` passing.

Nothing before task 9.1 can change runtime behaviour: `build_extraction_runtime` returns
`None` while `memory_extraction_enabled` is `False`, and until the API and CLI call the hooks
nothing invokes it. The first behavioural change lands in 9.1.

No task performs a real LLM call. `MemoryExtractor` takes a `model_factory` seam
(default `src.llm.provider.build_chat_model`) and every test injects a fake.

Already landed and verified in the working tree: the eight settings and their loader
plumbing, `.env.example` / `config/default.yaml` documentation, the extended
`tests/test_config.py`, `src/memory/transcript.py` and `src/memory/watermark.py` with their
test files, `MEMORY_EXTRACTION_PROMPT`, `EXTRACTION_TAG_PREFIX`, and `MEMORY_NOTE_MARKER`.
Those tasks are checked off below.

## Tasks

- [x] 1. Configuration and documentation
  - [x] 1.1 Add the eight extraction settings and their environment mapping
    - `Settings` in `src/config/settings.py` carries `memory_extraction_enabled` (False),
      `memory_extraction_on_session_start` (True), `memory_extraction_turn_interval` (10),
      `memory_extraction_max_candidates` (5), `memory_extraction_max_transcript_chars`
      (8000), `memory_extraction_timeout_seconds` (60), `memory_extraction_max_concurrency`
      (2), `memory_extraction_max_session_age_hours` (168)
    - Eight `SETTING_ENV_NAMES` entries in `src/config/loader.py`, the two flags appended to
      the existing `parse_bool` branch, the six integers added to `_MEMORY_INT_RANGES` so the
      existing raise-on-out-of-range branch covers them with no new code path
    - No extraction-specific model, provider, or sampling field
    - _Requirements: 8.1, 8.2, 8.4, 8.5, 8.8, 8.9, 8.10_

  - [x] 1.2 Document the extraction settings
    - Eight variables in `.env.example` beside the existing `MEMORY_*` block, each with its
      default and, for the six integers, its accepted range
    - The same eight keys with the same defaults in `config/default.yaml`
    - _Requirements: 8.3_

  - [x]* 1.3 Extend the configuration tests
    - `tests/test_config.py` covers defaults, env mapping, YAML loading, boolean accepted
      values, out-of-range failures naming field/value/range, non-integer failures, and that
      `.env.example` and `config/default.yaml` mention every name
    - Covers `memory_extraction_max_transcript_chars` and `memory_max_record_chars` loading
      in any in-range combination with no cross-field constraint
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.8, 8.9_

- [x] 2. Pure primitives, prompt, and provenance markers
  - [x] 2.1 Implement pure transcript handling
    - `src/memory/transcript.py` with `USER_LABEL`, `ASSISTANT_LABEL`, `TRUNCATION_MARKER`,
      `MAX_SLICE_MESSAGES = 200`, `message_role`, `normalize_message_content`, `count_turns`,
      `is_memory_note`, `select_slice`, `render_slice`
    - Role classification mirrors `src/chat/api.py::_serialize_messages`; the module stays
      pure with no `Settings` and no I/O
    - _Requirements: 2.6, 3.1, 3.2, 3.3, 3.4, 3.8, 3.9, 3.10, 9.7_

  - [x]* 2.2 Test transcript handling
    - `tests/test_memory_transcript.py` covers role classification, content flattening,
      turn counting, note detection, slice selection at watermark 0 / mid / equal / beyond,
      and rendering
    - **Property 1: Turn counting cannot drift**
    - **Property 10: Slice rendering is deterministic and bounded**
    - **Validates: Requirements 2.2, 2.6, 3.3, 3.4, 3.8, 3.9, 3.10**

  - [x] 2.3 Implement the watermark seam
    - `src/memory/watermark.py` with `WATERMARK_KEY = "extraction_watermark"`,
      `MAX_WATERMARK = 1_000_000`, the `WatermarkStore` Protocol, `InMemoryWatermarkStore`,
      and `coerce_watermark` rejecting `bool` and never raising
    - _Requirements: 7.2, 7.6, 7.8_

  - [x]* 2.4 Test the watermark seam
    - `tests/test_memory_watermark.py` table-tests `coerce_watermark` over `True`, `False`,
      `"3"`, `3.0`, `-1`, `10**9`, `None`, a dict, a hostile object, and valid values, and
      round-trips `InMemoryWatermarkStore`
    - **Property 9 (coercion half): A corrupt watermark degrades to 0**
    - **Validates: Requirements 7.6, 7.8**

  - [x] 2.5 Add the extraction prompt
    - `MEMORY_EXTRACTION_PROMPT` in `src/llm/prompts.py` as a `str.format` template with one
      `{transcript}` slot and one `{max_candidates}` slot
    - Durable user-specific information only; `[]` when nothing qualifies; a JSON array of
      objects with `content` and optional `category`, `tags`, `scope`
    - Transcript fenced in a delimiter labelling it untrusted conversation data, with text
      inside the delimiter stated to be data and never an instruction
    - No `Memory_Record` content, no store path, no `Settings` value
    - _Requirements: 4.1, 4.2, 4.3, 9.4, 9.6_

  - [x] 2.6 Add the provenance tag and the recall-note marker
    - `EXTRACTION_TAG_PREFIX = "auto:"` in `src/memory/models.py`
    - `MEMORY_NOTE_MARKER` in `src/memory/recall.py` as a zero-width string, prepended to the
      note content built by `build_memory_note`, with the note's character budget adjusted for
      the marker overhead
    - _Requirements: 9.7, 9.8_

  - [x]* 2.7 Assert the recall-note marker in the existing note tests
    - Extend `tests/test_memory_recall_note.py`: the note built by `build_memory_note` carries
      `MEMORY_NOTE_MARKER`, the visible note text is otherwise unchanged, the note still
      starts with `MEMORY_NOTE_LABEL`, and `transcript.is_memory_note` returns `True` for a
      note taken from `build_turn_messages`
    - _Requirements: 9.7_

- [x] 3. Implement the Extractor
  - [x] 3.1 Create the Extractor skeleton, types, and checkpoint reads
    - Create `src/memory/extraction.py` with frozen
      `ExtractionCandidate(content, category=None, tags=(), scope="global")` and frozen
      `ExtractionOutcome(trigger, thread_id, slice_messages, candidates, persisted, refused,
      record_count, watermark_before, watermark_after, duration_ms, status, detail="")`
    - Implement `MemoryExtractor.__init__(settings, *, checkpointer, watermarks, store=None,
      model_factory=build_chat_model, clock=time.monotonic)`, resolving the store through the
      existing `src.memory.store.get_memory_store` when none is passed
    - Implement `read_thread_messages(thread_id)` over
      `checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})` →
      `checkpoint["channel_values"]["messages"]`, returning an empty tuple when there is no
      checkpoint and never raising
    - Implement `turn_count(thread_id)` from `transcript.count_turns` and
      `should_extract(thread_id)` comparing it against
      `min(watermark, turn_count) + memory_extraction_turn_interval`
    - Add the module logger `logging.getLogger("src.memory.extraction")`
    - _Requirements: 2.1, 2.2, 2.4, 3.6, 3.7, 7.9_

  - [x] 3.2 Implement candidate parsing
    - Add `_parse_candidates(text)`: `json.loads` first, then the first `[...]` span located by
      a bracket scan, then zero candidates
    - Evaluate at most the first 100 entries; discard non-object entries and entries whose
      `content` is absent, non-string, or blank after trimming
    - Drop later duplicates whose content is equal after trimming, whitespace collapsing, and
      case folding, then truncate to `memory_extraction_max_candidates`
    - Leave an unusable `category` unset, keep only non-blank string `tags`, and coerce an
      absent or unknown `scope` to `global`
    - _Requirements: 4.4, 4.5, 4.7, 4.8, 4.9, 4.10_

  - [x]* 3.3 Write property test for round boundaries
    - Create `tests/test_memory_extraction.py` with a fake checkpointer returning a canned
      message list and an `InMemoryWatermarkStore`
    - **Property 2: Round boundaries are exact**
    - **Validates: Requirements 2.5**

  - [x]* 3.4 Write unit tests for candidate parsing
    - Extend `tests/test_memory_extraction.py`: empty array, unusable text, valid JSON that is
      not an array, a JSON array embedded in prose, non-object entries, blank and non-string
      `content`, duplicates differing only by case and internal whitespace, more than 100
      entries, more than `memory_extraction_max_candidates` survivors, unusable `category`,
      non-string `tags`, and unknown `scope`
    - _Requirements: 4.4, 4.5, 4.7, 4.8, 4.9, 4.10_

  - [x] 3.5 Implement the bounded LLM call
    - Add `_call_model(prompt)` running `model.invoke` in a nested daemon thread and waiting
      at most `memory_extraction_timeout_seconds`
    - On timeout return a no-response result, discard any late value, and leave the orphan
      thread to finish unobserved
    - Exactly one call per extraction, no retry, model built through the injected
      `model_factory` so no test reaches a provider
    - _Requirements: 4.1, 4.6_

  - [x] 3.6 Implement candidate persistence with the provenance tag
    - Add `_persist(candidates, thread_id, trigger)` calling the existing
      `MemoryStore.save(content=..., category=..., tags=..., scope=..., thread_id=...)` once
      per candidate, sequentially, in candidate order
    - Append `EXTRACTION_TAG_PREFIX + trigger` to each candidate's tags within the existing
      `MAX_TAGS` / `MAX_TAG_CHARS` limits, and pass the extracted thread's Thread_Id so a
      `session`-scoped candidate binds that thread and never the triggering one
    - Count persisted versus refused from `SaveOutcome.ok`, continue past a refusal, and never
      retry a refused candidate inside the same extraction
    - Use no memory-tool call budget and no second write path to the store file
    - _Requirements: 5.1, 5.2, 5.4, 5.5, 5.6, 5.7, 5.9, 9.2, 9.3, 9.8_

  - [x] 3.7 Implement `run` with the watermark policy and one log record
    - Implement `run(trigger, thread_id) -> ExtractionOutcome` orchestrating read, turn count,
      watermark clamp to the turn count, slice selection, render, model call, parse, persist,
      watermark update, and one log record
    - Advance the watermark when candidates were persisted, when the response was an empty
      array, when the response was unusable, when every candidate was refused, and when the
      slice was empty; never let it decrease
    - Hold the watermark when the model raised or timed out, when the checkpoint read failed,
      and when a store save failed for a reason other than refusal
    - Never raise: catch everything except `KeyboardInterrupt` and `SystemExit` and return
      `status="failed"`
    - Emit exactly one informational record per completed extraction and exactly one warning
      per failure, every message prefixed `memory_extraction`, carrying trigger, thread id,
      counts, both watermark values, and `duration_ms`, and no transcript, candidate, record,
      or response text
    - _Requirements: 4.6, 4.11, 5.3, 5.8, 6.2, 6.7, 7.4, 7.7, 9.1, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

  - [x]* 3.8 Write property test for the watermark policy
    - Create `tests/test_memory_extraction_properties.py` with one parametrized case per row
      of the design's advance-versus-hold table
    - **Property 3: Only transient failures hold the watermark**
    - **Validates: Requirements 4.6, 4.11, 5.8, 6.7**

  - [x]* 3.9 Write property test for the no-raise boundary
    - Extend `tests/test_memory_extraction_properties.py` with a checkpointer that raises, a
      model that raises, a store that raises, and a watermark store that raises
    - **Property 5: Extraction never raises into a turn**
    - **Validates: Requirements 6.2**

  - [x]* 3.10 Write property test for watermark clamping
    - Extend `tests/test_memory_extraction_properties.py`: a stored watermark above the current
      turn count is clamped, no LLM call is made solely because of it, and the reduced value is
      persisted on the next successful extraction
    - **Property 9 (clamping half): A corrupt watermark degrades to 0**
    - **Validates: Requirements 7.8, 7.9**

  - [x]* 3.11 Write property test for store safeguards
    - Extend `tests/test_memory_extraction_properties.py` with a real `MemoryStore` on
      `tmp_path`: a credential-bearing candidate is refused, a duplicate updates rather than
      appends, an over-length candidate is refused, capacity eviction is the store's decision,
      and the outcome counts match
    - **Property 11: Extraction cannot bypass store safeguards**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.5, 9.2**

  - [x]* 3.12 Write property test for scope binding
    - Extend `tests/test_memory_extraction_properties.py`: a `session`-scoped candidate from a
      `session_start` extraction carries the extracted thread's id
    - **Property 12: Scope binds to the extracted thread**
    - **Validates: Requirements 5.1, 5.9**

  - [x]* 3.13 Write property test for log discipline
    - Extend `tests/test_memory_extraction_properties.py` using `caplog`: every record starts
      with `memory_extraction`, exactly one informational record per completed extraction,
      exactly one warning per failure, and no 20-character substring of the transcript,
      a candidate, a record, or the response body appears in any record
    - **Property 14: Logs carry no content**
    - **Validates: Requirements 9.1, 10.1, 10.6**

- [x] 4. Checkpoint - Extractor verified
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement the bounded scheduler
  - [x] 5.1 Implement `ExtractionScheduler`
    - Create `src/memory/scheduler.py` with
      `ExtractionScheduler(*, max_concurrency, runner, thread_factory=threading.Thread)`
    - Guard an in-flight `set[str]` of thread ids with a lock and bound concurrency with a
      `threading.BoundedSemaphore`
    - In `submit(trigger, thread_id)` check the in-flight set **before** acquiring the
      semaphore, so a duplicate request for a busy thread is discarded at debug level without
      consuming a slot; then attempt a non-blocking acquire and discard with a warning naming
      the concurrency limit when no slot is free; return `True` only when a worker started
    - Start every worker as a daemon thread, join none of them, and do not use
      `concurrent.futures.ThreadPoolExecutor`
    - Catch every exception inside the worker body; release the semaphore and clear the
      in-flight entry in a `finally`
    - Add `shutdown()` making further `submit` calls no-ops without joining, and
      `wait_idle(timeout)` for tests only
    - _Requirements: 6.1, 6.3, 6.4, 6.5, 6.9, 6.10_

  - [x]* 5.2 Write property test for bounded concurrency
    - Create `tests/test_memory_scheduler.py` with a runner blocked on an event
    - **Property 6: Bounded concurrency with no queue**
    - **Validates: Requirements 6.3, 6.4, 6.9**

  - [x]* 5.3 Write property test for shutdown
    - Extend `tests/test_memory_scheduler.py`: every started worker has `daemon is True`,
      nothing joins them, and `shutdown()` makes subsequent submits no-ops
    - **Property 7: Shutdown is not delayed**
    - **Validates: Requirements 6.5**

  - [x]* 5.4 Write property test for non-blocking submission
    - Extend `tests/test_memory_scheduler.py`: with a runner held on an unset event the caller
      returns immediately and a runner that raises neither propagates nor leaks its slot,
      asserted without measuring elapsed time
    - **Property 4 (scheduler half): The reply path is never blocked**
    - **Validates: Requirements 6.1, 6.8**

- [x] 6. Persist the watermark through the session layer
  - [x] 6.1 Carry the watermark through `ChatSession` and session metadata
    - Add `extraction_watermark: int = 0` to `ChatSession` in `src/sessions/models.py`, leaving
      every existing field name, type, and default unchanged
    - Add `"extraction_watermark": int(session.extraction_watermark or 0)` to the `config`
      mapping built by `SessionMetadata.from_session` in `src/sessions/storage.py`, which
      rebuilds `config` wholesale on every save
    - Carry the persisted value back onto the restored session in `ChatSessionRegistry.restore`
      in `src/sessions/registry.py` by reading `metadata.config` through `coerce_watermark`
    - Leave `SQLiteStorage.SCHEMA_VERSION` at 1 and add no column
    - _Requirements: 7.1, 7.2, 7.3, 11.6, 11.7_

  - [x]* 6.2 Write property test for watermark survival
    - Extend `tests/test_sessions.py`: a watermark round-trips through
      `SessionMetadata.from_session` and `SQLiteStorage`, and `registry.get()` touching and
      re-saving a session leaves the persisted value equal to what it was before that save
    - **Property 8: The watermark survives unrelated saves**
    - **Validates: Requirements 7.3, 11.7**

  - [x]* 6.3 Write unit tests for legacy metadata and schema stability
    - Extend `tests/test_sessions.py`: metadata written without the key loads as 0 and keeps
      its other values after the next save, a `ChatSession` built without the field still
      constructs, and `SQLiteStorage.SCHEMA_VERSION` is still 1
    - _Requirements: 7.6, 11.6, 11.7, 11.8_

- [x] 7. Implement the trigger hooks
  - [x] 7.1 Implement the hooks module foundation
    - Create `src/chat/memory_hooks.py` with `SessionWatermarkStore(registry, storage)`
      implementing `WatermarkStore`, reading the persisted metadata value first and falling
      back to the in-memory registry, and writing through both the session object and the
      storage backend
    - Add an `ExtractionRuntime` container holding settings, extractor, scheduler, registry,
      and storage
    - Implement `build_extraction_runtime(settings, *, checkpointer, registry=None,
      storage=None)` returning `None` when `memory_enabled` or `memory_extraction_enabled` is
      disabled or the checkpointer is absent, and falling back to `InMemoryWatermarkStore`
      when no registry is supplied, so the registry-less CLI can still build a runtime
    - _Requirements: 1.9, 6.6, 7.2, 7.7, 8.6, 11.1, 11.2_

  - [x] 7.2 Implement the session-start trigger
    - Add `on_session_start(runtime, *, new_thread_id)` to `src/chat/memory_hooks.py`
    - Return immediately when the runtime is `None` or `memory_extraction_on_session_start` is
      disabled
    - Build the eligible set from persisted session metadata via `storage.list_metadata()`,
      excluding the new thread, excluding threads absent from the session registry, and
      requiring `turn_count >= 1` and `watermark < turn_count`
    - Order by `(last_accessed_at, thread_id)` descending and take the first, so an identical
      `last_accessed_at` breaks toward the lexicographically greatest thread id
    - When the winner's `last_accessed_at` is older than
      `memory_extraction_max_session_age_hours`, advance its watermark to its turn count, make
      no LLM call, and select no replacement
    - Otherwise submit exactly one `session_start` extraction and return
    - Wrap the whole body so no exception escapes into the request path
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.8, 1.9, 1.10, 8.7_

  - [x] 7.3 Implement the round trigger
    - Add `after_turn(runtime, *, thread_id)` to `src/chat/memory_hooks.py`
    - Return immediately when the runtime is `None`
    - Read the turn count from the checkpoint store after the turn's checkpoint has been
      persisted, never from in-memory graph state, and submit one `round_complete` extraction
      when `should_extract` is true
    - Wrap the whole body so no exception escapes into the request path
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.7, 2.9_

  - [x]* 7.4 Write property test for the disabled path
    - Create `tests/test_memory_extraction_integration.py` with a stub scheduler recording
      submissions, a fake checkpointer, and `tmp_path` for every store
    - **Property 13: Disabled means inert**
    - **Validates: Requirements 6.6, 8.6, 11.1, 11.2**

  - [x]* 7.5 Write property test for adoption
    - Extend `tests/test_memory_extraction_integration.py`: a metadata store holding 100
      un-extracted sessions yields exactly one submission for one session creation
    - **Property 15: Adoption is not a stampede**
    - **Validates: Requirements 1.5, 11.5**

  - [x]* 7.6 Write unit tests for previous-session selection
    - Extend `tests/test_memory_extraction_integration.py`: the most recently accessed eligible
      session wins, an identical `last_accessed_at` breaks toward the greatest thread id, a
      thread deleted from the registry is excluded, a zero-turn thread is excluded, an over-age
      winner has its watermark advanced with no submission and no replacement, and no
      submission happens when nothing is eligible
    - Assert `after_turn` submits at turn counts 10, 20, and 30 with interval 10 and submits
      nothing at 1 through 9 or 11 through 19
    - _Requirements: 1.2, 1.3, 1.4, 1.7, 1.8, 2.4, 2.5, 8.7_

- [x] 8. Checkpoint - Hooks and scheduler verified
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Wire the hooks into the chat entry points
  - [x] 9.1 Wire the hooks into the chat API
    - In the `lifespan` of `src/chat/api.py`, build the extraction runtime after the registry,
      storage, and `SQLiteMemorySaver` exist, store it on `app.state.extraction_runtime`, and
      call `runtime.scheduler.shutdown()` after the `yield` without joining
    - Call `on_session_start` at the end of `start_chat`, after the session and its metadata
      are persisted and immediately before returning the response
    - Call `after_turn` at the end of `post_message`, after the reply is extracted and
      immediately before returning `MessageResponse`
    - Call `after_turn` in the `finally` block of `event_iter` in `post_message_stream`,
      alongside the existing `_sync_graph_owned_web_sources` call
    - Keep every call best-effort so a hook failure cannot change the response
    - _Requirements: 1.1, 1.6, 2.1, 2.8, 2.10, 6.1, 6.5_

  - [x] 9.2 Wire the round trigger into the CLI REPL
    - In `src/chat/main.py`, build the extraction runtime once before the REPL loop using the
      already-built `checkpointer` and the `"cli"` thread id, and call `after_turn` at the end
      of each iteration after the answer has been printed
    - Call `scheduler.shutdown()` when the loop exits, including on `EOFError` and
      `KeyboardInterrupt`
    - _Requirements: 2.10, 6.5_

  - [x]* 9.3 Write integration tests for the three entry points
    - Extend `tests/test_memory_extraction_integration.py`: the non-streaming endpoint, the
      streaming endpoint, and the CLI path all reach `after_turn` for a completed turn, and a
      turn that fails before its checkpoint is persisted reaches no submission
    - Assert a hook whose scheduler raises does not change the endpoint response and that the
      runtime is shut down when the app lifespan exits
    - _Requirements: 2.7, 2.10, 6.2, 6.5_

  - [x]* 9.4 Write property test for the unblocked reply path
    - Extend `tests/test_memory_extraction_integration.py`: with a runner held on an unset event
      the message endpoint still returns its reply and `start_chat` still returns its thread id
    - **Property 4 (endpoint half): The reply path is never blocked**
    - **Validates: Requirements 1.6, 2.8, 6.1, 6.8**

- [x] 10. Documentation and final verification
  - [x] 10.1 Document the feature and export the new modules
    - Extend `src/memory/SKILL.md` with an extraction section covering the two triggers, the
      watermark contract and its `config` key, the advance-versus-hold policy table, the
      daemon-thread scheduler and why `ThreadPoolExecutor` is not used, the provenance tag, and
      the prompt-injection posture, and move automatic extraction out of its "not built yet"
      list
    - Note the two known issues: the timeout abandons rather than cancels, and the pre-existing
      thread backlog is never processed
    - Export `MemoryExtractor`, `ExtractionCandidate`, `ExtractionOutcome`,
      `ExtractionScheduler`, and the transcript and watermark helpers from
      `src/memory/__init__.py`, and mention the hooks in `src/chat/SKILL.md`
    - _Requirements: 6.5, 9.8, 10.6_

- [x] 11. Final checkpoint - Ensure all tests pass
  - Run `python -m pytest -p no:warnings -q`, `python -m compileall src tests`,
    `python -m ruff check src tests`, and `git diff --check`, fixing anything they surface
  - Ensure all tests pass, ask the user if questions arise.

## Notes

**Verification after every task.**

```
python -m pytest -p no:warnings -q
python -m compileall src tests
python -m ruff check src tests
git diff --check
```

**Property tests here are exhaustive parametrized tests, not generated ones.** The project
has no `hypothesis` dependency (`requirements.txt` pins `pytest` only), and every design
property has a small finite input space — the watermark policy table, the turn-count boundary
set, the scheduler's two discard paths. Each property task enumerates that space with
`pytest.mark.parametrize` rather than adding a dependency.

**Never make a real LLM call in a test.** `MemoryExtractor` takes `model_factory` precisely so
tests inject a fake returning canned text. A test that reaches DeepSeek or DashScope will be
slow, flaky, and will spend money.

**Never let a test write `memory/long_term_memory.json`.** Every test points the store at
`tmp_path`. The `.gitignore` rule hides damage to the real file from `git status`.

**Daemon threads are deliberate, not sloppy.** Task 5.1 must not use
`concurrent.futures.ThreadPoolExecutor`: its `atexit` hook joins non-daemon workers, so a
60-second extraction would hold the process open on Ctrl-C and violate Requirement 6
criterion 5. Bound concurrency with a non-blocking semaphore acquire instead, and do not join
on shutdown.

**Rejection order in `submit` matters.** Check the in-flight set before acquiring the
semaphore. Reversed, a duplicate request for a busy thread would take and release a slot and
log a warning instead of being a quiet debug-level no-op.

**The watermark advance policy is the subtle part.** Only a raised call, a timeout, a
checkpoint read failure, or a store write failure holds the watermark. An empty array, an
unusable response, and an all-refused candidate set all advance it. Getting this backwards
means one permanently unusable slice blocks every later Round for the life of the thread.

**`from_session` rebuilds `config` wholesale.** Task 6.1's one added line is load-bearing:
without it, every `registry.get()` touch silently resets the watermark to absent, and round
extraction would re-distil the same turns forever.

**The CLI has no session registry.** `src/chat/main.py` runs a single `"cli"` thread with a
checkpointer and no `ChatSessionRegistry` or storage backend, so
`build_extraction_runtime` must accept `registry=None` and fall back to
`InMemoryWatermarkStore`. The consequence is honest and small: the CLI's watermark does not
survive a process restart, so the first round after a restart re-extracts turns the previous
run already distilled. Session-start extraction does not apply there at all.

**Three deviations carried from the design, already agreed:**

1. The timeout abandons rather than cancels. `model.invoke` is not interruptible, so on
   timeout the orphan daemon thread runs to completion and its result is discarded. The real
   network bound stays the provider's own `dashscope_request_timeout`.
2. `MEMORY_NOTE_MARKER` is the only non-additive change to the existing memory feature.
   Role-based exclusion already covers today's behaviour, so this is defence against a future
   change in how notes are injected. It is already implemented; task 2.7 only adds the
   assertions.
3. Extraction uses the main chat model; there is no cheaper extraction model setting in this
   version.

**Adoption leaves the existing backlog unprocessed.** With 115 threads already in the
checkpoint store, only the most recent eligible session is ever selected per session creation
and over-age ones are marked done without a call. That is Requirement 1 criterion 5 working as
intended. Processing the backlog would need a separate one-off command, not a request hook.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.7", "3.1", "5.1", "6.1"] },
    { "id": 1, "tasks": ["3.2", "5.2", "6.2"] },
    { "id": 2, "tasks": ["3.5", "5.3", "6.3"] },
    { "id": 3, "tasks": ["3.6", "5.4"] },
    { "id": 4, "tasks": ["3.7"] },
    { "id": 5, "tasks": ["3.3", "7.1"] },
    { "id": 6, "tasks": ["3.4", "7.2"] },
    { "id": 7, "tasks": ["3.8", "7.3"] },
    { "id": 8, "tasks": ["3.9", "9.1"] },
    { "id": 9, "tasks": ["3.10", "9.2"] },
    { "id": 10, "tasks": ["3.11", "7.4"] },
    { "id": 11, "tasks": ["3.12", "7.5"] },
    { "id": 12, "tasks": ["3.13", "7.6"] },
    { "id": 13, "tasks": ["9.3"] },
    { "id": 14, "tasks": ["9.4"] },
    { "id": 15, "tasks": ["10.1"] }
  ]
}
```
