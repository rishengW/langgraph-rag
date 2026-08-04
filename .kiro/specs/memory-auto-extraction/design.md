# Design Document

## Overview

Automatic extraction adds three modules to `src/memory/` and one wiring module to
`src/chat/`. Nothing in the existing memory package changes behaviour; the Extractor is a
new consumer of the same `MemoryStore.save()` that the tools already use.

The shape is deliberately narrow:

```
trigger  →  scheduler (bounded, fire-and-forget)  →  extractor  →  MemoryStore.save()
```

Four decisions carry the design:

1. **Daemon threads, not `ThreadPoolExecutor`.** Requirement 6 criterion 5 forbids blocking
   shutdown. `concurrent.futures.ThreadPoolExecutor` registers an `atexit` hook that joins
   its non-daemon workers, so a 60-second extraction would hold the process open on Ctrl-C.
   The scheduler therefore spawns `threading.Thread(daemon=True)` and bounds concurrency
   with a non-blocking semaphore acquire. A rejected acquire discards the request rather
   than queueing it, which is exactly what criterion 4 asks for.

2. **The checkpoint store is the only source of turn state.** `Turn_Count` is counted from
   the thread's checkpointed `HumanMessage` list, read through `SQLiteMemorySaver.get_tuple`
   — verified to work without the thread's compiled graph. No independent counter exists to
   drift, and a previous or restored session needs no graph rebuild.

3. **The watermark rides in `SessionMetadata.config`, behind a Protocol.** No SQLite
   migration: `config` is already free-form JSON and the schema version stays 1. `src/memory/`
   must not import `src/sessions/`, so the Extractor depends on a small `WatermarkStore`
   Protocol and the concrete session-metadata adapter lives in `src/chat/`.

4. **Extraction proposes, the store enforces.** Every candidate goes through
   `MemoryStore.save()`, so validation order, duplicate detection, credential screening,
   capacity eviction, and atomic writes all apply unchanged and are already covered by 250
   existing tests.

Off by default (`memory_extraction_enabled=False`). With the flag off there is no extra LLM
call, no extra checkpoint read, and no extra store write.

## Architecture

```
src/memory/
├── transcript.py       NEW  pure: count_turns, select_slice, render_slice
├── extraction.py       NEW  ExtractionCandidate, MemoryExtractor.run()
├── scheduler.py        NEW  ExtractionScheduler (daemon threads + semaphore)
├── watermark.py        NEW  WatermarkStore Protocol + InMemoryWatermarkStore
├── store.py            (unchanged)
├── recall.py           + MEMORY_NOTE_MARKER for identifiable recall notes
└── models.py           + EXTRACTION_TAG_PREFIX

src/chat/
├── memory_hooks.py     NEW  session-metadata watermark adapter + trigger entry points
├── api.py              + 3 hook calls (start_chat, post_message, post_message_stream)
│                       + scheduler on app.state, shutdown in lifespan
└── main.py             + 1 hook call in the REPL loop

src/sessions/models.py  + ChatSession.extraction_watermark: int = 0
src/sessions/storage.py + from_session carries extraction_watermark into config
src/llm/prompts.py      + MEMORY_EXTRACTION_PROMPT
src/config/settings.py  + 8 memory_extraction_* fields
src/config/loader.py    + env names, bool branch, int-range branch
.env.example, config/default.yaml
```

### Flow

```
POST /chat  ─────────────► session created + metadata persisted
                              │
                              ├─ hooks.on_session_start(app_state, new_thread_id)
                              │     select Previous_Session from PERSISTED metadata
                              │     eligible? → scheduler.submit("session_start", tid)
                              │     too old?  → watermark := turn_count (no LLM call)
                              └─ return thread_id immediately         (R1.6)

POST /chat/{tid}/message ─► graph.invoke → reply extracted
                              │
                              ├─ hooks.after_turn(app_state, tid)
                              │     turn_count = count_turns(checkpoint)
                              │     turn_count >= watermark + interval?
                              │         → scheduler.submit("round_complete", tid)
                              └─ return reply immediately              (R2.8)

POST /chat/{tid}/message/stream ─► event_iter() finally block
                              └─ same hooks.after_turn call           (R2.10)

scheduler worker (daemon thread, semaphore-bounded)
   └─ MemoryExtractor.run(trigger, thread_id)
        1. messages = read_thread_messages(checkpointer, thread_id)   R3.6
        2. turn_count = count_turns(messages)
        3. watermark = min(watermark_store.get(tid), turn_count)      R7.9
        4. slice = select_slice(messages, watermark, trigger)          R3.1-3.3
        5. empty? → watermark := turn_count, return                    R3.5
        6. text = render_slice(slice, max_chars)                       R3.4, R3.8-3.10
        7. candidates = call LLM once, bounded wait                    R4
        8. for each candidate: MemoryStore.save(...)                   R5
        9. outcome decides watermark advance                           R4.11, R5.8, R6.7
       10. one log record                                             R10
```

## Components and Interfaces

### 1. `src/memory/transcript.py` — pure transcript handling

```python
USER_LABEL: Final[str] = "User:"
ASSISTANT_LABEL: Final[str] = "Assistant:"
TRUNCATION_MARKER: Final[str] = " ... [truncated]"
MAX_SLICE_MESSAGES: Final[int] = 200

def message_role(message: Any) -> str
    """'user' | 'assistant' | 'system' | 'tool' | other, from .type or class name."""

def normalize_message_content(message: Any) -> str
    """Content to trimmed str. Non-string content (list/dict blocks) is flattened by
    concatenating the text of every element that carries text, joined by one space."""

def count_turns(messages: Sequence[Any]) -> int
    """Number of user messages. The authoritative Turn_Count."""

def is_memory_note(message: Any) -> bool
    """True when the message carries injected recall content (marker-based)."""

def select_slice(
    messages: Sequence[Any], *, watermark: int, trigger: str
) -> list[Any]
    """Messages after the watermark-th user message, excluding system, tool, memory
    notes, and blank-content messages. Both triggers reduce to the same rule."""

def render_slice(messages: Sequence[Any], *, max_chars: int) -> str
    """Role-labelled blocks separated by one blank line. Drops whole messages from the
    START to fit; truncates a single oversized final message with TRUNCATION_MARKER."""
```

All pure, no `Settings`, no I/O. `select_slice` collapses Requirement 3 criteria 1 and 2
into one implementation: `round_complete` and `session_start` differ only in that the
former's upper bound happens to be the last turn, which is also the end of the list. That
removes a branch the requirements imply but do not need.

### 2. `src/memory/watermark.py` — the persistence seam

```python
class WatermarkStore(Protocol):
    def get(self, thread_id: str) -> int: ...
    def set(self, thread_id: str, value: int) -> None: ...

WATERMARK_KEY: Final[str] = "extraction_watermark"
MAX_WATERMARK: Final[int] = 1_000_000

def coerce_watermark(raw: object) -> int:
    """0 for anything not a whole number in 0..MAX_WATERMARK. Booleans are rejected
    because bool is an int subclass and True would otherwise read as 1."""

class InMemoryWatermarkStore:  # tests
```

A Protocol rather than a direct `src/sessions` import keeps `src/memory/` free of a
dependency on the session layer and gives tests a two-line fake. `coerce_watermark`
implements Requirement 7 criterion 8; the `bool` rejection is the subtle part.

### 3. `src/memory/extraction.py` — the Extractor

```python
@dataclass(frozen=True)
class ExtractionCandidate:
    content: str
    category: str | None = None
    tags: tuple[str, ...] = ()
    scope: str = "global"

@dataclass(frozen=True)
class ExtractionOutcome:
    trigger: str
    thread_id: str
    slice_messages: int
    candidates: int
    persisted: int
    refused: int
    record_count: int
    watermark_before: int
    watermark_after: int
    duration_ms: int
    status: str          # "extracted" | "no_information" | "failed" | "skipped"
    detail: str = ""

class MemoryExtractor:
    def __init__(
        self,
        settings: Settings,
        *,
        checkpointer: Any,
        watermarks: WatermarkStore,
        store: MemoryStore | None = None,
        model_factory: Callable[[Settings], Any] = build_chat_model,
        clock: Callable[[], float] = time.monotonic,
    ) -> None: ...

    def turn_count(self, thread_id: str) -> int
    def should_extract(self, thread_id: str) -> bool          # R2.1, R2.4
    def run(self, trigger: str, thread_id: str) -> ExtractionOutcome
```

`run` never raises. Every failure returns an `ExtractionOutcome` with `status="failed"`,
which is what makes Requirement 6 criterion 2 structural rather than a promise.

**Watermark advance policy** — the one place the requirements needed a decision, resolved in
Requirement 4 criterion 11, Requirement 5 criterion 8, and Requirement 6 criterion 7:

| Situation | Watermark | Why |
|---|---|---|
| Candidates persisted | advances | normal success |
| LLM returned `[]` | advances | model saw nothing durable; retrying is pointless |
| LLM response unparseable | advances | retrying the same slice will fail identically |
| Every candidate refused | advances | refusal is a decision, not a fault |
| Empty slice | advances | nothing to extract, ever |
| LLM raised or timed out | **holds** | transient; retry on next Round |
| Checkpoint read failed | **holds** | transient |
| Store write failed | **holds** | transient |

Without this split, one permanently un-extractable slice would block every later Round for
the life of the thread.

**LLM call with a bounded wait.** `model.invoke` is synchronous and not cancellable, so the
call runs in a nested daemon thread and the Extractor waits with
`memory_extraction_timeout_seconds`. On timeout the wait returns, the outcome is `failed`,
the watermark holds, and the orphan thread's result is discarded when it eventually
completes. The orphan is a daemon so it cannot delay shutdown. This is honest about the
limit: the underlying HTTP request is not aborted, only abandoned. The provider's own
`dashscope_request_timeout` remains the real network bound.

**Parsing.** `json.loads` first; on failure, the first `[...]` span found by a bracket scan;
on failure, zero candidates. Entries are filtered by Requirement 4 criteria 9 and 10
(non-objects and blank `content` dropped, then case-folded whitespace-collapsed duplicates
dropped), then truncated to `memory_extraction_max_candidates`.

**Provenance.** Every candidate gets `EXTRACTION_TAG_PREFIX + trigger` appended to its tags
(for example `auto:round_complete`), within the existing 10-tag / 40-char limits. Because
`MemoryStore.save` replaces the tag list on a duplicate update, the tag survives updates —
satisfying Requirement 9 criterion 8 with no store change.

### 4. `src/memory/scheduler.py` — bounded fire-and-forget

```python
class ExtractionScheduler:
    def __init__(
        self,
        *,
        max_concurrency: int,
        runner: Callable[[str, str], Any],
        thread_factory: Callable[..., threading.Thread] = threading.Thread,
    ) -> None:
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._inflight: set[str] = set()
        self._lock = threading.Lock()

    def submit(self, trigger: str, thread_id: str) -> bool:
        """False when discarded: already in flight for this thread, or no free slot."""

    def wait_idle(self, timeout: float | None = None) -> bool:   # tests only
    def shutdown(self) -> None:
```

Rejection order matters: the per-thread `_inflight` check comes **before** the semaphore
acquire, so a duplicate request for a busy thread is discarded at debug level (Requirement 6
criterion 3) rather than consuming and releasing a slot and logging a warning.

`submit` returns a bool so callers and tests can assert scheduling decisions without
inspecting logs.

`shutdown()` sets a flag that makes `submit` a no-op. It does **not** join: daemon threads
plus no join is what satisfies Requirement 6 criterion 5.

### 5. `src/chat/memory_hooks.py` — wiring

```python
class SessionWatermarkStore:
    """WatermarkStore backed by ChatSession + SessionMetadata.config."""

    def __init__(self, registry: ChatSessionRegistry, storage: Any | None) -> None: ...
    def get(self, thread_id: str) -> int      # persisted first, registry fallback (R1.9)
    def set(self, thread_id: str, value: int) -> None

def build_extraction_runtime(settings, *, checkpointer, registry, storage) -> ExtractionRuntime | None
    """None when memory_enabled or memory_extraction_enabled is off."""

def on_session_start(runtime, *, new_thread_id: str) -> None      # R1
def after_turn(runtime, *, thread_id: str) -> None                # R2
```

`on_session_start` implements the Previous_Session selection: read persisted metadata, filter
to sessions present in the registry (Requirement 1 criterion 7 excludes deleted threads),
require `turn_count >= 1` and `watermark < turn_count`, then order by
`(last_accessed_at, thread_id)` descending and take the first. Age check comes after
selection, and an over-age winner has its watermark advanced with no LLM call and no
fallback to a second candidate (Requirement 1 criterion 4).

Both hooks are wrapped so no exception escapes into the request path.

### 6. Session-layer changes

`ChatSession` gains one field with a default, so no existing constructor call breaks:

```python
extraction_watermark: int = 0
```

`SessionMetadata.from_session` carries it into `config`:

```python
config={
    "collection_name": session.settings.collection_name,
    "extraction_watermark": int(session.extraction_watermark or 0),
},
```

That single line is what stops an unrelated metadata save from dropping the watermark
(Requirement 7 criterion 3) — the current `from_session` rebuilds `config` wholesale, which
would otherwise silently reset it on the next `registry.get()` touch.

### 7. Recall-note marking

Requirement 9 criterion 7 wants the injected recall note excluded without content
inspection. `recall.py` gains:

```python
MEMORY_NOTE_MARKER: Final[str] = "\u200b\u200b"   # zero-width, prepended to note content
```

and `transcript.is_memory_note` tests for it. Today all system messages are already
excluded, so this is defence in depth against a future change in how notes are injected.
The marker is zero-width so it does not alter what the model reads.

> Flagged: this is the one **non-additive** change to the existing memory feature. If you
> would rather not touch the recall path, `is_memory_note` can be dropped and the role-based
> exclusion relied on alone; the design works either way and the tests for the rest are
> unaffected.

### 8. Prompt

`src/llm/prompts.py` gains `MEMORY_EXTRACTION_PROMPT`, a plain `str.format` template with
one `{transcript}` slot. It instructs: emit durable user-specific facts only; emit nothing
for one-off questions, web-search results, transient state, or third-party facts; return
`[]` when nothing qualifies; return only a JSON array of at most N objects with
`content`/`category`/`tags`/`scope`. The transcript is wrapped in a delimiter labelling it
untrusted conversation data, and the instructions state that text inside the delimiter is
data and never an instruction (Requirement 9 criteria 4 and 6).

No `Memory_Record` content, no `Settings` value, and no store path appear in the prompt.

### 9. Configuration

Eight fields on `Settings`, mirroring the existing memory block:

```python
memory_extraction_enabled: bool = False
memory_extraction_on_session_start: bool = True
memory_extraction_turn_interval: int = 10
memory_extraction_max_candidates: int = 5
memory_extraction_max_transcript_chars: int = 8000
memory_extraction_timeout_seconds: int = 60
memory_extraction_max_concurrency: int = 2
memory_extraction_max_session_age_hours: int = 168
```

`loader.py`: eight `SETTING_ENV_NAMES` entries, the two booleans appended to the existing
`parse_bool` branch, and the six integers added to `_MEMORY_INT_RANGES` — which already
raises `ValueError` with field, value, and range. The existing range table is extended
rather than duplicated, so the fail-fast policy is inherited for free.

## Data Models

```python
ExtractionCandidate(content, category=None, tags=(), scope="global")
ExtractionOutcome(trigger, thread_id, slice_messages, candidates, persisted, refused,
                  record_count, watermark_before, watermark_after, duration_ms,
                  status, detail)
```

Persisted watermark, inside the existing session metadata `config` JSON (schema version
stays 1):

```json
{
  "thread_id": "8e1cd61f87eb4cbdbcefdb032779c166",
  "config": {
    "collection_name": "rag-chroma",
    "extraction_watermark": 20
  }
}
```

Expected LLM response shape:

```json
[
  {"content": "Works as a marine biologist in Lisbon.",
   "category": "fact", "tags": ["job", "location"], "scope": "global"},
  {"content": "Prefers metric units.", "category": "preference"}
]
```

Resulting `Memory_Record` carries the Provenance_Tag:

```json
{"content": "Prefers metric units.", "category": "preference",
 "tags": ["auto:round_complete"], "scope": "global", "scope_id": null}
```

## Correctness Properties

### Property 1: Turn counting cannot drift

**Validates: Requirements 2.2, 2.6**

`count_turns(messages)` equals the number of user messages in the thread's checkpointed
list. No other counter exists, so no divergence between the counter and the transcript is
representable.

### Property 2: Round boundaries are exact

**Validates: Requirements 2.5**

With `interval = 10` and an initial watermark of 0, `should_extract` is true at Turn_Count
10, 20, 30 and false at 1–9 and 11–19. After the Turn_Count 20 extraction, the slice covers
turns 11 through 20 with no turn extracted twice and none skipped.

### Property 3: Only transient failures hold the watermark

**Validates: Requirements 4.6, 4.11, 5.8, 6.7**

The watermark advances for every outcome in which the LLM returned a response or the slice
was empty, and holds only for a raised call, a timeout, a checkpoint read failure, or a
store write failure. A permanently unusable slice therefore cannot block later Rounds.

### Property 4: The reply path is never blocked

**Validates: Requirements 1.6, 2.8, 6.1, 6.8**

With an extraction held indefinitely by a blocking fake runner, the turn's reply and the
session identifier are still returned. Assertable without measuring time.

### Property 5: Extraction never raises into a turn

**Validates: Requirements 6.2**

`MemoryExtractor.run` returns an `ExtractionOutcome` for every input, including a
checkpointer that raises, an LLM that raises, a store that raises, and a watermark store
that raises. `on_session_start` and `after_turn` likewise never raise.

### Property 6: Bounded concurrency with no queue

**Validates: Requirements 6.3, 6.4, 6.9**

At most `max_concurrency` runners execute at once; a request for a thread already in flight
is discarded before the semaphore is touched; a request with no free slot is discarded, not
queued. `submit` returns `False` in both discard cases.

### Property 7: Shutdown is not delayed

**Validates: Requirements 6.5**

Every worker is a daemon thread and nothing joins them, so no `atexit` join can hold the
process open. `shutdown()` makes further `submit` calls no-ops.

### Property 8: The watermark survives unrelated saves

**Validates: Requirements 7.3, 11.7**

After `registry.get(thread_id)` touches and re-saves a session, the persisted
`extraction_watermark` equals its value before that save. Metadata written before this
feature loads with a watermark of 0 and is preserved on the next save.

### Property 9: A corrupt watermark degrades to 0

**Validates: Requirements 7.8, 7.9**

`coerce_watermark` returns 0 for a non-integer, a boolean, a negative value, or a value
above 1,000,000, and a watermark above the current Turn_Count is clamped to Turn_Count
rather than triggering an extraction.

### Property 10: Slice rendering is deterministic and bounded

**Validates: Requirements 3.4, 3.8, 3.10**

Two renders of the same slice are character-identical; rendered length never exceeds
`memory_extraction_max_transcript_chars`; whole messages are dropped from the start; a single
oversized message is truncated with a marker rather than producing an empty slice.

### Property 11: Extraction cannot bypass store safeguards

**Validates: Requirements 5.1, 5.2, 5.3, 5.5, 9.2**

Every persisted candidate passes through `MemoryStore.save`, so credential refusal,
duplicate update, parameter limits, and capacity eviction behave identically to a manual
`save_memory` call. No other write path to the store file exists in the Extractor.

### Property 12: Scope binds to the extracted thread

**Validates: Requirements 5.1, 5.9**

A `session`-scoped candidate produced by a `session_start` extraction carries the
Previous_Session's Thread_Id, never the newly created session's.

### Property 13: Disabled means inert

**Validates: Requirements 6.6, 8.6, 11.1, 11.2**

With `memory_extraction_enabled` false, or `memory_enabled` false,
`build_extraction_runtime` returns `None`; the hooks are no-ops; zero LLM calls, zero
checkpoint reads, and zero store writes occur beyond what the feature-absent code performs;
turn message lists are unchanged.

### Property 14: Logs carry no content

**Validates: Requirements 9.1, 10.1, 10.6**

Every extraction log record starts with `memory_extraction`, carries counts and ids, and
contains no substring of 20+ consecutive characters from the transcript, a candidate, a
record, or the LLM response. Exactly one informational record per completed extraction and
exactly one warning per failure.

### Property 15: Adoption is not a stampede

**Validates: Requirements 1.5, 11.5**

Against a metadata store holding 100+ un-extracted sessions, one session creation schedules
at most one extraction.

## Error Handling

| Layer | Strategy |
|---|---|
| `transcript.py` | Pure; tolerates odd message shapes by normalizing rather than raising |
| `coerce_watermark` | Never raises; unusable values become 0 |
| `MemoryExtractor.run` | Sole boundary; returns `ExtractionOutcome(status="failed")` for everything |
| LLM call | Nested daemon thread + bounded wait; late results discarded |
| `ExtractionScheduler.submit` | Returns `False` on discard; the worker body catches everything |
| `on_session_start` / `after_turn` | Wrapped; log a warning and return, so the request path is unaffected |
| Store save | Refusals counted, not raised; a genuine write failure holds the watermark |
| Logging | `logging.getLogger("src.memory.extraction")`, every message prefixed `memory_extraction` |

## Testing Strategy

New files:

**`tests/test_memory_transcript.py`** — `count_turns` over mixed message kinds;
`select_slice` at watermark 0 / mid / equal-to-count; exclusion of system, tool, blank, and
marked memory-note messages; `render_slice` determinism, character bound, whole-message
dropping, and single-oversized-message truncation; non-string content flattening.

**`tests/test_memory_watermark.py`** — `coerce_watermark` table including `True`/`False`,
`"3"`, `3.0`, `-1`, `10**9`; `SessionWatermarkStore` persisted-over-registry precedence;
round-trip through `SessionMetadata.from_session` proving Property 8.

**`tests/test_memory_extraction.py`** — the bulk. A fake checkpointer returning a canned
message list, a fake model returning canned text, and a real `MemoryStore` on `tmp_path`:
- happy path persists candidates with the Provenance_Tag;
- empty array, unparseable text, JSON object instead of array, non-object entries,
  duplicate candidates, more than `max_candidates`;
- the full watermark table of Property 3, one case per row;
- credential-bearing candidate refused with no secret in logs;
- `session`-scoped candidate binds the extracted thread (Property 12);
- model raising and model exceeding the timeout;
- watermark above Turn_Count clamped;
- log assertions for Property 14 via `caplog`.

**`tests/test_memory_scheduler.py`** — per-thread dedupe, concurrency cap, no queueing,
`submit` return values, daemon-thread assertion for Property 7, and a blocking runner proving
Property 4 without timing.

**`tests/test_memory_extraction_integration.py`** — `on_session_start` selection including
tie-break, deleted-thread exclusion, age skip advancing the watermark without an LLM call,
and the 100-session no-stampede case; `after_turn` triggering at the right Turn_Count from
all three entry points with a stub scheduler recording submissions; disabled-flag inertness.

**`tests/test_config.py`** (extended) — the eight new fields: defaults, env mapping, YAML,
boolean accepted values, out-of-range and non-integer failures naming field/value/range,
and `.env.example` + `config/default.yaml` documentation presence.

Every test uses `tmp_path` for the store and a fake model factory. **No test performs a real
LLM call**, and no test writes `memory/long_term_memory.json`.

Verification: `python -m pytest -p no:warnings -q`, `python -m compileall src tests`,
`python -m ruff check src tests`, `git diff --check`.

## Requirements Traceability

| Req | Design components |
|---|---|
| 1. Session-boundary extraction | `memory_hooks.on_session_start`, `SessionWatermarkStore`, `start_chat` hook |
| 2. Round extraction | `memory_hooks.after_turn`, `MemoryExtractor.should_extract`, `transcript.count_turns`, 3 call sites |
| 3. Transcript slice | `transcript.select_slice`, `render_slice`, `normalize_message_content`, `is_memory_note` |
| 4. LLM candidates | `MemoryExtractor._call_model`, `_parse_candidates`, `MEMORY_EXTRACTION_PROMPT` |
| 5. Persisting candidates | `MemoryExtractor._persist`, existing `MemoryStore.save`, `EXTRACTION_TAG_PREFIX` |
| 6. Never degrades a turn | `ExtractionScheduler` (daemon + semaphore), `run` no-raise boundary, lifespan shutdown |
| 7. Watermark bookkeeping | `watermark.py`, `ChatSession.extraction_watermark`, `SessionMetadata.from_session` |
| 8. Configuration | `Settings` fields, `SETTING_ENV_NAMES`, `_MEMORY_INT_RANGES`, `.env.example`, `config/default.yaml` |
| 9. Privacy and safety | `MEMORY_EXTRACTION_PROMPT` delimiter, store-only screening, `MEMORY_NOTE_MARKER`, Provenance_Tag, log discipline |
| 10. Observability | `ExtractionOutcome`, single log record per outcome, `memory_extraction` prefix |
| 11. Backward compatibility | `build_extraction_runtime` returning `None`, defaulted `ChatSession` field, unchanged tool surface |

## Open Design Notes

1. **The timeout abandons rather than cancels.** `model.invoke` cannot be interrupted, so on
   timeout the orphan daemon thread runs to completion and its result is discarded. The real
   network bound remains the provider's own `dashscope_request_timeout`. If a hard bound
   matters, the alternative is a subprocess, which is a much larger change.
2. **`MEMORY_NOTE_MARKER` touches the existing recall path.** It is the only non-additive
   change. Role-based exclusion already covers today's behaviour; say the word and I will
   drop the marker.
3. **Extraction uses the main chat model.** No cheaper model setting in this version, per
   Requirement 8 criterion 10. At defaults that is one call per 10 turns plus one per new
   session, on the same DeepSeek/DashScope credentials.
4. **`session_start` extracts one session, not a backlog.** With 115 threads already on
   disk, first enable will leave most of them permanently un-extracted, because only the
   most recent eligible session is ever chosen and over-age ones are marked done. If you want
   the backlog processed, that needs a separate one-off command rather than a request hook.
