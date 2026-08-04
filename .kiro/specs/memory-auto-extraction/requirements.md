# Requirements Document

## Introduction

This feature makes the chat agent's long-term memory **self-updating**. Today a record is
written only when the model chooses to call `save_memory` during a turn, which is why a
conversation can pass without any memory changing: the user asked about the World Cup, no
durable fact about them was stated, and nothing was saved. Assumption 4 of the
`chat-long-term-memory` spec recorded that explicit-call-only behaviour as deliberate and
listed automatic extraction as out of scope. **This spec supersedes that assumption.**

Two triggers are added, modelled on a self-evolving agent loop:

- **Session-boundary extraction.** When a new chat session starts, the agent reviews the
  transcript of the previous session and distils durable facts from it.
- **Round extraction.** Every time the current session completes a round of
  `memory_extraction_turn_interval` (default 10) query/answer turns, the agent distils that
  round and then begins a new round.

Both triggers converge on the same path: read a bounded transcript slice, ask the LLM for
candidate memories as structured output, then persist each candidate through the existing
`MemoryStore.save()`. That reuse is the core of the design — extraction proposes, and the
store still enforces every invariant already specified and tested: fixed validation order,
duplicate detection, credential screening, capacity eviction, and atomic writes.

Grounding verified against the running system before writing this document:

- `SQLiteMemorySaver` inherits `MemorySaver.get_tuple`, so a thread's transcript is
  readable from `checkpointer.get_tuple({"configurable": {"thread_id": tid}})` →
  `checkpoint["channel_values"]["messages"]`, **without** needing that session's compiled
  graph. Confirmed against the live store: 115 threads, human-turn counts derivable.
- `SessionMetadata.config` is a free-form `dict[str, object]` persisted as JSON, so
  extraction bookkeeping needs **no SQLite schema migration**. However
  `SessionMetadata.from_session` rebuilds `config` on every save, so a matching field must
  be added to `ChatSession` or the bookkeeping is lost on the next save.
- The checkpoint store already holds 115 threads. Enabling this feature must not trigger a
  backlog stampede over all of them.
- `ChatSessionRegistry.delete` removes session metadata but does **not** call
  `checkpointer.delete_thread`, so a deleted session's transcript survives in the
  checkpoint store.

The feature is **additive and off by default** (`memory_extraction_enabled=False`), so no
existing behaviour, test, or deployment changes until the flag is set.

One change to the *existing* memory feature is required and is called out here because it
is not additive: Requirement 9 criterion 7 obliges the Chat_Agent to **mark** the injected
recall `SystemMessage` so the Extractor can exclude it without inspecting its content.
Without that mark, a prior turn's recalled memories could be re-extracted into new records,
compounding over time.

## Glossary

- **Extractor**: The component that turns a Transcript_Slice into Extraction_Candidates by
  one LLM call, then persists them through the Memory_Store.
- **Extraction_Candidate**: One proposed memory produced by the Extractor, carrying
  content and optionally category, tags, and scope, before the store accepts or refuses it.
- **Transcript_Slice**: The ordered list of user and assistant messages selected for one
  extraction, after the exclusions in Requirement 3 criterion 3.
- **Round**: A block of `memory_extraction_turn_interval` completed query/answer turns in
  one session, counted from the session's first turn.
- **Turn_Count**: The number of `HumanMessage` entries in a thread's checkpointed message
  list. The authoritative turn counter, because it cannot drift from the transcript.
- **Extraction_Watermark**: The Turn_Count up to and including which a thread has already
  been extracted. Persisted under the `extraction_watermark` key of the session metadata
  `config` mapping.
- **Extraction_Trigger**: One of `session_start` (Case 1) or `round_complete` (Case 2).
- **Previous_Session**: The session selected for session-boundary extraction when a new
  session is created.
- **Provenance_Tag**: A tag applied to every Memory_Record the Extractor creates or
  updates, identifying it as automatically extracted and naming its Extraction_Trigger, so
  extracted records can be listed and removed as a group without inspecting content.
- **Memory_Store**, **Memory_Record**, **Memory_Scope**, **Thread_Id**, **Settings**,
  **Secret_Pattern**, **Chat_Agent**: As defined in the `chat-long-term-memory` spec
  glossary.

## Requirements

### Requirement 1: Session-boundary extraction

**User Story:** As a chat user, I want the agent to learn from my last conversation when I
start a new one, so that it improves without me telling it what to remember.

#### Acceptance Criteria

1. WHILE `memory_enabled` and `memory_extraction_enabled` are both enabled, WHEN a new
   chat session has been created and its session metadata persisted, THE Extractor SHALL
   select at most one Previous_Session and SHALL schedule exactly one extraction with
   Extraction_Trigger `session_start` for that Previous_Session's Thread_Id within 1 second
   of that creation.
2. THE Extractor SHALL select as Previous_Session the eligible session with the greatest
   `last_accessed_at`, where a session is eligible only if its Thread_Id differs from the
   new session's Thread_Id, its Turn_Count is 1 or greater, and its Extraction_Watermark is
   less than its Turn_Count.
3. IF no session satisfies criterion 2 — including when the new session is the first
   session ever created, when the persisted session metadata holds no other session, and
   when every other session's Extraction_Watermark already equals its Turn_Count — THEN THE
   Extractor SHALL schedule no extraction, SHALL make no LLM call, SHALL leave every
   Memory_Record and every Extraction_Watermark unchanged, and THE Chat_Agent SHALL still
   return the new session's identifier as a successful creation rather than an error.
4. IF the Previous_Session's `last_accessed_at` precedes the new session's creation
   timestamp by more than `memory_extraction_max_session_age_hours` hours, both timestamps
   compared in UTC, THEN THE Extractor SHALL schedule no extraction for it, SHALL make no
   LLM call, SHALL advance that session's Extraction_Watermark to its Turn_Count so it is
   not reconsidered, and SHALL select no further Previous_Session for that session
   creation.
5. WHEN a new chat session is created, THE Extractor SHALL schedule at most one
   `session_start` extraction and SHALL cause at most one extraction LLM call for that
   creation, even when 100 or more sessions have an Extraction_Watermark below their
   Turn_Count, so that enabling the feature against an existing checkpoint store cannot
   start a backlog stampede.
6. WHEN a `session_start` extraction is scheduled, THE Chat_Agent SHALL return the new
   session's identifier to the client before that extraction's LLM call completes, and the
   time to return that identifier SHALL increase by no more than 100 milliseconds compared
   with the same creation performed while `memory_extraction_enabled` is disabled,
   regardless of how long the extraction runs.
7. WHERE a Previous_Session has been deleted from the session registry but its transcript
   remains in the checkpoint store, THE Extractor SHALL treat that Thread_Id as ineligible
   for selection under criterion 2 even when its checkpointed Turn_Count is 1 or greater
   and exceeds its last known Extraction_Watermark, and SHALL make no LLM call for that
   Thread_Id.
8. IF two or more sessions are eligible under criterion 2 and share an identical
   `last_accessed_at` value, THEN THE Extractor SHALL select the session whose Thread_Id
   sorts last in ascending lexicographic order and SHALL schedule no extraction for the
   others, so that repeated selections over an unchanged set of sessions yield the same
   Thread_Id.
9. WHEN the Extractor evaluates eligibility under criterion 2, THE Extractor SHALL read
   each candidate session's existence, `last_accessed_at`, and Extraction_Watermark from
   the persisted session metadata, and IF a persisted value differs from an in-memory
   registry value for the same session, THEN THE Extractor SHALL use the persisted value,
   so that selection immediately after a process restart yields the same Thread_Id as
   selection before that restart.
10. IF creating the new chat session or persisting its metadata fails, THEN THE Extractor
    SHALL schedule no `session_start` extraction for that attempt, SHALL make no LLM call,
    and SHALL leave every Memory_Record and every Extraction_Watermark unchanged, and THE
    Chat_Agent SHALL return an error indicating that session creation failed.

### Requirement 2: Round extraction inside a session

**User Story:** As a chat user, I want a long conversation to keep updating memory as it
goes, so that a 40-turn session does not wait until it ends to record anything.

#### Acceptance Criteria

1. WHILE `memory_enabled` and `memory_extraction_enabled` are both enabled, WHEN a chat
   turn completes — meaning the Chat_Agent has persisted that turn's checkpoint for the
   Thread_Id and has returned or finished emitting that turn's assistant reply without
   raising — and the thread's Turn_Count is greater than or equal to the
   Extraction_Watermark plus `memory_extraction_turn_interval`, THE Extractor SHALL
   schedule exactly one extraction with Extraction_Trigger `round_complete` for that
   Thread_Id.
2. WHEN the Extractor evaluates the criterion 1 trigger condition, THE Extractor SHALL read
   Turn_Count from the thread's checkpointed message list in the checkpoint store keyed by
   Thread_Id, read after that turn's checkpoint has been persisted, and SHALL NOT read it
   from the reply path's in-memory graph state or from any separate turn counter.
3. WHEN a `round_complete` extraction completes, THE Extractor SHALL set that thread's
   Extraction_Watermark to the Turn_Count observed when the Transcript_Slice was selected,
   so the next Round covers only turns after that point.
4. WHEN a chat turn completes and the thread's Turn_Count is less than the
   Extraction_Watermark plus `memory_extraction_turn_interval`, THE Extractor SHALL
   schedule no extraction, SHALL make no LLM call, and SHALL leave the
   Extraction_Watermark unchanged.
5. WHEN a `round_complete` extraction is scheduled at an observed Turn_Count of N, THE
   Extractor SHALL schedule no further `round_complete` extraction for that Thread_Id until
   Turn_Count reaches N plus `memory_extraction_turn_interval`, so that with
   `memory_extraction_turn_interval` of 10 and an initial Extraction_Watermark of 0 the
   triggers fall at Turn_Count 10, 20, and 30, no trigger occurs at Turn_Count 1 through 9
   or at Turn_Count 11 through 19, and the Turn_Count 20 extraction covers turns 11
   through 20.
6. THE Extractor SHALL count a turn toward Turn_Count if and only if that turn's
   `HumanMessage` is present in the thread's checkpointed message list, regardless of
   whether that turn produced a non-empty assistant reply, so a turn whose assistant reply
   was empty still advances the Round boundary while being excluded from the
   Transcript_Slice by Requirement 3 criterion 3.
7. IF a turn fails before its assistant reply is returned or before its checkpoint is
   persisted — including an LLM error, a timeout, a cancelled request, or a disconnected
   client — THEN THE Extractor SHALL schedule no extraction for that turn and SHALL leave
   the Extraction_Watermark unchanged, and the next turn that completes as defined in
   criterion 1 SHALL be evaluated against criteria 1 and 4 using the then-current
   Turn_Count.
8. WHEN a turn completes, THE Chat_Agent SHALL return that turn's assistant reply to the
   client without awaiting any scheduled extraction, so the reply is returned whether that
   extraction has not started, is still running for up to
   `memory_extraction_timeout_seconds` seconds, has completed, or has failed.
9. WHERE a session accumulates more than `memory_extraction_turn_interval` turns beyond its
   Extraction_Watermark while an extraction for that Thread_Id is still in flight, THE
   Extractor SHALL schedule no second extraction for that Thread_Id, and the first turn to
   complete after that extraction finishes SHALL be evaluated against criterion 1 using the
   updated Extraction_Watermark so the next scheduled extraction covers every turn between
   the updated Extraction_Watermark and the then-current Turn_Count.
10. WHEN a turn completes on the streaming message endpoint, on the non-streaming message
    endpoint, or in the CLI REPL, THE Extractor SHALL apply criteria 1 through 7
    identically for all three entry points, and for the streaming endpoint that turn
    completes when the final assistant content chunk has been emitted and that turn's
    checkpoint has been persisted.

### Requirement 3: Transcript slice selection

**User Story:** As a developer, I want the extracted transcript slice to be bounded and
deterministic, so that extraction cost and prompt size are predictable.

#### Acceptance Criteria

1. WHEN the Extractor selects a Transcript_Slice for Extraction_Trigger `round_complete`,
   THE Extractor SHALL include the user and assistant messages whose position follows the
   Extraction_Watermark-th `HumanMessage` and precedes or equals the Turn_Count-th
   `HumanMessage` together with that turn's assistant reply.
2. WHEN the Extractor selects a Transcript_Slice for Extraction_Trigger `session_start`,
   THE Extractor SHALL include the user and assistant messages following the
   Extraction_Watermark-th `HumanMessage` through the end of that thread's message list.
3. THE Extractor SHALL exclude from every Transcript_Slice each message whose role is
   system, each tool message, and each message whose normalized content as defined in
   criterion 9 contains zero non-whitespace characters after trimming, so that injected
   memory notes, upload notes, and tool-call carriers are never fed back into extraction.
4. THE Extractor SHALL limit a Transcript_Slice to at most 200 retained messages and at
   most `memory_extraction_max_transcript_chars` characters of rendered text as defined in
   criterion 8, SHALL apply the message limit before the character limit, and SHALL enforce
   both limits by dropping whole messages from the **start** of the slice, so the most
   recent turns are retained and no retained message is cut mid-content except as stated in
   criterion 10.
5. IF a Transcript_Slice contains zero user messages after the exclusions in criterion 3,
   or its rendered text contains zero non-whitespace characters, THEN THE Extractor SHALL
   perform no LLM call, SHALL leave every Memory_Record unchanged, and SHALL advance the
   Extraction_Watermark to the observed Turn_Count.
6. WHEN the Extractor reads a thread's message list, THE Extractor SHALL read it from the
   checkpoint store keyed by Thread_Id and SHALL require no compiled graph for that thread.
7. IF reading a thread's message list fails or returns no checkpoint, THEN THE Extractor
   SHALL perform no LLM call, SHALL leave every Memory_Record and every
   Extraction_Watermark unchanged, and SHALL log one warning naming the Thread_Id and the
   cause.
8. WHEN the Extractor renders a Transcript_Slice into prompt text, THE Extractor SHALL
   render each retained message as one block consisting of a fixed role label — one label
   for user messages and a different label for assistant messages — followed by that
   message's normalized content, SHALL order blocks by ascending original message position,
   and SHALL separate consecutive blocks by one blank line, so that two renderings of the
   same slice are character-identical and the character count in criterion 4 counts label,
   separator, and content characters as Unicode code points.
9. WHEN a message's content is not a string, THE Extractor SHALL normalize it to a string
   by concatenating, in original order, the text of every element that carries text,
   joining consecutive concatenated parts with a single space character, and discarding
   every element that carries no text, and THE Extractor SHALL apply no other
   transformation to string content beyond trimming leading and trailing whitespace.
10. IF one retained message's rendered text alone exceeds
    `memory_extraction_max_transcript_chars` characters after every earlier message has
    been dropped under criterion 4, THEN THE Extractor SHALL retain that message, SHALL
    truncate its rendered text to at most `memory_extraction_max_transcript_chars`
    characters by removing trailing characters, and SHALL end the truncated text with a
    fixed marker indicating that content was omitted, rather than dropping the message and
    producing an empty slice.

### Requirement 4: Candidate extraction by the LLM

**User Story:** As a chat user, I want the agent to record only what is durably true about
me, so that memory does not fill up with one-off questions and trivia.

#### Acceptance Criteria

1. WHEN the Extractor performs an extraction, THE Extractor SHALL make exactly one LLM
   call for that extraction, SHALL issue no retry call within that extraction, and SHALL
   request a JSON array of 0 to `memory_extraction_max_candidates` objects, each containing
   a `content` string and optionally `category`, `tags`, and `scope`.
2. THE Extractor prompt SHALL instruct the model to emit only durable, user-specific
   information — the user's identity, stated preferences, attributes, ongoing tasks, and
   long-lived context — and SHALL instruct it to emit nothing for one-off factual
   questions, web-search results, transient task state, or facts about third parties that
   the user did not claim as their own.
3. THE Extractor prompt SHALL instruct the model to return an empty array when the
   Transcript_Slice contains no durable user-specific information.
4. WHEN the LLM response parses to a JSON array, THE Extractor SHALL evaluate at most the
   first 100 entries of that array in array order, SHALL discard every entry rejected under
   criteria 9 and 10, SHALL keep the first `memory_extraction_max_candidates` surviving
   entries as Extraction_Candidates, and SHALL discard every entry beyond that count and
   beyond the first 100.
5. IF the LLM response does not parse as a JSON array, including a response that parses as
   valid JSON of any non-array type, THEN THE Extractor SHALL attempt to parse the first
   JSON array found within the response text, and IF that also fails, THEN THE Extractor
   SHALL produce zero Extraction_Candidates and SHALL log one warning naming the Thread_Id
   and stating that the response was unusable.
6. IF the LLM call raises an exception, or returns no complete response within
   `memory_extraction_timeout_seconds` seconds measured from the moment the call is issued,
   THEN THE Extractor SHALL produce zero Extraction_Candidates, SHALL ignore any response
   that arrives after that deadline, SHALL leave the Extraction_Watermark unchanged so the
   Round is retried on the next trigger, and SHALL log one warning naming the cause.
7. WHEN an Extraction_Candidate omits `category`, or supplies a `category` that is not a
   string of 1 or more characters after trimming, THE Extractor SHALL leave the category
   unset so the Memory_Store applies its documented default, and THE Extractor SHALL pass
   through only those `tags` entries that are strings of 1 or more characters after
   trimming, discarding any other `tags` value.
8. WHEN an Extraction_Candidate supplies a `scope` equal to one of the defined Memory_Scope
   values, THE Extractor SHALL pass it through unchanged, and WHEN it omits `scope` or
   supplies a value that is not a defined Memory_Scope value, THE Extractor SHALL request
   Memory_Scope `global` so that a distilled fact outlives the session it came from.
9. IF an entry of the parsed JSON array is not a JSON object, or its `content` field is
   absent, is not a string, or is empty after trimming leading and trailing whitespace,
   THEN THE Extractor SHALL discard that entry, SHALL continue evaluating the remaining
   entries, and SHALL NOT abandon the extraction.
10. IF two or more surviving entries carry `content` values that are equal after trimming,
    collapsing each run of internal whitespace to one space, and case-folding, THEN THE
    Extractor SHALL retain only the earliest such entry and SHALL discard the later ones
    before the `memory_extraction_max_candidates` limit of criterion 4 is applied.
11. WHEN the LLM call returns a response and the Extractor produces zero
    Extraction_Candidates from it — whether the response was an empty array, an unusable
    response under criterion 5, or an array whose every entry was discarded under criteria
    9 and 10 — THE Extractor SHALL advance the Extraction_Watermark to the Turn_Count
    observed when the Transcript_Slice was selected, so that only an extraction with no
    returned response, as in criterion 6, holds the watermark for retry.

### Requirement 5: Persisting candidates through the existing store

**User Story:** As an operator, I want automatically extracted memories held to exactly the
same rules as manually saved ones, so that automation cannot bypass the safeguards.

#### Acceptance Criteria

1. WHEN the Extractor persists an Extraction_Candidate, THE Extractor SHALL call the
   existing Memory_Store save operation with the Thread_Id of the thread whose
   Transcript_Slice produced that Extraction_Candidate, SHALL NOT substitute the Thread_Id
   of the session that triggered the extraction, and SHALL NOT write to the Memory_Store
   file by any other path.
2. WHEN an Extraction_Candidate duplicates an existing Memory_Record as defined by the
   Memory_Store's duplicate rule, THE Memory_Store SHALL update that record rather than
   create a second one, and the total Memory_Record count SHALL remain unchanged.
3. IF an Extraction_Candidate's content matches a Secret_Pattern, THEN THE Memory_Store
   SHALL refuse it, THE Extractor SHALL count it as refused, and no part of the matched
   text SHALL appear in any log record.
4. IF an Extraction_Candidate violates a Memory_Store parameter limit, THEN THE
   Memory_Store SHALL refuse it, THE Extractor SHALL count it as refused, THE Extractor
   SHALL NOT retry that Extraction_Candidate within the same extraction, and THE Extractor
   SHALL continue with the remaining Extraction_Candidates rather than abandoning the
   extraction.
5. WHEN persisting Extraction_Candidates would exceed `memory_max_records`, THE
   Memory_Store SHALL apply its existing capacity eviction, and THE Extractor SHALL make no
   separate eviction decision.
6. WHEN an extraction persists Extraction_Candidates, THE Extractor SHALL attempt at most
   `memory_extraction_max_candidates` save operations for that extraction, SHALL attempt
   them one at a time with no second save operation in flight for that extraction, and
   SHALL attempt them in the order the Extraction_Candidates were produced, so that the
   Memory_Record ordering and any capacity eviction outcome are reproducible for a given
   candidate list.
7. WHEN the Extractor persists Extraction_Candidates, THE Extractor SHALL NOT consume the
   per-turn Memory_Tool call budget, so an extraction cannot exhaust the budget available
   to the model's own tool calls in a concurrent turn.
8. WHEN every Extraction_Candidate of an extraction is refused by the Memory_Store and no
   save operation failed for a reason other than refusal, THE Extractor SHALL treat that
   extraction as complete, SHALL advance the Extraction_Watermark to the Turn_Count
   observed when the Transcript_Slice was selected, and SHALL NOT re-extract that
   Transcript_Slice on a later trigger, so that a refusal is distinguished from the write
   failure of Requirement 6 criterion 7.
9. WHERE an Extraction_Candidate requests Memory_Scope `session`, THE Extractor SHALL
   request that scope with the scope identifier set to the Thread_Id whose
   Transcript_Slice produced that Extraction_Candidate, so that a candidate extracted from
   a Previous_Session under Extraction_Trigger `session_start` is never bound to the newly
   created session's Thread_Id.

### Requirement 6: Extraction never degrades a chat turn

**User Story:** As a chat user, I want extraction to be invisible, so that a slow or broken
extraction never delays or breaks my conversation.

#### Acceptance Criteria

1. WHEN an extraction is scheduled, THE Chat_Agent SHALL hand that extraction to a
   background worker that the request path producing the user's reply does not await, and
   on that request path THE Chat_Agent SHALL make zero LLM calls, zero checkpoint reads,
   and zero Memory_Store writes on behalf of the scheduled extraction.
2. IF an extraction raises any exception, THEN THE Chat_Agent SHALL return the current
   turn's reply, SHALL propagate no extraction exception to the HTTP client or the CLI
   REPL, SHALL continue to serve every subsequent turn for that Thread_Id, and THE Extractor
   SHALL log exactly one record at warning level naming the Extraction_Trigger, the
   Thread_Id, and the cause.
3. WHILE an extraction is running for one Thread_Id, IF an extraction is requested for that
   same Thread_Id, THEN THE Extractor SHALL discard the new request, SHALL make no LLM call
   for it, SHALL leave that thread's Extraction_Watermark unchanged, and SHALL log at debug
   level or not at all.
4. THE Extractor SHALL run at most `memory_extraction_max_concurrency` extractions
   concurrently in one process and SHALL hold at most zero pending extraction requests, so
   no unbounded queue of extraction requests can form.
5. WHEN the chat application shuts down, THE Chat_Agent SHALL complete shutdown without
   awaiting any in-flight extraction, SHALL add at most 1 second to shutdown duration on
   account of in-flight extractions, and SHALL abandon each in-flight extraction rather
   than interrupting a Memory_Store save operation already in progress.
6. WHILE `memory_extraction_enabled` is disabled, THE Extractor SHALL make zero LLM calls,
   SHALL make zero checkpoint reads for extraction, and SHALL leave the Memory_Record
   count, every Memory_Record's content, and every Extraction_Watermark value unchanged.
7. IF a Memory_Store save operation for an Extraction_Candidate fails, THEN THE Extractor
   SHALL leave that thread's Extraction_Watermark unchanged so the Round is retried on the
   next trigger, SHALL continue with the remaining Extraction_Candidates of that
   extraction, and SHALL log exactly one record at warning level naming the
   Extraction_Trigger, the Thread_Id, and the cause.
8. WHILE a scheduled extraction has started and has not completed, THE Chat_Agent SHALL
   return that turn's reply, so a test can hold an extraction blocked for an unbounded
   duration, observe the reply returned, and assert non-blocking behaviour without
   measuring elapsed time.
9. IF an extraction is requested while `memory_extraction_max_concurrency` extractions are
   already running, THEN THE Extractor SHALL discard the request, SHALL make no LLM call for
   it, SHALL leave that thread's Extraction_Watermark unchanged so the Round is retried on
   the next trigger, and SHALL log exactly one record at warning level naming the
   Extraction_Trigger, the Thread_Id, and the concurrency limit as the cause.
10. IF process exit or a restart interrupts an in-flight extraction, THEN each
    Extraction_Candidate of that extraction SHALL be either fully present in the
    Memory_Store or entirely absent from it with no partially written Memory_Record, and
    that thread's Extraction_Watermark SHALL retain its value from before the extraction so
    the same Round is re-extracted on the next trigger.

### Requirement 7: Watermark bookkeeping and idempotency

**User Story:** As a developer, I want each round extracted exactly once, so that repeated
triggers do not re-distil the same turns or duplicate records.

#### Acceptance Criteria

1. WHEN a process restart occurs, THE Extractor SHALL read for each Thread_Id an
   Extraction_Watermark equal to the last value successfully persisted for that Thread_Id
   before the restart.
2. THE Extractor SHALL persist the Extraction_Watermark in the existing session metadata
   `config` mapping under the single key `extraction_watermark` as a whole number in the
   range 0 to 1000000 inclusive, THE Extractor SHALL add no other key to that mapping, and
   the session metadata schema version SHALL remain 1 so that no database migration is
   required.
3. WHEN a session's metadata is saved for any reason, including a save that carries no
   extraction change, THE Chat_Agent SHALL leave that session's persisted
   `extraction_watermark` value equal to the value persisted immediately before that save,
   so an unrelated metadata save cannot reset it to 0 or drop the key.
4. WHEN an extraction completes successfully, THE Extractor SHALL set the
   Extraction_Watermark to the Turn_Count observed when the Transcript_Slice was selected,
   and IF that observed Turn_Count is less than the current Extraction_Watermark, THEN THE
   Extractor SHALL leave the Extraction_Watermark unchanged, so the value never decreases.
5. WHEN an extraction is triggered a second time for one Thread_Id while that thread's
   Turn_Count is unchanged from the first trigger, THE Extractor SHALL perform exactly one
   LLM call across both triggers, and the Memory_Record count after the second trigger
   SHALL equal the count after the first.
6. WHERE a thread's session metadata `config` mapping contains no `extraction_watermark`
   key, or no session metadata row exists for that Thread_Id, THE Extractor SHALL use an
   Extraction_Watermark of 0 for that thread and SHALL create no session metadata row as a
   result of that read.
7. IF persisting the Extraction_Watermark fails, THEN THE Extractor SHALL log exactly one
   warning naming the Thread_Id and the cause, SHALL leave every Memory_Record already
   persisted by that extraction unchanged, and THE Chat_Agent SHALL complete the current
   and every subsequent turn, accepting that the Round may be extracted again.
8. IF a thread's persisted `extraction_watermark` value is not a whole number, is a
   boolean, is negative, or is greater than 1000000, THEN THE Extractor SHALL use an
   Extraction_Watermark of 0 for that thread, SHALL log exactly one warning naming the
   Thread_Id and stating that the stored value was unusable, SHALL raise no exception into
   the chat turn, and SHALL replace the stored value with a valid whole number on the next
   successful extraction for that Thread_Id.
9. IF a thread's Extraction_Watermark is greater than that thread's current Turn_Count,
   THEN THE Extractor SHALL use that thread's current Turn_Count in place of the stored
   value when evaluating an Extraction_Trigger and when selecting a Transcript_Slice, SHALL
   persist that reduced value on the next successful extraction for that Thread_Id, and
   SHALL make no LLM call solely because the stored value exceeded the Turn_Count.

### Requirement 8: Configuration

**User Story:** As an operator, I want automatic extraction configured the same way as the
rest of the memory feature, so that enabling and tuning it needs no new mechanism.

#### Acceptance Criteria

1. THE Settings dataclass SHALL define `memory_extraction_enabled` as a boolean defaulting
   to `false`, `memory_extraction_turn_interval` as an integer defaulting to 10,
   `memory_extraction_max_candidates` as an integer defaulting to 5,
   `memory_extraction_max_transcript_chars` as an integer defaulting to 8000,
   `memory_extraction_timeout_seconds` as an integer defaulting to 60,
   `memory_extraction_max_concurrency` as an integer defaulting to 2,
   `memory_extraction_max_session_age_hours` as an integer defaulting to 168, and
   `memory_extraction_on_session_start` as a boolean defaulting to `true`, and SHALL add no
   further extraction field, so that a Settings instance built from defaults alone leaves
   extraction disabled.
2. THE configuration loader SHALL map each field listed in criterion 1 to the environment
   variable whose name is that field name in upper case with its underscores preserved
   (`memory_extraction_enabled` to `MEMORY_EXTRACTION_ENABLED`,
   `memory_extraction_turn_interval` to `MEMORY_EXTRACTION_TURN_INTERVAL`, and likewise for
   the remaining six fields), SHALL read each YAML value under a key equal to the field
   name, and SHALL apply the existing precedence of programmatic overrides over environment
   variables over YAML configuration over dataclass defaults.
3. THE repository SHALL document every field listed in criterion 1 in both `.env.example`
   and `config/default.yaml`, in each case stating the field's default value from
   criterion 1, and SHALL state for each of the six integer fields the accepted range given
   in criterion 4.
4. IF `memory_extraction_turn_interval` is configured outside the range 1 to 1000,
   `memory_extraction_max_candidates` outside the range 1 to 20,
   `memory_extraction_max_transcript_chars` outside the range 200 to 100000,
   `memory_extraction_timeout_seconds` outside the range 1 to 600,
   `memory_extraction_max_concurrency` outside the range 1 to 16, or
   `memory_extraction_max_session_age_hours` outside the range 1 to 8760, THEN THE
   configuration loader SHALL fail to load, SHALL return no Settings instance, SHALL clamp
   no value to the nearest bound, and SHALL report the field name, the offending value, and
   the accepted range, consistent with the existing memory bounds policy.
5. IF `memory_extraction_turn_interval`, `memory_extraction_max_candidates`,
   `memory_extraction_max_transcript_chars`, `memory_extraction_timeout_seconds`,
   `memory_extraction_max_concurrency`, or `memory_extraction_max_session_age_hours` is
   configured with a value that is not parseable as an integer, including an empty value, a
   decimal value, and a non-numeric value, THEN THE configuration loader SHALL fail to
   load, SHALL return no Settings instance, and SHALL report the field name and the
   offending value, matching the reporting required by criterion 4.
6. WHILE `memory_enabled` is disabled, THE Extractor SHALL make no LLM call, SHALL read no
   checkpoint for extraction, and SHALL leave every Memory_Record and every
   Extraction_Watermark unchanged, even when `memory_extraction_enabled` is enabled.
7. WHILE `memory_extraction_on_session_start` is disabled, THE Extractor SHALL schedule no
   extraction with Extraction_Trigger `session_start`, SHALL advance no
   Extraction_Watermark on session creation, and SHALL continue to apply Requirement 2 for
   Extraction_Trigger `round_complete`.
8. WHEN `memory_extraction_enabled` or `memory_extraction_on_session_start` is supplied as
   an environment variable value or a YAML value, THE configuration loader SHALL treat that
   value as enabled if, after trimming leading and trailing whitespace and case folding, it
   equals one of `true`, `1`, `yes`, or `on`, SHALL treat every other value, including the
   empty string, as disabled, and SHALL load without failing for any supplied value.
9. THE configuration loader SHALL treat `memory_extraction_max_transcript_chars` and
   `memory_max_record_chars` as independent fields, SHALL enforce no ordering or ratio
   relationship between them, and SHALL load without failing for every combination in which
   each value is inside its own accepted range, so that
   `memory_extraction_max_transcript_chars` bounds only the Transcript_Slice fed to one LLM
   call while `memory_max_record_chars` remains the sole bound the Memory_Store applies to
   each persisted Memory_Record.
10. THE Settings dataclass SHALL define no extraction-specific model, provider, or sampling
    field in this version, and THE Extractor SHALL use the chat model the Chat_Agent is
    already configured with, so that a dedicated extraction model setting is out of scope
    for this version.

### Requirement 9: Privacy and safety

**User Story:** As an operator, I want automatic extraction to leak nothing, so that
distilling transcripts does not turn the memory file or the logs into a data-exposure
surface.

#### Acceptance Criteria

1. WHEN the Extractor logs any event, THE Extractor SHALL include the Extraction_Trigger,
   the Thread_Id, the Transcript_Slice message count, the candidate count, the persisted
   count, and the refused count as separate fields, and SHALL include no substring of 20 or
   more consecutive characters drawn from Transcript_Slice text, from Extraction_Candidate
   content, from Memory_Record content, or from an LLM response body.
2. WHEN the Extractor persists an Extraction_Candidate, THE Extractor SHALL submit that
   candidate's content unmodified to the existing Memory_Store save operation so that the
   existing Secret_Pattern check runs before the record is written, and SHALL apply no
   Secret_Pattern decision of its own that admits content the Memory_Store would refuse.
3. THE Extractor SHALL derive the Memory_Store path from Settings only, and IF a
   Transcript_Slice, an LLM response, or a request parameter supplies a path, a file name,
   or any other Memory_Store location value, THEN THE Extractor SHALL treat that value as
   ordinary text and SHALL read and write no location other than the Settings-derived one.
4. WHEN the Extractor builds its prompt, THE Extractor SHALL include exactly two parts —
   instruction text that is fixed at build time and identical for every extraction, and the
   Transcript_Slice enclosed in a delimiter that labels it as untrusted conversation data —
   and SHALL include no Memory_Record content, no Memory_Store path, and no Settings value,
   so an extraction cannot echo an existing memory or a configured value back into a new
   Memory_Record.
5. IF an Extraction_Candidate contains content that reads as an instruction to the agent,
   THEN THE Memory_Store SHALL store that content as ordinary record content with no
   special interpretation, and THE Extractor SHALL neither act on that content nor change
   any other Extraction_Candidate, Memory_Record, or Setting because of it.
6. IF the Transcript_Slice contains text directing the agent to save stated content, to
   modify or delete a Memory_Record, to disclose Memory_Store or Settings values, or to
   disregard the Extractor's instruction text, THEN THE Extractor SHALL still produce at
   most `memory_extraction_max_candidates` Extraction_Candidates, SHALL still route every
   candidate through the existing Memory_Store save operation, SHALL invoke no Memory_Store
   delete or forget operation, and SHALL leave every Setting unchanged for that extraction
   and every later extraction.
7. THE Extractor SHALL exclude from every Transcript_Slice each message that the Chat_Agent
   injected to carry recalled Memory_Record content, regardless of the message role under
   which that message is checkpointed, and WHERE the Chat_Agent injects recalled
   Memory_Record content into a turn, THE Chat_Agent SHALL mark that message so the
   exclusion is decidable without inspecting the message content, so that a prior turn's
   injected recall note cannot be re-extracted into a new Memory_Record.
8. WHEN the Extractor persists an Extraction_Candidate, THE Extractor SHALL request a
   Provenance_Tag on that Memory_Record identifying it as automatically extracted and
   naming the Extraction_Trigger, within the Memory_Store's existing tag count and tag
   length limits, and WHEN the Memory_Store updates an existing Memory_Record from an
   Extraction_Candidate, THE Extractor SHALL retain that Provenance_Tag on the updated
   record, so that every automatically created Memory_Record can be listed and removed as a
   group without inspecting its content.

### Requirement 10: Observability

**User Story:** As a developer, I want to see when extraction ran and what it decided, so
that I can tell a quiet extraction from a broken one.

#### Acceptance Criteria

1. WHEN an extraction completes without raising, THE Extractor SHALL log exactly one record
   at informational level, and no more than one, containing the Extraction_Trigger, the
   Thread_Id, the Transcript_Slice message count, the candidate count, the persisted count,
   the refused count, the resulting Memory_Record count, the Extraction_Watermark value
   observed when the Transcript_Slice was selected, the Extraction_Watermark value in effect
   after the extraction, and the elapsed wall-clock duration of the extraction expressed as
   a whole number of milliseconds.
2. WHEN an extraction completes and produces zero Extraction_Candidates, THE Extractor
   SHALL report that outcome within the single informational record required by criterion 1
   by recording a candidate count of 0 together with an outcome indicator distinguishing
   "no durable information found" from "failed", and SHALL log no additional record at
   informational level or above for that extraction, so that a completed extraction yields
   exactly one informational record whether or not candidates were found.
3. IF an extraction fails, raises, or times out, THEN THE Extractor SHALL log exactly one
   record at warning level containing the Extraction_Trigger, the Thread_Id, the exception
   type name, a cause description truncated to at most 200 characters, and whether the
   Extraction_Watermark was left unchanged, SHALL log no informational completion record for
   that same extraction attempt, and SHALL include no Transcript_Slice text,
   Extraction_Candidate content, or Memory_Record content in that record.
4. WHEN a chat turn completes or a session is created and no Extraction_Trigger condition is
   met, THE Extractor SHALL log zero records at informational level or above and SHALL log
   at most one record at debug level for that turn or session creation, so that normal turns
   produce no log noise above debug level.
5. THE Extractor SHALL emit every diagnostic through the `logging` module rather than
   through `print`.
6. THE Extractor SHALL emit every extraction diagnostic through one fixed logger name that
   is the same for all Extraction_Triggers and SHALL begin the message text of every
   extraction log record with the fixed prefix `memory_extraction`, so that a single
   case-sensitive text filter selects all extraction activity and no non-extraction record
   matches that prefix.
7. WHEN the Extractor logs the record required by criterion 1, THE Extractor SHALL report
   the candidate count, the persisted count, and the refused count as non-negative integers
   where the candidate count is between 0 and `memory_extraction_max_candidates` inclusive
   and the persisted count plus the refused count equals the candidate count, and SHALL
   report the Extraction_Watermark value after the extraction as greater than or equal to
   the value observed before it.

### Requirement 11: Backward compatibility

**User Story:** As an operator, I want to adopt automatic extraction without risk, so that
leaving it off keeps the system exactly as it is today.

#### Acceptance Criteria

1. WHILE `memory_extraction_enabled` is disabled, WHEN a turn is executed, THE Chat_Agent
   SHALL pass a graph input whose message list, injected memory note text, registered tool
   set, and tool names are identical to those the Chat_Agent produced for the same session
   state before this feature existed, with zero added, removed, or reordered messages.
2. WHILE `memory_extraction_enabled` is disabled, THE Chat_Agent SHALL make exactly zero
   additional LLM calls, exactly zero additional checkpoint reads, and exactly zero
   additional Memory_Store writes per turn and per session creation, relative to the counts
   observed with this feature's code absent.
3. THE existing explicit `save_memory`, `recall_memory`, and `forget_memory` tools SHALL
   remain callable by the model under their current names and parameter names, and SHALL
   retain their current behaviour — the same validation order, the same duplicate rule, the
   same refusal outcomes, and the same returned results as specified in the
   `chat-long-term-memory` spec — so automatic extraction supplements rather than replaces
   them.
4. WHEN the existing automatic recall injection runs, THE Chat_Agent SHALL leave every
   Memory_Record unchanged, SHALL leave each matched record's `last_recalled_at` at its
   prior value, and SHALL leave the total Memory_Record count unchanged, whether
   `memory_extraction_enabled` is enabled or disabled.
5. WHEN `memory_extraction_enabled` is enabled against a checkpoint store that already
   contains threads, THE Chat_Agent SHALL start at most one `session_start` extraction per
   session creation and at most one `round_complete` extraction per completed Round, and
   SHALL take each thread's retained Extraction_Watermark as the starting point for slice
   selection, so adoption produces no burst of LLM calls proportional to the existing thread
   count and no re-extraction of turns already extracted.
6. THE ChatSession field that carries the Extraction_Watermark SHALL be an optional field
   defaulting to 0, and THE Chat_Agent SHALL leave every pre-existing ChatSession field
   name, type, and required-or-optional status unchanged, so that any existing caller that
   constructs a ChatSession or reads session metadata without knowledge of that field
   continues to succeed with no error.
7. IF session metadata persisted before this feature existed is loaded, THEN THE Chat_Agent
   SHALL load it without error, SHALL preserve every value it already contained unchanged
   after the next save, and SHALL treat that session's Extraction_Watermark as 0.
8. WHILE `memory_extraction_enabled` is disabled after having previously been enabled, THE
   Chat_Agent SHALL retain each persisted Extraction_Watermark at its last successfully
   persisted value, SHALL neither advance nor reset it, and SHALL keep that session loadable
   and savable with no error.

## Open Questions / Assumptions

Recorded as initial decisions; confirm before the design phase.

1. **Turn counting uses the checkpoint, not a counter.** Turn_Count is the number of
   `HumanMessage` entries in the thread's checkpointed message list, so it cannot drift from
   the transcript. Only the Extraction_Watermark is stored. Confirm this over adding an
   independent per-session counter.
2. **Watermark lives in `SessionMetadata.config` under `extraction_watermark`.** That
   mapping is already free-form JSON, so no schema migration is needed and the schema
   version stays 1. It does require a matching field on `ChatSession`, because
   `SessionMetadata.from_session` rebuilds `config` on every save and would otherwise drop
   the value. Confirm.
3. **One `session_start` extraction per session creation, most-recent eligible session
   only.** The checkpoint store already holds 115 threads; extracting every un-extracted
   thread on first enable would fire 115 LLM calls. Sessions older than
   `memory_extraction_max_session_age_hours` (default 7 days) are marked extracted without a
   call. Confirm this bound, and confirm that silently skipping an old backlog is
   acceptable.
4. **Extraction runs on a bounded background worker, fire-and-forget, with no queue.** The
   reply is never delayed; a request arriving while the concurrency limit is reached is
   discarded rather than queued, and an in-flight extraction is abandoned at shutdown. Both
   cases are recovered by the watermark staying put, so the Round retries on the next
   trigger. Confirm this over a durable job queue.
5. **Scope defaults to `global` for extracted memories.** A fact distilled from a
   conversation is meant to outlive it, so extraction requests `global` even when
   `memory_default_scope` is configured to `session`. A `session`-scoped candidate is bound
   to the *extracted* thread, not the triggering one. Confirm.
6. **One LLM call per extraction on the existing chat model.** No dedicated extraction
   model, provider, or sampling setting in this version (Requirement 8 criterion 10). At
   defaults this is one extra call per 10 turns plus one per new session. Confirm.
7. **Only a missing response holds the watermark.** An empty array, an unparseable
   response, an all-discarded array, and an all-refused candidate set all count as
   completed and advance the watermark. Only a raised exception, a timeout, a checkpoint
   read failure, or a store write failure holds it for retry. This prevents a permanently
   un-extractable slice from blocking every later Round. Confirm.
8. **Deleted sessions are not extracted.** Their transcripts survive in the checkpoint
   store because `ChatSessionRegistry.delete` does not call `checkpointer.delete_thread`,
   but a user who deleted a conversation most likely does not want it distilled. Confirm.
9. **Injected recall notes must become identifiable.** Requirement 9 criterion 7 requires
   the Chat_Agent to mark the injected memory `SystemMessage` so extraction can exclude it
   without content inspection. This is a **change to the existing memory feature**, not
   purely additive. The current implementation already excludes all system messages, so the
   mark is defence against a future change in how notes are injected. Confirm whether to
   implement the mark now or rely on the role-based exclusion alone.
10. **Extracted records carry a Provenance_Tag.** Requirement 9 criterion 8 adds a tag
    naming the trigger, consuming one of the 10 available tag slots, so extracted memories
    can be audited or bulk-removed. Confirm, and confirm the tag text.
11. **No user-facing signal or review surface.** Extraction is silent; the user is not told
    that memory changed, and there is no endpoint to review or undo an extraction. Confirm,
    or a follow-up spec should add a review surface.
