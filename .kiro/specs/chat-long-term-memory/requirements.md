# Requirements Document

## Introduction

This feature gives the chat agent in the langgraph-rag project **long-term memory that
survives across chat sessions and server restarts**, persisted as a JSON file on the
local filesystem.

Today the chat agent has only *short-term* state: the LangGraph checkpoint
(`.chroma/chat/checkpoints.sqlite3`) holds the transcript for one `thread_id`, and
`SQLiteStorage` (`.chroma/chat/sessions.sqlite3`) holds session metadata (source URLs,
source mode, timestamps). Nothing carries a durable user fact or preference from one
`thread_id` to the next. Asking "what did I tell you my name was?" in a new session
fails.

The feature is purely **additive** and follows the established tool-authoring playbook in
`src/tools/SKILL.md`:

- A new module `src/tools/memory_tool.py` exposes agent-callable `StructuredTool`s built
  by `build_*_tool` factories that take `Settings` and an injectable seam for tests.
- A new `memory_enabled` feature flag in `Settings`, mapped in
  `src/config/loader.py::SETTING_ENV_NAMES`, documented in `.env.example` and
  `config/default.yaml`, defaulting to disabled so existing deployments are unchanged.
- Wiring in **both** tool-resolution sites in `src/graph/builder.py` (`_resolve_tools`
  and `_resolve_lightweight_tools`).
- A tool listing in `AGENT_SYSTEM_PROMPT` (`src/llm/prompts.py`) so the model knows the
  memory tools exist and when to call them.
- Automatic recall injected as a `SystemMessage` at the start of a chat turn, reusing the
  pattern already proven by the per-session upload-context note in
  `src/chat/api.py::_graph_inputs_for_turn` and `src/chat/uploads.py`.

Because the memory file is written by an agent loop and read back on every turn, this
document treats the JSON store as a **serializer/parser pair** and requires an explicit
round-trip property, per project testing guidance.

This document records initial decisions for the open design questions (tool surface,
default store location, scope model, and whether memories are written only by explicit
tool calls) as explicit assumptions. They are listed in the
"Open Questions / Assumptions" section and should be confirmed before the design phase.

## Glossary

- **Chat_Agent**: The existing LangGraph chat agent (`build_chat_graph` and
  `build_lightweight_graph` with `mode="chat"`) that serves the chat REPL and the chat
  FastAPI app.
- **Memory_Record**: One durable unit of remembered information (a fact, preference,
  entity, or task) with its identifier, scope, category, content, tags, and timestamps.
- **Memory_Store**: The JSON document on disk that holds a schema version and the list
  of Memory_Records, plus the code that owns reading and writing that document.
- **Memory_Serializer**: The component that converts in-memory Memory_Records into the
  JSON text of the Memory_Store.
- **Memory_Parser**: The component that converts the JSON text of the Memory_Store back
  into in-memory Memory_Records.
- **Memory_Tool**: Any one of the agent-callable LangChain `StructuredTool`s that the
  Chat_Agent invokes to write, read, or delete Memory_Records.
- **save_memory_tool**: The Memory_Tool that creates or updates one Memory_Record.
- **recall_memory_tool**: The Memory_Tool that returns the Memory_Records most relevant
  to a query string.
- **forget_memory_tool**: The Memory_Tool that deletes Memory_Records by identifier or by
  matching text.
- **Memory_Recall_Injector**: The chat-turn component that selects relevant
  Memory_Records and adds them to the turn as a `SystemMessage`, analogous to the
  existing upload-context note.
- **Memory_Scope**: The visibility of a Memory_Record, one of `global` (visible to every
  chat session) or `session` (visible only to the session whose `thread_id` matches the
  record's scope identifier).
- **Thread_Id**: The existing per-session identifier used by `ChatSessionRegistry` and
  the LangGraph checkpointer.
- **Settings**: The existing frozen configuration dataclass in `src/config/settings.py`.
- **Secret_Pattern**: A textual pattern that indicates credential material. The
  enumerated pattern set and its matching rules are defined in Requirement 9
  criterion 5.
- **Normalized_Content**: A Memory_Record's content after Unicode case folding,
  collapsing each run of whitespace to a single space, and trimming leading and trailing
  whitespace. Used for duplicate detection.
- **Relevance_Score**: The count of distinct query terms of 2 or more characters that
  appear in a Memory_Record's Normalized_Content or tags, used to rank recall results.
  Query-term derivation is defined in Requirement 3 criterion 1.

## Requirements

### Requirement 1: Memory tools available to the chat agent

**User Story:** As a chat user, I want the agent to have memory tools it can call, so
that it can remember and reuse what I tell it across sessions.

#### Acceptance Criteria

1. WHILE the `memory_enabled` setting is enabled, WHEN a Chat_Agent graph is built, THE
   Chat_Agent SHALL have exactly three Memory_Tools bound to it, named `save_memory`,
   `recall_memory`, and `forget_memory`, and SHALL leave the set of non-memory tools
   bound to that graph unchanged from the set bound when `memory_enabled` is disabled.
2. WHILE the `memory_enabled` setting is disabled, WHEN a Chat_Agent graph is built, THE
   Chat_Agent SHALL have zero Memory_Tools bound to it, and no tool name bound to that
   graph SHALL be `save_memory`, `recall_memory`, or `forget_memory`.
3. WHILE the `memory_enabled` setting is enabled, THE Chat_Agent SHALL expose the same
   three Memory_Tool names and the same three Memory_Tool input schemas on both the
   retrieval graph built by `_resolve_tools` and the lightweight web-search graph built
   by `_resolve_lightweight_tools`.
4. THE Chat_Agent system prompt SHALL contain each of the three Memory_Tool names, and
   for each name SHALL contain one description of 1 to 300 characters that states the
   condition under which the Chat_Agent calls that tool.
5. THE Memory_Tools SHALL each declare a Pydantic input schema that names every accepted
   parameter, marks each required parameter as required, rejects any parameter name not
   declared in that schema, and constrains each string parameter to a minimum length of 1
   character and to a declared maximum length of 1 or more characters.
6. WHEN a user message states, in the first person, the user's name, a preference, an
   attribute of the user, or an ongoing task, and does not limit that statement to the
   current turn, THE Chat_Agent SHALL call save_memory_tool at least once for that
   statement before emitting its final assistant message for that turn.
7. WHEN a user message asks for information about the user that does not appear in the
   current turn's message list, THE Chat_Agent SHALL call recall_memory_tool at least
   once before emitting its final assistant message for that turn.
8. WHILE the `memory_enabled` setting is disabled, THE Chat_Agent SHALL read no bytes
   from, write no bytes to, and create no file at the configured Memory_Store path.
9. WHILE the `memory_enabled` setting is enabled, THE Chat_Agent SHALL make at most 10
   Memory_Tool calls per chat turn, and IF a turn reaches that limit, THEN THE Chat_Agent
   SHALL emit its final assistant message for that turn without a further Memory_Tool
   call.

### Requirement 2: Saving a memory

**User Story:** As a chat user, I want the agent to store a fact or preference I state,
so that it does not ask me the same thing again in a later session.

#### Acceptance Criteria

1. WHEN save_memory_tool is called with content whose length after trimming leading and
   trailing whitespace is 1 to `memory_max_record_chars` characters and a category from
   the set {`fact`, `preference`, `entity`, `task`}, THE Memory_Store SHALL persist one
   Memory_Record whose content is that trimmed content, whose identifier is a
   32-character lowercase hexadecimal string unique among all persisted Memory_Records,
   and SHALL return a confirmation string that contains that identifier.
2. WHEN save_memory_tool is called with the category parameter omitted, null, or empty
   after trimming, THE Memory_Store SHALL persist the Memory_Record with category `fact`.
3. WHEN save_memory_tool is called with 0 to 10 tags, THE Memory_Store SHALL trim leading
   and trailing whitespace from each tag, SHALL discard every tag that is empty after
   trimming, SHALL keep only the first occurrence of tags whose trimmed values are equal
   after Unicode case folding, and SHALL persist the surviving tags, each of 1 to 40
   characters, in the order supplied.
4. WHEN save_memory_tool is called with content whose Normalized_Content, Memory_Scope,
   scope identifier, and category all match an existing Memory_Record, THE Memory_Store
   SHALL replace that existing Memory_Record's tag list with the tag list produced by
   criterion 3 and SHALL set its `updated_at` timestamp to the time of the call, SHALL
   leave that record's identifier, content, `created_at`, `last_recalled_at`,
   Memory_Scope, scope identifier, and category unchanged, SHALL leave the total
   Memory_Record count unchanged, and SHALL return a confirmation string that contains
   the existing identifier.
5. WHEN save_memory_tool persists a new Memory_Record, THE Memory_Store SHALL record
   `created_at` and `updated_at` as equal ISO-8601 timestamps in UTC with second or finer
   precision, and SHALL record `last_recalled_at` as null until that record is returned
   by a recall.
6. IF save_memory_tool is called with content that is omitted, null, empty, or consists
   only of whitespace, THEN THE Memory_Store SHALL leave the persisted Memory_Records
   unchanged and SHALL return an error string that identifies the content parameter as
   the cause.
7. IF save_memory_tool is called with content longer than `memory_max_record_chars`
   characters after trimming, with more than 10 tags supplied, with any single tag longer
   than 40 characters after trimming, or with a category outside the set {`fact`,
   `preference`, `entity`, `task`}, THEN THE Memory_Store SHALL leave the persisted
   Memory_Records unchanged and SHALL return an error string that names the parameter
   that violated its limit.
8. IF save_memory_tool is called with two or more parameter values that violate criterion
   6, criterion 7, or Requirement 10 criterion 6, THEN THE Memory_Store SHALL evaluate
   the parameters in the fixed order content presence, content length, category value,
   scope value, tag count, tag length, and SHALL return an error string that names only
   the first violating parameter in that order.

### Requirement 3: Recalling memories

**User Story:** As a chat user, I want the agent to look up what it remembers about me,
so that its answers reflect what I told it before.

#### Acceptance Criteria

1. WHEN recall_memory_tool is called with a query of 1 to 500 characters, THE
   Memory_Store SHALL derive the query terms by applying the Normalized_Content
   normalization to the query, splitting the normalized query on single spaces, and
   keeping the first 50 distinct resulting terms of 2 or more characters, and SHALL
   return the Memory_Records in scope for the current turn's Thread_Id whose
   Relevance_Score is 1 or greater, ordered by descending Relevance_Score, then by
   descending `updated_at`, then by ascending identifier, and limited to at most
   `memory_recall_top_k` records.
2. WHEN recall_memory_tool returns 1 or more Memory_Records, THE Memory_Store SHALL
   include each returned record's identifier, category, content, and `updated_at`
   timestamp in the returned string, SHALL exclude every other Memory_Record field from
   that string, and SHALL limit the returned string to at most
   `memory_context_max_chars` characters by dropping whole record entries from the end of
   the ordering defined in criterion 1 until the limit is met, so that the returned
   string contains no partial record entry.
3. WHEN recall_memory_tool is called with a query from which zero query terms of 2 or
   more characters are derived, or for which zero in-scope Memory_Records have a
   Relevance_Score of 1 or greater, THE Memory_Store SHALL return a string stating that
   no matching memory was found and SHALL leave every persisted Memory_Record field
   unchanged.
4. WHEN recall_memory_tool is called twice with the same query and the same Thread_Id and
   no intervening save_memory_tool or forget_memory_tool call changes any Memory_Record,
   THE Memory_Store SHALL return identical strings for both calls, and the
   `last_recalled_at` value written by the first call SHALL change neither returned
   string nor the ordering defined in criterion 1.
5. WHEN recall_memory_tool returns 1 or more Memory_Records, THE Memory_Store SHALL set
   each returned record's `last_recalled_at` field to an ISO-8601 timestamp in UTC with
   second or finer precision equal to the time of that call, and SHALL leave each
   returned record's identifier, scope, scope identifier, category, content, tags,
   `created_at`, and `updated_at` values unchanged.
6. IF recall_memory_tool is called with a query that is empty, consists only of
   whitespace, or is longer than 500 characters, THEN THE Memory_Store SHALL return an
   error string that identifies the query parameter as the cause, SHALL leave every
   persisted Memory_Record unchanged, and SHALL set no `last_recalled_at` value.
7. WHEN recall_memory_tool sets one or more `last_recalled_at` values, THE Memory_Store
   SHALL persist the updated Memory_Store document before recall_memory_tool returns,
   SHALL leave the total Memory_Record count unchanged, and SHALL prune no
   Memory_Record.
8. IF reading the Memory_Store fails while recall_memory_tool runs, THEN THE Memory_Store
   SHALL return an error string naming the read failure as the cause, SHALL leave the
   Memory_Store file content unchanged, and SHALL set no `last_recalled_at` value.

### Requirement 4: Forgetting memories

**User Story:** As a chat user, I want to tell the agent to forget something, so that I
stay in control of what it keeps about me.

#### Acceptance Criteria

1. WHEN forget_memory_tool is called with an identifier of exactly 32 hexadecimal
   characters that matches an in-scope Memory_Record, THE Memory_Store SHALL delete that
   Memory_Record, SHALL ignore any query parameter supplied in the same call, and SHALL
   return a confirmation string containing the deleted identifier.
2. WHEN forget_memory_tool is called with a query of 1 to 500 characters and no
   identifier, THE Memory_Store SHALL delete the in-scope Memory_Records whose
   Relevance_Score is 1 or greater, selecting them in the order defined in Requirement 3
   criterion 1 and stopping after 10 deleted records in one call, and SHALL return a
   confirmation string stating the number of records deleted.
3. WHEN forget_memory_tool deletes a Memory_Record, THE Memory_Store SHALL leave every
   other Memory_Record's identifier, content, category, tags, and `created_at` timestamp
   unchanged.
4. WHEN forget_memory_tool is called twice with the same identifier and no intervening
   write, THE Memory_Store SHALL return a confirmation string on the first call and a
   not-found string on the second call, and the persisted Memory_Record count after the
   second call SHALL equal the count after the first call.
5. IF forget_memory_tool is called with neither an identifier nor a query, where a
   parameter that is absent, empty, or consists only of whitespace counts as not
   supplied, THEN THE Memory_Store SHALL leave the persisted Memory_Records unchanged and
   SHALL return an error string stating that one of the two parameters is required.
6. IF forget_memory_tool is called with an identifier of exactly 32 hexadecimal
   characters that matches no in-scope Memory_Record, including the case where it matches
   only a Memory_Record that is out of scope for the current Thread_Id, THEN THE
   Memory_Store SHALL leave every persisted Memory_Record unchanged and SHALL return a
   string stating that the identifier was not found.
7. WHEN forget_memory_tool is called with a query of 1 to 500 characters and no
   identifier for which zero in-scope Memory_Records have a Relevance_Score of 1 or
   greater, THE Memory_Store SHALL leave every persisted Memory_Record unchanged and
   SHALL return a string stating that no matching memory was found.
8. IF forget_memory_tool is called with a non-empty identifier that is not exactly 32
   hexadecimal characters, or with a query longer than 500 characters, THEN THE
   Memory_Store SHALL leave the persisted Memory_Records unchanged and SHALL return an
   error string naming the parameter that violated its limit.

### Requirement 5: JSON store format and round-trip fidelity

**User Story:** As a developer, I want the memory file to be a documented JSON schema
with a verified round-trip, so that memories are not silently lost or corrupted between
runs.

#### Acceptance Criteria

1. THE Memory_Serializer SHALL write the Memory_Store as one UTF-8 encoded JSON document
   whose top-level value is a JSON object containing exactly three keys: `version`
   holding the integer 1, `updated_at` holding an ISO-8601 timestamp in UTC with second
   or finer precision, and `records` holding an array of 0 to `memory_max_records` record
   objects.
2. THE Memory_Serializer SHALL write each Memory_Record as a JSON object containing
   exactly the fields `id` (32-character hexadecimal string), `scope` (string equal to
   `global` or `session`), `scope_id` (string of 1 to 200 characters when `scope` is
   `session`, JSON null when `scope` is `global`), `category` (string from the set
   {`fact`, `preference`, `entity`, `task`}), `content` (string of 1 to
   `memory_max_record_chars` characters), `tags` (array of 0 to 10 strings, each 1 to 40
   characters), `created_at` and `updated_at` (ISO-8601 UTC timestamp strings), and
   `last_recalled_at` (ISO-8601 UTC timestamp string, or JSON null when the record has
   never been returned by a recall).
3. WHEN a Memory_Store document produced by the Memory_Serializer is read by the
   Memory_Parser and written again by the Memory_Serializer, THE Memory_Store SHALL
   produce a document whose `version` value and whose `records` array are equal, field by
   field including JSON null values and in the same order, to those of the original
   document, and this SHALL hold for a document whose `records` array is empty, a document
   containing records whose `scope_id` value is JSON null, a document containing records
   whose `last_recalled_at` value is JSON null, and a document containing
   `memory_max_records` records.
4. WHEN the Memory_Parser reads a record object that contains keys outside the field list
   in criterion 2, THE Memory_Parser SHALL ignore those keys, SHALL retain every
   recognized field of that record, and SHALL omit the ignored keys from every document
   the Memory_Serializer writes afterwards.
5. IF the Memory_Parser reads a record object in which `id`, `content`, or `scope` is
   absent, is JSON null, is a string consisting only of whitespace, or holds a JSON type
   other than string, or in which `scope` holds a string outside the set {`global`,
   `session`}, THEN THE Memory_Parser SHALL exclude that record from the returned
   Memory_Records, SHALL retain every well-formed record from the same document, and SHALL
   log one warning naming the excluded record's zero-based position in the `records` array
   and the offending field name, without including the record's content.
6. IF the Memory_Store file content is not parseable as JSON or its top-level value is not
   a JSON object, THEN THE Memory_Store SHALL rename the file by appending
   `.corrupt-<UTC timestamp>` to its name, SHALL continue with an empty Memory_Record
   list, SHALL log one warning naming the renamed file path, and SHALL return to the
   caller without raising an exception.
7. IF the Memory_Parser reads a document whose `version` field is absent, holds a JSON
   type other than integer, or holds an integer greater than 1, THEN THE Memory_Store
   SHALL return an empty Memory_Record list, SHALL leave the file unchanged, and SHALL log
   one warning naming the document's `version` value and the value 1.
8. IF the Memory_Parser reads a record object in which the value of `scope_id`,
   `category`, `tags`, `created_at`, `updated_at`, or `last_recalled_at` holds a JSON type
   other than the type stated for that field in criterion 2, THEN THE Memory_Parser SHALL
   discard that value, SHALL substitute `fact` for `category`, an empty array for `tags`,
   JSON null for `scope_id` and `last_recalled_at`, and the document's `updated_at` value
   for `created_at` and `updated_at`, SHALL retain that record in the returned
   Memory_Records, and SHALL log one warning naming the record's zero-based position and
   the offending field name.
9. IF the Memory_Store file does not exist when the Memory_Parser is asked to read it,
   THEN THE Memory_Store SHALL return an empty Memory_Record list, SHALL create no file,
   and SHALL log no warning.

### Requirement 6: Durable and atomic persistence

**User Story:** As a chat user, I want my remembered facts to survive a restart and
concurrent turns, so that memory is dependable rather than best-effort.

#### Acceptance Criteria

1. WHEN a Memory_Tool call changes any Memory_Record, THE Memory_Store SHALL write the
   complete Memory_Store document to disk before that Memory_Tool returns.
2. WHEN the Memory_Store writes the Memory_Store document, THE Memory_Store SHALL write
   the full content to a temporary file in the same directory as the target file, whose
   name is distinct from the target file name and from the temporary file name of every
   other write in flight, and SHALL then replace the target file with that temporary file
   in one operation that succeeds whether or not the target file already exists, so that a
   reader observes either the previous complete document or the new complete document and
   never a partially written document.
3. WHEN the Chat_Agent process restarts and the Memory_Store file exists, THE Memory_Store
   SHALL return every Memory_Record that was persisted before the restart, with each field
   value of each record equal to the value persisted before the restart, on the first read
   after the restart.
4. WHILE two or more chat turns run concurrently in one process, THE Memory_Store SHALL
   admit at most one read-modify-write operation at a time, SHALL apply each admitted
   operation to the document state left by the previously completed operation, and SHALL
   make each completed Memory_Tool write present in the persisted document; serialization
   across separate operating-system processes writing the same target path is outside this
   criterion, and the last completed write supersedes earlier concurrent cross-process
   writes.
5. WHEN the Memory_Store writes to a target path whose parent directory does not exist,
   THE Memory_Store SHALL create the missing parent directories before writing.
6. IF the Memory_Store cannot write the Memory_Store document because the path is not
   writable, is a directory, or the filesystem reports an error that is still present
   after the retry attempts defined in criterion 8, THEN THE Memory_Tool SHALL return an
   error string that states that the memory write failed and names the failure cause as
   one of not-writable path, path is a directory, or filesystem error, THE Memory_Store
   SHALL leave the previously persisted document unchanged, and the Chat_Agent SHALL
   continue serving the current turn and every subsequent turn.
7. IF a write fails after the temporary file is created, THEN THE Memory_Store SHALL
   delete the temporary file, SHALL leave the previous target file content unchanged, and
   SHALL log one warning naming the temporary file path if that deletion also fails.
8. IF the replace operation in criterion 2 fails with an error that indicates the target
   file is temporarily locked or shared by another reader or writer, THEN THE Memory_Store
   SHALL retry that replace operation at most 3 further times, waiting 50 milliseconds
   before the first retry and doubling the wait before each subsequent retry so that the
   total added wait is at most 350 milliseconds, and SHALL treat the operation as a write
   failure per criterion 6 if the last retry also fails.
9. IF the Memory_Store file does not exist when the Memory_Store reads it, THEN THE
   Memory_Store SHALL return an empty Memory_Record list, SHALL create no file and no
   parent directory during that read, and SHALL return no error string to the Chat_Agent.
10. WHERE the Memory_Store keeps an in-memory copy of the Memory_Store document between
    chat turns, THE Memory_Store SHALL discard that copy and re-read the file before every
    read-modify-write operation and whenever the target file's last-modification timestamp
    or byte size differs from the values observed when that copy was loaded, so that a
    read never returns a Memory_Record set that a completed write has superseded.

### Requirement 7: Automatic recall injection into a chat turn

**User Story:** As a chat user, I want the agent to already know my stored preferences
when a new session starts, so that I do not have to ask it to check its memory.

#### Acceptance Criteria

1. WHILE `memory_enabled` and `memory_auto_recall_enabled` are both enabled, WHEN a chat
   turn begins, THE Memory_Recall_Injector SHALL select the in-scope Memory_Records whose
   Relevance_Score against the first 500 characters of that turn's user message is 1 or
   greater, ordered by descending Relevance_Score, then by descending `updated_at`, then
   by ascending identifier as defined in Requirement 3 criterion 1, and limited to at most
   `memory_recall_top_k` records.
2. WHEN the Memory_Recall_Injector selects 1 or more Memory_Records, THE
   Memory_Recall_Injector SHALL add exactly one `SystemMessage` to that turn's messages,
   positioned ahead of the upload-context `SystemMessage` when one is present and ahead of
   that turn's `HumanMessage`, whose content begins with a fixed memory-context label of 1
   to 60 characters used by no other message the Chat_Agent adds to a turn, contains each
   selected record's identifier and content, and is at most `memory_context_max_chars`
   characters long after truncation.
3. WHEN the Memory_Recall_Injector selects zero Memory_Records, THE
   Memory_Recall_Injector SHALL add no message to that turn and SHALL leave the count and
   order of that turn's messages unchanged.
4. WHILE `memory_enabled` is enabled and `memory_auto_recall_enabled` is disabled, THE
   Memory_Recall_Injector SHALL add no message to any turn, SHALL read no bytes from the
   Memory_Store file for auto recall, and the Memory_Tools SHALL remain callable by the
   Chat_Agent.
5. WHEN the chat history endpoint serializes a transcript, THE Chat_Agent SHALL exclude
   every message whose role is system, including every `SystemMessage` added by the
   Memory_Recall_Injector identified by the label defined in criterion 2, and SHALL return
   every user and assistant message of that transcript in checkpointed order.
6. IF reading the Memory_Store fails, or IF selection does not complete within 2000
   milliseconds while a chat turn begins, THEN THE Memory_Recall_Injector SHALL add no
   message to that turn, SHALL log one warning naming the failure cause, and the
   Chat_Agent SHALL complete the turn and return its assistant reply.
7. WHEN the Memory_Recall_Injector selects Memory_Records for a turn, THE
   Memory_Recall_Injector SHALL write no bytes to the Memory_Store file and SHALL leave
   every Memory_Record's `last_recalled_at`, `updated_at`, content, category, and tags
   unchanged, so that only an explicit recall_memory_tool call updates `last_recalled_at`
   as required by Requirement 3 criterion 5, and two turns with the same user message and
   no intervening write select the same records in the same order.
8. WHEN the Memory_Recall_Injector adds its `SystemMessage` to a turn, THE Chat_Agent
   SHALL persist that message in that Thread_Id's LangGraph checkpoint alongside the
   turn's other messages, THE Memory_Recall_Injector SHALL add at most one such message
   per turn, and THE Memory_Recall_Injector SHALL leave memory `SystemMessage`s persisted
   by earlier turns unchanged and SHALL exclude them from the current turn's selection
   input.
9. WHILE `memory_enabled` and `memory_auto_recall_enabled` are both enabled, WHEN a chat
   turn begins in either the chat FastAPI app or the chat CLI REPL, THE
   Memory_Recall_Injector SHALL apply criteria 1 through 3 with the same selection,
   placement, and character limit in both entry points.

### Requirement 8: Capacity limits and pruning

**User Story:** As an operator, I want the memory file to stay bounded, so that it does
not grow without limit or overwhelm the model context.

#### Acceptance Criteria

1. THE Memory_Store SHALL persist at most `memory_max_records` Memory_Records in total,
   counted across every Memory_Scope and every scope identifier in the document, where
   `memory_max_records` is an integer of 1 or greater.
2. WHEN save_memory_tool would persist a new Memory_Record and the persisted total
   Memory_Record count is already equal to or greater than `memory_max_records`, THE
   Memory_Store SHALL delete Memory_Records selected from the whole store rather than from
   the current turn's in-scope records only, in ascending order of effective recency
   timestamp, where a record's effective recency timestamp is its `last_recalled_at` value
   or, when that field is absent or empty, its `created_at` value, until the total count
   equals `memory_max_records` minus 1, and SHALL then persist the new Memory_Record in
   the same write operation.
3. WHEN the Memory_Store deletes one or more Memory_Records to stay within
   `memory_max_records`, THE Memory_Store SHALL log exactly one record stating the number
   of deleted records, the identifier of each deleted record, and the resulting total
   Memory_Record count.
4. THE Memory_Store SHALL persist each Memory_Record's content with 1 to
   `memory_max_record_chars` characters, counted as Unicode code points, and SHALL reject
   rather than truncate content that exceeds that limit.
5. WHEN the Memory_Store selects Memory_Records for deletion under criterion 2 and two or
   more candidate records have equal effective recency timestamps, THE Memory_Store SHALL
   delete those candidates in ascending lexicographic order of identifier, so that the set
   of surviving Memory_Records is identical for identical inputs.
6. WHEN any Memory_Tool write completes successfully, THE Memory_Store SHALL leave a
   persisted document whose total Memory_Record count is at most `memory_max_records` and
   whose count is exactly the pre-write count plus the number of records added minus the
   number of records deleted.
7. IF the Memory_Parser reads a Memory_Store document whose total Memory_Record count
   exceeds `memory_max_records`, THEN THE Memory_Store SHALL return every well-formed
   parsed Memory_Record for that read without deleting any record and without writing to
   the file, SHALL log one warning stating the parsed count and the configured
   `memory_max_records` value, and SHALL reduce the persisted total count to at most
   `memory_max_records` on the next Memory_Tool write using the deletion order defined in
   criteria 2 and 5.
8. IF a save_memory_tool call updates an existing Memory_Record instead of creating one,
   as defined in Requirement 2 criterion 4, THEN THE Memory_Store SHALL delete zero
   Memory_Records for capacity reasons and SHALL leave the total Memory_Record count
   unchanged, even when that count equals `memory_max_records`.

### Requirement 9: Secret protection and privacy

**User Story:** As an operator, I want credential material kept out of the memory file, so
that a plain-text JSON file does not become a secret store.

#### Acceptance Criteria

1. IF save_memory_tool is called with a content value or any tag value that matches a
   Secret_Pattern, THEN THE Memory_Store SHALL leave the persisted Memory_Records
   unchanged, SHALL write no bytes to the Memory_Store file, and SHALL return an error
   string that states that credential-like content is refused, names whether the content
   parameter or the tags parameter caused the refusal, and contains none of the matched
   text.
2. WHEN the Memory_Store logs a read, write, prune, or refusal event, THE Memory_Store
   SHALL include the event kind, the Memory_Record count after the event, and the affected
   Memory_Record identifiers, and SHALL exclude Memory_Record content, tag values, recall
   and forget query text, and any text matched by a Secret_Pattern.
3. THE Memory_Store SHALL derive the path of the Memory_Store file exclusively from the
   `memory_store_path` Settings field, and THE Memory_Tools SHALL declare no input
   parameter that names a filesystem path, directory, or file name, so that no Memory_Tool
   parameter supplied by the Chat_Agent can change the file that is read or written.
4. WHEN the Memory_Store creates the Memory_Store file, THE Memory_Store SHALL create it
   at the path resolved from `memory_store_path`, resolving a relative value against the
   process working directory and using an absolute value as supplied, and SHALL create no
   file outside the directory of that resolved path.
5. THE Memory_Store SHALL treat a text as matching a Secret_Pattern when that text
   contains any one of the following: a PEM private-key header line; the prefix `sk-`
   followed by 16 or more characters that are letters, digits, hyphens, or underscores;
   the prefix `AKIA` or `ASIA` followed by 16 characters that are uppercase letters or
   digits; the word `bearer` followed by whitespace and 20 or more non-whitespace
   characters; or a key name containing `password`, `passwd`, `secret`, `api_key`, or
   `token` followed by `=` or `:` and 8 or more non-whitespace characters. THE
   Memory_Store SHALL perform every Secret_Pattern comparison case-insensitively.
6. WHEN save_memory_tool is called, THE Memory_Store SHALL apply the Secret_Pattern
   screening of criterion 1 after the parameter validations in Requirement 2 criteria 6
   and 7 and Requirement 10 criterion 6 and before the duplicate detection in
   Requirement 2 criterion 4, so that a call that violates both a parameter limit and a
   Secret_Pattern returns the parameter-limit error.
7. IF the configured `memory_store_path` contains a `..` path segment, THEN THE
   Memory_Store SHALL read no bytes from and write no bytes to any file, and every
   Memory_Tool SHALL return an error string naming `memory_store_path` as the cause.

### Requirement 10: Memory scope

**User Story:** As a chat user, I want memories to carry across my sessions by default,
while still being able to keep something local to one conversation.

#### Acceptance Criteria

1. WHEN save_memory_tool is called without a scope parameter, THE Memory_Store SHALL
   persist the Memory_Record with Memory_Scope equal to the configured
   `memory_default_scope` value resolved from Settings.
2. WHERE a Memory_Record has Memory_Scope `global`, THE Memory_Store SHALL include that
   record in the in-scope records for every Thread_Id and for every Memory_Tool
   invocation for which no Thread_Id is resolved.
3. WHERE a Memory_Record has Memory_Scope `session`, THE Memory_Store SHALL include that
   record in the in-scope records only when the Thread_Id resolved for the current
   invocation is character-for-character equal to that record's scope identifier, and
   SHALL exclude that record from the in-scope records for every other Thread_Id and for
   every invocation for which no Thread_Id is resolved.
4. WHEN save_memory_tool persists a Memory_Record with Memory_Scope `session`, THE
   Memory_Store SHALL set that record's scope identifier to the Thread_Id resolved for the
   current invocation, and SHALL ignore any scope identifier value supplied as a
   save_memory_tool parameter.
5. WHEN a chat session is deleted from the session registry, THE Memory_Store SHALL delete
   every Memory_Record whose Memory_Scope is `session` and whose scope identifier equals
   that session's Thread_Id, SHALL retain every Memory_Record whose Memory_Scope is
   `global`, and SHALL retain every Memory_Record whose scope identifier differs from that
   Thread_Id.
6. IF save_memory_tool is called with a scope value outside the set {`global`, `session`},
   THEN THE Memory_Store SHALL leave the persisted Memory_Records unchanged and SHALL
   return an error string naming the accepted scope values.
7. THE Memory_Store SHALL resolve the Thread_Id for a Memory_Tool invocation from the
   invocation context supplied by the calling turn, SHALL accept no Thread_Id value from
   any Memory_Tool parameter supplied by the Chat_Agent, and SHALL treat the Thread_Id as
   unresolved when that context carries no Thread_Id.
8. IF a save_memory_tool call resolves to Memory_Scope `session` while no Thread_Id is
   resolved for the invocation, THEN THE Memory_Store SHALL leave the persisted
   Memory_Records unchanged and SHALL return an error string stating that session-scoped
   memory requires an active chat session and naming `global` as the scope that is usable
   for that invocation.
9. IF the Memory_Store cannot delete the session-scoped Memory_Records while a chat
   session is being deleted from the session registry, THEN THE Memory_Store SHALL leave
   the persisted Memory_Records unchanged, SHALL log one warning naming that session's
   Thread_Id, and SHALL report the failure without preventing the session registry
   deletion from completing.

### Requirement 11: Configuration

**User Story:** As an operator, I want long-term memory configured the same way as every
other tool in this project, so that enabling and tuning it needs no new mechanism.

#### Acceptance Criteria

1. THE Settings dataclass SHALL define the fields `memory_enabled` as a boolean
   defaulting to `false`, `memory_store_path` as a string of 0 to 500 characters
   defaulting to the empty string, `memory_max_records` as an integer defaulting to 500,
   `memory_max_record_chars` as an integer defaulting to 1000, `memory_recall_top_k` as an
   integer defaulting to 5, `memory_context_max_chars` as an integer defaulting to 2000,
   `memory_default_scope` as a string defaulting to `global`, and
   `memory_auto_recall_enabled` as a boolean defaulting to `true`.
2. THE configuration loader SHALL map each field listed in criterion 1 to the environment
   variable whose name is that field name in upper case with its underscores preserved
   (`memory_enabled` to `MEMORY_ENABLED`, `memory_store_path` to `MEMORY_STORE_PATH`, and
   likewise for the remaining six fields), and SHALL apply the existing precedence of
   programmatic overrides over environment variables over YAML configuration over
   dataclass defaults.
3. THE repository SHALL document every field listed in criterion 1 in both `.env.example`
   and `config/default.yaml`, in each case stating the field's default value from
   criterion 1.
4. IF `memory_max_records` is configured outside the range 1 to 10000,
   `memory_max_record_chars` outside the range 1 to 10000, `memory_recall_top_k` outside
   the range 1 to 50, or `memory_context_max_chars` outside the range 1 to 20000, THEN THE
   configuration loader SHALL fail to load, SHALL return no Settings instance, and SHALL
   report the field name, the offending value, and the accepted range.
5. IF `memory_default_scope` is configured with a value that, after trimming leading and
   trailing whitespace and case folding, is outside the set {`global`, `session`}, THEN
   THE configuration loader SHALL fail to load, SHALL return no Settings instance, and
   SHALL report the offending value and the accepted values.
6. WHILE `memory_enabled` is enabled and `memory_store_path` is empty or consists only of
   whitespace, THE Memory_Store SHALL use the path `memory/long_term_memory.json`
   relative to the process working directory.
10. THE repository SHALL exclude the resolved default Memory_Store file and its
    `.corrupt-<UTC timestamp>` siblings from version control, and SHALL keep the tracked
    documentation files already present in the `memory/` directory tracked.
7. WHEN `memory_enabled` or `memory_auto_recall_enabled` is supplied as an environment
   variable value or a YAML value, THE configuration loader SHALL treat that value as
   enabled if, after trimming leading and trailing whitespace and case folding, it equals
   one of `true`, `1`, `yes`, or `on`, and SHALL treat every other value, including the
   empty string, as disabled.
8. IF `memory_max_records`, `memory_max_record_chars`, `memory_recall_top_k`, or
   `memory_context_max_chars` is configured with a value that is not parseable as an
   integer, THEN THE configuration loader SHALL fail to load, SHALL return no Settings
   instance, and SHALL report the field name and the offending value, matching the
   reporting required by criterion 4.
9. IF `memory_recall_top_k` is configured with a value greater than `memory_max_records`,
   THEN THE configuration loader SHALL load without failing and SHALL retain both
   configured values unchanged, so that each recall returns at most `memory_max_records`
   Memory_Records.

### Requirement 12: Error handling and observability

**User Story:** As a developer, I want memory failures to degrade the turn rather than
break it, so that a bad memory file never takes the chat agent down.

#### Acceptance Criteria

1. WHEN a Memory_Tool encounters any failure, including an exception type the tool does
   not anticipate, THE Memory_Tool SHALL return to the Chat_Agent a string of 1 to 500
   characters that begins with a fixed failure marker and names both the attempted memory
   operation and the failure cause, and SHALL propagate no exception to the graph
   executor, so that the agent loop continues.
2. WHEN a Memory_Tool completes with outcome `success`, THE Memory_Store SHALL log one
   record at informational level containing the tool name, the outcome as exactly
   `success`, and the resulting Memory_Record count.
3. IF a Memory_Tool completes with outcome `failure`, THEN THE Memory_Store SHALL log one
   record at warning level containing the tool name, the outcome as exactly `failure`, and
   the resulting Memory_Record count.
4. IF the logging facility raises an error while recording a Memory_Store event, THEN THE
   Memory_Tool SHALL discard that logging error, SHALL propagate no exception, SHALL leave
   the outcome of the memory operation itself unchanged, and SHALL return that operation's
   result string to the Chat_Agent.
5. WHEN a Memory_Tool completes with outcome `success`, THE Memory_Tool SHALL return a
   string that does not begin with the failure marker defined in criterion 1, so that the
   no-match result of Requirement 3 criterion 3 and the not-found results of Requirement 4
   criteria 4 and 6 are distinguishable from a failure and are logged with outcome
   `success`.
6. WHEN a Memory_Tool's result string would exceed `memory_context_max_chars` characters,
   THE Memory_Tool SHALL truncate the returned string to `memory_context_max_chars`
   characters and SHALL end the returned string with a marker indicating that the content
   was truncated.
7. THE Memory_Store SHALL emit every diagnostic output through the `logging` module rather
   than through `print`, consistent with the project logging strategy, using informational
   level for `success` outcomes and warning level for `failure` outcomes and for the
   malformed-record, corrupt-file, and version-mismatch events of Requirement 5.

## Open Questions / Assumptions

The following decisions are recorded as initial assumptions and should be confirmed before
the design phase:

1. **Tool surface**: Assumed three separate tools — `save_memory`, `recall_memory`, and
   `forget_memory` (Requirement 1) — rather than a single tool with an `action`
   parameter. Separate tools give the model clearer selection signals, matching the
   one-purpose-per-tool convention in `src/tools/SKILL.md`. Confirm whether a `list`
   tool is also wanted for a full inventory dump.
2. **Store location**: Confirmed. The default path is `memory/long_term_memory.json`
   (Requirement 11 criterion 6). The store file and its `.corrupt-*` siblings are
   git-ignored; the 7 documentation files already tracked under `memory/` stay tracked,
   because a `.gitignore` entry does not untrack files git already tracks.
3. **Scope model**: Assumed a two-value scope, `global` and `session`, with `global` as
   the default (Requirement 10). There is no per-user identity in the project today, so
   `global` effectively means "this deployment's single user". Confirm whether a
   `user_id` dimension is needed now or later.
4. **Write trigger**: Assumed memories are written only by an explicit
   `save_memory_tool` call made by the model (Requirement 1 criterion 6). Automatic
   LLM-driven extraction of memories from every turn is out of scope for this version.
   Confirm this is acceptable.
5. **Recall ranking**: Assumed deterministic lexical term-overlap scoring
   (Relevance_Score) rather than embedding similarity, to avoid an embedding call on
   every turn. Confirm; embedding-based recall could be added later behind a setting.
6. **QA mode**: Assumed the Memory_Tools are wired into both graph tool-resolution sites
   because `src/tools/SKILL.md` requires it, while automatic recall injection
   (Requirement 7) applies to chat turns only. Confirm whether single-shot QA mode should
   read memory at all.
7. **Auto-recall and recency**: Assumed automatic injection (Requirement 7) is read-only
   and does not update `last_recalled_at`, so only explicit `recall_memory` calls affect
   pruning recency. Confirm.
8. **Session-scope without a session**: Assumed a `session`-scoped save fails with an
   error when no Thread_Id is available (Requirement 10 criterion 8) rather than silently
   falling back to `global`. Confirm.
