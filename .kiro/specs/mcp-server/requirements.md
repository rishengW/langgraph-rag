# Requirements Document

## Introduction

This feature adds a Model Context Protocol (MCP) server to the existing LangGraph RAG
project ("only Subscribers") so that MCP-capable LLM clients (for example Claude
Desktop, Cursor, and Kiro) can invoke the RAG engine as callable tools. The MCP server
is purely **additive**: it is a thin adapter that reuses the existing graph layer
(`build_chat_graph`, `build_lightweight_graph`, `discover_urls_from_web`) and session
registry, and it does **not** replace, fork, or duplicate the existing FastAPI HTTP API
in `src/chat/api.py`.

The first deliverable exposes **stateless** RAG tools (ask a question against optional
explicit URLs, and answer a question grounded in live web search). Multi-turn session
support that maps an MCP conversation to a persisted `thread_id` is defined as an
**optional** capability so it can be deferred without blocking the stateless tools.

Because the team previously removed the `mcp` dependency during cleanup (see
`memory/refactor-daily-forms.md`), re-introducing it is a deliberate, scoped decision
recorded in these requirements.

This document records initial decisions for the four open questions (transport, tool
surface, stateless vs session, and authentication) as explicit assumptions. These are
flagged in the "Open Questions / Assumptions" section and should be confirmed before
the design phase.

## Glossary

- **MCP**: Model Context Protocol, an open protocol that lets LLM clients discover and
  invoke external tools, resources, and prompts over a defined transport.
- **MCP_Server**: The new component added by this feature that registers RAG
  capabilities as MCP tools and serves them to MCP clients.
- **MCP_Client**: An external LLM application (Claude Desktop, Cursor, Kiro, etc.) that
  connects to the MCP_Server and invokes its tools.
- **RAG_Engine**: The existing LangGraph-based retrieval-and-generation logic, reached
  through `build_chat_graph`, `build_lightweight_graph`, and the graph executor.
- **Graph_Adapter**: The MCP_Server-internal layer that translates MCP tool calls into
  RAG_Engine graph invocations and translates graph results back into MCP tool results.
- **Session_Registry**: The existing `ChatSessionRegistry` that maps a `thread_id` to a
  compiled graph, settings, and SQLite-persisted checkpoint state.
- **Stdio_Transport**: The MCP transport in which the MCP_Server runs as a local
  subprocess of the MCP_Client and communicates over standard input/output.
- **HTTP_Transport**: The MCP streamable-HTTP transport in which the MCP_Server is
  reachable over a network endpoint by remote MCP_Clients.
- **ask_rag_tool**: An MCP tool that answers a question, optionally against caller-supplied
  source URLs, using the RAG_Engine.
- **web_search_answer_tool**: An MCP tool that answers a question by first discovering
  source URLs via web search and then grounding the answer with the RAG_Engine.
- **API_Key**: The shared secret already used by the HTTP API (`require_api_key`) for
  bearer-token authentication.
- **Tool_Result**: The structured payload an MCP tool returns to the MCP_Client, including
  the answer text and any source metadata.

## Requirements

### Requirement 1: Expose RAG as an MCP server

**User Story:** As an MCP client user, I want the RAG engine exposed as an MCP server, so
that I can call it as a tool from my LLM client without using the HTTP API directly.

#### Acceptance Criteria

1. WHEN the MCP_Server completes startup, THE MCP_Server SHALL have registered at least
   one and at most 16 MCP tools, each of which invokes the RAG_Engine, and SHALL assign
   every registered tool a name that is unique among registered tools.
2. WHEN an MCP_Client requests the list of available tools, THE MCP_Server SHALL return
   each registered tool with a non-empty name of 1 to 128 characters, a non-empty
   human-readable description of 1 to 1024 characters, and an input schema that declares
   each accepted parameter and which parameters are required.
3. THE MCP_Server SHALL invoke the RAG_Engine through the existing graph entry points
   (`build_chat_graph`, `build_lightweight_graph`, `discover_urls_from_web`) without
   duplicating graph topology, node, or retrieval logic.
4. WHEN the MCP_Server entry point is started, THE MCP_Server SHALL NOT start the FastAPI
   HTTP application, and WHEN the FastAPI HTTP application is started, THE FastAPI HTTP
   application SHALL NOT start the MCP_Server.
5. IF a registered tool fails to initialize during MCP_Server startup, THEN THE
   MCP_Server SHALL exit with a non-zero status and emit a diagnostic message to standard
   error that identifies the failed tool, without partially serving the remaining tools.

### Requirement 2: Stateless single-question RAG tool

**User Story:** As an MCP client user, I want to ask a one-shot question optionally
scoped to specific URLs, so that I can get a grounded answer without managing a
conversation.

#### Acceptance Criteria

1. WHEN the MCP_Client invokes the ask_rag_tool with a question that contains at least one
   non-whitespace character and no more than 4,000 characters and supplies no URLs, THE
   MCP_Server SHALL produce an answer using the configured default source URLs loaded from
   the `Settings` configuration system.
2. WHERE the MCP_Client supplies between 1 and 50 source URLs to the ask_rag_tool, THE
   MCP_Server SHALL use the supplied URLs as the RAG_Engine sources for that call instead
   of the configured default source URLs.
3. WHEN the ask_rag_tool produces an answer, THE MCP_Server SHALL return a Tool_Result
   containing the answer text and the list of source URLs used to ground that answer.
4. IF the question field is empty or contains only whitespace, THEN THE MCP_Server SHALL
   return an MCP tool error that identifies the question field as the cause, and SHALL NOT
   invoke the RAG_Engine.
5. IF the question field exceeds 4,000 characters or the number of supplied source URLs
   exceeds 50, THEN THE MCP_Server SHALL return an MCP tool error that identifies the
   field that exceeded its limit as the cause, and SHALL NOT invoke the RAG_Engine.
6. IF the RAG_Engine raises an error while processing the ask_rag_tool call, THEN THE
   MCP_Server SHALL return an MCP tool error containing a message that describes the
   failure cause, SHALL preserve no partial answer in the Tool_Result, and SHALL continue
   serving subsequent tool calls.

### Requirement 3: Web-search-grounded answer tool

**User Story:** As an MCP client user, I want to ask a question that is answered from a
live web search, so that I can get current information without supplying URLs myself.

#### Acceptance Criteria

1. WHEN the MCP_Client invokes the web_search_answer_tool with a query that contains at
   least one non-whitespace character and no more than 4,096 characters, THE MCP_Server
   SHALL discover source URLs through `discover_urls_from_web` and then ground the answer
   with the RAG_Engine.
2. WHEN the web_search_answer_tool produces an answer, THE MCP_Server SHALL return a
   Tool_Result containing the answer text and the list of discovered source URLs used to
   ground the answer.
3. IF web search returns zero source URLs, THEN THE MCP_Server SHALL return a Tool_Result
   whose answer text indicates that no web sources were found and whose source URL list is
   empty.
4. WHILE the server setting `web_search_enabled` is disabled, THE MCP_Server SHALL omit
   the web_search_answer_tool from the advertised tool list.
5. IF web search fails with an error, THEN THE MCP_Server SHALL return an MCP tool error
   that indicates the web search step as the failure cause, SHALL leave the
   Session_Registry and checkpoint state unchanged, and SHALL continue serving subsequent
   tool calls.
6. IF the query field is empty or contains only whitespace, THEN THE MCP_Server SHALL
   return an MCP tool error that identifies the query field as the cause, and SHALL NOT
   invoke web search or the RAG_Engine.

### Requirement 4: Transport selection

**User Story:** As an operator, I want to choose how the MCP server is reached, so that I
can support both local LLM clients and remote deployments.

#### Acceptance Criteria

1. THE MCP_Server SHALL support the Stdio_Transport for local MCP_Clients.
2. WHERE the operator selects the HTTP_Transport, THE MCP_Server SHALL serve MCP requests
   over the configured host and the configured TCP port, where the port is an integer in
   the range 1 to 65535.
3. THE MCP_Server SHALL select the active transport from configuration from the set
   {Stdio_Transport, HTTP_Transport}, defaulting to the Stdio_Transport when no transport
   is specified.
4. WHEN the MCP_Server starts with the Stdio_Transport, THE MCP_Server SHALL write only
   MCP protocol messages to standard output and SHALL write diagnostic logs to standard
   error.
5. IF the configured transport value is not one of {Stdio_Transport, HTTP_Transport},
   THEN THE MCP_Server SHALL fail to start and SHALL write an error indicating the invalid
   transport value to standard error.
6. IF the HTTP_Transport is selected and the configured host is missing or the configured
   port is missing or outside the range 1 to 65535, THEN THE MCP_Server SHALL fail to
   start and SHALL write an error indicating the invalid host or port to standard error.

### Requirement 5: Authentication for remote transport

**User Story:** As an operator, I want remote MCP access protected by a shared key, so
that an exposed HTTP endpoint is not open to anonymous callers.

#### Acceptance Criteria

1. WHILE the HTTP_Transport is active and an API_Key is configured, THE MCP_Server SHALL
   require every MCP request to present the API_Key as a bearer token whose value exactly
   matches the configured API_Key in full.
2. IF the HTTP_Transport is active, an API_Key is configured, and a request presents no
   bearer token or a bearer token whose value does not exactly match the configured
   API_Key, THEN THE MCP_Server SHALL reject the request with an unauthorized result that
   includes a bearer authentication challenge, SHALL NOT invoke the RAG_Engine for that
   request, and SHALL continue serving subsequent requests.
3. WHILE the HTTP_Transport is active and no API_Key is configured, THE MCP_Server SHALL
   serve MCP requests without requiring an API_Key, consistent with the existing HTTP API
   behavior.
4. WHILE the Stdio_Transport is active, THE MCP_Server SHALL serve requests without an
   API_Key, because the transport is a local subprocess of the MCP_Client.
5. THE MCP_Server SHALL reuse the existing API_Key configuration source used by the HTTP
   API rather than defining a separate credential.

### Requirement 6: Optional multi-turn session tools

**User Story:** As an MCP client user, I want to optionally hold a multi-turn
conversation, so that follow-up questions keep context across calls.

#### Acceptance Criteria

1. WHERE multi-turn session support is enabled, WHEN the MCP_Client invokes the
   session-start tool, THE MCP_Server SHALL create a session in the Session_Registry and
   return a Tool_Result containing a `thread_id` that is unique among the currently active
   sessions.
2. WHERE multi-turn session support is enabled, WHEN the MCP_Client invokes a follow-up
   tool with a `thread_id` that matches an active session in the Session_Registry and a
   non-empty message, THE MCP_Server SHALL route the message to that existing session
   through the Session_Registry and return a Tool_Result containing the assistant reply
   text.
3. WHERE multi-turn session support is enabled, IF a follow-up tool is invoked with a
   `thread_id` that does not match any active session in the Session_Registry, THEN THE
   MCP_Server SHALL return an MCP tool error that identifies the `thread_id` as unknown and
   SHALL leave existing session state unchanged.
4. WHERE multi-turn session support is enabled, IF a follow-up tool is invoked with a
   message that is empty or contains only whitespace, THEN THE MCP_Server SHALL return an
   MCP tool error that identifies the message field as the cause and SHALL leave the
   addressed session state unchanged.
5. WHERE multi-turn session support is enabled, IF the RAG_Engine raises an error while
   processing a follow-up tool call, THEN THE MCP_Server SHALL return an MCP tool error
   containing a descriptive message, SHALL preserve the addressed session state, and SHALL
   continue serving subsequent tool calls.
6. WHERE multi-turn session support is disabled, THE MCP_Server SHALL omit the session
   tools from the advertised tool list.

### Requirement 7: Configuration and dependency management

**User Story:** As a developer, I want the MCP server configured through the existing
settings system, so that re-adding the MCP dependency stays consistent with the rest of
the project.

#### Acceptance Criteria

1. WHEN the MCP_Server starts, THE MCP_Server SHALL read its transport, host, port, and
   enablement values from the existing `Settings` configuration system, where transport is
   one of `stdio` or `http`, host is a non-empty string, port is an integer in the range 1
   to 65535, and enablement is a boolean.
2. THE project SHALL declare the `mcp` dependency in both `pyproject.toml` and
   `requirements.txt` using an identical exact-version pin (a `==` constraint) in each
   manifest.
3. WHERE the MCP_Server enablement setting is disabled, THE MCP_Server entry point SHALL
   not bind any transport, SHALL not advertise or serve any MCP tools, and SHALL log to
   standard error that the server is disabled.
4. THE MCP_Server SHALL load source URLs and web-search behavior from the same `Settings`
   values used by the HTTP API, and SHALL NOT define a separate configuration source for
   those values.
5. IF the transport value is not `stdio` or `http`, or the port is outside the range 1 to
   65535, or the host is empty, THEN THE MCP_Server SHALL fail to start, SHALL not serve
   any MCP requests, and SHALL log an error to standard error identifying the invalid
   setting.

### Requirement 8: Observability and graceful shutdown

**User Story:** As an operator, I want the MCP server to log its activity and shut down
cleanly, so that I can run and monitor it reliably.

#### Acceptance Criteria

1. WHEN the MCP_Server starts and before it accepts any tool call, THE MCP_Server SHALL
   write to standard error a single startup record containing the active transport name
   and the complete list of advertised tool names.
2. WHEN the MCP_Server receives an interrupt (SIGINT) or termination (SIGTERM) signal,
   THE MCP_Server SHALL stop accepting new tool calls and SHALL release the
   Session_Registry and checkpoint resources before exiting.
3. WHILE shutting down after an interrupt or termination signal, IF the release of the
   Session_Registry and checkpoint resources does not complete within 10 seconds, THEN
   THE MCP_Server SHALL exit without waiting further.
4. WHEN an MCP tool call completes, THE MCP_Server SHALL record, in the existing metrics
   or logging facility, the invoked tool name and an outcome value of exactly one of
   `success` or `failure`.
5. IF the metrics or logging facility raises an error while recording a tool outcome,
   THEN THE MCP_Server SHALL discard that recording error and SHALL continue serving
   subsequent tool calls without returning an error to the MCP_Client.

## Open Questions / Assumptions

The following decisions are recorded as initial assumptions and should be confirmed before
the design phase:

1. **Transport**: Assumed both Stdio_Transport (default, for local clients) and an
   optional HTTP_Transport are supported (Requirement 4). Confirm whether HTTP is needed
   in the first version or can be deferred.
2. **Tool surface**: Assumed two stateless tools first — `ask_rag` and
   `web_search_answer` (Requirements 2 and 3) — with optional session tools (Requirement
   6). Confirm whether additional tools (for example history retrieval) are wanted.
3. **Stateless vs session**: Assumed stateless tools are the required first deliverable and
   multi-turn session tools are optional/deferred (Requirement 6). Confirm priority.
4. **Authentication**: Assumed the existing API_Key bearer scheme is reused, enforced only
   on HTTP_Transport, and skipped on Stdio_Transport (Requirement 5). Confirm this model.
