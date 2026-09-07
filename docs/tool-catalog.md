# Immutable Tool Catalog and Policy Boundary

Phase 3 centralizes agent-visible tools under `src/backend/mcp` without enabling outbound MCP.

## Catalog lifecycle

`ToolDescriptor` and `ToolCatalogSnapshot` are frozen, deeply immutable values. A snapshot contains matching tuples of LangChain `BaseTool` objects and descriptors, plus a positive generation. Publication validates the complete candidate before replacing the active snapshot; failed publication retains the prior generation.

Validation bounds names, descriptions, schema bytes/depth/properties/enums, rejects unsupported schema constructs and external references, and detects collisions. Future remote tools must use `mcp__<server>__<tool>` with a matching `server_name`; local tools cannot claim that namespace.

## Providers and graph consistency

Providers exist for injected tools, the heavy retriever, required lightweight web search, memory, builtin read/compute tools, local document readers, and session-confined editors. `DisabledOutboundMCPProvider` is an explicit no-transport future seam.

Both graph builders call one provider composition. The full graph contributes the retriever and conditionally configured web search; the lightweight graph omits the retriever and requires web search. All remaining providers and ordering are shared. One immutable snapshot supplies both model binding and `ToolNode`; the compiled graph exposes `tool_catalog_snapshot`, `tool_catalog_generation`, and `tool_descriptors` for turn/executor integration.

## Execution policy

`ToolExecutionPipeline` preserves the LangChain `BaseTool` contract for sync and async tools and applies: authorization, input validation, deadline/concurrency limits, invocation, output bounds, redaction, audit, and outcome metrics. Principal/tenant allowlists and risk classes are immutable policy inputs. Errors and audit outcomes are stable and sanitized; arguments and results are never placed in audit metadata.

MCP-provided tools can mark `canonical_inbound_mcp_tool` metadata to avoid double wrapping. Session editor factories remain responsible for filesystem confinement and receive only the existing session root and trusted thread ID.

## Telemetry

Typed graph tool events add optional source server, catalog generation, bounded duration, and stable outcome fields while retaining compatible defaults. `GraphExecutor` can take a snapshot or descriptor tuple and tracks each tool call from start to result. `MetricsCollector` aggregates only bounded-cardinality outcome, duration, call-count, and generation values.
