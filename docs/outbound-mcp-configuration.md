# Outbound MCP Configuration Boundary

Outbound MCP is a separate bounded context under `src/backend/adapters/mcp_client`. It is not imported or started by the FastAPI process. Configuration contains only non-secret references and policy identifiers; transport connectors are injected into a separately launched lifecycle.

## Fail-closed activation

Outbound configuration defaults to disabled and no configuration file is discovered implicitly. Select a dedicated JSON document with `OUTBOUND_MCP_CONFIG_FILE` (or the explicit loader argument), and set `OUTBOUND_MCP_ENABLED=true` or `"enabled": true` in that document. `MCP_OUTBOUND_*` and `MCP_CLIENT_*` remain invalid inbound-server settings.

```json
{
  "version": 1,
  "enabled": true,
  "servers": [
    {
      "name": "approved_docs",
      "transport": "streamable_http",
      "endpoint": "https://mcp.example.com/service",
      "required": true,
      "authorization_secret_ref": {
        "provider": "env",
        "identifier": "APPROVED_DOCS_MCP_TOKEN"
      },
      "allowed_redirect_origins": [],
      "max_redirects": 0,
      "connect_timeout_seconds": 10,
      "invocation_timeout_seconds": 30,
      "shutdown_timeout_seconds": 5,
      "max_reconnect_attempts": 1,
      "reconnect_backoff_seconds": 0.1,
      "max_concurrent_invocations": 8,
      "max_request_bytes": 65536,
      "max_result_bytes": 262144
    }
  ]
}
```

The schema is closed. Remote endpoints require HTTPS and reject credentials, query strings, fragments, prohibited hosts, and unknown fields. Every initial and redirect target is resolved and checked; redirects are disabled in the connector and followed only through the exact configured origin allowlist. The connected peer must match an approved pre-resolved address, closing the DNS-rebinding window at the connector boundary. Private destinations are rejected unless runtime deployment policy supplies a typed approval for exact hosts and explicit private CIDRs; there is no general private-network switch.

## Bounded runtime

Each provider has hard connect, invocation, request/result byte, concurrency, reconnect/backoff, and shutdown bounds. A lost connection is re-established only within the configured attempt/deadline budget. An interrupted invocation is retried only when its caller explicitly marks it retry-safe. Reconnect cannot mutate the published tool generation. Shutdown first rejects work, then drains or cancels active calls, closes the MCP session, and finally closes the connector. Startup or publication failure closes every initialized but unpublished resource.

## Stdio process policy

JSON configuration never accepts an executable, shell string, arbitrary arguments, working directory, or environment. A stdio server references only a `command_template` identifier:

```json
{
  "name": "approved_local",
  "transport": "stdio",
  "command_template": "approved_local_server",
  "required": false
}
```

The separately supplied `ExecutableAllowlist` maps that identifier to one absolute executable plus fixed argv, working directory, and environment. `SecureProcessLauncher` uses direct exec with protocol pipes; it never invokes a shell, performs a PATH lookup, or inherits the ambient environment. Hosted-production stdio remains disabled by default and requires an explicit production approval on the allowlist. Termination and forced kill both have hard bounds.

## Secret resolution

`SecretReference` is persistable; `ResolvedSecret` is runtime-only and always redacted from its representation. A `SecretResolver` dispatches only typed references to provider instances explicitly supplied by the outbound lifecycle. `EnvironmentSecretProvider` additionally requires an identifier allowlist. Configuration loading never reads the referenced value, disabled settings resolve no credentials, and resolved values are not serialized into provider configuration, health, or errors.
