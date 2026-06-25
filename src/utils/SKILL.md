---
name: utils-architect
description: >
  Use this skill whenever working on shared utilities — retry logic, network
  helpers, URL input parsing, SSL configuration, or the Windows-safe directory
  removal helper. Trigger on mentions of invoke_with_retry, call_with_retry,
  dashscope_call_with_retry, remove_tree_with_retry, configure_ssl_from_env,
  parse_url_input, or "retryable error".
---

# Utils Architect — src/utils/

Domain: cross-cutting helpers shared by every other package. Pure functions
with no inbound application-code dependencies.
Parent: `SKILL.md` (root). Siblings: `src/llm/SKILL.md`, `src/config/SKILL.md`,
`src/web_search/SKILL.md`, `src/rag/SKILL.md`, `src/graph/SKILL.md`,
`src/api/SKILL.md`, `src/sessions/SKILL.md`, `src/tools/SKILL.md`.

## Quick Reference

| Module | Surface | Used by |
|---|---|---|
| `retry.py` | `invoke_with_retry`, `call_with_retry`, `dashscope_call_with_retry`, `remove_tree_with_retry`, `is_retryable_connection_error`, `RETRYABLE_HTTP_STATUSES` | Every LLM call site, embeddings, Chroma cleanup |
| `networking.py` | `configure_ssl_from_env`, `parse_dashscope_base_url`, `configure_dashscope_base_url`, `ensure_user_agent`, `DEFAULT_DASHSCOPE_HTTP_BASE_URL`, `DEFAULT_USER_AGENT` | `src/config/loader.py`, graph node SSL configuration |
| `urls.py` | `parse_url_input` | FastAPI request body normalization (`/query`, `/chat`) |

## File Map

```
src/utils/
├── __init__.py     # Empty docstring; helpers are imported directly from submodules
├── retry.py        # All retry variants + Windows-safe directory removal
├── networking.py   # SSL, DashScope base URL, USER_AGENT
└── urls.py         # parse_url_input — string-or-list normalization
```

## Retry Surface

Four entry points, picked based on what you're retrying:

| Function | When to use | Backoff |
|---|---|---|
| `invoke_with_retry(chain, input_data, *, max_retries, base_delay)` | LangChain runnables: `chain.invoke(input)`. Used by every LLM call. | Exponential with random jitter: `base_delay * 2**attempt + jitter` |
| `call_with_retry(operation, *, max_retries, base_delay, retryable, log_label)` | Arbitrary callables; you supply the retry predicate. | Plain exponential: `base_delay * 2**attempt` |
| `dashscope_call_with_retry(client, kwargs, *, max_retries, base_delay)` | Direct DashScope SDK calls that return a status code (e.g., embeddings). Inspects `response.status_code` against `RETRYABLE_HTTP_STATUSES`. | Plain exponential |
| `remove_tree_with_retry(path, *, max_retries, delay)` | Chroma DB directory cleanup on Windows where SQLite handles linger. | Linear delay; falls back to per-file `chmod` + `unlink` if `shutil.rmtree` fails |

### Retryable Error Predicate

`is_retryable_connection_error(exc)` returns True when the error message
contains any of:

```
SSL, CERTIFICATE, EOF, CONNECTION, MAX RETRIES, TIMEOUT, REMOTE END,
TEMPORARILY UNAVAILABLE
```

This is intentionally a substring match on `str(exc).upper()`. It catches the
common LangChain / `requests` / `urllib3` / `ssl` failure modes without
exhaustive type checking. Pure-message matching is fragile but observably
adequate; review the predicate when adding new providers.

### Retryable HTTP Statuses

`RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}`. Used only by
`dashscope_call_with_retry` since LangChain runnables raise rather than
returning status codes.

### Backoff Numbers

- Default `max_retries=3`, `base_delay=1.0`.
- `invoke_with_retry` adds up to ~1s of random jitter per attempt to avoid
  thundering-herd retries when many requests fail at once.
- Worst-case wait for 3 attempts: ~7s (1 + 2 + 4) plus jitter.
- DashScope SDK has its own retries (`dashscope_max_retries`) layered on top.
  Settings expose these so you can tune the combined effective budget.

## Windows-Safe Directory Removal

`remove_tree_with_retry` exists because Chroma's persistent SQLite leaves file
handles open on Windows briefly after close, and `shutil.rmtree` then fails
with `PermissionError`. The strategy:

1. `gc.collect()` to release Python-held references.
2. `shutil.rmtree` with an `onerror` that retries with `chmod(...IWUSR)` +
   `unlink`.
3. On final retry, walk the tree manually with `os.chmod`/`os.unlink`/`os.rmdir`
   in `topdown=False` order.

Used by Chroma rebuild paths in `src/rag/` and session cleanup in
`src/sessions/`. Linux/macOS hit the fast path almost always.

## Networking Surface

| Function | What it does |
|---|---|
| `configure_ssl_from_env()` | If `DISABLE_SSL_VERIFY=true`, clears `REQUESTS_CA_BUNDLE`/`CURL_CA_BUNDLE` and patches `ssl._create_default_https_context`. Debug-only escape hatch. Never call in production paths. |
| `parse_dashscope_base_url(raw)` | Strips trailing slash; validates `http://` or `https://` prefix; raises with a clear message on invalid input. |
| `configure_dashscope_base_url(base_url)` | Writes `DASHSCOPE_HTTP_BASE_URL` to `os.environ` and (if `dashscope` is importable) sets `dashscope.base_http_api_url`. Silent if SDK isn't installed yet. |
| `ensure_user_agent()` | Sets `USER_AGENT=only-subcribers/1.0` if unset. Stops LangChain `WebBaseLoader` from emitting a startup warning. |

## URL Input Parsing

`parse_url_input(raw_urls)` normalizes the messy union of inputs the API and
CLI receive into a clean `list[str] | None`:

| Input | Returns |
|---|---|
| `None` | `None` |
| `""` | `None` |
| `","` (whitespace/separators only) | `None` |
| `"https://a.com"` | `["https://a.com"]` |
| `"https://a.com, https://b.com"` | `["https://a.com", "https://b.com"]` |
| `["https://a.com"]` | `["https://a.com"]` |
| `["https://a, https://b"]` (CSV inside a list element) | `["https://a", "https://b"]` |
| `[]` | `None` |

Note: this does **not** validate URL syntax or filter by scheme. That's the
caller's responsibility (see `src/web_search/common.py:normalize_urls` for
scheme filtering).

## Known Issues

| # | Issue | Severity | Location | Fix |
|---|---|---|---|---|
| 1 | `is_retryable_connection_error` uses substring matching on the error message — fragile across library versions | Medium | `retry.py` | Augment with type checks (`isinstance(exc, ConnectionError | TimeoutError | ssl.SSLError | RequestException)` already done at the caller; predicate gates the *retry* decision) |
| 2 | `configure_ssl_from_env` patches the global `ssl._create_default_https_context`; affects DashScope and search alike | Medium | `networking.py` | Debug-only; log a prominent warning at startup |
| 3 | `remove_tree_with_retry` has Windows-specific logic but no platform guard | Low | `retry.py` | Harmless on POSIX (the retry path rarely fires); kept simple |
| 4 | No async variants | Medium | `retry.py` | Add `async_invoke_with_retry` when graph executor SSE path needs it |
| 5 | `parse_url_input` doesn't reject non-HTTP schemes | Low | `urls.py` | By design — `normalize_urls` in `src/web_search/common.py` enforces `http(s)://` later in the pipeline |

## Refactoring To-Do List

- [ ] **Async retry variant** — `async_invoke_with_retry` once any graph node goes async.
- [ ] **Structured retry events** — emit `RetryAttempt` events through `src/graph/events.py` so metrics can count retries per provider.
- [ ] **Type-based retryable predicate** — replace substring matching with an exception-type allowlist where feasible.
- [ ] **Platform guard for `remove_tree_with_retry`** — skip the chmod fallback on POSIX where `shutil.rmtree` reliably works.

## Testing Strategy

| Test | Approach |
|---|---|
| Retry succeeds on second attempt | Mock chain that raises once then returns; assert one retry happened |
| Non-retryable error raises immediately | Mock chain that raises `ValueError`; assert no retry |
| `is_retryable_connection_error` keywords | Parametrize with sample messages; assert each is classified correctly |
| `parse_url_input` empty / CSV / list shapes | Table-driven test of every input variant in the docs table above |
| `remove_tree_with_retry` happy path | Create tmp dir, remove it, assert gone |
| `remove_tree_with_retry` locked-file simulation | Hold a file open then close; assert eventual removal |
| Tests live in | `tests/test_retry.py`, `tests/test_urls.py` |

## Dependencies

- External: `requests` (for `RequestException` type), `dashscope` (lazy-imported in `configure_dashscope_base_url`).
- No inbound dependencies on `src/*` application code. Every other module can safely depend on `src/utils/`.
