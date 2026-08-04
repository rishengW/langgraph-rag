# Security Policy

## Supported Versions

This project is pre-release. Security fixes should target the active mainline
branch unless a release branch is explicitly created.

## Reporting A Vulnerability

Do not open public issues containing secrets, exploit details, private URLs, or
production data. Report vulnerabilities privately to the project maintainer or
repository owner with:

- A short description of the issue.
- Steps to reproduce.
- Affected endpoints, commands, or files.
- Whether credentials, source documents, Chroma data, or chat transcripts may be
  exposed.

If a secret was committed, logged, pasted into a prompt, or shared in a ticket,
rotate it immediately. Removing it from the working tree is not enough.

## Secrets

Required secrets, including `DASHSCOPE_API_KEY`, belong in `.env` or process
environment variables. Do not put secrets in YAML config files, Chroma backups,
screenshots, test fixtures, or documentation examples.

`AMAP_WEB_SERVICE_KEY` and `AMAP_JS_SECURITY_CODE` are server-side secrets.
`AMAP_JS_API_KEY` is intentionally browser-visible and should be restricted in
the AMap console to the application's deployed origins. The chat UI sends AMap
JS service traffic through `/_AMapService`; that route validates relative paths,
uses fixed AMap upstream hosts, overrides caller-supplied security codes, refuses
redirects, and bounds query and response sizes.

The old notebook had hard-coded keys that were removed from this Python project.
Assume any key that appeared in a notebook, chat transcript, or repository
history has been compromised and rotate it.

## Local Data

Chroma persists vector data under `.chroma/` by default. Explicit and
web-search chat sessions may create per-thread indexes under
`.chroma/chat/<thread_id>/`. Treat those directories as potentially sensitive
because chunks can contain source document text.

Stop the application before backing up or restoring `.chroma/` or a session
metadata SQLite database. Store backups somewhere with access controls
appropriate for the indexed source material.

## Current Security Gaps

API-key authentication is available for mutation endpoints when `API_KEY` is
set. Local development remains open when it is unset, so configure `API_KEY`
before exposing the QA or chat APIs outside a trusted workstation.

CORS is configurable through `cors_allow_origins` in YAML or
`CORS_ALLOW_ORIGINS` in the environment. Keep the allowed-origin list scoped to
known clients in shared or production deployments.

Dependency scanning is not configured yet. Before production use, add Dependabot
or a `pip-audit` workflow and triage vulnerabilities in the LangChain, FastAPI,
Chroma, and DashScope dependency chain.

See `SSL_FIX.md` for local certificate verification troubleshooting. Disabling
SSL verification should remain a local workaround only, not a production
default.
