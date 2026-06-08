# Contributor Guide

This project is a local LangGraph RAG application with two FastAPI entry
points: single-shot QA and multi-turn chat. Contributions should preserve the
existing CLI commands, HTTP endpoints, and compatibility import paths unless a
planned migration explicitly says otherwise.

## Local Setup

Use Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `DASHSCOPE_API_KEY` in `.env`. Keep real keys out of git, screenshots,
logs, and shared issue text.

## Running The Apps

Run the QA API:

```powershell
python -m src.qa.main serve
```

Run the chat API:

```powershell
python -m src.chat.main serve
```

Run a one-off QA query:

```powershell
python -m src.qa.main query "Your question here"
```

Use `--config <path>` with the QA or chat entry points when testing a custom
YAML config. Environment variables and CLI flags override YAML values.

## Verification

Run the existing test suite:

```powershell
python -m pytest -q
```

Check for whitespace errors before handing work off:

```powershell
git diff --check
```

The current project does not yet configure Ruff, MyPy, pre-commit, or coverage
thresholds. Treat those as company-readiness follow-up items, not required
local gates for this snapshot.

## Development Rules

- Keep behavior compatible with the public CLI and API contracts in `README.md`.
- Prefer existing provider, graph, session, and API abstractions over new
  module-level globals.
- Do not add dependencies unless they are already approved in `requirements.txt`
  or a readiness task explicitly includes dependency work.
- Add or update focused tests for source changes. Documentation-only changes
  should still pass `git diff --check`.
- Update `memory/refactor-daily-forms.md` at the end of each refactoring pass
  with completed, blocked, and remaining work.

## Commit Style

Use concise, imperative commit subjects when commits are requested:

```text
Add architecture readiness docs
Document Chroma backup restore flow
```

Group unrelated source, test, infrastructure, and documentation changes into
separate commits when practical.
