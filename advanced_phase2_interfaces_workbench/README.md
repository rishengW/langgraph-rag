# Advanced Phase 2 Interfaces Workbench

This folder marks the additional advanced refactoring workspace for the current
Phase 2 implementation pass.

Primary references:

- `REFACTORING_PLAN.md`
- `SKILL.md`
- `src/graph/SKILL.md`
- `src/rag/SKILL.md`
- `src/web_search/SKILL.md`
- `src/api/SKILL.md`
- `src/sessions/SKILL.md`

Delegation note:

- Each RAGRefactorDeveloper worker should own one subfolder `SKILL.md` scope.
- Workers must preserve existing API behavior and tests while building Phase 2
  interfaces and abstractions.

Current process snapshot:

- Status date: 2026-06-07.
- Phase 1 extraction baseline is implemented in the working tree.
- Phase 2 interface pass has completed across `src/web_search/`, `src/rag/`,
  `src/sessions/`, `src/graph/`, and `src/api/`.
- Compatibility shims remain in old `src/core/`, `src/qa/`, and `src/chat/`
  paths where needed.
- Passed checks: `git diff --check` and bundled Python `compileall` over
  `src` and `tests`.
- Blocked check: full pytest, because the workspace venv points to a missing
  Python 3.11 interpreter and the bundled Python lacks `pytest` plus compatible
  project dependencies.
