---
name: tools-architect
description: >
  Use this skill whenever working on agent-callable tools — currency conversion,
  stock quotes, weather forecasts, Wikipedia summaries — or when adding new
  LangChain StructuredTools. Covers the build_*_tool factory pattern, the
  shared _http JSON helper, Pydantic input schemas, error handling,
  feature-flag enablement, and the end-to-end "add a new tool" playbook with
  copy-paste templates. Trigger on mentions of StructuredTool, BaseTool,
  build_currency_tool, build_stock_tool, build_weather_tool,
  build_wikipedia_tool, _http, request_json, or adding agent tools.
---

# Tools Architect — src/tools/

Domain: agent-callable LangChain tools that fetch external data on demand.
Parent: `SKILL.md` (root). Siblings: `src/web_search/SKILL.md`,
`src/graph/SKILL.md`, `src/llm/SKILL.md`, `src/config/SKILL.md`,
`src/utils/SKILL.md`.

This skill is **executable**: the "Add a New Tool" section below is a
step-by-step playbook with copy-paste templates and verification commands.
An agent following it from top to bottom should be able to add a new tool,
wire it into both graph paths, expose it to the LLM, and verify it without
guessing.

## Quick Reference

| Fact | Value |
|---|---|
| Tool framework | `langchain_core.tools.StructuredTool.from_function` |
| Input schema | `pydantic.BaseModel` per tool (validated by LangChain) |
| HTTP helper | `src/tools/_http.py:request_json` (injectable `requester=` for tests) |
| Default HTTP client | `requests.get` (sync) |
| Enablement | Per-tool feature flag in `Settings`: `weather_enabled`, `stock_enabled`, `currency_enabled`, `wikipedia_enabled` (all default `False`) |
| Wiring sites | **TWO** functions in `src/graph/builder.py`: `_resolve_tools` (heavy/RAG graph) AND `_resolve_lightweight_tools` (lightweight web-search graph) |
| LLM-facing prompt | `AGENT_SYSTEM_PROMPT` in `src/llm/prompts.py` lists every tool the agent should know about |
| Live web search tool | Separate path under `src/web_search/tool.py` (`live_web_search`) — not part of this module |

## File Map

```
src/tools/
├── __init__.py         # Re-exports input schemas and build_*_tool factories
├── _http.py            # request_json — tiny injectable HTTP wrapper
├── _geocoding.py       # Shared place lookup: Open-Meteo cities + Photon POIs
├── currency.py         # convert_currency + Frankfurter exchange-rate API
├── directions.py       # get_directions + shared geocoder and OSRM routing
├── map_tool.py         # find_on_map + shared geocoder and OpenStreetMap links
├── stock.py            # get_stock_quote + yfinance ticker lookup
├── weather.py          # get_weather + Open-Meteo geocoding/forecast
└── wikipedia_tool.py   # search_wikipedia + Wikipedia MediaWiki API
```

## Shared Geocoding

`_geocoding.geocode_place` backs both `find_on_map` and `get_directions`.

1. `clean_place_query` removes request wording so a whole sentence
   ("在地图上找出上海的位置", "where is Shanghai on a map") becomes a place name.
2. Open-Meteo runs first for populated places, with `language=zh` for CJK input.
   Passing `language=en` returns nothing for Chinese place names.
3. Anything Open-Meteo cannot resolve confidently falls through to Photon
   (`photon.komoot.io`), which covers points of interest, campuses, and
   non-Latin names. No API key; send a descriptive `User-Agent`.
4. Each candidate carries a `match_score` (share of query terms present in the
   candidate name). Below `CONFIDENT_MATCH_SCORE` the map tool labels the result
   an APPROXIMATE MATCH; equally scored same-name places in different
   countries are labelled AMBIGUOUS. Both labels tell the agent to verify with a
   web search instead of asserting the coordinates.

Photon ranks by text similarity with no notion of prominence, so a plausible
name can be the wrong place ("Eiffel Tower" matches a peak in Alberta). Never
present a low-scoring or tied candidate as a confirmed location.

## Currently Available Tools

| Tool name (LLM-facing) | Module | Backing API | Network | Enable flag |
|---|---|---|---|---|
| `convert_currency` | `currency.py` | `api.frankfurter.dev` | HTTP JSON | `CURRENCY_ENABLED` |
| `get_stock_quote` | `stock.py` | `yfinance` Python library | yfinance internals | `STOCK_ENABLED` |
| `get_weather` | `weather.py` | Open-Meteo (geocoding + forecast) | HTTP JSON (two calls) | `WEATHER_ENABLED` |
| `find_on_map` | `map_tool.py` | Open-Meteo + Photon geocoding | HTTP JSON (one or two calls) | `MAP_ENABLED` |
| `get_directions` | `directions.py` | Shared geocoder + OSRM routing | HTTP JSON (two or three calls) | `DIRECTIONS_ENABLED` |
| `search_wikipedia` | `wikipedia_tool.py` | `en.wikipedia.org/w/api.php` | HTTP JSON (two calls) | `WIKIPEDIA_ENABLED` |

## Tool Authoring Pattern

Every tool follows the same shape. Match this when adding new tools.

```python
# 1. Input schema — Pydantic model with field descriptions and constraints
class FooInput(BaseModel):
    """Input schema for the foo tool."""
    query: str = Field(..., min_length=1, description="...")
    max_results: int = Field(default=1, ge=1, le=5, description="...")


# 2. Factory that returns a BaseTool, accepts Settings + injectable seam
def build_foo_tool(
    settings: Settings,
    *,
    requester: JsonRequester | None = None,
) -> BaseTool:
    """Create a foo tool."""

    def _run_foo(query: str, max_results: int = 1) -> str:
        return do_foo(query, max_results=max_results, requester=requester)

    return StructuredTool.from_function(
        func=_run_foo,
        name="foo_action",            # snake_case, LLM-facing
        description=(
            "When to use this tool. What it returns. "
            "Examples of triggering questions."
        ),
        args_schema=FooInput,
    )


# 3. Pure callable that the factory wraps — easy to unit-test
def do_foo(query: str, *, max_results: int = 1, requester=None) -> str:
    try:
        payload = request_json(URL, params={...}, requester=requester)
    except Exception as exc:
        return f"Foo failed for {query!r}: {exc}"

    # ... compose a string return value
    return "..."
```

### Why this shape

- **Factory takes `Settings`**: lets tools read project-wide config (timeouts,
  user agents, max char limits) without hardcoding.
- **`requester=` injection**: tests pass a fake that returns a dict; production
  uses `requests.get`. No HTTP mocking library needed.
- **Pure function return is a string**: LangChain tools must return strings to
  the LLM. Format multi-field results as line-delimited human-readable text.
- **Errors return error strings, not raise**: the agent sees the error message
  as the tool's output. Raising would force the graph executor to handle it
  globally and would interrupt the agent's loop ungracefully.

## The `_http` JSON Helper

```python
def request_json(
    url: str,
    *,
    params: dict[str, Any],
    requester: JsonRequester | None = None,
    timeout: int = 10,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
```

Tiny wrapper. Three reasons it exists:

1. **Test injection**: `requester=fake` lets tests return a `dict` or a
   `Response`-like object without monkeypatching `requests`.
2. **Consistent timeout**: every tool uses 10s by default; one place to bump it.
3. **Error envelope**: the wrapper handles both real `Response` objects (with
   `.raise_for_status()` + `.json()`) and plain dicts from test fakes.

If you need bespoke headers (e.g., Wikipedia requires `User-Agent` /
`Api-User-Agent`), pass them via `headers=`. The `wikipedia_tool` shows the
pattern.

## Tool Description Discipline

The `description=` string is what the LLM reads to decide whether to call the
tool. Bad descriptions cause the agent to ignore useful tools or call wrong
ones. Conventions:

- Start with what it does in one sentence.
- Follow with "Use for X, Y, and Z" listing categories of triggering questions.
- Don't pad with caveats — the LLM weights every word.
- Prefer concrete examples over abstract claims.

Compare:

> Bad: "A tool to look up weather information from an API."
>
> Good: "Get current weather and a short forecast for a city or latitude/longitude. Use for current temperature, humidity, wind, and weather forecast questions."

---

# 🛠 Playbook: Add a New Tool End-to-End

Follow these steps in order. Each step lists the **exact file**, the
**change**, a **copy-paste template** with placeholders (`<lowercase>`,
`<PascalCase>`, etc.), and a **verification command**.

## Naming Conventions (Fill These In Once)

Before you start, fix four names. The playbook references them everywhere.

| Placeholder | Meaning | Example (`get_stock_quote`) |
|---|---|---|
| `<lowercase>` | snake_case identifier; the Python module name | `stock` |
| `<tool_name>` | LLM-facing tool name (snake_case verb_object) | `get_stock_quote` |
| `<PascalCase>` | Pydantic schema class name | `Stock` |
| `<UPPER>` | env-var prefix | `STOCK` |

So for a new "translate" tool you might use: `<lowercase>=translate`,
`<tool_name>=translate_text`, `<PascalCase>=Translate`, `<UPPER>=TRANSLATE`.

---

## Step 1 — Create the tool module

**File:** `src/tools/<lowercase>.py` (new)

**Template** (copy and replace placeholders; this matches the shape of every
existing tool in `src/tools/`):

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._http import JsonRequester, request_json

if TYPE_CHECKING:
    from ..config import Settings

<UPPER>_API_URL = "https://example.invalid/api"


class <PascalCase>Input(BaseModel):
    """Input schema for the <lowercase> tool."""

    query: str = Field(..., min_length=1, description="...")
    # add more fields as needed; use ge=, le=, min_length=, max_length=
    # to make the LLM's tool call self-validating.


def build_<lowercase>_tool(
    settings: Settings,
    *,
    requester: JsonRequester | None = None,
) -> BaseTool:
    """Create a <lowercase> tool."""

    def _run_<lowercase>(query: str) -> str:
        return run_<lowercase>(query, requester=requester)

    return StructuredTool.from_function(
        func=_run_<lowercase>,
        name="<tool_name>",
        description=(
            "<One sentence: what the tool does.> "
            "Use for <category 1>, <category 2>, and <category 3> questions."
        ),
        args_schema=<PascalCase>Input,
    )


def run_<lowercase>(
    query: str,
    *,
    requester: JsonRequester | None = None,
) -> str:
    term = query.strip()
    if not term:
        return "<lowercase> requires a non-empty query."

    try:
        payload = request_json(
            <UPPER>_API_URL,
            params={"q": term},
            requester=requester,
        )
    except Exception as exc:
        return f"<lowercase> failed for {term!r}: {exc}"

    # Compose a human-readable string. The LLM will read this verbatim.
    result = payload.get("result") or "no result"
    return f"<lowercase> result for {term}: {result}"


__all__ = [
    "<PascalCase>Input",
    "build_<lowercase>_tool",
    "run_<lowercase>",
]
```

**Verify:** `python -c "from src.tools.<lowercase> import build_<lowercase>_tool"`
should exit 0.

---

## Step 2 — Add the feature flag and config fields to `Settings`

**File:** `src/config/settings.py`

**Change:** add a new field next to the other `*_enabled` flags
(`weather_enabled`, `stock_enabled`, `currency_enabled`, `wikipedia_enabled`):

```python
<lowercase>_enabled: bool = False
```

If your tool needs configurable values (max char limits, user agents, API
keys), add them too. Existing examples:

```python
wikipedia_max_summary_chars: int = 1500
wikipedia_user_agent: str = "langgraph-rag/1.0 (contact: ...)"
```

**Verify:** `python -c "from src.config.settings import Settings; s = Settings(dashscope_api_key='x'); print(s.<lowercase>_enabled)"` → `False`.

---

## Step 3 — Map the env var

**File:** `src/config/loader.py`

**Change:** add an entry to `SETTING_ENV_NAMES`:

```python
"<lowercase>_enabled": "<UPPER>_ENABLED",
# plus any extra config fields added in Step 2, e.g.
# "<lowercase>_max_chars": "<UPPER>_MAX_CHARS",
```

If the new field is anything other than a plain string (int, bool, list,
Path), check whether `_coerce_setting` in the same file already handles its
type. Booleans are handled by the generic `parse_bool` branch — you don't
need to touch `_coerce_setting` for a bool flag.

**Verify:**
```bash
<UPPER>_ENABLED=true python -c "from src.config import load_settings; print(load_settings().<lowercase>_enabled)"
```
→ `True`.

---

## Step 4 — Re-export from the tools package

**File:** `src/tools/__init__.py`

**Change:** add the new factory and input schema to the imports and
`__all__` list, alphabetically:

```python
from .<lowercase> import <PascalCase>Input, build_<lowercase>_tool

__all__ = [
    # ... existing entries ...
    "<PascalCase>Input",
    "build_<lowercase>_tool",
]
```

**Verify:** `python -c "from src.tools import build_<lowercase>_tool"` → no error.

---

## Step 5 — Wire enablement into BOTH graph paths

**File:** `src/graph/builder.py`

⚠️ There are **two** wiring functions. You must update both, or the tool
will work on one graph but not the other:

| Function | Used for |
|---|---|
| `_resolve_tools` | Heavy/RAG graph (Chroma retriever + grading) |
| `_resolve_lightweight_tools` | Lightweight web-search graph (used when `web_search_lightweight=True` and sources came from web search) |

**Change in `_resolve_tools`** — append after the existing `if settings.wikipedia_enabled:` block:

```python
if settings.<lowercase>_enabled:
    tools.append(tool_module.build_<lowercase>_tool(settings))
```

**Change in `_resolve_lightweight_tools`** — same line, after the same
existing block. The `tool_module` variable is already in scope in both
functions (it's `import_module("..tools", package=__package__)`).

**Verify:** with the flag enabled, the graph tool list should include your
tool:
```bash
<UPPER>_ENABLED=true python -c "
from src.config import load_settings
from src.graph.builder import _resolve_tools, GraphProviders
s = load_settings()
tools = _resolve_tools(s, GraphProviders(), rebuild_vectorstore=False)
print([t.name for t in tools])
"
```
Output should contain `<tool_name>`.

---

## Step 6 — Tell the LLM the tool exists

**File:** `src/llm/prompts.py`

**Change:** add a bullet to the `TOOLS AVAILABLE WHEN ENABLED:` list inside
`AGENT_SYSTEM_PROMPT`. Without this, even with the tool wired and enabled,
the agent won't know when to call it.

```python
"- <tool_name>: <one-sentence purpose>.\n"
```

If the new tool fits an existing "when to use" category (latest/current,
prices/financial, weather, etc.), the broader guidance already covers it.
If the tool opens a new category (e.g., translation, image analysis), add a
new bullet to the "DEFAULT to calling a tool for..." list too.

**Verify:** `python -c "from src.llm.prompts import AGENT_SYSTEM_PROMPT; assert '<tool_name>' in AGENT_SYSTEM_PROMPT"` → no error.

---

## Step 7 — Document the env var

Three files, all string-only edits:

| File | Edit |
|---|---|
| `.env.example` | Add `<UPPER>_ENABLED=false` next to the other tool flags |
| `config/default.yaml` | Add `<lowercase>_enabled: false` next to the other tool flags |
| `README.md` | If there's a "configuring tools" or env-var section, add the new flag |

**Verify:** `grep <UPPER>_ENABLED .env.example config/default.yaml` should
return both files.

---

## Step 8 — Add tests

**File:** `tests/test_tools.py` (or a new `tests/test_tools_<lowercase>.py`
if you prefer per-tool files).

Minimum test set (the existing tools cover all of these in their own tests):

```python
def test_<lowercase>_input_validation():
    """Pydantic rejects bad input before reaching the callable."""
    from src.tools.<lowercase> import <PascalCase>Input
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        <PascalCase>Input(query="")  # or whatever your constraints reject


def test_<lowercase>_pure_callable_happy_path():
    """The pure callable returns the formatted string."""
    from src.tools.<lowercase> import run_<lowercase>
    fake = lambda url, **kwargs: {"result": "ok"}
    out = run_<lowercase>("hello", requester=fake)
    assert "ok" in out


def test_<lowercase>_pure_callable_error_returns_string():
    """The tool must NOT raise — it returns an error string for the LLM."""
    from src.tools.<lowercase> import run_<lowercase>
    def boom(url, **kwargs):
        raise RuntimeError("network down")
    out = run_<lowercase>("hello", requester=boom)
    assert "failed" in out.lower()
    # crucially: the function returned, did not raise


def test_<lowercase>_factory_returns_structured_tool():
    """The factory produces a real LangChain BaseTool with the right name."""
    from src.config import load_settings
    from src.tools.<lowercase> import build_<lowercase>_tool
    s = load_settings()
    tool = build_<lowercase>_tool(s)
    assert tool.name == "<tool_name>"
    assert tool.description  # non-empty
```

**Verify:** `python -m pytest tests/test_tools.py -k <lowercase> -q`

---

## Step 9 — Final integration check

Run the full suite to make sure nothing else regressed:

```bash
python -m pytest -p no:warnings -q
python -m compileall src tests
git diff --check
```

All three should exit 0 (line-ending warnings on Windows are expected and
harmless).

---

## Worked Example — A Hypothetical "Translate" Tool

To make the playbook concrete, here's what each step looks like if you
were adding a translation tool wrapping a public translation API. Names:
`<lowercase>=translate`, `<tool_name>=translate_text`,
`<PascalCase>=Translate`, `<UPPER>=TRANSLATE`.

| Step | File | What you'd add |
|---|---|---|
| 1 | `src/tools/translate.py` | `TranslateInput`, `build_translate_tool`, `translate_text` (the pure callable) |
| 2 | `src/config/settings.py` | `translate_enabled: bool = False` |
| 3 | `src/config/loader.py` | `"translate_enabled": "TRANSLATE_ENABLED"` in `SETTING_ENV_NAMES` |
| 4 | `src/tools/__init__.py` | `from .translate import TranslateInput, build_translate_tool` + `__all__` entries |
| 5 | `src/graph/builder.py` | `if settings.translate_enabled: tools.append(tool_module.build_translate_tool(settings))` in BOTH `_resolve_tools` and `_resolve_lightweight_tools` |
| 6 | `src/llm/prompts.py` | `"- translate_text: translate text between languages.\n"` in `AGENT_SYSTEM_PROMPT` |
| 7 | `.env.example`, `config/default.yaml` | `TRANSLATE_ENABLED=false` / `translate_enabled: false` |
| 8 | `tests/test_tools.py` | Four tests using the templates in Step 8 |
| 9 | Run the suite | `pytest`, `compileall`, `git diff --check` all green |

---

## Known Issues

| # | Issue | Severity | Location | Fix |
|---|---|---|---|---|
| 1 | Stock tool depends on `yfinance` which is not in `requirements.txt`; raises a clear error on first use | Low | `stock.py:_default_ticker_factory` | Document install: `pip install yfinance`. Acceptable since the tool is opt-in |
| 2 | All tools use synchronous `requests`, blocking the FastAPI event loop when called | Medium | `_http.py` | Routed through `asyncio.to_thread` at the call site (graph executor); future fix is an async path |
| 3 | Wikipedia's default `User-Agent` includes a placeholder asking for contact info | Low | `wikipedia_tool.py` | Set `WIKIPEDIA_USER_AGENT` in `.env` per the Wikipedia API guidelines |
| 4 | No retry on transient HTTP failures; a single 503 fails the tool call | Low | `_http.py` | Wrap with `call_with_retry` from `src/utils/retry.py` if needed |
| 5 | Tool output isn't structured — the LLM has to parse a free-form string | Low | All tools | Acceptable for current models; structured tool outputs could be added if any model supports them |
| 6 | Two wiring sites in `src/graph/builder.py` must be kept in sync manually | Medium | `_resolve_tools` and `_resolve_lightweight_tools` | A `ToolBuilder` Protocol + auto-iteration would eliminate the duplication (see To-Do) |

## Refactoring To-Do List

- [ ] **Tool registry / Protocol** — define a `ToolBuilder.build(settings) -> BaseTool` Protocol and collect builders in a list. Both `_resolve_tools` and `_resolve_lightweight_tools` would then iterate one list, eliminating the duplicated `if settings.X_enabled` blocks (Known Issue #6).
- [ ] **Async HTTP path** — `httpx.AsyncClient` variant of `request_json` so tools don't block the event loop.
- [ ] **Retry wrapper** — add `call_with_retry` around each tool's outer try/except for transient HTTP errors.
- [ ] **Tool result caching** — `(tool_name, args)` → response with TTL for idempotent queries (currency rates, geocoding).
- [ ] **Pyproject extras** — `pip install langgraph-rag[stock]` to opt into `yfinance`.

## Testing Strategy

| Test | Approach |
|---|---|
| Input schema validation | Pydantic raises on invalid input (e.g. unknown ISO code, negative amount) |
| Pure callable | Pass `requester=fake_dict_returner` and assert formatted output |
| Error path | `requester=` raises; assert the tool returns an error string (not raises) |
| Settings injection | Pass `settings.wikipedia_max_summary_chars=200`; assert summary is trimmed |
| Factory shape | Tool name matches, description is non-empty, `args_schema` is set |
| Integration (skipped by default) | Real API call gated behind `@pytest.mark.network` |
| Tests live in | `tests/test_tools.py` (or `tests/test_tools_*.py` if you split per tool) |

## Dependencies

- `src/config/Settings` — feature flags + per-tool config (user agents, char limits, timeouts).
- `src/utils/retry.py` (optional) — `call_with_retry` for transient failure handling.
- External: `requests`, `pydantic`, `langchain_core.tools`. Optional: `yfinance` (stock tool only).
