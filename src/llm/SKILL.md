---
name: llm-architect
description: >
  Use this skill whenever working on LLM provider integration, chat-model
  construction, or prompt design. Covers the DashScope and DeepSeek provider
  implementations, the LLMProvider protocol, prompt templates (RAG, condense,
  grade, agent system prompt), the {current_date} pattern, and source-trust
  instructions. Trigger on mentions of ChatTongyi, DeepSeek, ChatOpenAI,
  qwen-plus, prompts, RAG_PROMPT, CONDENSE_PROMPT, GRADE_PROMPT, prompt
  engineering, or training-cutoff bias.
---

# LLM Architect — src/llm/

Domain: LLM provider seam, chat-model construction, prompt templates.
Parent: `SKILL.md` (root). Siblings: `src/graph/SKILL.md`, `src/rag/SKILL.md`,
`src/web_search/SKILL.md`, `src/api/SKILL.md`, `src/sessions/SKILL.md`,
`src/config/SKILL.md`, `src/tools/SKILL.md`, `src/utils/SKILL.md`.

## Quick Reference

| Fact | Value |
|---|---|
| Provider seam | `LLMProvider` Protocol in `provider.py` |
| Default provider | `DashScopeLLMProvider` (`ChatTongyi`, `qwen-plus`) |
| Alternate provider | `DeepSeekLLMProvider` (`ChatOpenAI`, `deepseek-v4-pro`, OpenAI-compatible) |
| Provider switch | `settings.llm_provider` (`"dashscope"` or `"deepseek"`) |
| Prompts | `RAG_PROMPT`, `CONDENSE_PROMPT`, `GRADE_PROMPT`, `AGENT_SYSTEM_PROMPT` |
| Date convention | `{current_date}` template variable, bound via `.partial()` |
| Date source | `datetime.date.today().isoformat()` (server local clock) |

## File Map

```
src/llm/
├── __init__.py    # Re-exports: build_chat_model, LLMProvider, providers
├── provider.py    # LLMProvider Protocol + DashScope + DeepSeek implementations
└── prompts.py     # RAG_PROMPT, CONDENSE_PROMPT, GRADE_PROMPT, AGENT_SYSTEM_PROMPT
```

### Detailed Responsibilities

| Symbol | Responsibility |
|---|---|
| `LLMProvider` (Protocol) | `chat_model(settings)` returns a configured chat model. Lets nodes depend on a seam, not on `ChatTongyi`. |
| `DashScopeLLMProvider` | Builds `ChatTongyi(model=settings.qwen_model, api_key=..., max_retries=..., model_kwargs={...})`. Honors `dashscope_http_base_url`. |
| `DeepSeekLLMProvider` | Builds `ChatOpenAI(model=settings.deepseek_model, api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url, ...)`. Lazy-imports `langchain_openai`. |
| `build_chat_model(settings, provider=None)` | Entry point used by every graph node. Routes by `settings.llm_provider` when no explicit provider is passed. |
| `RAG_PROMPT` | Final answer generation. Variables: `{current_date}`, `{question}`, `{context}`. |
| `CONDENSE_PROMPT` | Standalone-question rewrite for chat. Variables: `{current_date}`, `{history}`, `{question}`. |
| `GRADE_PROMPT` | Relevance grading on retrieved context. Variables: `{current_date}`, `{context}`, `{question}`. |
| `AGENT_SYSTEM_PROMPT` | Tool-use guidance for the agent. Variable: `{current_date}`. |

## Provider Seam Pattern

```python
# Every graph node calls this — never constructs ChatTongyi directly
from src.llm.provider import build_chat_model

model = build_chat_model(settings)
```

Switching providers is a single env var: `LLM_PROVIDER=deepseek`. No node code
changes. New providers should:

1. Implement `LLMProvider.chat_model(settings) -> Any`.
2. Add a branch in `build_chat_model` (or pass `provider=` explicitly in tests).
3. Add provider-specific settings fields (e.g. `deepseek_api_key`, `deepseek_base_url`) to `src/config/settings.py`.
4. Surface those fields in `SETTING_ENV_NAMES` in `src/config/loader.py` and add `apply_runtime_environment` env wiring if the SDK needs an env-var key.

## Prompt Design Conventions

### 1. Date awareness — every LLM-facing prompt gets today's date

All four prompts include `{current_date}`. The date is **bound at the call
site**, not at module import, so each invocation reflects the actual current
date.

Binding sites (must match this pattern for any new prompt):

| Prompt | Bound at | Pattern |
|---|---|---|
| `RAG_PROMPT` | `src/graph/nodes/common.py` `generate_factory` | `RAG_PROMPT.partial(current_date=date.today().isoformat())` |
| `CONDENSE_PROMPT` | `src/graph/nodes/condense.py` `condense_question_factory` | `CONDENSE_PROMPT.partial(current_date=date.today().isoformat())` |
| `GRADE_PROMPT` | `src/graph/nodes/common.py` `grade_documents_factory` | `GRADE_PROMPT.partial(current_date=date.today().isoformat())` |
| `AGENT_SYSTEM_PROMPT` | wherever the agent prompt is composed | `.format(current_date=date.today().isoformat())` |
| `rewrite_prompt` (inline) | `src/graph/nodes/common.py` `rewrite_factory` | f-string with `date.today().isoformat()` |

For lightweight web answers, see `src/web_search/prompt_builder.py:build_web_search_prompt` which f-strings the date directly via `(today or date.today()).isoformat()`.

### 2. Source-trust language

Every prompt that takes retrieved context tells the model the context may be
**more up to date than its training data** and instructs it to trust the
context on conflict. Phrasing pattern:

> Today's date is {current_date}. The context below was retrieved from
> sources that reflect the current state of the world and may be MORE UP TO
> DATE than your own training data. When the context conflicts with your
> prior knowledge, trust the context. Do not dismiss information as future,
> unreleased, or non-existent merely because it postdates your training
> cutoff.

This wording was added after observing `qwen-plus` (training cutoff ~2024)
silently overriding fresh 2026 source content with stale parametric knowledge.
Don't weaken this language without a verified replacement.

### 3. Semantic matching, not exact-phrase matching

`RAG_PROMPT` and `GRADE_PROMPT` both include explicit "match on meaning, not
exact phrasing" guidance with examples (`'latest model'` vs `'new release'` vs
`'V4 Pro launched in 2026'`). This was added because the grader was rejecting
relevant docs on vocabulary mismatch alone.

### 4. Agent tool-use bias

`AGENT_SYSTEM_PROMPT` defaults the agent toward calling tools whenever the
answer **could have changed since the training cutoff**. The categories
("latest/newest/current", "model names/versions", "prices/stocks",
"weather", "current events", anything with a specific year) are intentionally
broad. Direct-answer is allowed only for math/logic, timeless CS concepts,
chitchat, and long-settled encyclopedic facts.

## Adding a New Prompt — Checklist

1. Define it in `src/llm/prompts.py` with `{current_date}` as a variable.
2. List `current_date` in `input_variables=` (for `PromptTemplate`).
3. At every call site, bind the date via `.partial(current_date=date.today().isoformat())`.
4. Include source-trust language if the prompt takes retrieved context.
5. Include semantic-matching guidance if the prompt grades or generates from context.
6. Add `from datetime import date` to the call-site module.
7. Add a test that confirms the rendered prompt contains today's date.

## Known Issues

| # | Issue | Severity | Location | Fix |
|---|---|---|---|---|
| 1 | `date.today()` reflects server local timezone | Low | Every call site | Switch to `datetime.now(timezone.utc).date().isoformat()` if running across timezones |
| 2 | `DashScopeLLMProvider` uses `model_kwargs["base_address"]` for base URL; not a documented `ChatTongyi` kwarg | Low | `provider.py` | The real override goes through `dashscope.base_http_api_url`; the `base_address` kwarg is a belt-and-braces and is silently ignored if unrecognized |
| 3 | DeepSeek path lazy-imports `langchain_openai`; install isn't enforced by `requirements.txt` unless `LLM_PROVIDER=deepseek` | Low | `provider.py` | Import is wrapped; failure surfaces at provider invocation |
| 4 | Prompts are English-only; UI users may send Chinese questions but the system instructions are English | Medium | `prompts.py` | The model handles cross-lingual prompts, but consider language-aware prompts for production |

## Refactoring To-Do List

- [ ] **Add a prompt registry** so tests can enumerate prompts and assert each one binds `current_date`.
- [ ] **Centralize the source-trust paragraph** as a shared string constant instead of duplicating across `RAG_PROMPT` and `GRADE_PROMPT`.
- [ ] **Add streaming-aware provider methods** (`stream_chat_model`) once `src/graph/executor.py` SSE path needs token streaming.
- [ ] **Add `OpenAILLMProvider`** as a generic OpenAI-compatible provider if more endpoints (Azure OpenAI, local vLLM) are needed.

## Testing Strategy

| Test | Approach |
|---|---|
| Provider switching | Set `LLM_PROVIDER=deepseek` in settings; assert `build_chat_model` returns the DeepSeek class |
| Provider injection | Pass `provider=FakeProvider()` to `build_chat_model`; assert it bypasses the env-based routing |
| Date binding | Bind today's date into each prompt and assert it renders in the formatted output |
| Source-trust copy present | Assert each context-bearing prompt contains the literal "MORE UP TO DATE" phrase |
| Tests live in | `tests/test_graph_state_and_nodes.py`, `tests/test_config.py`, `tests/test_web_search_lightweight_primitives.py` |

## Dependencies

- `src/config/settings.py` — `llm_provider`, `qwen_model`, `dashscope_*`, `deepseek_*`, retry/timeout fields.
- `src/utils/retry.py` — every prompt invocation goes through `invoke_with_retry`.
- External: `langchain-core`, `langchain-community` (DashScope), `langchain-openai` (DeepSeek, optional).
